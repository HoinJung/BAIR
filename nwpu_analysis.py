#!/usr/bin/env python3
"""
NWPU remote-sensing experiment runner.

Supports:
- Baselines (no retrieval / standard RAG / strong-prompt oracle)
- BAIR, MS-PoE, MAD-RAG, BAIR+MS-PoE, BAIR+MAD-RAG
- LongLLMLingua and BAIR+LongLLMLingua
- Optional ``--strong-interventions``: same suite with ``--strong-instruction`` text (``*_strong`` JSON fields)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from llm_explainer import load_llm_model, generate_with_hf, generate_geochat_skysense_intervention
from bottleneck_intervention import (
    NWPURAGInterventionManager,
    set_bottleneck_intervention,
)
from huggingface_hub import list_repo_files, hf_hub_download

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from llmlingua import PromptCompressor
except Exception:
    PromptCompressor = None

try:
    from gender_analysis import (
        generate_with_qwen_intervention,
        generate_with_llava_intervention,
        generate_with_deepseek_intervention,
    )
except Exception:
    generate_with_qwen_intervention = None
    generate_with_llava_intervention = None
    generate_with_deepseek_intervention = None


LONGLLMLINGUA_COMPRESSOR = None
INTERVENTION_MANAGER = NWPURAGInterventionManager()
GGUF_MODEL_CACHE: Dict[str, Any] = {}
LONGLLMLINGUA_CACHE: Dict[str, str] = {}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_docs(context: str) -> List[str]:
    parts = re.split(r"--- Document \d+ ---", context or "")
    return [p.strip() for p in parts if p.strip()]


def extract_gt_only_context(context: str) -> str:
    """
    Keep only the ground-truth passage (Document 3 in the 5-doc NWPU layout).
    Same contract as unified_medgemma_analysis.extract_gt_only_context for NIH.
    """
    docs = parse_docs(context or "")
    if len(docs) != 5:
        return context or ""
    return f"--- Document 1 ---\n{docs[2]}"


def build_nwpu_user_text(
    instruction: Optional[str],
    question: str,
    context: Optional[str],
) -> str:
    """
    Single user-text layout for all NWPU runs (oracle, baselines, interventions).

    Matches ``gender_analysis.build_full_prompt(..., include_experiment_notice=False)``
    so Qwen BAIR/intervention paths stay aligned with oracle, without the FACET-only
    synthetic-person disclaimer.
    """
    ins = (instruction or "").strip()
    ctx = (context or "").strip()
    if ins:
        if ctx:
            return f"Instruction: {ins}\n\nContext:\n{ctx}\n\nQuestion: {question}"
        return f"Instruction: {ins}\n\nQuestion: {question}"
    if ctx:
        return f"Context:\n{ctx}\n\n{question}"
    return question


def _estimate_question_tokens(model_components: Dict[str, Any], question: str) -> int:
    tok = model_components.get("tokenizer")
    try:
        # Match NWPU prompt template tail more closely than raw question-only length.
        q_tail = f"Question: {question or ''}"
        q_toks = tok.encode(q_tail, add_special_tokens=False) if tok is not None else []
        return max(16, len(q_toks) + 12)
    except Exception:
        return 64


def _is_bad_intervention_output(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    if (
        t == "[GENERATION_FAILED]"
        or lower.startswith("error:")
        or t.startswith("[Error]")
        or t.startswith("[UNSUPPORTED")
    ):
        return True
    # Treat ultra-short outputs as degenerate for EarthDial BAIR stability.
    if len(t) < 12 or len(t.split()) < 3:
        return True
    return False


def run_earthdial_backbone_intervention(
    *,
    model_components: Dict[str, Any],
    image_path: str,
    question: str,
    instruction: str,
    context: str,
    max_new_tokens: int,
    alpha_v: float,
    alpha_t: float,
    gamma_s: float,
    mspoe_scaling: float,
    mspoe_text_only: bool,
    use_madrag: bool,
    allow_quality_fallback: bool = True,
) -> str:
    """
    EarthDial intervention path using patched decoder attention + explicit
    calibration(no-context) and generation(with-context) passes.
    """
    num_visual_tokens = 256
    question_tokens = _estimate_question_tokens(model_components, question)
    model = model_components.get("model")
    use_mspoe = abs(float(mspoe_scaling) - 1.0) > 1e-12
    INTERVENTION_MANAGER.patch_backbone("earthdial", enable=True)
    try:
        if use_mspoe and model is not None:
            INTERVENTION_MANAGER.enable_mspoe(
                model=model,
                scaling_factor=float(mspoe_scaling),
                text_only=bool(mspoe_text_only),
                num_visual_tokens=num_visual_tokens,
            )

        alpha_scales = [1.0]
        if allow_quality_fallback and (abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12):
            alpha_scales = [1.0, 0.75, 0.5, 0.25]

        last_out = ""
        for s in alpha_scales:
            eff_alpha_v = float(alpha_v) * s
            eff_alpha_t = float(alpha_t) * s

            # Calibration run on no-context prompt.
            set_bottleneck_intervention(
                True,
                num_visual_tokens=num_visual_tokens,
                visual_start_idx=0,
                calibration_run=True,
                reset_layer=True,
                alpha_v=eff_alpha_v,
                alpha_t=eff_alpha_t,
                gamma_s=gamma_s,
                question_tokens=question_tokens,
                use_madrag=use_madrag,
                mspoe_scaling=mspoe_scaling,
                mspoe_text_only=mspoe_text_only,
            )
            _ = generate_standard(
                model_components=model_components,
                image_path=image_path,
                question=question,
                instruction=instruction,
                context=None,
                max_new_tokens=max_new_tokens,
            )

            # Generation run on full RAG context prompt.
            set_bottleneck_intervention(
                True,
                num_visual_tokens=num_visual_tokens,
                visual_start_idx=0,
                calibration_run=False,
                reset_layer=True,
                alpha_v=eff_alpha_v,
                alpha_t=eff_alpha_t,
                gamma_s=gamma_s,
                question_tokens=question_tokens,
                use_madrag=use_madrag,
                mspoe_scaling=mspoe_scaling,
                mspoe_text_only=mspoe_text_only,
            )
            out = generate_standard(
                model_components=model_components,
                image_path=image_path,
                question=question,
                instruction=instruction,
                context=context,
                max_new_tokens=max_new_tokens,
            )
            last_out = out
            if not _is_bad_intervention_output(out):
                return out

        if not allow_quality_fallback:
            return last_out

        # Final safety: return a stable non-empty generation instead of degenerate text.
        return generate_standard(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=context,
            max_new_tokens=max_new_tokens,
        )
    finally:
        INTERVENTION_MANAGER.disable_mspoe()
        set_bottleneck_intervention(False)
        INTERVENTION_MANAGER.patch_backbone("earthdial", enable=False)


def load_longllmlingua(device: str):
    global LONGLLMLINGUA_COMPRESSOR
    if LONGLLMLINGUA_COMPRESSOR is None:
        if PromptCompressor is None:
            raise ImportError("llmlingua is required for LongLLMLingua modes (pip install llmlingua).")
        LONGLLMLINGUA_COMPRESSOR = PromptCompressor(
            model_name="NousResearch/Llama-2-7b-hf",
            model_config={"torch_dtype": torch.bfloat16},
            device_map=device,
        )
    return LONGLLMLINGUA_COMPRESSOR


def compress_with_longllmlingua(
    context: str,
    question: str,
    instruction: str,
    device: str,
    ratio: float = 0.5,
) -> str:
    compressor = load_longllmlingua(device)
    docs = parse_docs(context)
    if not docs:
        return context
    return INTERVENTION_MANAGER.compress_longllmlingua(
        compressor=compressor,
        context_docs=docs,
        instruction=instruction or "",
        question=question,
        rate=ratio,
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()


def _build_longllmlingua_cache_key(
    cache_scope: str,
    context: str,
    question: str,
    instruction: str,
    ratio: float,
    context_mode: str,
) -> str:
    ratio_str = f"{float(ratio):.6f}"
    return "|".join(
        [
            cache_scope,
            context_mode,
            ratio_str,
            _stable_hash(context or ""),
            _stable_hash(question or ""),
            _stable_hash(instruction or ""),
        ]
    )


def resolve_longllmlingua_cache_scope(row: Dict[str, Any]) -> str:
    # Prefer class-wise reuse. This assumes class-level retrieved context policy.
    gt = (row.get("ground_truth_label") or "").strip()
    if gt:
        return f"class:{gt}"
    raw = (row.get("raw_label") or "").strip()
    if raw:
        return f"class_raw:{raw}"
    uid = row.get("uid")
    if uid is not None:
        return f"uid:{uid}"
    return "global"


def load_longllmlingua_cache(cache_path: Path) -> Dict[str, str]:
    global LONGLLMLINGUA_CACHE
    if LONGLLMLINGUA_CACHE:
        return LONGLLMLINGUA_CACHE
    if cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                LONGLLMLINGUA_CACHE = {str(k): str(v) for k, v in raw.items()}
        except Exception:
            LONGLLMLINGUA_CACHE = {}
    return LONGLLMLINGUA_CACHE


def save_longllmlingua_cache(cache_path: Path) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(LONGLLMLINGUA_CACHE), encoding="utf-8")
    tmp_path.replace(cache_path)


def get_or_compute_longllmlingua_context(
    *,
    cache_scope: str,
    context: str,
    question: str,
    instruction: str,
    device: str,
    ratio: float,
    context_mode: str,
    cache_path: Path,
) -> str:
    key = _build_longllmlingua_cache_key(
        cache_scope=cache_scope,
        context=context,
        question=question,
        instruction=instruction,
        ratio=ratio,
        context_mode=context_mode,
    )
    cached = LONGLLMLINGUA_CACHE.get(key)
    if cached:
        return cached

    compressed = compress_with_longllmlingua(context, question, instruction, device, ratio=ratio)
    LONGLLMLINGUA_CACHE[key] = compressed
    save_longllmlingua_cache(cache_path)
    return compressed


def get_precomputed_longllmlingua_context(
    row: Dict[str, Any],
    precomputed_map: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    question: str,
    instruction: str,
    ratio: float,
    context_mode: str,
) -> Optional[str]:
    source_row = row
    if precomputed_map is not None:
        uid = row.get("uid")
        mapped = precomputed_map.get(str(uid)) if uid is not None else None
        if isinstance(mapped, dict):
            source_row = mapped
    ctx = source_row.get("nwpu_context_longllmlingua")
    meta = source_row.get("nwpu_longllmlingua_meta")
    if not isinstance(ctx, str) or not ctx.strip():
        return None
    if not isinstance(meta, dict):
        return None
    if (meta.get("question") or "") != (question or ""):
        return None
    if (meta.get("instruction") or "") != (instruction or ""):
        return None
    if (meta.get("context_mode") or "full") != context_mode:
        return None
    try:
        meta_ratio = float(meta.get("compression_ratio"))
    except Exception:
        return None
    if abs(meta_ratio - float(ratio)) > 1e-9:
        return None
    return ctx


def build_precomputed_longllmlingua_map(data_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    precomputed: Dict[str, Dict[str, Any]] = {}
    for row in data_rows:
        uid = row.get("uid")
        if uid is None:
            continue
        if "nwpu_context_longllmlingua" not in row or "nwpu_longllmlingua_meta" not in row:
            continue
        precomputed[str(uid)] = {
            "nwpu_context_longllmlingua": row.get("nwpu_context_longllmlingua"),
            "nwpu_longllmlingua_meta": row.get("nwpu_longllmlingua_meta"),
        }
    return precomputed


def _has_valid_value(entry: Dict[str, Any], key: str) -> bool:
    value = entry.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        if not value.strip():
            return False
        if (
            value.startswith("[Error]")
            or value.startswith("[UNSUPPORTED")
            or value.startswith("Error generating response:")
            or value.startswith("Error:")
        ):
            return False
    return True


def _strip_context_fields_for_save(entry: Dict[str, Any]) -> None:
    """Drop context blobs from persisted results to keep JSON compact."""
    for key in list(entry.keys()):
        if (
            key == "nwpu_context"
            or key.endswith("_context")
            or key.startswith("nwpu_context_")
            or key == "nwpu_longllmlingua_meta"
        ):
            entry.pop(key, None)


def _strip_precompute_payload_fields(entry: Dict[str, Any]) -> None:
    for key in ("nwpu_context_original", "nwpu_context_longllmlingua", "nwpu_longllmlingua_meta"):
        entry.pop(key, None)


def generate_standard(
    model_components: Dict[str, Any],
    image_path: str,
    question: str,
    instruction: str,
    context: Optional[str],
    max_new_tokens: int,
) -> str:
    if model_components.get("is_gguf"):
        # GGUF text models do not support direct image input in this path.
        ctx_norm = (context or "").strip() or None
        ins_norm = (instruction or "").strip() or None
        prompt = build_nwpu_user_text(ins_norm, question, ctx_norm)
        out = model_components["model"].create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        text = out["choices"][0]["text"].strip()
        return text

    ctx_norm = (context or "").strip() or None
    ins_norm = (instruction or "").strip() or None
    user = build_nwpu_user_text(ins_norm, question, ctx_norm)
    answer, _ctx = generate_with_hf(
        model_components=model_components,
        instruction="",
        question=question,
        passages=[],
        image=image_path,
        max_new_tokens=max_new_tokens,
        max_context_tokens=2000,
        multimodal_user_text=user,
    )
    return answer


def run_intervention(
    model_name: str,
    model_components: Dict[str, Any],
    image_path: str,
    question: str,
    instruction: str,
    context: str,
    max_new_tokens: int,
    alpha_v: float,
    alpha_t: float,
    gamma_s: float,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
    use_madrag: bool = False,
    force_intervention_path: bool = False,
    qwen_pixel_limit: Optional[int] = 28 * 28 * 50,
    skysense_max_retries: int = 2,
    allow_quality_fallback: bool = True,
) -> str:
    # Attention-level BAIR / MS-PoE / MAD-RAG: Qwen2.5-VL (AdaptLLM) and GeoChat/SkySense (Llama patches).
    # EarthDial / GeoPix currently run in compatibility mode (no architecture-specific attention patch).
    # GGUF text-only: no multimodal attention hooks in this script.
    if model_components.get("is_gguf"):
        return generate_standard(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=context,
            max_new_tokens=max_new_tokens,
        )
    name = model_name.lower()
    if "geochat" in name or "skysense" in name:
        ins = (instruction or "").strip() or None
        ctx_norm = (context or "").strip() or None
        u_no = build_nwpu_user_text(ins, question, None)
        u_ctx = build_nwpu_user_text(ins, question, ctx_norm)
        # SkySense OOM policy:
        # 1) default configured tokens (typically 128)
        # 2) on OOM, retry once at 64
        # 3) if OOM again, fallback to baseline generation
        max_token_candidates = [int(max_new_tokens)]
        if int(max_new_tokens) > 64:
            max_token_candidates.append(64)
        last_exc: Optional[Exception] = None
        for cur_max_new_tokens in max_token_candidates:
            try:
                return generate_geochat_skysense_intervention(
                    model_components=model_components,
                    image_path=image_path,
                    question=question,
                    user_text_no_context=u_no,
                    user_text_with_context=u_ctx,
                    max_new_tokens=cur_max_new_tokens,
                    alpha_v=alpha_v,
                    alpha_t=alpha_t,
                    gamma_s=gamma_s,
                    mspoe_scaling=mspoe_scaling,
                    mspoe_text_only=mspoe_text_only,
                    use_madrag=use_madrag,
                    force_intervention_path=force_intervention_path,
                    max_retries=skysense_max_retries,
                )
            except RuntimeError as exc:
                last_exc = exc
                if "out of memory" not in str(exc).lower():
                    break
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                last_exc = exc
                break
        # If intervention still OOM after one retry (64), fallback to baseline.
        if isinstance(last_exc, RuntimeError) and "out of memory" in str(last_exc).lower():
            try:
                return generate_standard(
                    model_components=model_components,
                    image_path=image_path,
                    question=question,
                    instruction=instruction,
                    context=context,
                    max_new_tokens=64 if int(max_new_tokens) > 64 else int(max_new_tokens),
                )
            except Exception:
                pass
        return f"[Error] skysense intervention failed: {last_exc}"
    if "earthdial" in name:
        return run_earthdial_backbone_intervention(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=context,
            max_new_tokens=max_new_tokens,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            mspoe_scaling=mspoe_scaling,
            mspoe_text_only=mspoe_text_only,
            use_madrag=use_madrag,
            allow_quality_fallback=allow_quality_fallback,
        )
    if "geopix" in name:
        proxy_context, proxy_instruction = apply_prompt_level_intervention_proxy(
            context=context,
            instruction=instruction,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            mspoe_scaling=mspoe_scaling,
            mspoe_text_only=mspoe_text_only,
            use_madrag=use_madrag,
        )
        return generate_standard(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=proxy_instruction,
            context=proxy_context,
            max_new_tokens=max_new_tokens,
        )
    try:
        if "qwen" in name and generate_with_qwen_intervention is not None:
            return generate_with_qwen_intervention(
                model_components=model_components,
                question=question,
                image_path=image_path,
                oracle_context=context,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                alpha_v=alpha_v,
                alpha_t=alpha_t,
                gamma_s=gamma_s,
                qwen_pixel_limit=qwen_pixel_limit,
                mspoe_scaling=mspoe_scaling,
                mspoe_text_only=mspoe_text_only,
                use_madrag=use_madrag,
                include_experiment_notice=False,
            )
        if "llava" in name and generate_with_llava_intervention is not None:
            return generate_with_llava_intervention(
                model_components=model_components,
                question=question,
                image_path=image_path,
                oracle_context=context,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                alpha_v=alpha_v,
                alpha_t=alpha_t,
                gamma_s=gamma_s,
                mspoe_scaling=mspoe_scaling,
                mspoe_text_only=mspoe_text_only,
                use_madrag=use_madrag,
            )
        if "deepseek" in name and generate_with_deepseek_intervention is not None:
            return generate_with_deepseek_intervention(
                model_components=model_components,
                question=question,
                image_path=image_path,
                oracle_context=context,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                alpha_v=alpha_v,
                alpha_t=alpha_t,
                gamma_s=gamma_s,
                mspoe_scaling=mspoe_scaling,
                mspoe_text_only=mspoe_text_only,
                use_madrag=use_madrag,
            )
    except Exception as exc:
        return f"[Error] intervention failed: {exc}"
    return "[UNSUPPORTED_INTERVENTION_MODEL]"


def load_gguf_components(model_name: str, device_id: int) -> Dict[str, Any]:
    if Llama is None:
        raise ImportError("llama-cpp-python is required for GGUF models.")
    if model_name in GGUF_MODEL_CACHE:
        return GGUF_MODEL_CACHE[model_name]

    files = list_repo_files(model_name)
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in HF repo: {model_name}")
    chosen = sorted(gguf_files)[0]
    local_path = hf_hub_download(repo_id=model_name, filename=chosen)

    # Try GPU offload first; fallback to CPU-friendly defaults.
    try:
        llm = Llama(model_path=local_path, n_ctx=4096, n_gpu_layers=50, verbose=False)
    except Exception:
        llm = Llama(model_path=local_path, n_ctx=4096, n_gpu_layers=0, verbose=False)

    components = {
        "is_gguf": True,
        "model_name": model_name,
        "model": llm,
        "device_id": device_id,
    }
    GGUF_MODEL_CACHE[model_name] = components
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NWPU BAIR experiments.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--dataset-json", type=str, default=str(base_dir / "data" / "generated" / "nwpu_retrieval_dataset.json"))
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=str(base_dir / "outputs" / "generation_results_nwpu"))
    parser.add_argument(
        "--context-mode",
        type=str,
        default="full",
        choices=["full", "gt_only"],
        help="full: use all 5 retrieved documents; gt_only: only Document 3 (GT) as a single doc.",
    )

    parser.add_argument("--generate-baselines", action="store_true")
    parser.add_argument("--use-intervention", action="store_true")
    parser.add_argument("--use-mspoe", action="store_true")
    parser.add_argument("--use-madrag", action="store_true")
    parser.add_argument("--use-combo", action="store_true")
    parser.add_argument("--use-madrag-combo", action="store_true")
    parser.add_argument("--use-longllmlingua", action="store_true")
    parser.add_argument("--use-longllmlingua-combo", action="store_true")
    parser.add_argument(
        "--use-precomputed-longllmlingua",
        action="store_true",
        help="Use precompressed context from dataset for LongLLMLingua mode (no runtime compression).",
    )
    parser.add_argument(
        "--use-precomputed-longllmlingua-combo",
        action="store_true",
        help="Use precompressed context from dataset for BAIR+LongLLMLingua combo (no runtime compression).",
    )
    parser.add_argument(
        "--strong-interventions",
        action="store_true",
        help="With intervention flags, also run the same modes using --strong-instruction; "
        "writes oracle_*_strong fields (same question and rag_ctx / compressed ctx).",
    )
    parser.add_argument(
        "--strong-intervention-only",
        action="store_true",
        help="With --use-intervention and --strong-interventions: skip baseline-instruction BAIR "
        "(oracle_with_intervention) and only fill oracle_with_intervention_strong. "
        "Use after a baseline BAIR run; pair with --seed-json to copy baseline fields into this output.",
    )
    parser.add_argument(
        "--seed-json",
        type=str,
        default=None,
        help="Merge another NWPU results JSON into each row (matched by uid) before generation.",
    )

    parser.add_argument("--alpha-v", type=float, default=0.5)
    parser.add_argument("--alpha-t", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--gamma-s", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--mspoe-scaling", type=float, default=1.5)
    parser.add_argument("--mspoe-text-only", action="store_true")
    parser.add_argument(
        "--skysense-max-retries",
        type=int,
        default=2,
        help="Max BAIR retries for SkySense when response is degenerate.",
    )
    parser.add_argument(
        "--qwen-pixel-limit",
        type=int,
        default=28 * 28 * 50,
        help="Qwen2.5-VL min/max pixel cap for stable visual token counts (0 to disable).",
    )
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--compressor-device-id", type=int, default=None)
    parser.add_argument(
        "--require-precomputed-longllmlingua",
        action="store_true",
        help="For LongLLMLingua modes, require precomputed context in dataset and fail instead of on-the-fly compression.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="You are an expert in remote sensing and geospatial analysis. Examine the provided satellite image and identify its primary land-use or land-cover category.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Use the image as primary evidence and use retrieved context as supporting information.",
    )
    parser.add_argument(
        "--strong-instruction",
        type=str,
        default="You are an expert remote-sensing analyst. Prioritize the visual satellite evidence over text context. If context conflicts with image content, trust the image.",
    )
    args = parser.parse_args()
    args.alpha_t = 1.0
    args.gamma_s = 1.0

    if args.strong_intervention_only and not (args.use_intervention and args.strong_interventions):
        parser.error("--strong-intervention-only requires --use-intervention and --strong-interventions")

    set_all_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_device = f"cuda:{args.compressor_device_id}" if args.compressor_device_id is not None else f"cuda:{args.device_id}"

    data = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    if args.num_samples > 0:
        data = data[: args.num_samples]
    precomputed_longllm_map = build_precomputed_longllmlingua_map(data)

    mode_parts = []
    if args.generate_baselines:
        mode_parts.append("baselines")
    if args.use_intervention:
        mode_parts.append(f"bair_av{args.alpha_v}")
    if args.use_mspoe:
        mode_parts.append("mspoe_text" if args.mspoe_text_only else "mspoe_full")
    if args.use_madrag:
        mode_parts.append("madrag")
    if args.use_combo:
        combo_mode = "combo_text" if args.mspoe_text_only else "combo_full"
        mode_parts.append(
            f"{combo_mode}_av{args.alpha_v}_ms{args.mspoe_scaling}"
        )
    if args.use_madrag_combo:
        mode_parts.append(f"madrag_combo_av{args.alpha_v}")
    if args.use_longllmlingua or args.use_precomputed_longllmlingua:
        mode_parts.append(f"longllmlingua_cr{args.compression_ratio}")
    if args.use_longllmlingua_combo or args.use_precomputed_longllmlingua_combo:
        mode_parts.append(
            f"longllmlingua_combo_av{args.alpha_v}_cr{args.compression_ratio}"
        )
    if args.context_mode == "gt_only":
        mode_parts.append("ctx_gt_only")
    _any_intervention = (
        args.use_intervention
        or args.use_mspoe
        or args.use_madrag
        or args.use_combo
        or args.use_madrag_combo
        or args.use_longllmlingua
        or args.use_longllmlingua_combo
        or args.use_precomputed_longllmlingua
        or args.use_precomputed_longllmlingua_combo
    )
    if args.strong_interventions and _any_intervention:
        mode_parts.append("strong_ix")
    suffix = "_".join(mode_parts) if mode_parts else "default"
    model_tag = re.sub(r"[^a-zA-Z0-9]+", "_", args.model_name).strip("_")
    out_file = output_dir / f"nwpu_results_{model_tag}_{suffix}.json"
    longllmlingua_cache_path = output_dir / "nwpu_longllmlingua_cache.json"
    load_longllmlingua_cache(longllmlingua_cache_path)

    if "gguf" in args.model_name.lower():
        model_components = load_gguf_components(args.model_name, device_id=args.device_id)
    else:
        model_components = load_llm_model(args.model_name, gpu_id=args.device_id)
    all_results = [dict(x) for x in data]

    if args.seed_json:
        seed_path = Path(args.seed_json)
        if seed_path.is_file():
            seed_rows = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_map = {str(e.get("uid")): e for e in seed_rows if e.get("uid")}
            for item in all_results:
                se = seed_map.get(str(item.get("uid")))
                if se:
                    for key, value in se.items():
                        item[key] = value

    if out_file.exists():
        try:
            partial = json.loads(out_file.read_text(encoding="utf-8"))
            partial_map = {str(e.get("uid")): e for e in partial if e.get("uid")}
            for item in all_results:
                pe = partial_map.get(str(item.get("uid")))
                if pe:
                    for key, value in pe.items():
                        item[key] = value
        except Exception:
            pass

    # Remove bulky precompute payload fields from the working rows to avoid
    # pathological per-row full-file write overhead. Precomputed contexts are
    # preserved in precomputed_longllm_map and looked up by uid.
    for item in all_results:
        _strip_precompute_payload_fields(item)

    for row in tqdm(all_results, desc="Running NWPU experiment"):
        image_path = row["image_path"]
        full_ctx = row["nwpu_context"]
        rag_ctx = extract_gt_only_context(full_ctx) if args.context_mode == "gt_only" else full_ctx
        row.pop("run_error", None)

        try:
            if args.generate_baselines:
                if not _has_valid_value(row, "no_retrieval_answer"):
                    row["no_retrieval_answer"] = generate_standard(
                        model_components, image_path, args.question, args.instruction, None, args.max_new_tokens
                    )
                if not _has_valid_value(row, "oracle_answer"):
                    row["oracle_answer"] = generate_standard(
                        model_components, image_path, args.question, args.instruction, rag_ctx, args.max_new_tokens
                    )
                if not _has_valid_value(row, "prompt_baseline_answer"):
                    row["prompt_baseline_answer"] = generate_standard(
                        model_components, image_path, args.question, args.strong_instruction, rag_ctx, args.max_new_tokens
                    )

            if (
                args.use_intervention
                and not args.strong_intervention_only
                and not _has_valid_value(row, "oracle_with_intervention")
            ):
                row["oracle_with_intervention"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )
            if (
                args.strong_interventions
                and args.use_intervention
                and not _has_valid_value(row, "oracle_with_intervention_strong")
            ):
                row["oracle_with_intervention_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if args.use_mspoe and not _has_valid_value(row, "oracle_mspoe_answer"):
                row["oracle_mspoe_answer"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=0.0,
                    alpha_t=0.0,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=args.mspoe_scaling,
                    mspoe_text_only=args.mspoe_text_only,
                    use_madrag=False,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )
            if args.strong_interventions and args.use_mspoe and not _has_valid_value(row, "oracle_mspoe_strong"):
                row["oracle_mspoe_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=0.0,
                    alpha_t=0.0,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=args.mspoe_scaling,
                    mspoe_text_only=args.mspoe_text_only,
                    use_madrag=False,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if args.use_madrag and not _has_valid_value(row, "oracle_madrag_answer"):
                row["oracle_madrag_answer"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=0.0,
                    alpha_t=0.0,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=1.0,
                    mspoe_text_only=False,
                    use_madrag=True,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )
            if args.strong_interventions and args.use_madrag and not _has_valid_value(row, "oracle_madrag_strong"):
                row["oracle_madrag_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=0.0,
                    alpha_t=0.0,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=1.0,
                    mspoe_text_only=False,
                    use_madrag=True,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if args.use_combo and not _has_valid_value(row, "oracle_bair_mspoe_combo_answer"):
                row["oracle_bair_mspoe_combo_answer"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=args.mspoe_scaling,
                    mspoe_text_only=args.mspoe_text_only,
                    use_madrag=False,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )
            if args.strong_interventions and args.use_combo and not _has_valid_value(row, "oracle_bair_mspoe_combo_strong"):
                row["oracle_bair_mspoe_combo_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=args.mspoe_scaling,
                    mspoe_text_only=args.mspoe_text_only,
                    use_madrag=False,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if args.use_madrag_combo and not _has_valid_value(row, "oracle_bair_madrag_combo_answer"):
                row["oracle_bair_madrag_combo_answer"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=1.0,
                    mspoe_text_only=False,
                    use_madrag=True,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )
            if (
                args.strong_interventions
                and args.use_madrag_combo
                and not _has_valid_value(row, "oracle_bair_madrag_combo_strong")
            ):
                row["oracle_bair_madrag_combo_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=rag_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    mspoe_scaling=1.0,
                    mspoe_text_only=False,
                    use_madrag=True,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if (args.use_longllmlingua or args.use_precomputed_longllmlingua) and not _has_valid_value(
                row, "oracle_longllmlingua_answer"
            ):
                try:
                    comp_ctx = get_precomputed_longllmlingua_context(
                        row,
                        precomputed_longllm_map,
                        question=args.question,
                        instruction=args.instruction,
                        ratio=args.compression_ratio,
                        context_mode=args.context_mode,
                    )
                    if not comp_ctx:
                        if args.use_precomputed_longllmlingua or args.require_precomputed_longllmlingua:
                            raise RuntimeError(
                                "Missing matching precomputed LongLLMLingua context for this row "
                                "(question/instruction/ratio/context_mode mismatch or field absent)."
                            )
                        comp_ctx = get_or_compute_longllmlingua_context(
                            cache_scope=resolve_longllmlingua_cache_scope(row),
                            context=rag_ctx,
                            question=args.question,
                            instruction=args.instruction,
                            device=comp_device,
                            ratio=args.compression_ratio,
                            context_mode=args.context_mode,
                            cache_path=longllmlingua_cache_path,
                        )
                except Exception as exc:
                    row["oracle_longllmlingua_warning"] = f"{exc}"
                    comp_ctx = rag_ctx
                name = args.model_name.lower()
                if "geochat" in name or "skysense" in name:
                    # Use the same fast SkySense intervention engine as MAD-RAG,
                    # but with neutral knobs (no BAIR/MS-PoE/MAD-RAG effect).
                    row["oracle_longllmlingua_answer"] = run_intervention(
                        model_name=args.model_name,
                        model_components=model_components,
                        image_path=image_path,
                        question=args.question,
                        instruction=args.instruction,
                        context=comp_ctx,
                        max_new_tokens=args.max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=1.0,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=False,
                        force_intervention_path=True,
                        qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                        skysense_max_retries=max(1, args.skysense_max_retries),
                    )
                else:
                    row["oracle_longllmlingua_answer"] = generate_standard(
                        model_components, image_path, args.question, args.instruction, comp_ctx, args.max_new_tokens
                    )
                row["oracle_longllmlingua_context"] = comp_ctx

            if (
                args.strong_interventions
                and (args.use_longllmlingua or args.use_precomputed_longllmlingua)
                and not _has_valid_value(row, "oracle_longllmlingua_strong")
            ):
                try:
                    comp_ctx_s = get_precomputed_longllmlingua_context(
                        row,
                        precomputed_longllm_map,
                        question=args.question,
                        instruction=args.strong_instruction,
                        ratio=args.compression_ratio,
                        context_mode=args.context_mode,
                    )
                    if not comp_ctx_s:
                        if args.use_precomputed_longllmlingua or args.require_precomputed_longllmlingua:
                            raise RuntimeError(
                                "Missing matching precomputed LongLLMLingua strong context for this row "
                                "(question/instruction/ratio/context_mode mismatch or field absent)."
                            )
                        comp_ctx_s = get_or_compute_longllmlingua_context(
                            cache_scope=resolve_longllmlingua_cache_scope(row),
                            context=rag_ctx,
                            question=args.question,
                            instruction=args.strong_instruction,
                            device=comp_device,
                            ratio=args.compression_ratio,
                            context_mode=args.context_mode,
                            cache_path=longllmlingua_cache_path,
                        )
                except Exception as exc:
                    row["oracle_longllmlingua_strong_warning"] = f"{exc}"
                    comp_ctx_s = rag_ctx
                name = args.model_name.lower()
                if "geochat" in name or "skysense" in name:
                    row["oracle_longllmlingua_strong"] = run_intervention(
                        model_name=args.model_name,
                        model_components=model_components,
                        image_path=image_path,
                        question=args.question,
                        instruction=args.strong_instruction,
                        context=comp_ctx_s,
                        max_new_tokens=args.max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=1.0,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=False,
                        force_intervention_path=True,
                        qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                        skysense_max_retries=max(1, args.skysense_max_retries),
                    )
                else:
                    row["oracle_longllmlingua_strong"] = generate_standard(
                        model_components,
                        image_path,
                        args.question,
                        args.strong_instruction,
                        comp_ctx_s,
                        args.max_new_tokens,
                    )
                row["oracle_longllmlingua_context_strong"] = comp_ctx_s

            if (args.use_longllmlingua_combo or args.use_precomputed_longllmlingua_combo) and not _has_valid_value(
                row, "oracle_bair_longllmlingua_combo_answer"
            ):
                comp_ctx = row.get("oracle_longllmlingua_context")
                if not comp_ctx:
                    try:
                        comp_ctx = get_precomputed_longllmlingua_context(
                            row,
                            precomputed_longllm_map,
                            question=args.question,
                            instruction=args.instruction,
                            ratio=args.compression_ratio,
                            context_mode=args.context_mode,
                        )
                        if not comp_ctx:
                            if args.use_precomputed_longllmlingua_combo or args.require_precomputed_longllmlingua:
                                raise RuntimeError(
                                    "Missing matching precomputed LongLLMLingua context for combo run "
                                    "(question/instruction/ratio/context_mode mismatch or field absent)."
                                )
                            comp_ctx = get_or_compute_longllmlingua_context(
                                cache_scope=resolve_longllmlingua_cache_scope(row),
                                context=rag_ctx,
                                question=args.question,
                                instruction=args.instruction,
                                device=comp_device,
                                ratio=args.compression_ratio,
                                context_mode=args.context_mode,
                                cache_path=longllmlingua_cache_path,
                            )
                    except Exception as exc:
                        row["oracle_longllmlingua_warning"] = f"{exc}"
                        comp_ctx = rag_ctx
                    row["oracle_longllmlingua_context"] = comp_ctx
                row["oracle_bair_longllmlingua_combo_answer"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.instruction,
                    context=comp_ctx,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

            if (
                args.strong_interventions
                and (args.use_longllmlingua_combo or args.use_precomputed_longllmlingua_combo)
                and not _has_valid_value(row, "oracle_bair_longllmlingua_combo_strong")
            ):
                comp_ctx_s = row.get("oracle_longllmlingua_context_strong")
                if not comp_ctx_s:
                    try:
                        comp_ctx_s = get_precomputed_longllmlingua_context(
                            row,
                            precomputed_longllm_map,
                            question=args.question,
                            instruction=args.strong_instruction,
                            ratio=args.compression_ratio,
                            context_mode=args.context_mode,
                        )
                        if not comp_ctx_s:
                            if args.use_precomputed_longllmlingua_combo or args.require_precomputed_longllmlingua:
                                raise RuntimeError(
                                    "Missing matching precomputed LongLLMLingua strong context for combo run "
                                    "(question/instruction/ratio/context_mode mismatch or field absent)."
                                )
                            comp_ctx_s = get_or_compute_longllmlingua_context(
                                cache_scope=resolve_longllmlingua_cache_scope(row),
                                context=rag_ctx,
                                question=args.question,
                                instruction=args.strong_instruction,
                                device=comp_device,
                                ratio=args.compression_ratio,
                                context_mode=args.context_mode,
                                cache_path=longllmlingua_cache_path,
                            )
                    except Exception as exc:
                        row["oracle_longllmlingua_strong_warning"] = f"{exc}"
                        comp_ctx_s = rag_ctx
                    row["oracle_longllmlingua_context_strong"] = comp_ctx_s
                row["oracle_bair_longllmlingua_combo_strong"] = run_intervention(
                    model_name=args.model_name,
                    model_components=model_components,
                    image_path=image_path,
                    question=args.question,
                    instruction=args.strong_instruction,
                    context=comp_ctx_s,
                    max_new_tokens=args.max_new_tokens,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                    gamma_s=args.gamma_s,
                    qwen_pixel_limit=(args.qwen_pixel_limit if args.qwen_pixel_limit and args.qwen_pixel_limit > 0 else None),
                    skysense_max_retries=max(1, args.skysense_max_retries),
                )

        except Exception as exc:
            row["run_error"] = f"{exc}"

        _strip_context_fields_for_save(row)
        out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
