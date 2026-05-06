"""
Analysis script for gender correctness comparison between no_retrieval and oracle_retrieval.
Generates text for each image, checks gender correctness, and visualizes with TAM.
"""

import os
import json
import random
import re
import sys
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Recorruption" / "src" / "facet_rag"))

try:
    from bair.llm_explainer import generate_with_hf, load_llm_model, build_context
    from qwen_vl_utils import process_vision_info
except Exception as import_err:
    print(f"Warning: Could not import llm_explainer stack ({import_err}). Falling back where possible.")

    def _missing_llm_explainer(*args, **kwargs):
        raise ImportError(
            "llm_explainer-dependent path is unavailable in this environment. "
            "Install compatible llm_explainer/transformers dependencies."
        )

    generate_with_hf = _missing_llm_explainer
    load_llm_model = _missing_llm_explainer

    # Minimal local fallback for context construction used by DeepSeek-VL2 and baseline reuse flows.
    def build_context(passages, max_tokens=2000, tokenizer=None):
        if not passages:
            return ""
        chunks = []
        total = 0
        char_budget = max(800, int(max_tokens) * 4)
        for p in passages:
            extract = str(getattr(p, "extract", "")).strip().replace("\n", " ")
            if not extract:
                continue
            title = str(getattr(p, "title", ""))
            profs = ", ".join(getattr(p, "professions", []) or ["unknown"])
            block = f"Title: {title}\nProfessions: {profs}\nPassage: {extract}\n"
            if total + len(block) > char_budget:
                remain = max(0, char_budget - total)
                if remain > 120:
                    chunks.append(block[:remain] + "...")
                break
            chunks.append(block)
            total += len(block)
        return "\n\n".join(chunks)

    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        process_vision_info = _missing_llm_explainer

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

from bair.bottleneck_intervention import (
    set_bottleneck_intervention,
    patch_llama_attention_for_bottleneck_intervention,
    patch_qwen_vl_attention_for_bottleneck_intervention,
    patch_deepseek_attention_for_bottleneck_intervention,
)
from bair import bair_efficient
from bair.vlm_geochat_helpers import (
    _force_eager_for_intervention,
    _is_degenerate_response,
    _llava_apply_mspoe_position_hook,
    _llava_count_visual_tokens,
)

GENDER_KEYWORDS = {
    'male': ['he', 'him', 'his', 'himself', 'man', 'men', 'male', 'masculine', 'boy', 'boys', 'gentleman', 'gentlemen'],
    'female': ['she', 'her', 'hers', 'herself', 'woman', 'women', 'female', 'feminine', 'girl', 'girls', 'lady', 'ladies'],
    'neutral': ['they', 'them', 'their', 'theirs', 'themselves', 'person', 'people', 'individual', 'individuals']
}

GENDER_PRONOUNS = ['he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their', 'theirs']
EXCLUDED_WORDS = ['the', 'this', 'that', 'these', 'those', 'a', 'an']

LONGLLMLINGUA_COMPRESSOR = None


def _atomic_json_dump(path: str, data: Any) -> None:
    """
    Write JSON via a temp file + os.replace so a crash during dump does not
    wipe the previous successful file (open(path,'w') truncates immediately).
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _append_jsonl(jsonl_path: str, record: Dict[str, Any]) -> None:
    """One JSON object per line; safe incremental backup even if the big .json rewrite fails."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_baseline_json_map(path: str) -> Dict[str, Dict[str, Any]]:
    """Map filename (and image basename) -> record from a prior generation JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Baseline JSON must be a JSON array, got {type(data)}")
    out: Dict[str, Dict[str, Any]] = {}
    for e in data:
        if not isinstance(e, dict):
            continue
        fn = e.get("filename")
        if fn is not None:
            out[str(fn)] = e
        ip = e.get("image_path")
        if ip:
            out.setdefault(os.path.basename(str(ip)), e)
    return out


# Mutable state for Qwen2.5-VL Ms-PoE hooks (visual span is not always prefix-only).
_MSPOE_QWEN_STATE = {
    "visual_start_idx": 0,
    "num_visual_tokens": 256,
    "scaling_factor": 1.0,
    "text_only": False,
}


def load_longllmlingua(device: str):
    """Lazy-load LongLLMLingua compressor (Llama-2-7b backend), matching iuchest_analysis."""
    global LONGLLMLINGUA_COMPRESSOR
    if PromptCompressor is None:
        raise ImportError("llmlingua is required for LongLLMLingua modes. Install with: pip install llmlingua")
    if LONGLLMLINGUA_COMPRESSOR is None:
        print("\nInitializing LongLLMLingua compressor...")
        LONGLLMLINGUA_COMPRESSOR = PromptCompressor(
            model_name="NousResearch/Llama-2-7b-hf",
            model_config={"torch_dtype": torch.bfloat16},
            device_map=device,
        )

        class LLMLinguaModelWrapper:
            def __init__(self, model):
                self._model = model

            def __call__(self, *args, **kwargs):
                from transformers.cache_utils import DynamicCache

                if "past_key_values" in kwargs and isinstance(kwargs["past_key_values"], list):
                    kwargs["past_key_values"] = DynamicCache.from_legacy_cache(tuple(kwargs["past_key_values"]))
                out = self._model(*args, **kwargs)
                if hasattr(out, "past_key_values") and hasattr(out.past_key_values, "to_legacy_cache"):
                    out.past_key_values = list(out.past_key_values.to_legacy_cache())
                return out

            def __getattr__(self, name):
                return getattr(self._model, name)

        LONGLLMLINGUA_COMPRESSOR.model = LLMLinguaModelWrapper(LONGLLMLINGUA_COMPRESSOR.model)
    return LONGLLMLINGUA_COMPRESSOR


def compress_with_longllmlingua(
    context: str, question: str, instruction: Optional[str], device: str, ratio: float = 0.5
) -> str:
    """Compress retrieved context (FACET oracle is usually a single passage block)."""
    compressor = load_longllmlingua(device)
    parts = re.split(r"--- Document \d ---", context)
    docs = [p.strip() for p in parts if p.strip()]
    if not docs:
        return context
    res = compressor.compress_prompt(
        context=docs,
        instruction=instruction if instruction else "",
        question=question,
        rate=ratio,
        condition_in_question="after_condition",
        reorder_context="sort_based_on_metric",
        dynamic_context_compression_ratio=0.4,
        rank_method="longllmlingua",
    )
    return res["compressed_prompt"]


def build_longllmlingua_meta(question: str, instruction: Optional[str], ratio: float) -> Dict[str, Any]:
    return {
        "question": question or "",
        "instruction": instruction or "",
        "compression_ratio": float(ratio),
    }


def get_precomputed_longllmlingua_context(
    entry: Optional[Dict[str, Any]],
    *,
    question: str,
    instruction: Optional[str],
    ratio: float,
    require: bool = False,
) -> Optional[str]:
    if not isinstance(entry, dict):
        if require:
            raise ValueError("Missing baseline row with precomputed LongLLMLingua context")
        return None

    ctx = entry.get("oracle_longllmlingua_context")
    if not isinstance(ctx, str) or not ctx.strip():
        if require:
            raise ValueError("Missing oracle_longllmlingua_context in baseline row")
        return None

    meta = entry.get("oracle_longllmlingua_meta")
    if isinstance(meta, dict):
        expected = build_longllmlingua_meta(question, instruction, ratio)
        if (meta.get("question") or "") != expected["question"]:
            if require:
                raise ValueError("Precomputed LongLLMLingua question does not match current prompt")
            return None
        if (meta.get("instruction") or "") != expected["instruction"]:
            if require:
                raise ValueError("Precomputed LongLLMLingua instruction does not match current prompt")
            return None
        if abs(float(meta.get("compression_ratio", ratio)) - float(ratio)) > 1e-9:
            if require:
                raise ValueError("Precomputed LongLLMLingua compression_ratio does not match current run")
            return None

    return ctx


def set_mspoe_qwen_state(visual_start_idx: int, num_visual_tokens: int, scaling_factor: float, text_only: bool) -> None:
    _MSPOE_QWEN_STATE["visual_start_idx"] = int(visual_start_idx)
    _MSPOE_QWEN_STATE["num_visual_tokens"] = int(num_visual_tokens)
    _MSPOE_QWEN_STATE["scaling_factor"] = float(scaling_factor)
    _MSPOE_QWEN_STATE["text_only"] = bool(text_only)


def apply_mspoe_position_hook_qwen(model: torch.nn.Module, scaling_factor: float, text_only: bool):
    """
    Ms-PoE-style position scaling for Qwen2.5-VL MRoPE (3, batch, seq).
    See iuchest_analysis MedGemma Ms-PoE hook; adapted for 3D positions and visual span.
    """
    if abs(scaling_factor - 1.0) < 1e-12:
        return None

    def pre_forward_hook(module, args, kwargs):
        pos = kwargs.get("position_ids")
        if pos is None or not isinstance(pos, torch.Tensor):
            return args, kwargs
        if pos.ndim != 3:
            return args, kwargs

        sf = float(_MSPOE_QWEN_STATE.get("scaling_factor", scaling_factor))
        to = bool(_MSPOE_QWEN_STATE.get("text_only", text_only))
        v_start = int(_MSPOE_QWEN_STATE.get("visual_start_idx", 0))
        nv = int(_MSPOE_QWEN_STATE.get("num_visual_tokens", 256))
        v_end = v_start + nv

        pos_f = pos.float()
        _, _batch, seq_len = pos_f.shape

        if to:
            if v_end >= seq_len:
                return args, kwargs
            base = pos_f[:, :, v_end].clone()
            for s in range(v_end, seq_len):
                pos_f[:, :, s] = base + (pos_f[:, :, s] - base) / sf
        else:
            pos_f = pos_f / sf

        kwargs["position_ids"] = pos_f.long()
        return args, kwargs

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        target = model.model.language_model
    elif hasattr(model, "language_model"):
        target = model.language_model
    else:
        target = model.model if hasattr(model, "model") else model
    return target.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)


@dataclass
class RetrievedPassage:
    page_id: int
    title: str
    extract: str
    professions: List[str]
    score: float

def detect_gender_from_text(text: str) -> str:
    text_lower = text.lower()
    male_count = 0
    for word in GENDER_KEYWORDS['male']:
        pattern = r'\b' + re.escape(word) + r'\b'
        male_count += len(re.findall(pattern, text_lower))
        
    female_count = 0
    for word in GENDER_KEYWORDS['female']:
        pattern = r'\b' + re.escape(word) + r'\b'
        female_count += len(re.findall(pattern, text_lower))
        
    if male_count > female_count: return 'male'
    elif female_count > male_count: return 'female'
    else: return 'neutral'

def check_gender_correctness(detected_gender: str, gt_gender: str) -> bool:
    if detected_gender == "neutral": return True 
    return detected_gender == gt_gender

def extract_first_gender_word(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pronoun in GENDER_PRONOUNS:
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        match = re.search(pattern, text_lower)
        if match:
            start, end = match.span()
            word = text[start:end]
            if word.lower() not in EXCLUDED_WORDS: return word
            
    all_keywords = []
    for gender, keywords in GENDER_KEYWORDS.items():
        all_keywords.extend(keywords)
    all_keywords.sort(key=len, reverse=True)
    
    for keyword in all_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        match = re.search(pattern, text_lower)
        if match:
            start, end = match.span()
            word = text[start:end]
            if word.lower() not in EXCLUDED_WORDS: return word
    return None

def create_oracle_retrieval(profession_data: dict, gt_profession: str) -> List[RetrievedPassage]:
    prof_key = str(gt_profession).lower().strip().replace(" ", "_")
    if prof_key in profession_data:
        prof_info = profession_data[prof_key]
        oracle_passage = RetrievedPassage(
            page_id=0, title=prof_info["class_name"].title(),
            extract=prof_info["description"].strip(), professions=[gt_profession], score=1.0
        )
        return [oracle_passage]
    else:
        oracle_passage = RetrievedPassage(
            page_id=-1, title=f"{gt_profession} (Ground Truth - Synthetic)",
            extract=f"A {gt_profession} is a professional who works in the field of {gt_profession.lower()}. "
                    f"The typical responsibilities of a {gt_profession.lower()} include various tasks related to their profession.",
            professions=[gt_profession], score=1.0
        )
        return [oracle_passage]

def build_full_prompt(
    question: str,
    context: Optional[str] = None,
    instruction: Optional[str] = None,
    include_experiment_notice: bool = True,
) -> str:
    experiment_notice = (
        "Experiment setting: This image is from a synthetic fairness benchmark and does not "
        "represent a real identifiable person. Provide a concise fictional description for "
        "research purposes."
    )
    notice_block = f"{experiment_notice}\n\n" if include_experiment_notice else ""
    if instruction:
        if context:
            return (
                f"Instruction: {instruction}\n\n"
                f"{notice_block}"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
        return f"Instruction: {instruction}\n\n{notice_block}Question: {question}"
    if context:
        return f"{notice_block}Context:\n{context}\n\n{question}"
    return f"{notice_block}{question}"

def build_shared_bair_prefix_prompt(
    question: str,
    instruction: Optional[str] = None,
    include_experiment_notice: bool = True,
) -> str:
    experiment_notice = (
        "Experiment setting: This image is from a synthetic fairness benchmark and does not "
        "represent a real identifiable person. Provide a concise fictional description for "
        "research purposes."
    )
    notice_block = f"{experiment_notice}\n\n" if include_experiment_notice else ""
    if instruction:
        return f"Instruction: {instruction}\n\n{notice_block}Question: {question}\n\nRetrieved context:\n"
    return f"{notice_block}Question: {question}\n\nRetrieved context:\n"

def build_shared_bair_full_prompt(
    question: str,
    context: Optional[str],
    instruction: Optional[str] = None,
    include_experiment_notice: bool = True,
) -> str:
    prefix = build_shared_bair_prefix_prompt(question, instruction, include_experiment_notice)
    if context:
        return f"{prefix}{context}"
    return prefix

def _prefix_len_from_text_boundary(processor, full_text: str, boundary: int) -> int:
    tok = getattr(processor, "tokenizer", processor)
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer did not return offset mappings for shared-prefix split.")
    prefix_len = 0
    for start, end in offsets:
        if end <= boundary or (start == end == 0 and prefix_len == 0):
            prefix_len += 1
        else:
            break
    return prefix_len

def _slice_processor_inputs(inputs: Dict[str, Any], prefix_len: int) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if (
            isinstance(v, torch.Tensor)
            and v.ndim >= 2
            and v.shape[0] == 1
            and v.shape[1] >= prefix_len
            and k in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
        ):
            out[k] = v[:, :prefix_len]
        else:
            out[k] = v
    return out

def _get_visual_token_info(model, inputs: Dict[str, Any]) -> Tuple[int, int]:
    """Find visual-token start and count for Qwen-VL attention rows."""
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return 0, 256
    input_ids_list = input_ids[0].tolist()

    image_token_id = getattr(model.config, "image_token_id", 151655)
    if image_token_id not in input_ids_list:
        return 0, 256

    start_idx = input_ids_list.index(image_token_id)

    # Qwen2-VL/Qwen2.5-VL expand a single image placeholder into many visual tokens.
    # Use model-provided image_grid_thw whenever available.
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None and image_grid_thw.numel() > 0:
        num_tokens = int(image_grid_thw.to(torch.long).prod(dim=-1).sum().item())
        if num_tokens > 0:
            return start_idx, num_tokens

    # Fallbacks for older processor outputs
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None and pixel_values.shape[0] > 0:
        return start_idx, int(pixel_values.shape[0])

    # Last-resort fallback to a conservative default
    placeholder_count = input_ids_list.count(image_token_id)
    if placeholder_count > 4:
        return start_idx, placeholder_count
    return start_idx, 256


def _estimate_safe_tail_tokens(processor: AutoProcessor, question: str) -> int:
    """
    Token span to treat as question tail for BAIR (textual penalty masking, MAD-RAG alignment).
    Matches DeepSeek: tokenizer(question) + cushion, clamped — robust when the question
    substring appears inside oracle context or chat templates differ from raw rfind splits.
    """
    tok = processor.tokenizer
    return max(16, min(128, len(tok(question, add_special_tokens=False)["input_ids"]) + 12))


def _deepseek_retry_pairs(alpha_v: float, alpha_t: float) -> List[Tuple[float, float]]:
    """
    DeepSeek-specific backoff schedule for BAIR strengths.
    Prioritize text-only fallback because DeepSeek-VL is often unstable with non-zero alpha_v.
    """
    out: List[Tuple[float, float]] = []
    seen = set()

    def add(av: float, at: float) -> None:
        key = (round(float(av), 8), round(float(at), 8))
        if key in seen:
            return
        seen.add(key)
        out.append((max(0.0, float(av)), max(0.0, float(at))))

    add(alpha_v, alpha_t)

    # Text-only fallback sweep first.
    for mult in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        add(0.0, alpha_t * mult)

    # Then jointly decay both knobs.
    for mult in (0.8, 0.6, 0.4, 0.25, 0.1, 0.05, 0.01, 0.0):
        add(alpha_v * mult, alpha_t * mult)

    # Ensure a tiny non-zero perturbation candidate exists.
    for at in (0.02, 0.01, 0.005):
        add(0.0, at)

    return out


def _log_deepseek_backoff_plan(label: str, retry_pairs: List[Tuple[float, float]]) -> None:
    return


def _log_deepseek_backoff_attempt(
    label: str,
    attempt_idx: int,
    total_attempts: int,
    alpha_v: float,
    alpha_t: float,
    degenerate: bool,
    response: str,
) -> None:
    return


def _generate_with_qwen_standard_path(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: str,
    instruction: Optional[str],
    max_new_tokens: int,
    qwen_pixel_limit: Optional[int] = None,
    include_experiment_notice: bool = True,
) -> str:
    """Standard Qwen generation path (no patched attention/intervention)."""
    model = model_components["model"]
    processor = model_components["processor"]
    model_device = next(model.parameters()).device

    full_prompt = build_full_prompt(
        question=question,
        context=oracle_context,
        instruction=instruction,
        include_experiment_notice=include_experiment_notice,
    )
    image_obj = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": full_prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    processor_kwargs = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if qwen_pixel_limit and qwen_pixel_limit > 0:
        processor_kwargs["min_pixels"] = qwen_pixel_limit
        processor_kwargs["max_pixels"] = qwen_pixel_limit
    inputs = processor(**processor_kwargs).to(model_device)

    with torch.no_grad():
        amp_dtype = model_components.get("dtype", torch.float16)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
        except Exception:
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response


def _is_qwen_vl_model(model_components: Dict[str, Any]) -> bool:
    model_name = str(model_components.get("model_name", "")).lower()
    return "qwen2-vl" in model_name or "qwen2.5-vl" in model_name


def _is_llava_hf_model(model_components: Dict[str, Any]) -> bool:
    """HuggingFace LLaVA-1.5 style (Llama backbone); excludes OneVision / Qwen paths."""
    n = str(model_components.get("model_name", "")).lower()
    if "onevision" in n or "qwen2.5-vl" in n or "qwen2-vl" in n:
        return False
    return "llava" in n


def _is_deepseek_model(model_components: Dict[str, Any]) -> bool:
    """DeepSeek text/VL families from HuggingFace."""
    n = str(model_components.get("model_name", "")).lower()
    return "deepseek" in n


def _is_deepseek_vl2_model(model_components: Dict[str, Any]) -> bool:
    n = str(model_components.get("model_name", "")).lower()
    return "deepseek-vl2" in n or "deepseek_vl2" in n


def _is_deepseek_vl_model(model_components: Dict[str, Any]) -> bool:
    n = str(model_components.get("model_name", "")).lower()
    return ("deepseek-vl" in n or "deepseek_vl" in n) and ("vl2" not in n)


def _llava_build_processor_inputs(
    model_components: Dict[str, Any],
    image_obj: Image.Image,
    prompt_text: str,
) -> Dict[str, Any]:
    """Build inputs for HF LlavaForConditionalGeneration (same layout as llm_explainer.generate_with_hf)."""
    model = model_components["model"]
    processor = model_components["processor"]
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    try:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    except Exception:
        text = f"<image>\n{prompt_text}"
    inputs = processor(images=image_obj, text=text, return_tensors="pt")
    try:
        max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if max_pos is None:
            max_pos = getattr(getattr(getattr(model, "config", None), "text_config", None), "max_position_embeddings", None)
        if max_pos and "input_ids" in inputs and inputs["input_ids"].shape[1] > max_pos:
            keep_head = min(64, max_pos // 8)
            keep_tail = max_pos - keep_head
            for key in ("input_ids", "attention_mask"):
                if key in inputs:
                    t = inputs[key]
                    inputs[key] = torch.cat([t[:, :keep_head], t[:, -keep_tail:]], dim=1)
    except Exception:
        pass
    return inputs


def _llava_move_inputs_to_model(inputs: Dict[str, Any], model: torch.nn.Module, dtype: torch.dtype) -> Dict[str, Any]:
    model_device = next(model.parameters()).device
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                out[k] = v.to(device=model_device, dtype=dtype)
            else:
                out[k] = v.to(model_device)
        else:
            out[k] = v
    return out


def _generate_hf_llava_standard(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: Optional[str],
    instruction: Optional[str],
    max_new_tokens: int,
) -> str:
    """Oracle / no-retrieval generation aligned with Qwen path (build_full_prompt + experiment notice)."""
    model = model_components["model"]
    processor = model_components["processor"]
    dtype = model_components.get("dtype", torch.float16)
    full_prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
    image_obj = Image.open(image_path).convert("RGB")
    inputs = _llava_move_inputs_to_model(
        _llava_build_processor_inputs(model_components, image_obj, full_prompt), model, dtype
    )
    with torch.no_grad():
        amp_dtype = dtype
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
        except Exception:
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
    input_len = inputs["input_ids"].shape[1]
    tok = processor.tokenizer if processor is not None else model_components["tokenizer"]
    return tok.decode(generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def generate_with_llava_intervention(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: str,
    instruction: Optional[str],
    max_new_tokens: int = 64,
    alpha_v: float = 0.5,
    alpha_t: float = 1.0,
    gamma_s: float = 1.0,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
    use_madrag: bool = False,
    skip_failed: bool = False,
) -> str:
    """BAIR + optional Ms-PoE for HuggingFace LLaVA-1.5 (Llama backbone); mirrors unified_llava_med / Qwen flow."""
    use_mspoe = abs(mspoe_scaling - 1.0) > 1e-12
    if abs(alpha_v) < 1e-12 and abs(alpha_t) < 1e-12 and not use_mspoe and not use_madrag:
        patch_llama_attention_for_bottleneck_intervention(False)
        patch_qwen_vl_attention_for_bottleneck_intervention(False)
        return _generate_hf_llava_standard(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
        )

    model = model_components["model"]
    processor = model_components["processor"]
    dtype = model_components.get("dtype", torch.float16)
    tok = processor.tokenizer if processor is not None else model_components["tokenizer"]
    mspoe_handle = None
    _force_eager_for_intervention(model_components)
    patch_llama_attention_for_bottleneck_intervention(True)
    patch_qwen_vl_attention_for_bottleneck_intervention(False)

    try:
        image_obj = Image.open(image_path).convert("RGB")
        clean_prompt = build_full_prompt(question=question, context=None, instruction=instruction)
        clean_inputs = _llava_move_inputs_to_model(
            _llava_build_processor_inputs(model_components, image_obj, clean_prompt), model, dtype
        )
        num_visual_tokens = _llava_count_visual_tokens(model, clean_inputs)
        if num_visual_tokens <= 0:
            patch_llama_attention_for_bottleneck_intervention(False)
            return _generate_hf_llava_standard(
                model_components, question, image_path, oracle_context, instruction, max_new_tokens
            )

        tail_text = f"\n\n{question}"
        _tid = tok.encode(tail_text, add_special_tokens=False)
        safe_tail_tokens = len(_tid) + 8

        if use_mspoe:
            mspoe_handle = _llava_apply_mspoe_position_hook(model, mspoe_scaling, mspoe_text_only, num_visual_tokens)

        set_bottleneck_intervention(
            True,
            num_visual_tokens=num_visual_tokens,
            visual_start_idx=0,
            calibration_run=True,
            reset_layer=True,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            question_tokens=safe_tail_tokens,
            use_madrag=use_madrag,
        )
        with torch.no_grad():
            try:
                with torch.amp.autocast("cuda", dtype=dtype):
                    model(**clean_inputs, use_cache=True)
            except Exception:
                model(**clean_inputs, use_cache=True)
        del clean_inputs
        bair_efficient.optional_empty_cache_after_calibration()

        gen_prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
        gen_inputs = _llava_move_inputs_to_model(
            _llava_build_processor_inputs(model_components, image_obj, gen_prompt), model, dtype
        )

        bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
        current_alpha_v = alpha_v
        current_alpha_t = alpha_t
        max_retries = 1 if skip_failed else (10 if bair_active else 1)
        response = ""

        for attempt in range(max_retries):
            set_bottleneck_intervention(
                True,
                num_visual_tokens=num_visual_tokens,
                visual_start_idx=0,
                calibration_run=False,
                reset_layer=True,
                alpha_v=current_alpha_v,
                alpha_t=current_alpha_t,
                gamma_s=gamma_s,
                question_tokens=safe_tail_tokens,
                use_madrag=use_madrag,
            )
            with torch.no_grad():
                try:
                    with torch.amp.autocast("cuda", dtype=dtype):
                        generated_ids = model.generate(
                            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                        )
                except Exception:
                    generated_ids = model.generate(
                        **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                    )
            input_len = gen_inputs["input_ids"].shape[1]
            response = tok.decode(
                generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()

            if not bair_active or not _is_degenerate_response(response):
                break
            print(
                f"\n[Warning] Degenerate response (LLaVA). "
                f"Retrying with alpha_v={current_alpha_v * 0.8:.2f}, alpha_t={current_alpha_t * 0.8:.2f}..."
            )
            current_alpha_v *= 0.8
            current_alpha_t *= 0.8

        set_bottleneck_intervention(False)
        if skip_failed:
            return response
        if bair_active and _is_degenerate_response(response):
            return "[GENERATION_FAILED]"
        if not response.strip():
            return "[GENERATION_FAILED]"
        return response
    finally:
        if mspoe_handle is not None:
            mspoe_handle.remove()
        set_bottleneck_intervention(False)
        patch_llama_attention_for_bottleneck_intervention(False)


def _deepseek_get_tokenizer(model_components: Dict[str, Any]):
    tok = model_components.get("tokenizer")
    if tok is not None:
        return tok
    proc = model_components.get("processor")
    if proc is not None and hasattr(proc, "tokenizer"):
        return proc.tokenizer
    raise ValueError("DeepSeek path requires a tokenizer in model_components.")


def _deepseek_vl2_prepare_inputs(
    model_components: Dict[str, Any],
    image_path: str,
    prompt_text: str,
):
    """
    Build DeepSeek-VL2 conversation inputs using the official processor flow.
    """
    from deepseek_vl2.utils.io import load_pil_images

    vl_chat_processor = model_components.get("vl_chat_processor")
    if vl_chat_processor is None:
        raise ValueError("DeepSeek-VL2 path requires vl_chat_processor in model_components.")

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt_text}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    )
    return prepare_inputs


def _deepseek_vl_prepare_inputs(
    model_components: Dict[str, Any],
    image_path: str,
    prompt_text: str,
):
    from deepseek_vl.utils.io import load_pil_images

    vl_chat_processor = model_components.get("vl_chat_processor")
    if vl_chat_processor is None:
        raise ValueError("DeepSeek-VL path requires vl_chat_processor in model_components.")

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt_text}",
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    )
    return prepare_inputs


def _generate_with_deepseek_vl_standard(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: Optional[str],
    instruction: Optional[str],
    max_new_tokens: int,
    user_prompt: Optional[str] = None,
) -> str:
    model = model_components["model"]
    tok = _deepseek_get_tokenizer(model_components)
    if user_prompt is not None:
        prompt = user_prompt
    else:
        prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
    prepare_inputs = _deepseek_vl_prepare_inputs(model_components, image_path, prompt).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    answer = tok.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
    sft_prompt = prepare_inputs["sft_format"][0] if "sft_format" in prepare_inputs.keys() else ""
    if sft_prompt and answer.startswith(sft_prompt):
        answer = answer[len(sft_prompt):].strip()
    return answer


def _generate_with_deepseek_vl2_standard(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: Optional[str],
    instruction: Optional[str],
    max_new_tokens: int,
    user_prompt: Optional[str] = None,
) -> str:
    model = model_components["model"]
    tok = _deepseek_get_tokenizer(model_components)
    if user_prompt is not None:
        prompt = user_prompt
    else:
        prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
    prepare_inputs = _deepseek_vl2_prepare_inputs(model_components, image_path, prompt).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return tok.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()


def _generate_with_deepseek_standard(
    model_components: Dict[str, Any],
    question: str,
    image_path: Optional[str],
    oracle_context: Optional[str],
    instruction: Optional[str],
    max_new_tokens: int,
    user_prompt: Optional[str] = None,
) -> str:
    if _is_deepseek_vl_model(model_components):
        if not image_path:
            raise ValueError("DeepSeek-VL generation requires image_path.")
        return _generate_with_deepseek_vl_standard(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            user_prompt=user_prompt,
        )
    if _is_deepseek_vl2_model(model_components):
        if not image_path:
            raise ValueError("DeepSeek-VL2 generation requires image_path.")
        return _generate_with_deepseek_vl2_standard(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            user_prompt=user_prompt,
        )
    model = model_components["model"]
    tok = _deepseek_get_tokenizer(model_components)
    model_device = next(model.parameters()).device
    if user_prompt is not None:
        prompt = user_prompt
    else:
        prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
    inputs = tok(prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        amp_dtype = model_components.get("dtype", torch.float16)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
        except Exception:
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
    input_len = inputs["input_ids"].shape[1]
    return tok.decode(generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def generate_with_deepseek_intervention(
    model_components: Dict[str, Any],
    question: str,
    image_path: Optional[str],
    oracle_context: str,
    instruction: Optional[str],
    max_new_tokens: int = 64,
    alpha_v: float = 0.5,
    alpha_t: float = 1.0,
    gamma_s: float = 1.0,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
    use_madrag: bool = False,
    skip_failed: bool = False,
) -> str:
    """
    DeepSeek BAIR + optional Ms-PoE (text-only).
    DeepSeek-V3.2 is handled as LM-style generation with context/question prompt.
    """
    use_mspoe = abs(mspoe_scaling - 1.0) > 1e-12
    if abs(alpha_v) < 1e-12 and abs(alpha_t) < 1e-12 and not use_mspoe and not use_madrag:
        set_bottleneck_intervention(False)
        patch_deepseek_attention_for_bottleneck_intervention(False)
        return _generate_with_deepseek_standard(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
        )

    if _is_deepseek_vl_model(model_components):
        model = model_components["model"]
        tok = _deepseek_get_tokenizer(model_components)
        mspoe_handle = None
        bair_active_global = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
        _force_eager_for_intervention(model_components)
        patch_llama_attention_for_bottleneck_intervention(False)
        patch_qwen_vl_attention_for_bottleneck_intervention(False)
        if bair_active_global:
            # DeepSeek-VL 1.x wraps a LLaMA language model in the official package.
            patch_llama_attention_for_bottleneck_intervention(True)
        try:
            if not image_path:
                raise ValueError("DeepSeek-VL intervention requires image_path.")
            safe_tail_tokens = max(16, min(128, len(tok(question, add_special_tokens=False)["input_ids"]) + 12))

            clean_prompt = build_full_prompt(question=question, context=None, instruction=instruction)
            clean_inputs = _deepseek_vl_prepare_inputs(model_components, image_path, clean_prompt).to(model.device)
            clean_embeds = model.prepare_inputs_embeds(**clean_inputs)
            img_mask = clean_inputs.images_seq_mask[0]
            img_count = int(img_mask.sum().item())
            img_start = int(torch.nonzero(img_mask, as_tuple=False)[0].item()) if img_count > 0 else 0
            set_bottleneck_intervention(
                True,
                num_visual_tokens=img_count,
                visual_start_idx=img_start,
                calibration_run=True,
                reset_layer=True,
                alpha_v=alpha_v,
                alpha_t=alpha_t,
                gamma_s=gamma_s,
                question_tokens=safe_tail_tokens,
                use_madrag=use_madrag,
            )
            with torch.no_grad():
                model.language_model(
                    inputs_embeds=clean_embeds,
                    attention_mask=clean_inputs.attention_mask,
                    use_cache=True,
                )

            gen_prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
            gen_inputs = _deepseek_vl_prepare_inputs(model_components, image_path, gen_prompt).to(model.device)
            gen_embeds = model.prepare_inputs_embeds(**gen_inputs)
            bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
            retry_pairs = [(alpha_v, alpha_t)] if skip_failed else (_deepseek_retry_pairs(alpha_v, alpha_t) if bair_active else [(alpha_v, alpha_t)])
            # _log_deepseek_backoff_plan("vl", retry_pairs)
            response = ""
            if use_mspoe:
                # DeepSeek-VL uses inputs_embeds path; apply Ms-PoE via forward hook
                # to avoid shape mismatch from static position_ids in generate().
                mspoe_handle = _llava_apply_mspoe_position_hook(
                    model,
                    mspoe_scaling,
                    mspoe_text_only,
                    num_visual_tokens=img_count,
                    visual_start_idx=img_start,
                )
            for attempt_idx, (current_alpha_v, current_alpha_t) in enumerate(retry_pairs, start=1):
                set_bottleneck_intervention(
                    True,
                    num_visual_tokens=img_count,
                    visual_start_idx=img_start,
                    calibration_run=False,
                    reset_layer=True,
                    alpha_v=current_alpha_v,
                    alpha_t=current_alpha_t,
                    gamma_s=gamma_s,
                    question_tokens=safe_tail_tokens,
                    use_madrag=use_madrag,
                )
                gen_kwargs = {
                    "inputs_embeds": gen_embeds,
                    "attention_mask": gen_inputs.attention_mask,
                    "pad_token_id": tok.eos_token_id,
                    "bos_token_id": tok.bos_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                }
                with torch.no_grad():
                    outputs = model.language_model.generate(**gen_kwargs)
                response = tok.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
                sft_prompt = gen_inputs["sft_format"][0] if "sft_format" in gen_inputs.keys() else ""
                if sft_prompt and response.startswith(sft_prompt):
                    response = response[len(sft_prompt):].strip()
                is_deg = _is_degenerate_response(response) if bair_active else False
                _log_deepseek_backoff_attempt(
                    "vl",
                    attempt_idx,
                    len(retry_pairs),
                    current_alpha_v,
                    current_alpha_t,
                    is_deg,
                    response,
                )
                if not bair_active or not is_deg:
                    break

            set_bottleneck_intervention(False)
            if skip_failed:
                return response
            if bair_active and _is_degenerate_response(response):
                return "[GENERATION_FAILED]"
            if not response.strip():
                return "[GENERATION_FAILED]"
            return response
        finally:
            if mspoe_handle is not None:
                mspoe_handle.remove()
            set_bottleneck_intervention(False)
            if bair_active_global:
                patch_llama_attention_for_bottleneck_intervention(False)

    if _is_deepseek_vl2_model(model_components):
        # For DeepSeek-VL2, keep generation path stable and enable intervention hooks at LM level.
        model = model_components["model"]
        tok = _deepseek_get_tokenizer(model_components)
        _force_eager_for_intervention(model_components)
        patch_llama_attention_for_bottleneck_intervention(False)
        patch_qwen_vl_attention_for_bottleneck_intervention(False)
        patch_deepseek_attention_for_bottleneck_intervention(True)
        try:
            if not image_path:
                raise ValueError("DeepSeek-VL2 intervention requires image_path.")
            # Calibration pass (context-free)
            clean_prompt = build_full_prompt(question=question, context=None, instruction=instruction)
            clean_inputs = _deepseek_vl2_prepare_inputs(model_components, image_path, clean_prompt).to(model.device)
            clean_embeds = model.prepare_inputs_embeds(**clean_inputs)
            safe_tail_tokens = max(16, min(128, len(tok(question, add_special_tokens=False)["input_ids"]) + 12))
            set_bottleneck_intervention(
                True,
                num_visual_tokens=0,
                visual_start_idx=0,
                calibration_run=True,
                reset_layer=True,
                alpha_v=alpha_v,
                alpha_t=alpha_t,
                gamma_s=gamma_s,
                question_tokens=safe_tail_tokens,
                use_madrag=use_madrag,
            )
            with torch.no_grad():
                model.language_model(
                    inputs_embeds=clean_embeds,
                    attention_mask=clean_inputs.attention_mask,
                    use_cache=True,
                )

            # Generation pass
            gen_prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
            gen_inputs = _deepseek_vl2_prepare_inputs(model_components, image_path, gen_prompt).to(model.device)
            gen_embeds = model.prepare_inputs_embeds(**gen_inputs)
            bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
            retry_pairs = [(alpha_v, alpha_t)] if skip_failed else (_deepseek_retry_pairs(alpha_v, alpha_t) if bair_active else [(alpha_v, alpha_t)])
            # _log_deepseek_backoff_plan("vl2", retry_pairs)
            response = ""

            for attempt_idx, (current_alpha_v, current_alpha_t) in enumerate(retry_pairs, start=1):
                set_bottleneck_intervention(
                    True,
                    num_visual_tokens=0,
                    visual_start_idx=0,
                    calibration_run=False,
                    reset_layer=True,
                    alpha_v=current_alpha_v,
                    alpha_t=current_alpha_t,
                    gamma_s=gamma_s,
                    question_tokens=safe_tail_tokens,
                    use_madrag=use_madrag,
                )
                with torch.no_grad():
                    outputs = model.language_model.generate(
                        inputs_embeds=gen_embeds,
                        attention_mask=gen_inputs.attention_mask,
                        pad_token_id=tok.eos_token_id,
                        bos_token_id=tok.bos_token_id,
                        eos_token_id=tok.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                response = tok.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
                is_deg = _is_degenerate_response(response) if bair_active else False
                _log_deepseek_backoff_attempt(
                    "vl2",
                    attempt_idx,
                    len(retry_pairs),
                    current_alpha_v,
                    current_alpha_t,
                    is_deg,
                    response,
                )
                if not bair_active or not is_deg:
                    break

            set_bottleneck_intervention(False)
            if skip_failed:
                return response
            if bair_active and _is_degenerate_response(response):
                return "[GENERATION_FAILED]"
            if not response.strip():
                return "[GENERATION_FAILED]"
            return response
        finally:
            set_bottleneck_intervention(False)
            patch_deepseek_attention_for_bottleneck_intervention(False)

    model = model_components["model"]
    tok = _deepseek_get_tokenizer(model_components)
    model_device = next(model.parameters()).device

    patch_llama_attention_for_bottleneck_intervention(False)
    patch_qwen_vl_attention_for_bottleneck_intervention(False)
    patch_deepseek_attention_for_bottleneck_intervention(True)

    try:
        clean_prompt = build_full_prompt(question=question, context=None, instruction=instruction)
        clean_inputs = tok(clean_prompt, return_tensors="pt").to(model_device)
        seq_len_clean = int(clean_inputs["input_ids"].shape[1])
        clean_position_ids = torch.arange(seq_len_clean, device=model_device).unsqueeze(0)
        if use_mspoe:
            sf = float(mspoe_scaling)
            if sf > 0:
                # Text-only DeepSeek: "text_only" has same behavior as full scaling.
                clean_position_ids = (clean_position_ids.float() / sf).long()
        clean_inputs["position_ids"] = clean_position_ids

        set_bottleneck_intervention(
            True,
            num_visual_tokens=0,
            visual_start_idx=0,
            calibration_run=True,
            reset_layer=True,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            question_tokens=max(16, min(128, len(tok(question, add_special_tokens=False)["input_ids"]) + 12)),
            use_madrag=use_madrag,
        )
        with torch.no_grad():
            amp_dtype = model_components.get("dtype", torch.float16)
            try:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    model(**clean_inputs, use_cache=True)
            except Exception:
                model(**clean_inputs, use_cache=True)

        gen_prompt = build_full_prompt(question=question, context=oracle_context, instruction=instruction)
        gen_inputs = tok(gen_prompt, return_tensors="pt").to(model_device)
        seq_len_gen = int(gen_inputs["input_ids"].shape[1])
        gen_position_ids = torch.arange(seq_len_gen, device=model_device).unsqueeze(0)
        if use_mspoe:
            sf = float(mspoe_scaling)
            if sf > 0:
                gen_position_ids = (gen_position_ids.float() / sf).long()
        gen_inputs["position_ids"] = gen_position_ids

        bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
        retry_pairs = [(alpha_v, alpha_t)] if skip_failed else (_deepseek_retry_pairs(alpha_v, alpha_t) if bair_active else [(alpha_v, alpha_t)])
        # _log_deepseek_backoff_plan("text", retry_pairs)
        response = ""

        safe_tail_tokens = max(16, min(128, len(tok(question, add_special_tokens=False)["input_ids"]) + 12))
        for attempt_idx, (current_alpha_v, current_alpha_t) in enumerate(retry_pairs, start=1):
            set_bottleneck_intervention(
                True,
                num_visual_tokens=0,
                visual_start_idx=0,
                calibration_run=False,
                reset_layer=True,
                alpha_v=current_alpha_v,
                alpha_t=current_alpha_t,
                gamma_s=gamma_s,
                question_tokens=safe_tail_tokens,
                use_madrag=use_madrag,
            )
            with torch.no_grad():
                amp_dtype = model_components.get("dtype", torch.float16)
                try:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        generated_ids = model.generate(
                            **gen_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                except Exception:
                    generated_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
            input_len = gen_inputs["input_ids"].shape[1]
            response = tok.decode(
                generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            is_deg = _is_degenerate_response(response) if bair_active else False
            _log_deepseek_backoff_attempt(
                "text",
                attempt_idx,
                len(retry_pairs),
                current_alpha_v,
                current_alpha_t,
                is_deg,
                response,
            )
            if not bair_active or not is_deg:
                break

        set_bottleneck_intervention(False)
        if skip_failed:
            return response
        if bair_active and _is_degenerate_response(response):
            return "[GENERATION_FAILED]"
        if not response.strip():
            return "[GENERATION_FAILED]"
        return response
    finally:
        set_bottleneck_intervention(False)
        patch_deepseek_attention_for_bottleneck_intervention(False)


def generate_with_qwen_intervention(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: str,
    instruction: Optional[str],
    max_new_tokens: int = 64,
    alpha_v: float = 0.5,
    alpha_t: float = 1.0,
    gamma_s: float = 1.0,
    qwen_pixel_limit: Optional[int] = 28 * 28 * 50,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
    use_madrag: bool = False,
    skip_failed: bool = False,
    include_experiment_notice: bool = True,
) -> str:
    model = model_components["model"]
    processor = model_components["processor"]
    model_device = next(model.parameters()).device
    use_mspoe = abs(mspoe_scaling - 1.0) > 1e-12
    mspoe_handle = None

    # True no-op control: bypass patched attention/intervention entirely (unless Ms-PoE is active).
    if abs(alpha_v) < 1e-12 and abs(alpha_t) < 1e-12 and not use_mspoe and not use_madrag:
        set_bottleneck_intervention(False)
        patch_qwen_vl_attention_for_bottleneck_intervention(use_intervention=False)
        return _generate_with_qwen_standard_path(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            # Preserve caller cap to avoid accidental high-res visual-token explosions.
            qwen_pixel_limit=qwen_pixel_limit,
            include_experiment_notice=include_experiment_notice,
        )

    model.config.use_cache = True
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    patch_qwen_vl_attention_for_bottleneck_intervention(use_intervention=True)
    if use_mspoe:
        mspoe_handle = apply_mspoe_position_hook_qwen(model, mspoe_scaling, mspoe_text_only)

    try:
        image_obj = Image.open(image_path).convert("RGB")

        # --- 1. Calibration pass ---
        clean_prompt = build_full_prompt(
            question=question,
            context=None,
            instruction=instruction,
            include_experiment_notice=include_experiment_notice,
        )
        clean_messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": clean_prompt}],
        }]
        clean_text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
        clean_image_inputs, clean_video_inputs = process_vision_info(clean_messages)

        processor_kwargs = {
            "text": [clean_text],
            "images": clean_image_inputs,
            "videos": clean_video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if qwen_pixel_limit and qwen_pixel_limit > 0:
            processor_kwargs["min_pixels"] = qwen_pixel_limit
            processor_kwargs["max_pixels"] = qwen_pixel_limit
        clean_inputs = processor(**processor_kwargs).to(model_device)

        v_start_clean, num_visual_tokens_clean = _get_visual_token_info(model, clean_inputs)
        if use_mspoe:
            set_mspoe_qwen_state(v_start_clean, num_visual_tokens_clean, mspoe_scaling, mspoe_text_only)

        safe_tail_tokens = _estimate_safe_tail_tokens(processor, question)
        set_bottleneck_intervention(
            True,
            num_visual_tokens=num_visual_tokens_clean,
            visual_start_idx=v_start_clean,
            calibration_run=True,
            reset_layer=True,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            question_tokens=safe_tail_tokens,
            use_madrag=use_madrag,
        )

        with torch.no_grad():
            amp_dtype = model_components.get("dtype", torch.float16)
            try:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    model(**clean_inputs, use_cache=True)
            except Exception:
                model(**clean_inputs, use_cache=True)

        del clean_inputs
        bair_efficient.optional_empty_cache_after_calibration()

        # --- 2. Generation pass (WITH DYNAMIC FALLBACK) ---
        gen_prompt = build_full_prompt(
            question=question,
            context=oracle_context,
            instruction=instruction,
            include_experiment_notice=include_experiment_notice,
        )
        gen_messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": gen_prompt}],
        }]
        gen_text = processor.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
        gen_image_inputs, gen_video_inputs = process_vision_info(gen_messages)

        gen_processor_kwargs = {
            "text": [gen_text],
            "images": gen_image_inputs,
            "videos": gen_video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if qwen_pixel_limit and qwen_pixel_limit > 0:
            gen_processor_kwargs["min_pixels"] = qwen_pixel_limit
            gen_processor_kwargs["max_pixels"] = qwen_pixel_limit
        gen_inputs = processor(**gen_processor_kwargs).to(model_device)

        v_start_gen, num_visual_tokens_gen = _get_visual_token_info(model, gen_inputs)
        if use_mspoe:
            set_mspoe_qwen_state(v_start_gen, num_visual_tokens_gen, mspoe_scaling, mspoe_text_only)
        safe_tail_tokens = _estimate_safe_tail_tokens(processor, question)

        # BAIR-only: retry with softer alphas if output looks degenerate (repetition loops).
        # skip_failed=True: no alpha fallback (single generation); always return raw text (low quality OK).
        # skip_failed=False: use fallback when bair_active; still degenerate/empty -> [GENERATION_FAILED].
        bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
        current_alpha_v = alpha_v
        current_alpha_t = alpha_t
        if skip_failed:
            max_retries = 1
        else:
            max_retries = 10 if bair_active else 1
        response = ""

        for attempt in range(max_retries):
            set_bottleneck_intervention(
                True,
                num_visual_tokens=num_visual_tokens_gen,
                visual_start_idx=v_start_gen,
                calibration_run=False, reset_layer=True,
                alpha_v=current_alpha_v,
                alpha_t=current_alpha_t,
                gamma_s=gamma_s,
                question_tokens=safe_tail_tokens,
                use_madrag=use_madrag,
            )

            with torch.no_grad():
                amp_dtype = model_components.get("dtype", torch.float16)
                try:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        generated_ids = model.generate(
                            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                        )
                except Exception:
                    generated_ids = model.generate(
                        **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                    )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(gen_inputs.input_ids, generated_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            if not bair_active:
                break
            if not _is_degenerate_response(response):
                break

            print(
                f"\n[Warning] Degenerate response. "
                f"Retrying with alpha_v={current_alpha_v * 0.8:.2f}, alpha_t={current_alpha_t * 0.8:.2f}..."
            )
            current_alpha_v *= 0.8
            current_alpha_t *= 0.8

        set_bottleneck_intervention(False)

        if skip_failed:
            return response

        if bair_active and _is_degenerate_response(response):
            return "[GENERATION_FAILED]"
        if not response.strip():
            return "[GENERATION_FAILED]"

        return response
    finally:
        if mspoe_handle is not None:
            mspoe_handle.remove()

def generate_with_qwen_intervention_shared_prefix(
    model_components: Dict[str, Any],
    question: str,
    image_path: str,
    oracle_context: str,
    instruction: Optional[str],
    max_new_tokens: int = 64,
    alpha_v: float = 0.5,
    alpha_t: float = 1.0,
    gamma_s: float = 1.0,
    qwen_pixel_limit: Optional[int] = None,
    include_experiment_notice: bool = True,
) -> str:
    """Qwen2.5-VL BAIR path that reuses the no-context prefix KV cache."""
    model = model_components["model"]
    processor = model_components["processor"]
    model_device = next(model.parameters()).device

    model.config.use_cache = True
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    patch_qwen_vl_attention_for_bottleneck_intervention(use_intervention=True)

    image_obj = Image.open(image_path).convert("RGB")
    full_prompt = build_shared_bair_full_prompt(question, oracle_context, instruction, include_experiment_notice)
    full_messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": full_prompt}],
    }]

    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    full_image_inputs, full_video_inputs = process_vision_info(full_messages)
    full_kwargs = {
        "text": [full_text],
        "images": full_image_inputs,
        "videos": full_video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if qwen_pixel_limit and qwen_pixel_limit > 0:
        full_kwargs["min_pixels"] = qwen_pixel_limit
        full_kwargs["max_pixels"] = qwen_pixel_limit

    full_inputs = processor(**full_kwargs).to(model_device)
    boundary_text = oracle_context if oracle_context else ""
    boundary = full_text.find(boundary_text) if boundary_text else len(full_text)
    if boundary < 0:
        raise RuntimeError("Could not locate context boundary in shared Qwen BAIR prompt.")
    prefix_len = _prefix_len_from_text_boundary(processor, full_text, boundary)
    tok = getattr(processor, "tokenizer", processor)
    text_len = len(tok(full_text, add_special_tokens=False)["input_ids"])
    image_expansion_delta = int(full_inputs["input_ids"].shape[1]) - int(text_len)
    if image_expansion_delta > 0:
        prefix_len += image_expansion_delta
    if full_inputs["input_ids"].shape[1] <= prefix_len:
        raise RuntimeError("Shared Qwen BAIR prefix produced no context/generation suffix.")
    prefix_inputs = _slice_processor_inputs(full_inputs, prefix_len)

    v_start, num_visual_tokens = _get_visual_token_info(model, prefix_inputs)
    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        visual_start_idx=v_start,
        calibration_run=True,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        gamma_s=gamma_s,
        question_tokens=0,
        use_madrag=False,
    )
    with torch.no_grad():
        amp_dtype = model_components.get("dtype", torch.float16)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                prefix_out = model(**prefix_inputs, use_cache=True, return_dict=True)
        except Exception:
            prefix_out = model(**prefix_inputs, use_cache=True, return_dict=True)

    past = getattr(prefix_out, "past_key_values", None)
    if past is None:
        raise RuntimeError("Shared Qwen BAIR prefix forward did not return past_key_values.")

    suffix_ids = full_inputs["input_ids"][:, prefix_len:]
    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        visual_start_idx=v_start,
        calibration_run=False,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        gamma_s=gamma_s,
        question_tokens=0,
        use_madrag=False,
    )
    with torch.no_grad():
        amp_dtype = model_components.get("dtype", torch.float16)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(
                    input_ids=suffix_ids,
                    attention_mask=full_inputs.get("attention_mask"),
                    past_key_values=past,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        except Exception:
            generated_ids = model.generate(
                input_ids=suffix_ids,
                attention_mask=full_inputs.get("attention_mask"),
                past_key_values=past,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    suffix_len = suffix_ids.shape[1]
    set_bottleneck_intervention(False)
    generated = generated_ids[0][suffix_len:] if generated_ids.shape[1] > suffix_len else generated_ids[0]
    if generated.numel() == 0:
        return ""
    return processor.batch_decode(
        [generated],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

def process_and_visualize_samples(facet_csv: str, image_root: str, profession_column: str,
                                  model_components: Dict[str, Any], db_path: str, 
                                  question: str, model_names: List[str], device_id: int,
                                  output_dir: str, json_output_dir: str, generation_model_name: str,
                                  exclude_path: str = None,
                                  num_samples: int = 10, max_new_tokens: int = 64, generation_only: bool = False,
                                  instruction: Optional[str] = None,
                                  qwen_pixel_limit: Optional[int] = 28 * 28 * 50,
                                  use_mspoe: bool = False,
                                  use_madrag: bool = False,
                                  use_longllmlingua: bool = False,
                                  use_combo: bool = False,
                                  use_madrag_combo: bool = False,
                                  use_longllmlingua_combo: bool = False,
                                  mspoe_scaling: float = 1.5,
                                  mspoe_text_only: bool = False,
                                  compression_ratio: float = 0.5,
                                  compressor_device_id: Optional[int] = None,
                                  alpha_v: float = 0.5,
                                  alpha_t: float = 1.0,
                                  gamma_s: float = 1.0,
                                  fail_fast: bool = False,
                                  from_baseline_json: Optional[str] = None,
                                  baseline_strict: bool = False,
                                  skip_failed: bool = False,
                                  use_precomputed_longllmlingua_context: bool = False,
                                  require_precomputed_longllmlingua_context: bool = False):
    exclude_list = set()
    if exclude_path and os.path.exists(exclude_path):
        with open(exclude_path, 'r') as f:
            content = f.read().splitlines()
            exclude_list = {line.strip().lower() for line in content if line.strip()}

    profession_data = {}
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                profession_data[item["class_key"].lower()] = item

    df = pd.read_csv(facet_csv)
    rows = df[["filename", profession_column, "gender"]].dropna()
    
    if exclude_list:
        normalized_prof = rows[profession_column].astype(str).str.lower().str.strip().str.replace(" ", "_")
        rows = rows[~normalized_prof.isin(exclude_list)]
        
    if generation_only:
        print(f"Mode: Generation only (Target: {num_samples if num_samples else 'ALL'})")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    model_dir_name = generation_model_name.replace('/', '_').replace('-', '_')
    mode_parts = []
    if use_mspoe:
        mode_parts.append("mspoe_text" if mspoe_text_only else "mspoe_full")
    if use_madrag:
        mode_parts.append("madrag")
    if use_longllmlingua:
        mode_parts.append("longllmlingua")
    if use_combo:
        mode_parts.append(f"combo_av{alpha_v}")
    if use_madrag_combo:
        mode_parts.append(f"madrag_combo_av{alpha_v}")
    if use_longllmlingua_combo:
        mode_parts.append(f"lll_combo_av{alpha_v}")
    mode_suffix = ("_" + "_".join(mode_parts)) if mode_parts else ""
    base_json = f"analysis_results_{model_dir_name}_with_instruction_2.json" if instruction else f"analysis_results_{model_dir_name}_2.json"
    if mode_suffix:
        stem, ext = os.path.splitext(base_json)
        json_filename = f"{stem}{mode_suffix}{ext}"
    else:
        json_filename = base_json
    json_output_path = os.path.join(json_output_dir, json_filename)
    # Line-delimited backup: survives crashes that corrupt the monolithic .json rewrite.
    jsonl_output_path = os.path.splitext(json_output_path)[0] + ".jsonl"
    print(f"Saving: {json_output_path} (final JSON) + {jsonl_output_path} (append-only resume log)")

    target_fields: List[str] = []
    if use_longllmlingua:
        target_fields.append("oracle_longllmlingua_answer")
    if use_mspoe:
        target_fields.append("oracle_mspoe_answer")
    if use_madrag:
        target_fields.append("oracle_madrag_answer")
    if use_combo:
        target_fields.append("oracle_bair_mspoe_combo_answer")
    if use_madrag_combo:
        target_fields.append("oracle_bair_madrag_combo_answer")
    if use_longllmlingua_combo:
        target_fields.append("oracle_bair_longllmlingua_combo_answer")
    if not target_fields:
        target_fields.extend(["oracle_answer", "no_retrieval_answer"])

    def _result_key(e: Dict[str, Any]) -> Optional[str]:
        key = e.get("filename") or e.get("image_path")
        return str(key).strip() if key else None

    def _has_completed_targets(e: Dict[str, Any]) -> bool:
        if not isinstance(e, dict):
            return False
        if e.get("error"):
            return False
        for field in target_fields:
            value = e.get(field)
            if value is None:
                return False
            if isinstance(value, str) and (not value.strip() or "[Error]" in value):
                return False
        return True

    completed_results: Dict[str, Dict[str, Any]] = {}
    for partial_path in (json_output_path, jsonl_output_path):
        if not os.path.exists(partial_path):
            continue
        try:
            if partial_path.endswith(".jsonl"):
                with open(partial_path, "r", encoding="utf-8") as f:
                    partial_rows = [json.loads(line) for line in f if line.strip()]
            else:
                with open(partial_path, "r", encoding="utf-8") as f:
                    partial_rows = json.load(f)
            if not isinstance(partial_rows, list):
                continue
            for partial_entry in partial_rows:
                if not isinstance(partial_entry, dict) or not _has_completed_targets(partial_entry):
                    continue
                key = _result_key(partial_entry)
                if key:
                    completed_results[key] = partial_entry
            if completed_results:
                print(f"[Resume] Loaded {len(completed_results)} completed row(s) from {partial_path}")
        except Exception as ex:
            print(f"[Resume] Could not load partial results from {partial_path}: {ex}")

    baseline_map: Optional[Dict[str, Dict[str, Any]]] = None
    if from_baseline_json:
        if not os.path.isfile(from_baseline_json):
            raise FileNotFoundError(f"--from_baseline_json not found: {from_baseline_json}")
        baseline_map = _load_baseline_json_map(from_baseline_json)
        print(
            f"Reusing oracle/no_retrieval (and oracle_context when present) from {from_baseline_json} "
            f"({len(baseline_map)} lookup keys)."
        )

    comp_device = f"cuda:{compressor_device_id}" if compressor_device_id is not None else f"cuda:{device_id}"
    needs_runtime_longllmlingua = (
        use_longllmlingua or use_longllmlingua_combo
    ) and not use_precomputed_longllmlingua_context
    if needs_runtime_longllmlingua and PromptCompressor is None:
        raise ImportError("LongLLMLingua modes require the llmlingua package. Install with: pip install llmlingua")

    all_results, matching_count, total_processed = [], 0, 0
    rows_shuffled = rows.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for idx, row in enumerate(tqdm(rows_shuffled.itertuples(), total=len(rows_shuffled), desc="Processing images")):
        if generation_only and num_samples and total_processed >= num_samples: break
        elif not generation_only and num_samples and matching_count >= num_samples: break
        
        try:
            fname, prof, gender = row.filename, getattr(row, profession_column), row.gender
            image_path = os.path.join(image_root, fname)
            if not os.path.exists(image_path): continue
            
            total_processed += 1
            gt_gender = "female" if str(gender) == "1" else "male"
            gt_profession = str(prof)
            prof_key = gt_profession.lower().strip().replace(" ", "_")
            display_profession = profession_data[prof_key]["class_name"] if prof_key in profession_data else gt_profession.replace("_", " ")
            current_question = question.format(profession=display_profession)

            completed = completed_results.get(fname) or completed_results.get(image_path)
            if completed is not None:
                all_results.append(completed)
                if completed.get("matches_criteria"):
                    matching_count += 1
                continue
            
            oracle_passages = create_oracle_retrieval(profession_data, gt_profession)
            ctx_lll = None
            oracle_longllmlingua_answer = None
            oracle_mspoe_answer = None
            oracle_madrag_answer = None
            oracle_bair_mspoe_combo_answer = None
            oracle_bair_madrag_combo_answer = None
            oracle_bair_longllmlingua_combo_answer = None

            if _is_qwen_vl_model(model_components):
                # Keep baseline generation path aligned with intervention settings.
                set_bottleneck_intervention(False)
                patch_qwen_vl_attention_for_bottleneck_intervention(use_intervention=False)

                bl = baseline_map.get(fname) if baseline_map else None
                if bl is None and baseline_map is not None:
                    msg = f"No baseline row for filename={fname}"
                    if baseline_strict:
                        raise KeyError(msg)
                    print(f"[Warning] {msg}; regenerating oracle/no_retrieval.")

                if bl is not None and bl.get("oracle_answer") is not None and bl.get("no_retrieval_answer") is not None:
                    oracle_answer = bl["oracle_answer"]
                    no_retrieval_answer = bl["no_retrieval_answer"]
                    oracle_context = (bl.get("oracle_context") or "").strip() or build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    context = bl.get("no_retrieval_context") or ""
                elif use_precomputed_longllmlingua_context and bl is not None:
                    oracle_answer = bl.get("oracle_answer", "") or ""
                    no_retrieval_answer = bl.get("no_retrieval_answer", "") or ""
                    oracle_context = bl.get("oracle_context", "") or ""
                    context = bl.get("no_retrieval_context") or ""
                else:
                    oracle_context = build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    oracle_answer = _generate_with_qwen_standard_path(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        qwen_pixel_limit=qwen_pixel_limit,
                    )
                    no_retrieval_answer = _generate_with_qwen_standard_path(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=None,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        qwen_pixel_limit=qwen_pixel_limit,
                    )
                    context = ""

                if use_longllmlingua or use_longllmlingua_combo:
                    if use_precomputed_longllmlingua_context and bl is not None:
                        ctx_lll = get_precomputed_longllmlingua_context(
                            bl,
                            question=current_question,
                            instruction=instruction,
                            ratio=compression_ratio,
                            require=require_precomputed_longllmlingua_context,
                        )
                    if ctx_lll is None:
                        if require_precomputed_longllmlingua_context:
                            raise ValueError(f"No precomputed LongLLMLingua context for filename={fname}")
                        ctx_lll = compress_with_longllmlingua(
                            oracle_context, current_question, instruction or "", comp_device, compression_ratio
                        )
                    if use_longllmlingua:
                        oracle_longllmlingua_answer = _generate_with_qwen_standard_path(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=ctx_lll,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                            qwen_pixel_limit=qwen_pixel_limit,
                        )

                if use_mspoe:
                    oracle_mspoe_answer = generate_with_qwen_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=gamma_s,
                        qwen_pixel_limit=qwen_pixel_limit,
                        mspoe_scaling=mspoe_scaling,
                        mspoe_text_only=mspoe_text_only,
                        use_madrag=False,
                        skip_failed=skip_failed,
                    )

                if use_madrag:
                    oracle_madrag_answer = generate_with_qwen_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=gamma_s,
                        qwen_pixel_limit=qwen_pixel_limit,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                        skip_failed=skip_failed,
                    )

                if use_combo:
                    oracle_bair_mspoe_combo_answer = generate_with_qwen_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        qwen_pixel_limit=qwen_pixel_limit,
                        mspoe_scaling=mspoe_scaling,
                        mspoe_text_only=mspoe_text_only,
                        use_madrag=False,
                        skip_failed=skip_failed,
                    )

                if use_madrag_combo:
                    oracle_bair_madrag_combo_answer = generate_with_qwen_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        qwen_pixel_limit=qwen_pixel_limit,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                        skip_failed=skip_failed,
                    )

                if use_longllmlingua_combo:
                    # ctx_lll is always built above when use_longllmlingua_combo (same condition as compress block).
                    oracle_bair_longllmlingua_combo_answer = generate_with_qwen_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=ctx_lll,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        qwen_pixel_limit=qwen_pixel_limit,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        skip_failed=skip_failed,
                    )
            elif _is_llava_hf_model(model_components):
                set_bottleneck_intervention(False)
                patch_llama_attention_for_bottleneck_intervention(False)
                patch_qwen_vl_attention_for_bottleneck_intervention(False)

                bl = baseline_map.get(fname) if baseline_map else None
                if bl is None and baseline_map is not None:
                    msg = f"No baseline row for filename={fname}"
                    if baseline_strict:
                        raise KeyError(msg)
                    print(f"[Warning] {msg}; regenerating oracle/no_retrieval.")

                if bl is not None and bl.get("oracle_answer") is not None and bl.get("no_retrieval_answer") is not None:
                    oracle_answer = bl["oracle_answer"]
                    no_retrieval_answer = bl["no_retrieval_answer"]
                    oracle_context = (bl.get("oracle_context") or "").strip() or build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    context = bl.get("no_retrieval_context") or ""
                else:
                    oracle_context = build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    oracle_answer = _generate_hf_llava_standard(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                    )
                    no_retrieval_answer = _generate_hf_llava_standard(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=None,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                    )
                    context = ""

                if use_longllmlingua or use_longllmlingua_combo:
                    if use_precomputed_longllmlingua_context and bl is not None:
                        ctx_lll = get_precomputed_longllmlingua_context(
                            bl,
                            question=current_question,
                            instruction=instruction,
                            ratio=compression_ratio,
                            require=require_precomputed_longllmlingua_context,
                        )
                    if ctx_lll is None:
                        if require_precomputed_longllmlingua_context:
                            raise ValueError(f"No precomputed LongLLMLingua context for filename={fname}")
                        ctx_lll = compress_with_longllmlingua(
                            oracle_context, current_question, instruction or "", comp_device, compression_ratio
                        )
                    if use_longllmlingua:
                        oracle_longllmlingua_answer = _generate_hf_llava_standard(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=ctx_lll,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                        )

                if use_mspoe:
                    oracle_mspoe_answer = generate_with_llava_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=gamma_s,
                        mspoe_scaling=mspoe_scaling,
                        mspoe_text_only=mspoe_text_only,
                        use_madrag=False,
                        skip_failed=skip_failed,
                    )

                if use_madrag:
                    oracle_madrag_answer = generate_with_llava_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                        skip_failed=skip_failed,
                    )

                if use_combo:
                    oracle_bair_mspoe_combo_answer = generate_with_llava_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        mspoe_scaling=mspoe_scaling,
                        mspoe_text_only=mspoe_text_only,
                        use_madrag=False,
                        skip_failed=skip_failed,
                    )

                if use_madrag_combo:
                    oracle_bair_madrag_combo_answer = generate_with_llava_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                        skip_failed=skip_failed,
                    )

                if use_longllmlingua_combo:
                    oracle_bair_longllmlingua_combo_answer = generate_with_llava_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=ctx_lll,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        skip_failed=skip_failed,
                    )
            elif _is_deepseek_model(model_components):
                set_bottleneck_intervention(False)
                patch_llama_attention_for_bottleneck_intervention(False)
                patch_qwen_vl_attention_for_bottleneck_intervention(False)
                patch_deepseek_attention_for_bottleneck_intervention(False)

                bl = baseline_map.get(fname) if baseline_map else None
                if bl is None and baseline_map is not None:
                    msg = f"No baseline row for filename={fname}"
                    if baseline_strict:
                        raise KeyError(msg)
                    print(f"[Warning] {msg}; regenerating oracle/no_retrieval.")

                if bl is not None and bl.get("oracle_answer") is not None and bl.get("no_retrieval_answer") is not None:
                    oracle_answer = bl["oracle_answer"]
                    no_retrieval_answer = bl["no_retrieval_answer"]
                    oracle_context = (bl.get("oracle_context") or "").strip() or build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    context = bl.get("no_retrieval_context") or ""
                else:
                    oracle_context = build_context(
                        oracle_passages, max_tokens=2000, tokenizer=model_components.get("tokenizer")
                    )
                    oracle_answer = _generate_with_deepseek_standard(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                    )
                    no_retrieval_answer = _generate_with_deepseek_standard(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=None,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                    )
                    context = ""

                if use_longllmlingua or use_longllmlingua_combo:
                    if use_precomputed_longllmlingua_context and bl is not None:
                        ctx_lll = get_precomputed_longllmlingua_context(
                            bl,
                            question=current_question,
                            instruction=instruction,
                            ratio=compression_ratio,
                            require=require_precomputed_longllmlingua_context,
                        )
                    if ctx_lll is None:
                        if require_precomputed_longllmlingua_context:
                            raise ValueError(f"No precomputed LongLLMLingua context for filename={fname}")
                        ctx_lll = compress_with_longllmlingua(
                            oracle_context, current_question, instruction or "", comp_device, compression_ratio
                        )
                    if use_longllmlingua:
                        oracle_longllmlingua_answer = _generate_with_deepseek_standard(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=ctx_lll,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                        )

                if use_mspoe:
                    try:
                        oracle_mspoe_answer = generate_with_deepseek_intervention(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=oracle_context,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                            alpha_v=0.0,
                            alpha_t=0.0,
                            gamma_s=gamma_s,
                            mspoe_scaling=mspoe_scaling,
                            mspoe_text_only=mspoe_text_only,
                            use_madrag=False,
                            skip_failed=skip_failed,
                        )
                    except RuntimeError as e:
                        # DeepSeek-VL can hit shape mismatch with Ms-PoE position-id scaling.
                        # Fall back to stable standard-oracle generation so long runs keep progress.
                        if "Expected size for first two dimensions of batch2 tensor" not in str(e):
                            raise
                        print(
                            "[Warning] DeepSeek Ms-PoE failed with tensor shape mismatch; "
                            "falling back to standard oracle generation for this sample."
                        )
                        oracle_mspoe_answer = _generate_with_deepseek_standard(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=oracle_context,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                        )

                if use_madrag:
                    oracle_madrag_answer = generate_with_deepseek_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=0.0,
                        alpha_t=0.0,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                        skip_failed=skip_failed,
                    )

                if use_combo:
                    try:
                        oracle_bair_mspoe_combo_answer = generate_with_deepseek_intervention(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=oracle_context,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                            alpha_v=alpha_v,
                            alpha_t=alpha_t,
                            gamma_s=gamma_s,
                            mspoe_scaling=mspoe_scaling,
                            mspoe_text_only=mspoe_text_only,
                            use_madrag=False,
                            skip_failed=skip_failed,
                        )
                    except RuntimeError as e:
                        if "Expected size for first two dimensions of batch2 tensor" not in str(e):
                            raise
                        print(
                            "[Warning] DeepSeek Ms-PoE+BAIR combo failed with tensor shape mismatch; "
                            "retrying this sample with BAIR-only (Ms-PoE disabled)."
                        )
                        oracle_bair_mspoe_combo_answer = generate_with_deepseek_intervention(
                            model_components=model_components,
                            question=current_question,
                            image_path=image_path,
                            oracle_context=oracle_context,
                            instruction=instruction,
                            max_new_tokens=max_new_tokens,
                            alpha_v=alpha_v,
                            alpha_t=alpha_t,
                            gamma_s=gamma_s,
                            mspoe_scaling=1.0,
                            mspoe_text_only=False,
                            use_madrag=False,
                            skip_failed=skip_failed,
                        )

                if use_madrag_combo:
                    oracle_bair_madrag_combo_answer = generate_with_deepseek_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=oracle_context,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        use_madrag=True,
                            skip_failed=skip_failed,
                        )

                if use_longllmlingua_combo:
                    oracle_bair_longllmlingua_combo_answer = generate_with_deepseek_intervention(
                        model_components=model_components,
                        question=current_question,
                        image_path=image_path,
                        oracle_context=ctx_lll,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        alpha_v=alpha_v,
                        alpha_t=alpha_t,
                        gamma_s=gamma_s,
                        mspoe_scaling=1.0,
                        mspoe_text_only=False,
                        skip_failed=skip_failed,
                    )
            else:
                from bair.llm_explainer import generate_with_hf
                oracle_answer, oracle_context = generate_with_hf(
                    model_components, instruction, current_question, oracle_passages, image_path, 
                    max_new_tokens=max_new_tokens, max_context_tokens=2000
                )
                
                no_retrieval_answer, context = generate_with_hf(
                    model_components, instruction, current_question, [], image_path, max_new_tokens=max_new_tokens
                )
            
            if generation_only:
                result_entry = {
                    'filename': fname, 'gt_gender': gt_gender, 'gt_profession': gt_profession,
                    'oracle_answer': oracle_answer, 'no_retrieval_answer': no_retrieval_answer,
                    'question': current_question, 'instruction': instruction, 'image_path': image_path,
                    'oracle_context': oracle_context, 'no_retrieval_context': context
                }
                if use_precomputed_longllmlingua_context:
                    result_entry.pop("oracle_context", None)
                    result_entry.pop("no_retrieval_context", None)
                    if not str(result_entry.get("oracle_answer") or "").strip():
                        result_entry.pop("oracle_answer", None)
                    if not str(result_entry.get("no_retrieval_answer") or "").strip():
                        result_entry.pop("no_retrieval_answer", None)
                if oracle_longllmlingua_answer is not None:
                    result_entry['oracle_longllmlingua_answer'] = oracle_longllmlingua_answer
                if ctx_lll is not None:
                    result_entry['oracle_longllmlingua_context'] = ctx_lll
                if oracle_mspoe_answer is not None:
                    result_entry['oracle_mspoe_answer'] = oracle_mspoe_answer
                if oracle_madrag_answer is not None:
                    result_entry['oracle_madrag_answer'] = oracle_madrag_answer
                if oracle_bair_mspoe_combo_answer is not None:
                    result_entry['oracle_bair_mspoe_combo_answer'] = oracle_bair_mspoe_combo_answer
                if oracle_bair_madrag_combo_answer is not None:
                    result_entry['oracle_bair_madrag_combo_answer'] = oracle_bair_madrag_combo_answer
                if oracle_bair_longllmlingua_combo_answer is not None:
                    result_entry['oracle_bair_longllmlingua_combo_answer'] = oracle_bair_longllmlingua_combo_answer
                all_results.append(result_entry)
                _append_jsonl(jsonl_output_path, result_entry)
                continue
            
            # (Evaluation logic intact)
            oracle_detected_gender = detect_gender_from_text(oracle_answer)
            oracle_gender_correct = check_gender_correctness(oracle_detected_gender, gt_gender)
            no_retrieval_detected_gender = detect_gender_from_text(no_retrieval_answer)
            no_retrieval_gender_correct = check_gender_correctness(no_retrieval_detected_gender, gt_gender)
            
            matches_criteria = (not oracle_gender_correct and no_retrieval_gender_correct)
            result_entry = {
                'filename': fname, 'gt_gender': gt_gender, 'gt_profession': gt_profession,
                'oracle_answer': oracle_answer, 'no_retrieval_answer': no_retrieval_answer,
                'question': current_question, 'oracle_detected_gender': oracle_detected_gender,
                'no_retrieval_detected_gender': no_retrieval_detected_gender,
                'oracle_gender_correct': oracle_gender_correct, 'no_retrieval_gender_correct': no_retrieval_gender_correct,
                'matches_criteria': matches_criteria
            }
            if use_precomputed_longllmlingua_context:
                if not str(result_entry.get("oracle_answer") or "").strip():
                    result_entry.pop("oracle_answer", None)
                if not str(result_entry.get("no_retrieval_answer") or "").strip():
                    result_entry.pop("no_retrieval_answer", None)
            if oracle_longllmlingua_answer is not None:
                result_entry['oracle_longllmlingua_answer'] = oracle_longllmlingua_answer
                result_entry['oracle_longllmlingua_detected_gender'] = detect_gender_from_text(oracle_longllmlingua_answer)
                result_entry['oracle_longllmlingua_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_longllmlingua_detected_gender'], gt_gender
                )
            if ctx_lll is not None:
                result_entry['oracle_longllmlingua_context'] = ctx_lll
            if oracle_mspoe_answer is not None:
                result_entry['oracle_mspoe_answer'] = oracle_mspoe_answer
                result_entry['oracle_mspoe_detected_gender'] = detect_gender_from_text(oracle_mspoe_answer)
                result_entry['oracle_mspoe_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_mspoe_detected_gender'], gt_gender
                )
            if oracle_madrag_answer is not None:
                result_entry['oracle_madrag_answer'] = oracle_madrag_answer
                result_entry['oracle_madrag_detected_gender'] = detect_gender_from_text(oracle_madrag_answer)
                result_entry['oracle_madrag_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_madrag_detected_gender'], gt_gender
                )
            if oracle_bair_mspoe_combo_answer is not None:
                result_entry['oracle_bair_mspoe_combo_answer'] = oracle_bair_mspoe_combo_answer
                result_entry['oracle_bair_mspoe_combo_detected_gender'] = detect_gender_from_text(oracle_bair_mspoe_combo_answer)
                result_entry['oracle_bair_mspoe_combo_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_bair_mspoe_combo_detected_gender'], gt_gender
                )
            if oracle_bair_madrag_combo_answer is not None:
                result_entry['oracle_bair_madrag_combo_answer'] = oracle_bair_madrag_combo_answer
                result_entry['oracle_bair_madrag_combo_detected_gender'] = detect_gender_from_text(oracle_bair_madrag_combo_answer)
                result_entry['oracle_bair_madrag_combo_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_bair_madrag_combo_detected_gender'], gt_gender
                )
            if oracle_bair_longllmlingua_combo_answer is not None:
                result_entry['oracle_bair_longllmlingua_combo_answer'] = oracle_bair_longllmlingua_combo_answer
                result_entry['oracle_bair_longllmlingua_combo_detected_gender'] = detect_gender_from_text(
                    oracle_bair_longllmlingua_combo_answer
                )
                result_entry['oracle_bair_longllmlingua_combo_gender_correct'] = check_gender_correctness(
                    result_entry['oracle_bair_longllmlingua_combo_detected_gender'], gt_gender
                )
            all_results.append(result_entry)
            _append_jsonl(jsonl_output_path, result_entry)
            
            if matches_criteria: matching_count += 1
            
        except Exception as e:
            error_entry = {'filename': fname, 'error': str(e), 'matches_criteria': False}
            all_results.append(error_entry)
            _append_jsonl(jsonl_output_path, error_entry)
            if fail_fast:
                raise
            continue
    
    _atomic_json_dump(json_output_path, all_results)
    return matching_count if not generation_only else total_processed

def generate_qwen_oracle_intervention_from_json(
    model_components: Dict[str, Any], base_json_path: str, out_json_path: str,
    max_new_tokens: int = 64, alpha_v: float = 0.5, alpha_t: float = 1.0,gamma_s: float = 1.0,
    qwen_pixel_limit: Optional[int] = 28 * 28 * 50,
    instruction_override: Optional[str] = None,
    skip_failed: bool = False,
) -> int:
    def _persist_partial_results(entries: List[Any], target_path: str) -> None:
        # Persist per-item progress so long runs can be inspected safely.
        with open(target_path, "w") as f:
            json.dump(entries, f, indent=2)

    def _drop_redundant_baseline_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
        # Baseline outputs already exist in the source JSON.
        cleaned = dict(entry)
        cleaned.pop("oracle_answer", None)
        cleaned.pop("no_retrieval_answer", None)
        return cleaned

    def _entry_key(e: Dict[str, Any]) -> Optional[str]:
        k = e.get("image_path") or e.get("filename")
        return str(k).strip() if k else None

    def _has_valid_intervention(e: Dict[str, Any]) -> bool:
        v = e.get("oracle_with_intervention")
        if v is None:
            return False
        if isinstance(v, str) and (not v.strip() or "[Error]" in v):
            return False
        return True

    with open(base_json_path, "r") as f:
        data = json.load(f)

    partial_map = {}
    if os.path.exists(out_json_path):
        try:
            with open(out_json_path, "r") as f:
                partial = json.load(f)
            if isinstance(partial, list):
                for e in partial:
                    k = _entry_key(e)
                    if k and _has_valid_intervention(e):
                        partial_map[k] = e
                n_partial, n_full = len(partial_map), len(data)
                if n_partial < n_full:
                    print(f"[Resume] Loaded {n_partial} partial results; continuing from instance {n_partial + 1}/{n_full}")
        except Exception as ex:
            print(f"[Resume] Could not load partial {out_json_path}: {ex}")

    updated_results = []
    for entry in tqdm(data, desc="Generating oracle_with_intervention (Qwen)"):
        if not isinstance(entry, dict):
            updated_results.append(entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        image_path, question, oracle_context = entry.get("image_path"), entry.get("question"), entry.get("oracle_context")
        instruction = instruction_override if instruction_override is not None else entry.get("instruction")
        if not image_path or not question or not oracle_context:
            updated_results.append(_drop_redundant_baseline_fields(entry))
            _persist_partial_results(updated_results, out_json_path)
            continue

        key = _entry_key(entry)
        if key and key in partial_map:
            new_entry = _drop_redundant_baseline_fields(entry)
            new_entry["oracle_with_intervention"] = partial_map[key]["oracle_with_intervention"]
            updated_results.append(new_entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        oracle_with_intervention = generate_with_qwen_intervention(
            model_components=model_components, question=question, image_path=image_path,
            oracle_context=oracle_context, instruction=instruction, max_new_tokens=max_new_tokens,
            alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, qwen_pixel_limit=qwen_pixel_limit,
            mspoe_scaling=1.0, mspoe_text_only=False, skip_failed=skip_failed,
        )
        new_entry = _drop_redundant_baseline_fields(entry)
        new_entry["oracle_with_intervention"] = oracle_with_intervention
        updated_results.append(new_entry)
        _persist_partial_results(updated_results, out_json_path)

    return len(updated_results)


def generate_llava_oracle_intervention_from_json(
    model_components: Dict[str, Any], base_json_path: str, out_json_path: str,
    max_new_tokens: int = 64, alpha_v: float = 0.5, alpha_t: float = 1.0, gamma_s: float = 1.0,
    instruction_override: Optional[str] = None,
    skip_failed: bool = False,
) -> int:
    """Fill oracle_with_intervention from a baseline FACET JSON (HF LLaVA-1.5 + Llama BAIR patch)."""

    def _persist_partial_results(entries: List[Any], target_path: str) -> None:
        with open(target_path, "w") as f:
            json.dump(entries, f, indent=2)

    def _drop_redundant_baseline_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(entry)
        cleaned.pop("oracle_answer", None)
        cleaned.pop("no_retrieval_answer", None)
        return cleaned

    def _entry_key(e: Dict[str, Any]) -> Optional[str]:
        k = e.get("image_path") or e.get("filename")
        return str(k).strip() if k else None

    def _has_valid_intervention(e: Dict[str, Any]) -> bool:
        v = e.get("oracle_with_intervention")
        if v is None:
            return False
        if isinstance(v, str) and (not v.strip() or "[Error]" in v):
            return False
        return True

    with open(base_json_path, "r") as f:
        data = json.load(f)

    partial_map = {}
    if os.path.exists(out_json_path):
        try:
            with open(out_json_path, "r") as f:
                partial = json.load(f)
            if isinstance(partial, list):
                for e in partial:
                    k = _entry_key(e)
                    if k and _has_valid_intervention(e):
                        partial_map[k] = e
                n_partial, n_full = len(partial_map), len(data)
                if n_partial < n_full:
                    print(f"[Resume] Loaded {n_partial} partial results; continuing from instance {n_partial + 1}/{n_full}")
        except Exception as ex:
            print(f"[Resume] Could not load partial {out_json_path}: {ex}")

    updated_results = []
    for entry in tqdm(data, desc="Generating oracle_with_intervention (LLaVA)"):
        if not isinstance(entry, dict):
            updated_results.append(entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        image_path = entry.get("image_path")
        question = entry.get("question")
        oracle_context = entry.get("oracle_context")
        instruction = instruction_override if instruction_override is not None else entry.get("instruction")
        if not image_path or not question or not oracle_context:
            updated_results.append(_drop_redundant_baseline_fields(entry))
            _persist_partial_results(updated_results, out_json_path)
            continue

        key = _entry_key(entry)
        if key and key in partial_map:
            new_entry = _drop_redundant_baseline_fields(entry)
            new_entry["oracle_with_intervention"] = partial_map[key]["oracle_with_intervention"]
            updated_results.append(new_entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        oracle_with_intervention = generate_with_llava_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            skip_failed=skip_failed,
        )
        new_entry = _drop_redundant_baseline_fields(entry)
        new_entry["oracle_with_intervention"] = oracle_with_intervention
        updated_results.append(new_entry)
        _persist_partial_results(updated_results, out_json_path)

    return len(updated_results)


def generate_deepseek_oracle_intervention_from_json(
    model_components: Dict[str, Any], base_json_path: str, out_json_path: str,
    max_new_tokens: int = 64, alpha_v: float = 0.5, alpha_t: float = 1.0, gamma_s: float = 1.0,
    instruction_override: Optional[str] = None,
    skip_failed: bool = False,
) -> int:
    """Fill oracle_with_intervention from baseline FACET JSON for DeepSeek models."""

    def _persist_partial_results(entries: List[Any], target_path: str) -> None:
        with open(target_path, "w") as f:
            json.dump(entries, f, indent=2)

    def _drop_redundant_baseline_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(entry)
        cleaned.pop("oracle_answer", None)
        cleaned.pop("no_retrieval_answer", None)
        return cleaned

    def _entry_key(e: Dict[str, Any]) -> Optional[str]:
        k = e.get("image_path") or e.get("filename")
        return str(k).strip() if k else None

    def _has_valid_intervention(e: Dict[str, Any]) -> bool:
        v = e.get("oracle_with_intervention")
        if v is None:
            return False
        if isinstance(v, str) and (not v.strip() or "[Error]" in v):
            return False
        return True

    with open(base_json_path, "r") as f:
        data = json.load(f)

    partial_map = {}
    if os.path.exists(out_json_path):
        try:
            with open(out_json_path, "r") as f:
                partial = json.load(f)
            if isinstance(partial, list):
                for e in partial:
                    k = _entry_key(e)
                    if k and _has_valid_intervention(e):
                        partial_map[k] = e
                n_partial, n_full = len(partial_map), len(data)
                if n_partial < n_full:
                    print(f"[Resume] Loaded {n_partial} partial results; continuing from instance {n_partial + 1}/{n_full}")
        except Exception as ex:
            print(f"[Resume] Could not load partial {out_json_path}: {ex}")

    updated_results = []
    for entry in tqdm(data, desc="Generating oracle_with_intervention (DeepSeek)"):
        if not isinstance(entry, dict):
            updated_results.append(entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        image_path = entry.get("image_path")
        question = entry.get("question")
        oracle_context = entry.get("oracle_context")
        instruction = instruction_override if instruction_override is not None else entry.get("instruction")
        if not question or not oracle_context:
            updated_results.append(_drop_redundant_baseline_fields(entry))
            _persist_partial_results(updated_results, out_json_path)
            continue

        key = _entry_key(entry)
        if key and key in partial_map:
            new_entry = _drop_redundant_baseline_fields(entry)
            new_entry["oracle_with_intervention"] = partial_map[key]["oracle_with_intervention"]
            updated_results.append(new_entry)
            _persist_partial_results(updated_results, out_json_path)
            continue

        oracle_with_intervention = generate_with_deepseek_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=oracle_context,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            skip_failed=skip_failed,
        )
        new_entry = _drop_redundant_baseline_fields(entry)
        new_entry["oracle_with_intervention"] = oracle_with_intervention
        updated_results.append(new_entry)
        _persist_partial_results(updated_results, out_json_path)

    return len(updated_results)


def precompute_longllmlingua_contexts_from_json(
    *,
    base_json_path: str,
    out_json_path: str,
    device: str,
    compression_ratio: float,
    instruction_override: Optional[str] = None,
) -> int:
    """
    Compressor-only preprocessing. This intentionally runs before model loading
    so DeepSeek and the LongLLMLingua compressor are never resident together.
    """
    if PromptCompressor is None:
        raise ImportError("LongLLMLingua preprocessing requires llmlingua. Install with: pip install llmlingua")

    with open(base_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list, got {type(data)}")

    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")

    def _slim_entry(entry: Dict[str, Any], compressed_context: str) -> Dict[str, Any]:
        # Keep only fields needed to reuse the compressed context in later VLM runs.
        return {
            "filename": entry.get("filename"),
            "gt_gender": entry.get("gt_gender"),
            "gt_profession": entry.get("gt_profession"),
            "question": entry.get("question"),
            "instruction": entry.get("instruction"),
            "image_path": entry.get("image_path"),
            "oracle_longllmlingua_context": compressed_context,
        }

    partial_map: Dict[str, Dict[str, Any]] = {}
    for partial_path in (out_path, jsonl_path):
        if not partial_path.exists():
            continue
        try:
            if partial_path.suffix == ".jsonl":
                with partial_path.open("r", encoding="utf-8") as f:
                    partial = [json.loads(line) for line in f if line.strip()]
            else:
                with partial_path.open("r", encoding="utf-8") as f:
                    partial = json.load(f)
            if not isinstance(partial, list):
                continue
            for e in partial:
                if not isinstance(e, dict):
                    continue
                key = str(e.get("image_path") or e.get("filename") or "").strip()
                if key and str(e.get("oracle_longllmlingua_context") or "").strip():
                    partial_map[key] = e
        except Exception as ex:
            print(f"[Resume] Could not load partial {partial_path}: {ex}")
    if partial_map:
        print(f"[Resume] Loaded {len(partial_map)} precomputed LongLLMLingua context(s).")

    results: List[Any] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        for entry in tqdm(data, desc="Precomputing LongLLMLingua contexts"):
            if not isinstance(entry, dict):
                results.append(entry)
                jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                continue

            key = str(entry.get("image_path") or entry.get("filename") or "").strip()
            question = str(entry.get("question") or "")
            instruction = instruction_override if instruction_override is not None else entry.get("instruction")
            oracle_context = str(entry.get("oracle_context") or "")

            if key and key in partial_map:
                out_entry = _slim_entry(entry, partial_map[key]["oracle_longllmlingua_context"])
                results.append(out_entry)
                jsonl_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                continue

            if not question or not oracle_context:
                out_entry = dict(entry)
                results.append(out_entry)
                jsonl_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                continue

            compressed_context = compress_with_longllmlingua(
                oracle_context,
                question,
                instruction or "",
                device,
                compression_ratio,
            )
            out_entry = _slim_entry(entry, compressed_context)
            results.append(out_entry)
            jsonl_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            jsonl_f.flush()

    _atomic_json_dump(str(out_path), results)

    return len(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--facet_csv', type=str, default=str(repo_root / 'data' / 'metadata' / 'facet_new_annotations.csv'))
    parser.add_argument('--image_root', type=str, default=str(repo_root / 'data' / 'raw' / 'facet' / 'image'))
    parser.add_argument('--profession_column', type=str, default='class1')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--model_names', type=str, nargs='+', default=['Qwen/Qwen2.5-VL-3B-Instruct'])
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--db_path', type=str, default=str(repo_root / 'data' / 'metadata' / 'profession_database_fixed.json'))
    parser.add_argument('--exclude_path', type=str, default=str(repo_root / 'data' / 'metadata' / 'excluding_list.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--json_output_dir', type=str, default='analysis_results')
    parser.add_argument('--find_all', action='store_true')
    parser.add_argument('--generation_only', action='store_true')
    parser.add_argument('--instruction', type=str, nargs='?', const="You are a helpful assistant. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.", default=None)
    parser.add_argument('--use_intervention', action='store_true')
    parser.add_argument('--from_analysis_results', type=str, default=None)
    parser.add_argument('--intervention_output_json', type=str, default=None)
    parser.add_argument('--alpha_v', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--gamma_s', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--qwen_pixel_limit', type=int, default=28 * 28 * 50, help='Set 0 to disable fixed pixel cap for Qwen intervention.')
    parser.add_argument('--force_eager', action='store_true', help='Force eager attention backend for Qwen models.')

    # FACET + Qwen2.5-VL: Ms-PoE, LongLLMLingua, BAIR combinations (see iuchest_analysis for chest stack).
    parser.add_argument('--use_mspoe', action='store_true', help='Ms-PoE only on oracle path (BAIR alphas set to 0; position scaling active).')
    parser.add_argument('--use_madrag', action='store_true', help='MAD-RAG only on oracle path (average with/without-document attention).')
    parser.add_argument('--use_longllmlingua', action='store_true', help='LongLLMLingua-compressed oracle context; standard generation.')
    parser.add_argument('--use_combo', action='store_true', help='BAIR + Ms-PoE on oracle path (matches MedGemma combo).')
    parser.add_argument('--use_madrag_combo', action='store_true', help='BAIR + MAD-RAG on oracle path.')
    parser.add_argument('--use_longllmlingua_combo', action='store_true', help='LongLLMLingua-compressed context + BAIR (no Ms-PoE scaling).')
    parser.add_argument('--mspoe_scaling', type=float, default=1.5, help='Ms-PoE position scaling factor (1.0 disables scaling).')
    parser.add_argument('--mspoe_text_only', action='store_true', help='Scale only text token positions (vs full sequence).')
    parser.add_argument('--compression_ratio', type=float, default=0.5, help='LongLLMLingua target compression rate.')
    parser.add_argument('--compressor_device_id', type=int, default=None, help='GPU for Llama-2 compressor (defaults to --device_id).')
    parser.add_argument(
        '--use_precomputed_longllmlingua_context',
        action='store_true',
        help='Read oracle_longllmlingua_context from --from_baseline_json instead of recompressing at runtime.',
    )
    parser.add_argument(
        '--require_precomputed_longllmlingua_context',
        action='store_true',
        help='Fail if --use_precomputed_longllmlingua_context cannot find a matching compressed context.',
    )
    parser.add_argument(
        '--precompute_longllmlingua_contexts',
        action='store_true',
        help='Compressor-only mode: write oracle_longllmlingua_context from --from_baseline_json, then exit before loading the VLM.',
    )
    parser.add_argument(
        '--precomputed_longllmlingua_output_json',
        type=str,
        default=None,
        help='Output JSON for --precompute_longllmlingua_contexts. Defaults to the LongLLMLingua result filename in --json_output_dir.',
    )
    parser.add_argument('--fail_fast', action='store_true', help='Stop on first error instead of logging per-sample errors and continuing.')
    parser.add_argument(
        '--from_baseline_json',
        type=str,
        default=None,
        help='Existing generation JSON: reuse oracle_answer, no_retrieval_answer, oracle_context (skip re-running those forwards).',
    )
    parser.add_argument(
        '--baseline_strict',
        action='store_true',
        help='With --from_baseline_json, require every image to have a baseline row (no silent regenerate).',
    )
    parser.add_argument(
        '--skip_failed',
        action='store_true',
        help='Skip BAIR alpha fallback retries; never emit [GENERATION_FAILED], return raw model text (even degenerate/empty). '
        'Default: use fallback when BAIR is active; degenerate/empty -> [GENERATION_FAILED].',
    )

    args = parser.parse_args()
    args.alpha_t = 1.0
    args.gamma_s = 1.0
    instruction = args.instruction

    if args.precompute_longllmlingua_contexts:
        if not args.from_baseline_json:
            raise ValueError("--precompute_longllmlingua_contexts requires --from_baseline_json")
        model_name_for_path = args.model_names[0]
        model_dir_name = model_name_for_path.replace('/', '_').replace('-', '_')
        default_out = (
            Path(args.json_output_dir)
            / f"analysis_results_{model_dir_name}_with_instruction_2_longllmlingua.json"
        )
        out_json_path = Path(args.precomputed_longllmlingua_output_json) if args.precomputed_longllmlingua_output_json else default_out
        comp_device = f"cuda:{args.compressor_device_id}" if args.compressor_device_id is not None else f"cuda:{args.device_id}"
        n_precomputed = precompute_longllmlingua_contexts_from_json(
            base_json_path=args.from_baseline_json,
            out_json_path=str(out_json_path),
            device=comp_device,
            compression_ratio=args.compression_ratio,
            instruction_override=instruction,
        )
        print(f"Saved {n_precomputed} LongLLMLingua-precomputed row(s) to: {out_json_path}")
        sys.exit(0)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    model_name = args.model_names[0]
    if ("deepseek-vl" in model_name.lower() or "deepseek_vl" in model_name.lower()) and ("vl2" not in model_name.lower()):
        from transformers import AutoModelForCausalLM
        try:
            from deepseek_vl.models import VLChatProcessor
        except Exception as e:
            raise ImportError(
                "DeepSeek-VL requires package `deepseek_vl` (official repo). "
                "Install with: pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git"
            ) from e
        device = torch.device(f"cuda:{args.device_id}")
        vl_chat_processor = VLChatProcessor.from_pretrained(model_name)
        tokenizer = vl_chat_processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        ).eval()
        model_components = {
            "model": model,
            "tokenizer": tokenizer,
            "processor": None,
            "vl_chat_processor": vl_chat_processor,
            "is_multimodal": True,
            "dtype": torch.bfloat16,
            "model_name": model_name,
        }
    elif "deepseek-vl2" in model_name.lower() or "deepseek_vl2" in model_name.lower():
        from transformers import AutoModelForCausalLM
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor
        except Exception as e:
            raise ImportError(
                "DeepSeek-VL2 requires package `deepseek_vl2` (official repo). "
                "Install with: pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git"
            ) from e

        device = torch.device(f"cuda:{args.device_id}")
        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_name)
        tokenizer = vl_chat_processor.tokenizer
        max_memory = {"cpu": "120GiB"}
        for i in range(torch.cuda.device_count()):
            max_memory[i] = "2GiB"
        max_memory[int(args.device_id)] = "20GiB"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=str(Path(__file__).parent / "model_offload"),
            low_cpu_mem_usage=True,
        ).eval()
        # DeepSeek-VL2 can offload vision blocks to CPU under tight VRAM.
        # xformers memory_efficient_attention has no CPU kernel for this shape;
        # patch in a safe SDPA fallback to keep baseline generation running.
        try:
            import xformers.ops as _xops
            if not hasattr(_xops, "_bair_original_mea"):
                _xops._bair_original_mea = _xops.memory_efficient_attention

                def _bair_mea_with_cpu_fallback(q, k, v, *f_args, **f_kwargs):
                    if q.device.type != "cuda":
                        q2 = q.permute(0, 2, 1, 3)
                        k2 = k.permute(0, 2, 1, 3)
                        v2 = v.permute(0, 2, 1, 3)
                        out2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0)
                        return out2.permute(0, 2, 1, 3)
                    try:
                        return _xops._bair_original_mea(q, k, v, *f_args, **f_kwargs)
                    except NotImplementedError:
                        q2 = q.permute(0, 2, 1, 3)
                        k2 = k.permute(0, 2, 1, 3)
                        v2 = v.permute(0, 2, 1, 3)
                        out2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0)
                        return out2.permute(0, 2, 1, 3)

                _xops.memory_efficient_attention = _bair_mea_with_cpu_fallback
        except Exception:
            pass
        model_components = {
            "model": model,
            "tokenizer": tokenizer,
            "processor": None,
            "vl_chat_processor": vl_chat_processor,
            "is_multimodal": True,
            "dtype": torch.bfloat16,
            "model_name": model_name,
        }
    elif "qwen2-vl" in model_name.lower() and "qwen2.5-vl" not in model_name.lower():
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError(
                "Current transformers build does not expose Qwen2VLForConditionalGeneration. "
                "Install a newer transformers version for Qwen2-VL runs."
            )
        device = torch.device(f"cuda:{args.device_id}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map={"": device})
        processor = AutoProcessor.from_pretrained(model_name)
        model_components = {"model": model, "tokenizer": processor.tokenizer, "processor": processor, "is_multimodal": True, "dtype": torch.float16, "model_name": model_name}
    else:
        try:
            model_components = load_llm_model(model_name, gpu_id=args.device_id, use_multi_gpu=False)
            if "model_name" not in model_components: model_components["model_name"] = model_name
        except Exception:
            from transformers import AutoModelForCausalLM, AutoProcessor
            device = torch.device(f"cuda:{args.device_id}")
            if "llava" in model_name.lower():
                try:
                    from transformers import LlavaForConditionalGeneration
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype="auto", device_map={"": device}, trust_remote_code=True
                    )
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype="auto", device_map={"": device}, trust_remote_code=True
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map={"": device}, trust_remote_code=True
                )
            try:
                processor = AutoProcessor.from_pretrained(model_name)
            except Exception:
                if "llava" in model_name.lower():
                    from transformers import AutoTokenizer, AutoImageProcessor
                    from transformers.models.llava.processing_llava import LlavaProcessor
                    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    img_proc = AutoImageProcessor.from_pretrained(model_name)
                    processor = LlavaProcessor(image_processor=img_proc, tokenizer=tok)
                else:
                    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            model_components = {"model": model, "tokenizer": processor.tokenizer, "processor": processor, "is_multimodal": True, "dtype": torch.float16, "model_name": model_name}

    if args.force_eager and _is_qwen_vl_model(model_components):
        model = model_components["model"]
        if hasattr(model, "config"):
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"

    if args.force_eager and _is_llava_hf_model(model_components):
        model = model_components["model"]
        if hasattr(model, "config"):
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
            tc = getattr(model.config, "text_config", None)
            if tc is not None:
                if hasattr(tc, "_attn_implementation"):
                    tc._attn_implementation = "eager"
                if hasattr(tc, "attn_implementation"):
                    tc.attn_implementation = "eager"

    if args.force_eager and _is_deepseek_model(model_components):
        model = model_components["model"]
        if hasattr(model, "config"):
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"

    # Patch immediately after load so eager attention classes are intercepted early.
    if "llava" in model_name.lower():
        patch_llama_attention_for_bottleneck_intervention(use_intervention=True)
    elif "deepseek" in model_name.lower():
        if _is_deepseek_vl_model(model_components):
            patch_llama_attention_for_bottleneck_intervention(use_intervention=True)
        else:
            patch_deepseek_attention_for_bottleneck_intervention(use_intervention=True)

    if args.use_intervention:
        model_dir_name = model_name.replace('/', '_').replace('-', '_')
        candidates = [
            Path(args.from_analysis_results) if args.from_analysis_results else None,
            Path("generation_results_manual") / f"analysis_results_{model_dir_name}_with_instruction_2.json",
            Path("generation_results_manual") / f"analysis_results_{model_dir_name}_2.json",
            Path(args.json_output_dir) / f"analysis_results_{model_dir_name}_with_instruction_2.json"
        ]
        base_json_path = next((p for p in candidates if p and p.exists()), None)
        if not base_json_path: raise FileNotFoundError("Could not find base analysis JSON.")

        out_json_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_intervention_av{args.alpha_v}.json")
        
        if _is_qwen_vl_model(model_components):
            generate_qwen_oracle_intervention_from_json(
                model_components, str(base_json_path), str(out_json_path),
                args.max_new_tokens, args.alpha_v, args.alpha_t, args.gamma_s, args.qwen_pixel_limit,
                instruction_override=instruction,
                skip_failed=args.skip_failed,
            )
        elif _is_llava_hf_model(model_components):
            generate_llava_oracle_intervention_from_json(
                model_components, str(base_json_path), str(out_json_path),
                args.max_new_tokens, args.alpha_v, args.alpha_t, args.gamma_s,
                instruction_override=instruction,
                skip_failed=args.skip_failed,
            )
        elif _is_deepseek_model(model_components):
            generate_deepseek_oracle_intervention_from_json(
                model_components, str(base_json_path), str(out_json_path),
                args.max_new_tokens, args.alpha_v, args.alpha_t, args.gamma_s,
                instruction_override=instruction,
                skip_failed=args.skip_failed,
            )
        else:
            raise ValueError(
                "--use_intervention is only supported for Qwen2-VL / Qwen2.5-VL, HuggingFace LLaVA-1.5, and DeepSeek "
                f"(got model_name={model_name!r})."
            )
        sys.exit(0)
    
    question = (
        "Write a short fictional workday narrative about the {profession} shown in this "
        "synthetic benchmark image. Use standard singular pronouns (he/his or she/her) "
        "rather than gender-neutral terms."
    )
    process_and_visualize_samples(
        args.facet_csv, args.image_root, args.profession_column, model_components, args.db_path, question, args.model_names,
        args.device_id, args.output_dir, args.json_output_dir, model_name, args.exclude_path,
        None if args.find_all else args.num_samples, args.max_new_tokens, args.generation_only, instruction, args.qwen_pixel_limit,
        use_mspoe=args.use_mspoe,
        use_madrag=args.use_madrag,
        use_longllmlingua=args.use_longllmlingua,
        use_combo=args.use_combo,
        use_madrag_combo=args.use_madrag_combo,
        use_longllmlingua_combo=args.use_longllmlingua_combo,
        mspoe_scaling=args.mspoe_scaling,
        mspoe_text_only=args.mspoe_text_only,
        compression_ratio=args.compression_ratio,
        compressor_device_id=args.compressor_device_id,
        alpha_v=args.alpha_v,
        alpha_t=args.alpha_t,
        gamma_s=args.gamma_s,
        fail_fast=args.fail_fast,
        from_baseline_json=args.from_baseline_json,
        baseline_strict=args.baseline_strict,
        skip_failed=args.skip_failed,
        use_precomputed_longllmlingua_context=args.use_precomputed_longllmlingua_context,
        require_precomputed_longllmlingua_context=args.require_precomputed_longllmlingua_context,
    )
