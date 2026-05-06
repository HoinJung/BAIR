"""
IU-Chest runner for MedGemma and CheXagent.

Baselines, BAIR, Ms-PoE, positional RAG, and LongLLMLingua-style modes.
Select backbone with:  --model medgemma | chexagent
"""

import os
import json
import random
import re
import sys
import traceback
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from PIL import Image

try:
    from llmlingua import PromptCompressor
    _LLMLINGUA_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _e:
    PromptCompressor = None  # type: ignore[misc, assignment]
    _LLMLINGUA_IMPORT_ERROR = _e

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Recorruption" / "src" / "medical_rag"))

from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from bair import bair_efficient
from bair.bottleneck_intervention import (
    set_bottleneck_intervention,
    patch_gemma3_attention_for_bottleneck_intervention,
    patch_mistral_attention_for_bottleneck_intervention,
)

MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MG_MODEL, MG_PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS = None, None, None

CHEXAGENT_ID = "StanfordAIMI/CheXagent-8b"
CHEX_MODEL, CHEX_PROCESSOR, CHEX_GENERATION_CONFIG = None, None, None

LONGLLMLINGUA_COMPRESSOR = None

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_consistent_subset(records: List[Dict], fraction: float, split_seed: int) -> List[Dict]:
    """
    Pick a deterministic subset so all experiment modes run on the same samples.
    Selection is based on sorted UIDs + fixed RNG seed.
    """
    if fraction >= 1.0:
        return records
    if fraction <= 0.0:
        raise ValueError(f"--test_fraction must be in (0, 1], got {fraction}")
    if not records:
        return records

    records_sorted = sorted(records, key=lambda x: str(x.get("uid", "")))
    k = max(1, int(len(records_sorted) * fraction))
    rng = random.Random(split_seed)
    selected_indices = sorted(rng.sample(range(len(records_sorted)), k))
    return [records_sorted[i] for i in selected_indices]

def _fraction_tag(value: float) -> str:
    # 0.2 -> "f20p0", 0.125 -> "f12p5"
    return f"f{value * 100:.1f}".replace(".", "p")

def _resolve_image_path(image_value, data_dir: Path, image_subdir: str = "images_normalized") -> str:
    path = Path(str(image_value))
    if path.is_absolute() or path.exists():
        return str(path)
    if len(path.parts) > 1:
        candidate = data_dir / path
        if candidate.exists():
            return str(candidate)
        return str(path)
    return str(data_dir / image_subdir / path)

def get_unified_dataset(args) -> List[Dict]:
    data_dir = Path(args.data_dir)
    repo_root = Path(__file__).resolve().parent.parent
    records = []
    if args.dataset == "iuchest":
        db_path = repo_root / "data" / "generated" / "iuchest_nih_retrieval_dataset.json"
        with open(db_path, "r") as f: data = json.load(f)
        for item in data:
            records.append({
                "uid": str(item["uid"]),
                # Store as plain string so progress JSON is always serializable.
                "image_path": _resolve_image_path(item["image_filename"], data_dir),
                "context": item["nih_context"],
                "gt_report": item["ground_truth_report"],
                "gt_problems": item["ground_truth_problems"]
            })
    return records

def reorder_nih_context(context: str, gt_target_index: int) -> str:
    parts = re.split(r'--- Document \d ---', context)
    docs = [p.strip() for p in parts if p.strip()]
    if len(docs) != 5: return context 
    gt_doc = docs[2]
    new_docs = [docs[0], docs[1], docs[3], docs[4]]
    new_docs.insert(gt_target_index, gt_doc)
    return "\n\n".join([f"--- Document {i+1} ---\n{doc}" for i, doc in enumerate(new_docs)])

def extract_gt_only_context(context: str) -> str:
    """
    Keep only the GT report document (Document 3) from NIH 5-document context.
    Falls back to original context if parsing does not match expected format.
    """
    parts = re.split(r'--- Document \d ---', context)
    docs = [p.strip() for p in parts if p.strip()]
    if len(docs) != 5:
        return context
    return f"--- Document 1 ---\n{docs[2]}"

def load_medgemma_intervention_model(device: str):
    global MG_MODEL, MG_PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS
    if MG_MODEL is None:
        print(f"Loading MedGemma Intervention Model on {device}...")
        MG_PROCESSOR = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
        MG_MODEL = Gemma3ForConditionalGeneration.from_pretrained(
            MEDGEMMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="eager"
        )
        MG_MODEL.eval()
        INTERVENTION_NUM_VISUAL_TOKENS = getattr(MG_MODEL.config, "mm_tokens_per_image", 256)
        patch_gemma3_attention_for_bottleneck_intervention(use_intervention=True)
    return MG_MODEL, MG_PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS

def load_longllmlingua(device: str):
    global LONGLLMLINGUA_COMPRESSOR
    if LONGLLMLINGUA_COMPRESSOR is None:
        print("\nInitializing LongLLMLingua Compressor...")
        LONGLLMLINGUA_COMPRESSOR = PromptCompressor(model_name="NousResearch/Llama-2-7b-hf", model_config={"torch_dtype": torch.bfloat16}, device_map=device)
        class LLMLinguaModelWrapper:
            def __init__(self, model): self._model = model
            def __call__(self, *args, **kwargs):
                from transformers.cache_utils import DynamicCache
                if 'past_key_values' in kwargs and isinstance(kwargs['past_key_values'], list):
                    kwargs['past_key_values'] = DynamicCache.from_legacy_cache(tuple(kwargs['past_key_values']))
                out = self._model(*args, **kwargs)
                if hasattr(out, 'past_key_values') and hasattr(out.past_key_values, 'to_legacy_cache'):
                    out.past_key_values = list(out.past_key_values.to_legacy_cache())
                return out
            def __getattr__(self, name): return getattr(self._model, name)
        LONGLLMLINGUA_COMPRESSOR.model = LLMLinguaModelWrapper(LONGLLMLINGUA_COMPRESSOR.model)
    return LONGLLMLINGUA_COMPRESSOR

def build_full_prompt(question: str, context: Optional[str] = None, instruction: Optional[str] = None) -> str:
    if instruction:
        if context: return f"Instruction: {instruction}\n\nContext:\n{context}\n\nQuestion: {question}"
        return f"Instruction: {instruction}\n\nQuestion: {question}"
    if context: return f"Context:\n{context}\n\n{question}"
    return question

def compress_with_longllmlingua(context: str, question: str, instruction: str, device: str, ratio: float = 0.5) -> str:
    compressor = load_longllmlingua(device)
    parts = re.split(r'--- Document \d ---', context)
    docs = [p.strip() for p in parts if p.strip()]
    if not docs: return context
    res = compressor.compress_prompt(
        context=docs, instruction=instruction if instruction else "", question=question,
        rate=ratio, condition_in_question='after_condition', reorder_context="sort_based_on_metric",
        dynamic_context_compression_ratio=0.4, rank_method="longllmlingua" 
    )
    return res["compressed_prompt"]

def apply_mspoe_position_hook_medgemma(model, scaling_factor: float, text_only: bool, num_visual_tokens: int = 256):
    if scaling_factor == 1.0: return None
    def pre_forward_hook(module, args, kwargs):
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            pos_ids = kwargs["position_ids"].float()
            if text_only:
                mask = pos_ids >= num_visual_tokens
                pos_ids[mask] = num_visual_tokens + (pos_ids[mask] - num_visual_tokens) / scaling_factor
            else: pos_ids = pos_ids / scaling_factor
            kwargs["position_ids"] = pos_ids.long() 
        return args, kwargs
    target = model.model if hasattr(model, "model") else (model.language_model.model if hasattr(model, "language_model") else model)
    return target.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

def generate_standard_medgemma(image_path: str, question: str, context: Optional[str], max_new_tokens: int, instruction: Optional[str], device: str) -> str:
    """Standard generation bypassing all interventions for baseline & LLMLingua."""
    model, processor, _ = load_medgemma_intervention_model(device)
    image = Image.open(image_path).convert("RGB")
    full_prompt = build_full_prompt(question, context, instruction)
    
    text_with_placeholder = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}], tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=text_with_placeholder, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    set_bottleneck_intervention(False)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def generate_with_bair_and_mspoe_medgemma(
    image_path: str, question: str, context: Optional[str], max_new_tokens: int, 
    instruction: Optional[str], alpha_v: float, alpha_t: float, gamma_s: float, device: str, 
    scaling_factor: float, text_only: bool, use_madrag: bool = False
) -> str:
    model, processor, num_visual_tokens = load_medgemma_intervention_model(device)
    hook_handle = apply_mspoe_position_hook_medgemma(model, scaling_factor, text_only, num_visual_tokens)
    image = Image.open(image_path).convert("RGB")

    clean_prompt = build_full_prompt(question, None, instruction)
    clean_text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "image"}]}], tokenize=False, add_generation_prompt=True)
    clean_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processor(images=image, text=clean_text, return_tensors="pt", padding=True).items()}
    
    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=True, reset_layer=True, alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, use_madrag=use_madrag)
    with torch.no_grad(): model(**clean_inputs)

    full_prompt = build_full_prompt(question, context, instruction)
    text_with_placeholder = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}], tokenize=False, add_generation_prompt=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processor(images=image, text=text_with_placeholder, return_tensors="pt", padding=True).items()}

    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=False, reset_layer=True, alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, use_madrag=use_madrag)
    with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if hook_handle: hook_handle.remove()
    set_bottleneck_intervention(False)
    
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_chexagent_intervention_model(device: str):
    global CHEX_MODEL, CHEX_PROCESSOR, CHEX_GENERATION_CONFIG
    if CHEX_MODEL is None:
        print(f"Loading CheXagent Intervention Model on {device}...")

        import transformers.models.mistral.modeling_mistral as mistral_mod

        if hasattr(mistral_mod, "MISTRAL_ATTENTION_CLASSES"):
            mistral_mod.MISTRAL_ATTENTION_CLASSES["sdpa"] = mistral_mod.MistralAttention
            mistral_mod.MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = mistral_mod.MistralAttention

        if hasattr(mistral_mod, "MistralSdpaAttention"):
            mistral_mod.MistralSdpaAttention = mistral_mod.MistralAttention
        if hasattr(mistral_mod, "MistralFlashAttention2"):
            mistral_mod.MistralFlashAttention2 = mistral_mod.MistralAttention

        CHEX_PROCESSOR = AutoProcessor.from_pretrained(CHEXAGENT_ID, trust_remote_code=True)
        CHEX_GENERATION_CONFIG = GenerationConfig.from_pretrained(CHEXAGENT_ID)
        CHEX_GENERATION_CONFIG.use_cache = True

        config = AutoConfig.from_pretrained(CHEXAGENT_ID, trust_remote_code=True)
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = "eager"

        CHEX_MODEL = AutoModelForCausalLM.from_pretrained(
            CHEXAGENT_ID,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="eager",
        )
        CHEX_MODEL.eval()

        patch_mistral_attention_for_bottleneck_intervention(use_intervention=True)

        if hasattr(CHEX_MODEL, "language_model") and hasattr(CHEX_MODEL.language_model.model, "layers"):
            layer_type = type(CHEX_MODEL.language_model.model.layers[0].self_attn).__name__
            print(f"\n[BAIR Security Check] Layer 0 Attention Type = {layer_type}")
            if layer_type != "MistralAttention":
                print(">> [WARNING] Eager override failed! Intervention might be silent!")
            else:
                print(">> [SUCCESS] SDPA successfully blocked. BAIR is fully active.")

    return CHEX_MODEL, CHEX_PROCESSOR, CHEX_GENERATION_CONFIG


def format_chex_prompt(question: str, context: Optional[str] = None, instruction: Optional[str] = None) -> str:
    user_prompt = question
    if context:
        user_prompt = f"Context:\n{context}\n\n{user_prompt}"
    if instruction:
        user_prompt = f"{instruction}\n\n{user_prompt}"
    return f" USER: <s>{user_prompt} ASSISTANT: <s>"


def apply_mspoe_position_hook_chexagent(model, scaling_factor: float, text_only: bool, num_visual_tokens: int):
    if scaling_factor == 1.0:
        return None

    target_module = model.language_model.model
    target_module._mspoe_delta = 0

    def pre_forward_hook(module, args, kwargs):
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            pos_ids = kwargs["position_ids"].clone()
            seq_len = pos_ids.shape[-1]

            if seq_len > 1:
                float_pos = pos_ids.float()
                if text_only:
                    mask = float_pos >= num_visual_tokens
                    float_pos[mask] = num_visual_tokens + (float_pos[mask] - num_visual_tokens) / scaling_factor
                else:
                    float_pos = float_pos / scaling_factor

                scaled_pos = float_pos.long()
                real_last = pos_ids[0, -1].item()
                scaled_last = scaled_pos[0, -1].item()
                module._mspoe_delta = real_last - scaled_last

                kwargs["position_ids"] = scaled_pos
            else:
                delta = getattr(module, "_mspoe_delta", 0)
                kwargs["position_ids"] = pos_ids - delta

        return args, kwargs

    return target_module.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)


def generate_standard_chexagent(
    image_path: str,
    question: str,
    context: Optional[str],
    max_new_tokens: int,
    instruction: Optional[str],
    device: str,
) -> str:
    model, processor, gen_config = load_chexagent_intervention_model(device)
    image = Image.open(image_path).convert("RGB")
    full_prompt = format_chex_prompt(question, context, instruction)

    inputs = processor(images=[image], text=full_prompt, return_tensors="pt")
    inputs = {
        k: v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device) for k, v in inputs.items()
    }

    set_bottleneck_intervention(False)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )[0]

    res = processor.tokenizer.decode(output, skip_special_tokens=True)
    if "ASSISTANT: <s>" in res:
        return res.split("ASSISTANT: <s>", 1)[1].split("</s>")[0].split("USER:")[0].strip()
    return res.strip()


def generate_with_bair_and_mspoe_chexagent(
    image_path: str,
    question: str,
    context: Optional[str],
    max_new_tokens: int,
    instruction: Optional[str],
    alpha_v: float,
    alpha_t: float,
    gamma_s: float,
    device: str,
    scaling_factor: float,
    text_only: bool,
    use_madrag: bool = False,
) -> str:
    model, processor, gen_config = load_chexagent_intervention_model(device)
    image = Image.open(image_path).convert("RGB")

    clean_prompt = format_chex_prompt(question, None, instruction)
    clean_inputs = processor(images=[image], text=clean_prompt, return_tensors="pt")
    clean_inputs = {
        k: v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device)
        for k, v in clean_inputs.items()
    }

    total_seq_len = [0]

    def catch_seq_len(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        total_seq_len[0] = hidden_states.shape[1]
        raise RuntimeError("Caught Seq Len")

    handle = model.language_model.model.layers[0].register_forward_pre_hook(catch_seq_len, with_kwargs=True)

    try:
        with torch.no_grad():
            model(**clean_inputs)
    except RuntimeError as e:
        if "Caught Seq Len" not in str(e):
            raise e
    finally:
        handle.remove()

    if total_seq_len[0] == 0:
        raise ValueError("[BAIR Error] Hook bypassed! Vision encoder might have crashed.")

    text_len = clean_inputs["input_ids"].shape[1]
    num_visual_tokens = total_seq_len[0] - text_len

    if num_visual_tokens <= 0:
        raise ValueError(f"[BAIR Error] Invalid visual token count: {num_visual_tokens}")

    tail_text = f"\n\n{question} ASSISTANT: <s>"
    safe_tail_tokens = len(processor.tokenizer(tail_text).input_ids) + 10

    hook_handle = apply_mspoe_position_hook_chexagent(model, scaling_factor, text_only, num_visual_tokens)

    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=True,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        gamma_s=gamma_s,
        question_tokens=safe_tail_tokens,
        use_madrag=use_madrag,
    )
    with torch.no_grad():
        model(**clean_inputs)
    del clean_inputs
    bair_efficient.optional_empty_cache_after_calibration()

    full_prompt = format_chex_prompt(question, context, instruction)
    gen_inputs = processor(images=[image], text=full_prompt, return_tensors="pt")
    gen_inputs = {
        k: v.to(device).to(torch.bfloat16) if k == "pixel_values" else v.to(device)
        for k, v in gen_inputs.items()
    }

    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=False,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        gamma_s=gamma_s,
        question_tokens=safe_tail_tokens,
        use_madrag=use_madrag,
    )

    with torch.no_grad():
        output = model.generate(
            **gen_inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )[0]

    if hook_handle:
        hook_handle.remove()
    set_bottleneck_intervention(False)

    res = processor.tokenizer.decode(output, skip_special_tokens=True)
    if "ASSISTANT: <s>" in res:
        return res.split("ASSISTANT: <s>", 1)[1].split("</s>")[0].split("USER:")[0].strip()
    return res.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IU-Chest runner for MedGemma or CheXagent (BAIR + baselines + ablations)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["medgemma", "chexagent"],
        help="VLM backbone: google/medgemma-4b-it or StanfordAIMI/CheXagent-8b.",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["iuchest"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--uids",
        type=str,
        default=None,
        help="Comma-separated UIDs to process (overrides test_fraction/num_samples when set).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate deterministically (0,1].",
    )
    parser.add_argument("--test_split_seed", type=int, default=42)
    parser.add_argument("--device_id", type=int, default=3)
    parser.add_argument("--compressor_device_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: generation_results_medgemma or generation_results_chexagent).",
    )

    parser.add_argument("--generate_baselines", action="store_true")
    parser.add_argument("--use_intervention", action="store_true")
    parser.add_argument(
        "--intervention_use_strong_instruction",
        action="store_true",
        help="With --use_intervention, use --strong_instruction and tag output filename.",
    )
    parser.add_argument("--use_mspoe", action="store_true")
    parser.add_argument("--use_madrag", action="store_true")
    parser.add_argument("--use_combo", action="store_true")
    parser.add_argument("--use_madrag_combo", action="store_true")
    parser.add_argument("--use_longllmlingua", action="store_true")
    parser.add_argument("--use_longllmlingua_combo", action="store_true")

    parser.add_argument("--context_mode", type=str, default="five_docs", choices=["five_docs", "gt_only"])
    parser.add_argument("--alpha_v", type=float, default=0.5)
    parser.add_argument("--alpha_t", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--gamma_s", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--combo_text_only", action="store_true")
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument(
        "--question",
        type=str,
        default="Based on the visual evidence, what are the primary impressions for this chest radiograph?",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a radiologist. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.",
    )
    parser.add_argument(
        "--strong_instruction",
        type=str,
        default="You are an expert radiologist. Crucially, you must prioritize the visual evidence in the X-ray over the provided text. The text contains historical distractors. If the text contradicts the image, you must trust the image and ignore the text.",
    )

    args = parser.parse_args()
    args.alpha_t = 1.0
    args.gamma_s = 1.0
    if args.output_dir is None:
        args.output_dir = (
            "generation_results_medgemma" if args.model == "medgemma" else "generation_results_chexagent"
        )

    if args.intervention_use_strong_instruction and not args.use_intervention:
        parser.error("--intervention_use_strong_instruction requires --use_intervention")

    if (args.use_longllmlingua or args.use_longllmlingua_combo) and PromptCompressor is None:
        msg = "LongLLMLingua requires the `llmlingua` package. Install with: pip install llmlingua\n"
        if _LLMLINGUA_IMPORT_ERROR is not None:
            msg += f"Original import error: {_LLMLINGUA_IMPORT_ERROR!r}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    set_all_seeds(args.seed)

    device = f"cuda:{args.device_id}"
    comp_device = f"cuda:{args.compressor_device_id}" if args.compressor_device_id is not None else device
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset.upper()} dataset (model={args.model})...")
    records = get_unified_dataset(args)
    if args.uids:
        want = {str(x).strip().replace(".0", "") for x in args.uids.split(",") if x.strip()}
        records = [r for r in records if str(r.get("uid", "")).strip().replace(".0", "") in want]
        print(f"Using {len(records)} records (filtered by --uids: {sorted(want)})")
    else:
        records = select_consistent_subset(records, args.test_fraction, args.test_split_seed)
        random.shuffle(records)
        if args.num_samples:
            records = records[: args.num_samples]
        print(
            f"Using {len(records)} records after deterministic split "
            f"(fraction={args.test_fraction}, split_seed={args.test_split_seed})."
        )

    mode_parts = []
    if args.generate_baselines:
        mode_parts.append("baselines")
    if args.use_intervention:
        if args.intervention_use_strong_instruction:
            mode_parts.append(f"new_bair_strong_prompt_av{args.alpha_v}_mid")
        else:
            mode_parts.append(f"new_bair_av{args.alpha_v}_mid")
    if args.use_mspoe:
        mspoe_variant = "mspoe_text" if args.combo_text_only else "mspoe_full"
        mode_parts.append(f"{mspoe_variant}_mid")
    if args.use_madrag:
        mode_parts.append("madrag_mid")
    if args.use_combo:
        combo_variant = "text" if args.combo_text_only else "full"
        mode_parts.append(f"combo_{combo_variant}_av{args.alpha_v}_mid")
    if args.use_madrag_combo:
        mode_parts.append(f"madrag_combo_av{args.alpha_v}_mid")
    if args.use_longllmlingua:
        mode_parts.append("longllmlingua_mid")
    if args.use_longllmlingua_combo:
        mode_parts.append(f"lll_combo_av{args.alpha_v}_mid")
    if args.context_mode == "gt_only":
        mode_parts.append("ctx_gt_only")

    suffix = "_" + "_".join(mode_parts) if mode_parts else "_default"
    if args.test_fraction < 1.0 and not args.uids:
        suffix += f"_{_fraction_tag(args.test_fraction)}_seed{args.test_split_seed}"

    backbone_tag = "medgemma" if args.model == "medgemma" else "chexagent"
    out_file = Path(args.output_dir) / f"{args.dataset}_{backbone_tag}_results{suffix}.json"

    def _clean_uid(v):
        return str(v).strip().replace(".0", "")

    def _has_valid_value(entry, key):
        v = entry.get(key)
        if v is None:
            return False
        if isinstance(v, str) and (not v.strip() or "[Error]" in v):
            return False
        if args.model == "chexagent" and isinstance(v, str) and "Cons Cons" in v:
            return False
        return True

    target_keys = []
    if args.generate_baselines:
        target_keys.extend(["no_retrieval_answer", "oracle_answer", "prompt_baseline_answer"])
    if args.use_intervention:
        target_keys.append("oracle_with_intervention")
    if args.use_mspoe:
        mspoe_prefix = "mspoe_text" if args.combo_text_only else "mspoe_full"
        target_keys.append(f"{mspoe_prefix}_answer")
    if args.use_madrag:
        target_keys.append("madrag_answer")
    if args.use_combo:
        target_keys.append("combo_mid_answer")
    if args.use_madrag_combo:
        target_keys.append("madrag_combo_mid_answer")
    if args.use_longllmlingua:
        target_keys.append("longllmlingua_mid_answer")
    if args.use_longllmlingua_combo:
        target_keys.append("longllmlingua_combo_mid_answer")

    all_results = [dict(r) for r in records]
    partial_map = {}
    if out_file.exists():
        try:
            partial = json.load(open(out_file))
            if isinstance(partial, list):
                for e in partial:
                    uid = _clean_uid(e.get("uid") or e.get("image", ""))
                    if uid:
                        partial_map[uid] = e
                n_partial, n_full = len(partial_map), len(records)
                if n_partial < n_full:
                    print(f"[Resume] Loaded {n_partial} partial results; continuing from instance {n_partial + 1}/{n_full}")
        except Exception as ex:
            print(f"[Resume] Could not load partial {out_file}: {ex}")

    for entry in all_results:
        uid = _clean_uid(entry.get("uid") or entry.get("image", ""))
        pe = partial_map.get(uid)
        if pe:
            for k in target_keys:
                if _has_valid_value(pe, k):
                    entry[k] = pe[k]

    data_root = Path(args.data_dir)
    for entry in all_results:
        raw_path = entry.get("image_path")
        if raw_path is None:
            continue
        entry["image_path"] = _resolve_image_path(raw_path, data_root)

    is_mg = args.model == "medgemma"

    for entry in tqdm(all_results, desc=f"Processing {args.dataset} ({args.model})"):
        if all(_has_valid_value(entry, k) for k in target_keys):
            continue

        image_path = str(entry["image_path"])
        q, ins, s_ins = args.question, args.instruction, args.strong_instruction
        base_ctx = entry["context"]

        if args.context_mode == "gt_only":
            ctx_mid = extract_gt_only_context(base_ctx)
            target_ctx = ctx_mid
        else:
            ctx_mid = reorder_nih_context(base_ctx, 2)
            target_ctx = reorder_nih_context(base_ctx, 2)

        try:
            if args.generate_baselines:
                if not _has_valid_value(entry, "no_retrieval_answer"):
                    entry["no_retrieval_answer"] = (
                        generate_standard_medgemma(image_path, q, None, args.max_new_tokens, ins, device)
                        if is_mg
                        else generate_standard_chexagent(image_path, q, None, args.max_new_tokens, ins, device)
                    )
                if not _has_valid_value(entry, "oracle_answer"):
                    entry["oracle_answer"] = (
                        generate_standard_medgemma(image_path, q, ctx_mid, args.max_new_tokens, ins, device)
                        if is_mg
                        else generate_standard_chexagent(image_path, q, ctx_mid, args.max_new_tokens, ins, device)
                    )
                if not _has_valid_value(entry, "prompt_baseline_answer"):
                    entry["prompt_baseline_answer"] = (
                        generate_standard_medgemma(image_path, q, ctx_mid, args.max_new_tokens, s_ins, device)
                        if is_mg
                        else generate_standard_chexagent(image_path, q, ctx_mid, args.max_new_tokens, s_ins, device)
                    )

            if args.use_intervention:
                key = "oracle_with_intervention"
                if not _has_valid_value(entry, key):
                    intervention_ins = s_ins if args.intervention_use_strong_instruction else ins
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            intervention_ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            intervention_ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                        )
                    )

            if args.use_mspoe:
                prefix = "mspoe_text" if args.combo_text_only else "mspoe_full"
                key = f"{prefix}_answer"
                if not _has_valid_value(entry, key):
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            0.0,
                            0.0,
                            1.0,
                            device,
                            1.5,
                            args.combo_text_only,
                            use_madrag=False,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            0.0,
                            0.0,
                            1.0,
                            device,
                            1.5,
                            args.combo_text_only,
                            use_madrag=False,
                        )
                    )

            if args.use_madrag:
                key = "madrag_answer"
                if not _has_valid_value(entry, key):
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            0.0,
                            0.0,
                            1.0,
                            device,
                            1.0,
                            False,
                            use_madrag=True,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            0.0,
                            0.0,
                            1.0,
                            device,
                            1.0,
                            False,
                            use_madrag=True,
                        )
                    )

            if args.use_combo:
                key = "combo_mid_answer"
                if not _has_valid_value(entry, key):
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.5,
                            args.combo_text_only,
                            use_madrag=False,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.5,
                            args.combo_text_only,
                            use_madrag=False,
                        )
                    )

            if args.use_madrag_combo:
                key = "madrag_combo_mid_answer"
                if not _has_valid_value(entry, key):
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                            use_madrag=True,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            target_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                            use_madrag=True,
                        )
                    )

            if args.use_longllmlingua:
                key = "longllmlingua_mid_answer"
                if not _has_valid_value(entry, key):
                    comp_ctx = compress_with_longllmlingua(target_ctx, q, ins, comp_device, args.compression_ratio)
                    entry[key] = (
                        generate_standard_medgemma(image_path, q, comp_ctx, args.max_new_tokens, ins, device)
                        if is_mg
                        else generate_standard_chexagent(image_path, q, comp_ctx, args.max_new_tokens, ins, device)
                    )

            if args.use_longllmlingua_combo:
                key = "longllmlingua_combo_mid_answer"
                if not _has_valid_value(entry, key):
                    comp_ctx = compress_with_longllmlingua(target_ctx, q, ins, comp_device, args.compression_ratio)
                    entry[key] = (
                        generate_with_bair_and_mspoe_medgemma(
                            image_path,
                            q,
                            comp_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                        )
                        if is_mg
                        else generate_with_bair_and_mspoe_chexagent(
                            image_path,
                            q,
                            comp_ctx,
                            args.max_new_tokens,
                            ins,
                            args.alpha_v,
                            args.alpha_t,
                            args.gamma_s,
                            device,
                            1.0,
                            False,
                        )
                    )

        except Exception as e:
            print(f"\n[Error] Failed on {entry['uid']}: {e}")
            if args.model == "chexagent":
                traceback.print_exc()

        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
