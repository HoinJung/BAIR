"""
Unified MedGemma Analysis for IU-Chest and MIMIC-CXR datasets.
Includes Baselines, BAIR, Ms-PoE, Positional RAG, and LongLLMLingua interventions.
"""

import os
import json
import random
import re
import sys
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from PIL import Image

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Recorruption" / "src" / "medical_rag"))

from transformers import AutoProcessor
from custom_medgemma_model import MedGemmaForConditionalGenerationCustom
from bottleneck_intervention import (
    set_bottleneck_intervention,
    patch_gemma3_attention_for_bottleneck_intervention,
)

MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MODEL, PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS = None, None, None
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
    base_dir = Path(__file__).resolve().parent
    records = []
    if args.dataset == "iuchest":
        db_path = base_dir / "data" / "generated" / "iuchest_nih_retrieval_dataset.json"
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
    elif args.dataset == "mimic":
        db_path = base_dir / "data" / "generated" / "mimic_nih_retrieval_dataset_findings_only.json"
        if not db_path.exists(): raise FileNotFoundError(f"Missing {db_path} for 5-document mode.")
        with open(db_path, "r") as f: data = json.load(f)
        for item in data:
            records.append({
                "uid": str(item["uid"]),
                "image_path": _resolve_image_path(item["image_path"], data_dir),
                "context": item["nih_context"],
                "gt_report": item["ground_truth_report"]
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
    global MODEL, PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS
    if MODEL is None:
        print(f"Loading MedGemma Intervention Model on {device}...")
        PROCESSOR = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
        MODEL = MedGemmaForConditionalGenerationCustom.from_pretrained(
            MEDGEMMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="eager"
        )
        MODEL.eval()
        INTERVENTION_NUM_VISUAL_TOKENS = getattr(MODEL.config, "mm_tokens_per_image", 256)
        patch_gemma3_attention_for_bottleneck_intervention(use_intervention=True)
    return MODEL, PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS

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

def apply_mspoe_position_hook(model, scaling_factor: float, text_only: bool, num_visual_tokens: int = 256):
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
    hook_handle = apply_mspoe_position_hook(model, scaling_factor, text_only, num_visual_tokens)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['iuchest', 'mimic'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--test_fraction', type=float, default=1.0, help="Fraction of dataset to evaluate deterministically (0,1].")
    parser.add_argument('--test_split_seed', type=int, default=42, help="Seed used for deterministic subset selection.")
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--compressor_device_id', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='generation_results_medgemma')
    
    # Modes
    parser.add_argument('--generate_baselines', action='store_true')
    parser.add_argument('--use_intervention', action='store_true')
    parser.add_argument(
        '--intervention_use_strong_instruction',
        action='store_true',
        help='With --use_intervention, generate using --strong_instruction and add strong_prompt to the output filename.',
    )
    parser.add_argument('--use_mspoe', action='store_true')
    parser.add_argument('--use_madrag', action='store_true')
    parser.add_argument('--use_combo', action='store_true')
    parser.add_argument('--use_madrag_combo', action='store_true')
    parser.add_argument('--use_longllmlingua', action='store_true')
    parser.add_argument('--use_longllmlingua_combo', action='store_true')
    
    # Params
    parser.add_argument('--gt_position', type=str, default='mid', choices=['mid', 'first', 'last'])
    parser.add_argument('--context_mode', type=str, default='five_docs', choices=['five_docs', 'gt_only'])
    parser.add_argument('--alpha_v', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--gamma_s', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--combo_text_only', action='store_true')
    parser.add_argument('--compression_ratio', type=float, default=0.5)
    parser.add_argument('--question', type=str, default="Based on the visual evidence, what are the primary impressions for this chest radiograph?")
    parser.add_argument('--instruction', type=str, default="You are a radiologist. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.")
    parser.add_argument('--strong_instruction', type=str, default="You are an expert radiologist. Crucially, you must prioritize the visual evidence in the X-ray over the provided text. The text contains historical distractors. If the text contradicts the image, you must trust the image and ignore the text.")

    args = parser.parse_args()
    args.alpha_t = 1.0
    args.gamma_s = 1.0
    if args.intervention_use_strong_instruction and not args.use_intervention:
        parser.error('--intervention_use_strong_instruction requires --use_intervention')
    set_all_seeds(args.seed)
    
    device = f"cuda:{args.device_id}"
    comp_device = f"cuda:{args.compressor_device_id}" if args.compressor_device_id is not None else device
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset.upper()} dataset...")
    records = get_unified_dataset(args)
    records = select_consistent_subset(records, args.test_fraction, args.test_split_seed)
    random.shuffle(records)
    if args.num_samples: records = records[:args.num_samples]
    print(f"Using {len(records)} records after deterministic split (fraction={args.test_fraction}, split_seed={args.test_split_seed}).")

    # Build explicit, collision-resistant filename suffix by mode.
    mode_parts = []
    if args.generate_baselines:
        mode_parts.append("baselines")
    if args.use_intervention:
        if args.intervention_use_strong_instruction:
            mode_parts.append(
                f"new_bair_strong_prompt_av{args.alpha_v}_{args.gt_position}"
            )
        else:
            mode_parts.append(f"new_bair_av{args.alpha_v}_{args.gt_position}")
    if args.use_mspoe:
        mspoe_variant = "mspoe_text" if args.combo_text_only else "mspoe_full"
        mode_parts.append(f"{mspoe_variant}_{args.gt_position}")
    if args.use_madrag:
        mode_parts.append(f"madrag_{args.gt_position}")
    if args.use_combo:
        combo_variant = "text" if args.combo_text_only else "full"
        mode_parts.append(f"combo_{combo_variant}_av{args.alpha_v}_{args.gt_position}")
    if args.use_madrag_combo:
        mode_parts.append(f"madrag_combo_av{args.alpha_v}_{args.gt_position}")
    if args.use_longllmlingua:
        mode_parts.append(f"longllmlingua_{args.gt_position}")
    if args.use_longllmlingua_combo:
        mode_parts.append(f"lll_combo_av{args.alpha_v}_{args.gt_position}")
    if args.context_mode == "gt_only":
        mode_parts.append("ctx_gt_only")

    suffix = "_" + "_".join(mode_parts) if mode_parts else "_default"
    if args.test_fraction < 1.0:
        suffix += f"_{_fraction_tag(args.test_fraction)}_seed{args.test_split_seed}"
    out_file = Path(args.output_dir) / f"{args.dataset}_medgemma_results{suffix}.json"

    def _clean_uid(v):
        return str(v).strip().replace(".0", "")

    def _has_valid_value(entry, key):
        v = entry.get(key)
        if v is None:
            return False
        if isinstance(v, str) and (not v.strip() or "[Error]" in v):
            return False
        return True

    target_keys = []
    if args.generate_baselines:
        target_keys.extend(["no_retrieval_answer", "oracle_answer", "prompt_baseline_answer"])
    if args.use_intervention:
        target_keys.append("oracle_with_intervention" if args.gt_position == "mid" else f"intervention_{args.gt_position}_answer")
    if args.use_mspoe:
        mspoe_prefix = "mspoe_text" if args.combo_text_only else "mspoe_full"
        target_keys.append(f"{mspoe_prefix}_answer" if args.gt_position == "mid" else f"{mspoe_prefix}_{args.gt_position}_answer")
    if args.use_madrag:
        target_keys.append("madrag_answer" if args.gt_position == "mid" else f"madrag_{args.gt_position}_answer")
    if args.use_combo:
        target_keys.append(f"combo_{args.gt_position}_answer")
    if args.use_madrag_combo:
        target_keys.append(f"madrag_combo_{args.gt_position}_answer")
    if args.use_longllmlingua:
        target_keys.append(f"longllmlingua_{args.gt_position}_answer")
    if args.use_longllmlingua_combo:
        target_keys.append(f"longllmlingua_combo_{args.gt_position}_answer")

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

    # Normalize legacy/relative paths and guarantee JSON-serializable strings.
    data_root = Path(args.data_dir)
    for entry in all_results:
        raw_path = entry.get("image_path")
        if raw_path is None:
            continue
        entry["image_path"] = _resolve_image_path(raw_path, data_root)
    
    for entry in tqdm(all_results, desc=f"Processing {args.dataset} (MedGemma)"):
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
            target_ctx = reorder_nih_context(base_ctx, 0 if args.gt_position == 'first' else 4 if args.gt_position == 'last' else 2)
        
        try:
            if args.generate_baselines:
                if not _has_valid_value(entry, "no_retrieval_answer"):
                    entry["no_retrieval_answer"] = generate_standard_medgemma(image_path, q, None, args.max_new_tokens, ins, device)
                if not _has_valid_value(entry, "oracle_answer"):
                    entry["oracle_answer"] = generate_standard_medgemma(image_path, q, ctx_mid, args.max_new_tokens, ins, device)
                if not _has_valid_value(entry, "prompt_baseline_answer"):
                    entry["prompt_baseline_answer"] = generate_standard_medgemma(image_path, q, ctx_mid, args.max_new_tokens, s_ins, device)
            
            if args.use_intervention:
                key = "oracle_with_intervention" if args.gt_position == 'mid' else f"intervention_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    intervention_ins = s_ins if args.intervention_use_strong_instruction else ins
                    ans = generate_with_bair_and_mspoe_medgemma(
                        image_path, q, target_ctx, args.max_new_tokens, intervention_ins, args.alpha_v, args.alpha_t, args.gamma_s, device, 1.0, False
                    )
                    entry[key] = ans
                
            if args.use_mspoe:
                prefix = "mspoe_text" if args.combo_text_only else "mspoe_full"
                key = f"{prefix}_answer" if args.gt_position == 'mid' else f"{prefix}_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    ans = generate_with_bair_and_mspoe_medgemma(image_path, q, target_ctx, args.max_new_tokens, ins, 0.0, 0.0, 1.0, device, 1.5, args.combo_text_only, use_madrag=False)
                    entry[key] = ans

            if args.use_madrag:
                key = "madrag_answer" if args.gt_position == 'mid' else f"madrag_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    ans = generate_with_bair_and_mspoe_medgemma(image_path, q, target_ctx, args.max_new_tokens, ins, 0.0, 0.0, 1.0, device, 1.0, False, use_madrag=True)
                    entry[key] = ans
                
            if args.use_combo:
                key = f"combo_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    ans = generate_with_bair_and_mspoe_medgemma(image_path, q, target_ctx, args.max_new_tokens, ins, args.alpha_v, args.alpha_t, args.gamma_s, device, 1.5, args.combo_text_only, use_madrag=False)
                    entry[key] = ans

            if args.use_madrag_combo:
                key = f"madrag_combo_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    ans = generate_with_bair_and_mspoe_medgemma(image_path, q, target_ctx, args.max_new_tokens, ins, args.alpha_v, args.alpha_t, args.gamma_s, device, 1.0, False, use_madrag=True)
                    entry[key] = ans
                
            if args.use_longllmlingua:
                key = f"longllmlingua_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    comp_ctx = compress_with_longllmlingua(target_ctx, q, ins, comp_device, args.compression_ratio)
                    ans = generate_standard_medgemma(image_path, q, comp_ctx, args.max_new_tokens, ins, device)
                    entry[key] = ans
                
            if args.use_longllmlingua_combo:
                key = f"longllmlingua_combo_{args.gt_position}_answer"
                if not _has_valid_value(entry, key):
                    comp_ctx = compress_with_longllmlingua(target_ctx, q, ins, comp_device, args.compression_ratio)
                    ans = generate_with_bair_and_mspoe_medgemma(image_path, q, comp_ctx, args.max_new_tokens, ins, args.alpha_v, args.alpha_t, args.gamma_s, device, 1.0, False)
                    entry[key] = ans
                
        except Exception as e:
            print(f"\n[Error] Failed on {entry['uid']}: {e}")
            
        with open(out_file, "w") as f: json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
