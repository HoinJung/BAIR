"""
Analysis script for medical hallucination analysis comparing no_retrieval and oracle_retrieval.
Optimized for pure generation (Standard RAG, Prompt Baseline, No Retrieval) 
and isolated BAIR, Ms-PoE, Positional (First/Last), and LongLLMLingua interventions.
"""

import os
import json
import random
import re
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from PIL import Image

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None
    print("Warning: llmlingua not installed. Run 'pip install llmlingua' to use LongLLMLingua features.")

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Recorruption" / "src" / "medical_rag"))

from transformers import pipeline, AutoProcessor
from custom_medgemma_model import MedGemmaForConditionalGenerationCustom
from bottleneck_intervention import (
    set_bottleneck_intervention,
    patch_gemma3_attention_for_bottleneck_intervention,
)
import bair_efficient

# Paths / configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "iuchest"
RESULTS_DIR = BASE_DIR / "outputs" / "medical_rag"
IMAGE_DIR = DATA_DIR / "images_normalized"
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
DEVICE_ID = 3
DEVICE = f"cuda:{DEVICE_ID}"

NIH_DATABASE_JSON = BASE_DIR / "data" / "generated" / "iuchest_nih_retrieval_dataset.json"
DEFAULT_GENERATION_RESULTS_JSON = BASE_DIR / "generation_results" / "analysis_results_google_medgemma_4b_it_with_instruction.json"

GENERATOR_PIPE = None
INTERVENTION_MODEL = None
INTERVENTION_PROCESSOR = None
INTERVENTION_NUM_VISUAL_TOKENS = None
LONGLLMLINGUA_COMPRESSOR = None

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_medgemma_generator():
    global GENERATOR_PIPE
    if GENERATOR_PIPE is None:
        # Eager attention aligns with intervention paths; KV cache enabled during decode (HF defaults).
        try:
            GENERATOR_PIPE = pipeline(
                "image-text-to-text",
                model=MEDGEMMA_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device=DEVICE,
                model_kwargs={"attn_implementation": "eager"},
            )
        except TypeError:
            GENERATOR_PIPE = pipeline(
                "image-text-to-text",
                model=MEDGEMMA_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device=DEVICE,
            )
    return GENERATOR_PIPE

def load_longllmlingua(device: str):
    global LONGLLMLINGUA_COMPRESSOR
    if LONGLLMLINGUA_COMPRESSOR is None:
        if PromptCompressor is None:
            raise ImportError("Please install llmlingua: pip install llmlingua")
        print("\nInitializing LongLLMLingua Compressor...")
        # Note: Uses a smaller model by default to save VRAM alongside MedGemma.
        LONGLLMLINGUA_COMPRESSOR = PromptCompressor(
            model_name="NousResearch/Llama-2-7b-hf", 
            model_config={"torch_dtype": torch.bfloat16},
            device_map=device
        )
        
        # =====================================================================
        # FIX: Monkeypatch the model to handle modern Transformers Cache objects
        # This prevents the 'list object has no attribute get_seq_length' error
        # =====================================================================
        class LLMLinguaModelWrapper:
            def __init__(self, model):
                self._model = model
                
            def __call__(self, *args, **kwargs):
                from transformers.cache_utils import DynamicCache
                
                # 1. Convert LLMLingua's legacy list into HuggingFace's DynamicCache
                if 'past_key_values' in kwargs and isinstance(kwargs['past_key_values'], list):
                    kwargs['past_key_values'] = DynamicCache.from_legacy_cache(tuple(kwargs['past_key_values']))
                    
                out = self._model(*args, **kwargs)
                
                # 2. Convert DynamicCache back to a list so LLMLingua can slice it natively
                if hasattr(out, 'past_key_values') and hasattr(out.past_key_values, 'to_legacy_cache'):
                    out.past_key_values = list(out.past_key_values.to_legacy_cache())
                    
                return out
                
            def __getattr__(self, name):
                return getattr(self._model, name)
                
        # Apply the wrapper to the compressor's model
        LONGLLMLINGUA_COMPRESSOR.model = LLMLinguaModelWrapper(LONGLLMLINGUA_COMPRESSOR.model)
        # =====================================================================
        
    return LONGLLMLINGUA_COMPRESSOR

def build_full_prompt(question: str, context: Optional[str] = None, instruction: Optional[str] = None) -> str:
    if instruction:
        if context:
            return f"Instruction: {instruction}\n\nContext:\n{context}\n\nQuestion: {question}"
        return f"Instruction: {instruction}\n\nQuestion: {question}"
    if context:
        return f"Context:\n{context}\n\n{question}"
    return question

def build_shared_bair_prefix_prompt(question: str, instruction: Optional[str] = None) -> str:
    """Prompt prefix used by the experimental shared-cache BAIR path.

    The retrieved context is appended after this prefix, so calibration can
    populate the same visual/text prefix KV cache that generation continues.
    """
    if instruction:
        return f"Instruction: {instruction}\n\nQuestion: {question}\n\nRetrieved context:\n"
    return f"Question: {question}\n\nRetrieved context:\n"

def build_shared_bair_full_prompt(question: str, context: Optional[str], instruction: Optional[str] = None) -> str:
    prefix = build_shared_bair_prefix_prompt(question, instruction)
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
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[0] == 1 and v.shape[1] >= prefix_len and k in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}:
            out[k] = v[:, :prefix_len]
        else:
            out[k] = v
    return out

def _shared_prefix_len_from_processor(processor, image: Image.Image, full_text: str, boundary: int, device: str) -> int:
    prefix_text = full_text[:boundary]
    prefix_inputs = processor(images=image, text=prefix_text, return_tensors="pt", padding=True)
    prefix_ids = prefix_inputs["input_ids"]
    full_ids = processor(images=image, text=full_text, return_tensors="pt", padding=True)["input_ids"]
    prefix_len = int(prefix_ids.shape[1])
    if prefix_len <= 0 or full_ids.shape[1] <= prefix_len:
        raise RuntimeError("Shared BAIR prefix produced no context/generation suffix.")
    if not torch.equal(prefix_ids[0], full_ids[0, :prefix_len]):
        raise RuntimeError("Shared BAIR prefix tokenization is not a prefix of the full prompt.")
    return prefix_len

def compress_with_longllmlingua(context: str, question: str, instruction: str, device: str, ratio: float = 0.5) -> str:
    compressor = load_longllmlingua(device)
    
    parts = re.split(r'--- Document \d ---', context)
    docs = [p.strip() for p in parts if p.strip()]
    if not docs:
        return context
        
    results = compressor.compress_prompt(
        context=docs,
        instruction=instruction if instruction else "",
        question=question,
        rate=ratio, # Use native rate instead of manual target_token math
        condition_in_question='after_condition',
        reorder_context="sort_based_on_metric",
        dynamic_context_compression_ratio=0.4,
        rank_method="longllmlingua" # CRUCIAL: Forces the question-aware ranking metric
    )
    return results["compressed_prompt"]

def generate_with_bair_and_mspoe(
    image_path: str, question: str, context: Optional[str], max_new_tokens: int,
    instruction: Optional[str], alpha_v: float, alpha_t: float, device: str,
    scaling_factor: float, text_only: bool, num_visual_tokens: int = 256
) -> str:
    model, processor, _ = load_medgemma_intervention_model(device)
    
    hook_handle = apply_mspoe_position_hook(model, scaling_factor, text_only, num_visual_tokens)
    
    image = Image.open(image_path).convert("RGB")

    clean_prompt = build_full_prompt(question, context=None, instruction=instruction)
    conversation_clean = [{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "image"}]}]
    clean_text = processor.apply_chat_template(conversation_clean, tokenize=False, add_generation_prompt=True)

    clean_inputs = processor(images=image, text=clean_text, return_tensors="pt", padding=True)
    clean_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clean_inputs.items()}
    question_suffix = f"\n\nQuestion: {question}"
    
    # Get the text up to the moment the question starts
    text_up_to_question = clean_prompt.split(question_suffix)[0]
    
    # Tokenize both to find the exact delta (this automatically includes 
    # the chat template's assistant tokens in the tail count)
    full_prompt_tokens = len(processor.tokenizer(clean_prompt)["input_ids"])
    prefix_tokens = len(processor.tokenizer(text_up_to_question)["input_ids"])
    
    exact_tail_tokens = full_prompt_tokens - prefix_tokens

    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=True, reset_layer=True, alpha_v=alpha_v, alpha_t=alpha_t, question_tokens=exact_tail_tokens)
    with torch.no_grad():
        model(**clean_inputs, use_cache=True)

    full_prompt = build_full_prompt(question, context=context, instruction=instruction)
    conversation_gen = [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}]
    text_with_placeholder = processor.apply_chat_template(conversation_gen, tokenize=False, add_generation_prompt=True)

    inputs = processor(images=image, text=text_with_placeholder, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    bair_efficient.reuse_medgemma_pixel_values_if_efficient(clean_inputs, inputs)

    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=False, reset_layer=True, alpha_v =alpha_v, alpha_t=alpha_t, question_tokens=exact_tail_tokens)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    if hook_handle:
        hook_handle.remove()

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

def reorder_nih_context(context: str, gt_target_index: int) -> str:
    parts = re.split(r'--- Document \d ---', context)
    docs = [p.strip() for p in parts if p.strip()]
    if len(docs) != 5:
        return context 
    
    gt_doc = docs[2]
    distractors = [docs[0], docs[1], docs[3], docs[4]]
    
    new_docs = distractors.copy()
    new_docs.insert(gt_target_index, gt_doc)
    
    new_context = []
    for i, doc in enumerate(new_docs):
        new_context.append(f"--- Document {i+1} ---\n{doc}")
        
    return "\n\n".join(new_context)

def load_medgemma_intervention_model(device: str):
    global INTERVENTION_MODEL, INTERVENTION_PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS
    if INTERVENTION_MODEL is None or INTERVENTION_PROCESSOR is None:
        print("\nInitializing Custom MedGemma Model for Intervention...")
        INTERVENTION_PROCESSOR = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
        INTERVENTION_MODEL = MedGemmaForConditionalGenerationCustom.from_pretrained(
            MEDGEMMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="eager"
        )
        INTERVENTION_MODEL.eval()
        INTERVENTION_NUM_VISUAL_TOKENS = getattr(INTERVENTION_MODEL.config, "mm_tokens_per_image", 256)
        patch_gemma3_attention_for_bottleneck_intervention(use_intervention=True)
    return INTERVENTION_MODEL, INTERVENTION_PROCESSOR, INTERVENTION_NUM_VISUAL_TOKENS

def generate_with_medgemma_intervention(
    image_path: str, question: str, context: Optional[str], max_new_tokens: int,
    instruction: Optional[str], alpha_v: float, alpha_t: float, device: str,
) -> str:
    model, processor, num_visual_tokens = load_medgemma_intervention_model(device)
    image = Image.open(image_path).convert("RGB")

    # 1. Calibration Pass (No need to shield question here since there is no context to penalize anyway)
    clean_prompt = build_full_prompt(question, context=None, instruction=instruction)
    conversation_clean = [{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "image"}]}]
    clean_text = processor.apply_chat_template(conversation_clean, tokenize=False, add_generation_prompt=True)
    clean_inputs = processor(images=image, text=clean_text, return_tensors="pt", padding=True)
    clean_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clean_inputs.items()}

    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=True, reset_layer=True, alpha_v=alpha_v, alpha_t=alpha_t)
    with torch.no_grad():
        model(**clean_inputs, use_cache=True)

    # 2. Generation Pass (Calculate the exact tokens on the FULL text)
    full_prompt = build_full_prompt(question, context=context, instruction=instruction)
    conversation_gen = [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}]
    text_with_placeholder = processor.apply_chat_template(conversation_gen, tokenize=False, add_generation_prompt=True)
    
    # --- MOVED CALCULATION BLOCK ---
    question_suffix = f"\n\nQuestion: {question}"
    text_up_to_question = text_with_placeholder.split(question_suffix)[0]
    full_prompt_tokens = len(processor.tokenizer(text_with_placeholder)["input_ids"])
    prefix_tokens = len(processor.tokenizer(text_up_to_question)["input_ids"])
    exact_tail_tokens = full_prompt_tokens - prefix_tokens

    inputs = processor(images=image, text=text_with_placeholder, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    bair_efficient.reuse_medgemma_pixel_values_if_efficient(clean_inputs, inputs)

    set_bottleneck_intervention(True, num_visual_tokens=num_visual_tokens, calibration_run=False, reset_layer=True, alpha_v=alpha_v, alpha_t=alpha_t, question_tokens=exact_tail_tokens)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

def generate_with_medgemma_intervention_shared_prefix(
    image_path: str, question: str, context: Optional[str], max_new_tokens: int,
    instruction: Optional[str], alpha_v: float, alpha_t: float, device: str,
) -> str:
    """BAIR generation that reuses the no-context visual/text prefix cache.

    This is an efficiency path for computational-cost experiments.  It makes the
    no-context calibration prompt a strict prefix of the context prompt, stores
    the calibration ``past_key_values``, then feeds only the context suffix to
    ``generate``.  If the local Transformers/model version cannot continue from
    that cache, callers should fall back to ``generate_with_medgemma_intervention``.
    """
    model, processor, num_visual_tokens = load_medgemma_intervention_model(device)
    image = Image.open(image_path).convert("RGB")

    full_prompt = build_shared_bair_full_prompt(question, context, instruction)

    full_conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt}]}]

    full_text = processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=True)

    full_inputs = processor(images=image, text=full_text, return_tensors="pt", padding=True)
    full_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}

    boundary_text = context if context else ""
    boundary = full_text.find(boundary_text) if boundary_text else len(full_text)
    if boundary < 0:
        raise RuntimeError("Could not locate context boundary in shared BAIR prompt.")
    prefix_len = _shared_prefix_len_from_processor(processor, image, full_text, boundary, device)
    if full_inputs["input_ids"].shape[1] <= prefix_len:
        raise RuntimeError("Shared BAIR prefix produced no context/generation suffix.")
    prefix_inputs = _slice_processor_inputs(full_inputs, prefix_len)

    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=True,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        question_tokens=0,
    )
    with torch.no_grad():
        prefix_out = model(**prefix_inputs, use_cache=True, return_dict=True)

    past = getattr(prefix_out, "past_key_values", None)
    if past is None:
        raise RuntimeError("Shared BAIR prefix forward did not return past_key_values.")

    suffix_ids = full_inputs["input_ids"][:, prefix_len:]
    full_attention = full_inputs.get("attention_mask")
    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=False,
        reset_layer=True,
        alpha_v=alpha_v,
        alpha_t=alpha_t,
        question_tokens=0,
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=suffix_ids,
            attention_mask=full_attention,
            past_key_values=past,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # With a supplied cache, HF generate returns the suffix plus newly generated
    # tokens in some versions, and only generated tokens in others.
    suffix_len = suffix_ids.shape[1]
    set_bottleneck_intervention(False)
    generated = out[0][suffix_len:] if out.shape[1] > suffix_len else out[0]
    if generated.numel() == 0:
        return ""
    return processor.decode(generated, skip_special_tokens=True).strip()

def apply_mspoe_position_hook(model, scaling_factor: float, text_only: bool, num_visual_tokens: int = 256):
    if scaling_factor == 1.0:
        return None

    def pre_forward_hook(module, args, kwargs):
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            pos_ids = kwargs["position_ids"].float()
            if text_only:
                mask = pos_ids >= num_visual_tokens
                pos_ids[mask] = num_visual_tokens + (pos_ids[mask] - num_visual_tokens) / scaling_factor
            else:
                pos_ids = pos_ids / scaling_factor
            kwargs["position_ids"] = pos_ids.long() 
        return args, kwargs

    if hasattr(model, "model"):
        return model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
    else:
        return model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

def generate_with_mspoe(
    image_path: str, question: str, context: str, max_new_tokens: int,
    instruction: str, device: str, scaling_factor: float = 1.5, text_only: bool = False
) -> str:
    model, processor, num_visual_tokens = load_medgemma_intervention_model(device)
    set_bottleneck_intervention(False) 
    
    hook_handle = apply_mspoe_position_hook(model, scaling_factor, text_only, num_visual_tokens)
    
    image = Image.open(image_path).convert("RGB")
    full_prompt = build_full_prompt(question, context=context, instruction=instruction)
    conversation_gen = [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}]
    text_with_placeholder = processor.apply_chat_template(conversation_gen, tokenize=False, add_generation_prompt=True)

    inputs = processor(images=image, text=text_with_placeholder, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    if hook_handle:
        hook_handle.remove()

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

def generate_with_medgemma(image_path: str, prompt_text: str, context: Optional[str] = None, max_new_tokens: int = 512, instruction: Optional[str] = None) -> str:
    pipe = load_medgemma_generator()
    image = Image.open(image_path).convert("RGB")
    full_prompt = build_full_prompt(prompt_text, context, instruction)
    messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image", "image": image}]}]
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    return output[0]["generated_text"][-1]["content"]

def process_samples(
    data_dir: Path, json_output_dir: str, num_samples: Optional[int] = None, 
    max_new_tokens: int = 64, preferred_question: Optional[str] = None,
    instruction: Optional[str] = None, strong_instruction: Optional[str] = None
):
    os.makedirs(json_output_dir, exist_ok=True)
    
    if not NIH_DATABASE_JSON.exists():
        raise FileNotFoundError(f"Missing {NIH_DATABASE_JSON}. Please run build_nih_database.py first.")
        
    with open(NIH_DATABASE_JSON, "r") as f:
        nih_dataset = json.load(f)
        
    print(f"Loaded {len(nih_dataset)} cases from pre-computed NiH database.")
    random.shuffle(nih_dataset)
    
    model_dir_name = MEDGEMMA_MODEL_ID.replace('/', '_').replace('-', '_')
    json_filename = f"analysis_results_{model_dir_name}_with_instruction.json" if instruction else f"analysis_results_{model_dir_name}.json"
    json_output_path = os.path.join(json_output_dir, json_filename)
    
    all_results, total_processed = [], 0
    question = preferred_question if preferred_question else "Based on the visual evidence, what are the primary impressions for this chest radiograph?"

    for data_item in tqdm(nih_dataset, desc="Generating Core Baselines"):
        if num_samples and total_processed >= num_samples: break
        
        try:
            uid = data_item["uid"]
            image_path = data_dir / "images_normalized" / data_item["image_filename"]
            if not image_path.exists(): continue
            
            ground_truth_report = data_item["ground_truth_report"]
            ground_truth_problems = data_item["ground_truth_problems"]
            oracle_context = data_item["nih_context"]
            
            total_processed += 1
            
            oracle_answer = generate_with_medgemma(str(image_path), question, oracle_context, max_new_tokens=max_new_tokens, instruction=instruction)
            prompt_baseline_answer = generate_with_medgemma(str(image_path), question, oracle_context, max_new_tokens=max_new_tokens, instruction=strong_instruction)
            no_retrieval_answer = generate_with_medgemma(str(image_path), question, None, max_new_tokens=max_new_tokens, instruction=instruction)
            
            result_entry = {
                'uid': uid, 
                'image_path': str(image_path), 
                'ground_truth_report': ground_truth_report,
                'ground_truth_problems': ground_truth_problems, 
                'question': question, 
                'instruction': instruction, 
                'strong_instruction': strong_instruction,
                'oracle_answer': oracle_answer, 
                'prompt_baseline_answer': prompt_baseline_answer,
                'no_retrieval_answer': no_retrieval_answer,
                'retrieved_context': oracle_context 
            }
            all_results.append(result_entry)
            
            with open(json_output_path, 'w') as f: 
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            continue
    
    return total_processed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=str(BASE_DIR / 'data' / 'raw' / 'iuchest'))
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--json_output_dir', type=str, default='generation_results')
    parser.add_argument('--find_all', action='store_true')
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--generation_only', action='store_true')
    parser.add_argument('--instruction', type=str, nargs='?', const="You are a radiologist. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.", default="You are a radiologist. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.")
    parser.add_argument('--strong_instruction', type=str, default="You are an expert radiologist. Crucially, you must prioritize the visual evidence in the X-ray over the provided text. The text contains historical distractors. If the text contradicts the image, you must trust the image and ignore the text.")
    
    parser.add_argument('--use_intervention', action='store_true')
    parser.add_argument('--use_mspoe', action='store_true')
    parser.add_argument('--use_positional_rag', action='store_true')
    parser.add_argument('--gt_position', type=str, default='mid', choices=['mid', 'first', 'last'])
    parser.add_argument('--from_analysis_results', type=str, default=None)
    parser.add_argument('--intervention_output_json', type=str, default=None)
    parser.add_argument('--alpha_v', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=1.0, help=argparse.SUPPRESS)
    
    parser.add_argument('--use_combo', action='store_true')
    parser.add_argument('--combo_text_only', action='store_true')

    # New LongLLMLingua Flags
    parser.add_argument('--use_longllmlingua', action='store_true', help="Run standalone LongLLMLingua compression baseline")
    parser.add_argument('--use_longllmlingua_combo', action='store_true', help="Run LongLLMLingua followed by BAIR intervention")
    parser.add_argument('--compression_ratio', type=float, default=0.5, help="Target ratio of tokens to retain during compression")
    parser.add_argument('--compressor_device_id', type=int, default='0', help="GPU ID for LongLLMLingua")
    
    args = parser.parse_args()
    args.alpha_t = 1.0
    set_all_seeds(args.seed)

    # ---------------------------------------------------------
    # LongLLMLingua Execution Blocks
    # ---------------------------------------------------------
    if args.use_longllmlingua or args.use_longllmlingua_combo:
        base_json_path = Path(args.from_analysis_results) if args.from_analysis_results else DEFAULT_GENERATION_RESULTS_JSON
        tag = "combo" if args.use_longllmlingua_combo else "baseline"
        out_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_longllmlingua_{tag}_{args.gt_position}.json")

        with base_json_path.open("r") as f:
            base_results = json.load(f)

        device = f"cuda:{args.device_id}"
        compressor_device = f"cuda:{args.compressor_device_id}" if args.compressor_device_id is not None else device
        updated_results = []
        
        with open(NIH_DATABASE_JSON, "r") as f:
            nih_db = {item["uid"]: item["nih_context"] for item in json.load(f)}

        desc_tag = "Combo (+BAIR)" if args.use_longllmlingua_combo else "Baseline"
        for entry in tqdm(base_results, desc=f"Generating LongLLMLingua {desc_tag} (GT: {args.gt_position})"):
            uid = entry.get("uid")
            image_path = entry.get("image_path")
            question = entry.get("question", "Based on the visual evidence, what are the primary impressions for this chest radiograph?")
            
            if not uid or not image_path or uid not in nih_db:
                updated_results.append(entry)
                continue

            oracle_context = nih_db[uid] 
            if args.gt_position == 'first':
                target_context = reorder_nih_context(oracle_context, gt_target_index=0)
            elif args.gt_position == 'last':
                target_context = reorder_nih_context(oracle_context, gt_target_index=4)
            else:
                target_context = oracle_context
            
            try:
                # 1. Compress the context using LongLLMLingua
                compressed_context = compress_with_longllmlingua(
                    context=target_context, question=question, instruction=entry.get("instruction"),
                    device=compressor_device, ratio=args.compression_ratio
                )
                
                # 2. Pass compressed text to either standard model or BAIR intervention
                if args.use_longllmlingua_combo:
                    ans = generate_with_medgemma_intervention(
                        image_path=str(image_path), question=question, context=compressed_context,
                        max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"),
                        alpha_v=args.alpha_v, alpha_t=args.alpha_t, device=device,
                    )
                else:
                    ans = generate_with_medgemma(
                        image_path=str(image_path), prompt_text=question, context=compressed_context,
                        max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction")
                    )

                new_entry = dict(entry)
                key_prefix = "longllmlingua_combo" if args.use_longllmlingua_combo else "longllmlingua"
                
                if args.gt_position == 'first':
                    new_entry[f"{key_prefix}_first_answer"] = ans
                elif args.gt_position == 'last':
                    new_entry[f"{key_prefix}_last_answer"] = ans
                else:
                    new_entry[f"{key_prefix}_mid_answer"] = ans
                    
                updated_results.append(new_entry)
            except Exception as e:
                print(f"\nError generating LongLLMLingua for {uid}: {str(e)}")
                import traceback
                traceback.print_exc()
                updated_results.append(entry)

        with out_path.open("w") as f: json.dump(updated_results, f, indent=2)
        return

    if args.use_combo:
        base_json_path = Path(args.from_analysis_results) if args.from_analysis_results else DEFAULT_GENERATION_RESULTS_JSON
        combo_type_str = "TEXT" if args.combo_text_only else "FULL"
        out_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_COMBO_{combo_type_str}_{args.gt_position}_av{args.alpha_v}.json")

        with base_json_path.open("r") as f:
            base_results = json.load(f)

        device = f"cuda:{args.device_id}"
        updated_results = []
        
        with open(NIH_DATABASE_JSON, "r") as f:
            nih_db = {item["uid"]: item["nih_context"] for item in json.load(f)}

        for entry in tqdm(base_results, desc=f"Generating BAIR + MS-PoE Combo (GT: {args.gt_position}, MS-PoE: {combo_type_str})"):
            uid = entry.get("uid")
            image_path = entry.get("image_path")
            question = entry.get("question", "Based on the visual evidence, what are the primary impressions for this chest radiograph?")
            
            if not uid or not image_path or uid not in nih_db:
                updated_results.append(entry)
                continue

            oracle_context = nih_db[uid] 
            if args.gt_position == 'first':
                target_context = reorder_nih_context(oracle_context, gt_target_index=0)
            elif args.gt_position == 'last':
                target_context = reorder_nih_context(oracle_context, gt_target_index=4)
            else:
                target_context = oracle_context
            
            try:
                ans = generate_with_bair_and_mspoe(
                    image_path=str(image_path), question=question, context=target_context,
                    max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"),
                    alpha_v=args.alpha_v, alpha_t=args.alpha_t, device=device,
                    scaling_factor=1.5, text_only=args.combo_text_only
                )
                new_entry = dict(entry)
                
                if args.gt_position == 'first':
                    new_entry["combo_first_answer"] = ans
                elif args.gt_position == 'last':
                    new_entry["combo_last_answer"] = ans
                else:
                    new_entry["combo_mid_answer"] = ans
                    
                updated_results.append(new_entry)
            except Exception as e:
                print(f"\nError generating Combo for {uid}: {str(e)}")
                updated_results.append(entry)

        with out_path.open("w") as f: json.dump(updated_results, f, indent=2)
        return
        
    if args.use_positional_rag:
        base_json_path = Path(args.from_analysis_results) if args.from_analysis_results else DEFAULT_GENERATION_RESULTS_JSON
        out_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_positional_baseline.json")

        with base_json_path.open("r") as f:
            base_results = json.load(f)

        updated_results = []
        with open(NIH_DATABASE_JSON, "r") as f:
            nih_db = {item["uid"]: item["nih_context"] for item in json.load(f)}

        for entry in tqdm(base_results, desc="Generating Positional Baselines (First/Last)"):
            uid = entry.get("uid")
            image_path = entry.get("image_path")
            question = entry.get("question", "Based on the visual evidence, what are the primary impressions for this chest radiograph?")
            
            if not uid or not image_path or uid not in nih_db:
                updated_results.append(entry)
                continue

            oracle_context = nih_db[uid] 
            context_first = reorder_nih_context(oracle_context, gt_target_index=0)
            context_last = reorder_nih_context(oracle_context, gt_target_index=4)
            
            try:
                oracle_first = generate_with_medgemma(str(image_path), question, context_first, max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"))
                oracle_last = generate_with_medgemma(str(image_path), question, context_last, max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"))
                
                new_entry = dict(entry)
                new_entry["oracle_first_answer"] = oracle_first
                new_entry["oracle_last_answer"] = oracle_last
                updated_results.append(new_entry)
            except Exception:
                updated_results.append(entry)

        with out_path.open("w") as f: json.dump(updated_results, f, indent=2)
        return
        
    if args.use_mspoe:
        base_json_path = Path(args.from_analysis_results) if args.from_analysis_results else DEFAULT_GENERATION_RESULTS_JSON
        out_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_mspoe_{args.gt_position}_baseline.json")

        with base_json_path.open("r") as f:
            base_results = json.load(f)

        device = f"cuda:{args.device_id}"
        updated_results = []
        
        with open(NIH_DATABASE_JSON, "r") as f:
            nih_db = {item["uid"]: item["nih_context"] for item in json.load(f)}

        for entry in tqdm(base_results, desc=f"Generating Ms-PoE Baselines (GT: {args.gt_position})"):
            uid = entry.get("uid")
            image_path = entry.get("image_path")
            question = entry.get("question", "Based on the visual evidence, what are the primary impressions for this chest radiograph?")
            
            if not uid or not image_path or uid not in nih_db:
                updated_results.append(entry)
                continue

            oracle_context = nih_db[uid] 
            if args.gt_position == 'first':
                target_context = reorder_nih_context(oracle_context, gt_target_index=0)
            elif args.gt_position == 'last':
                target_context = reorder_nih_context(oracle_context, gt_target_index=4)
            else:
                target_context = oracle_context
            
            try:
                mspoe_full = generate_with_mspoe(
                    image_path=str(image_path), question=question, context=target_context,
                    max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"),
                    device=device, scaling_factor=1.5, text_only=False
                )
                mspoe_text = generate_with_mspoe(
                    image_path=str(image_path), question=question, context=target_context,
                    max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"),
                    device=device, scaling_factor=1.5, text_only=True
                )
                
                new_entry = dict(entry)
                
                if args.gt_position == 'first':
                    new_entry["mspoe_full_first_answer"] = mspoe_full
                    new_entry["mspoe_text_first_answer"] = mspoe_text
                elif args.gt_position == 'last':
                    new_entry["mspoe_full_last_answer"] = mspoe_full
                    new_entry["mspoe_text_last_answer"] = mspoe_text
                else:
                    new_entry["mspoe_full_answer"] = mspoe_full
                    new_entry["mspoe_text_answer"] = mspoe_text
                    
                updated_results.append(new_entry)
            except Exception as e:
                print(f"\nError generating Ms-PoE for {uid}: {str(e)}") 
                updated_results.append(entry)

        with out_path.open("w") as f: json.dump(updated_results, f, indent=2)
        return

    if args.use_intervention:
        base_json_path = Path(args.from_analysis_results) if args.from_analysis_results else DEFAULT_GENERATION_RESULTS_JSON
        out_path = Path(args.intervention_output_json) if args.intervention_output_json else base_json_path.with_name(base_json_path.stem + f"_intervention_{args.gt_position}_av{args.alpha_v}.json")

        with base_json_path.open("r") as f:
            base_results = json.load(f)

        device = f"cuda:{args.device_id}"
        updated_results = []
        
        with open(NIH_DATABASE_JSON, "r") as f:
            nih_db = {item["uid"]: item["nih_context"] for item in json.load(f)}

        for entry in tqdm(base_results, desc=f"Generating BAIR intervention (GT: {args.gt_position})"):
            uid = entry.get("uid")
            image_path = entry.get("image_path")
            question = entry.get("question", "Based on the visual evidence, what are the primary impressions for this chest radiograph?")
            
            if not uid or not image_path or uid not in nih_db:
                updated_results.append(entry)
                continue

            oracle_context = nih_db[uid] 
            if args.gt_position == 'first':
                target_context = reorder_nih_context(oracle_context, gt_target_index=0)
            elif args.gt_position == 'last':
                target_context = reorder_nih_context(oracle_context, gt_target_index=4)
            else:
                target_context = oracle_context
            
            try:
                ans = generate_with_medgemma_intervention(
                    image_path=str(image_path), question=question, context=target_context,
                    max_new_tokens=args.max_new_tokens, instruction=entry.get("instruction"),
                    alpha_v=args.alpha_v, alpha_t=args.alpha_t, device=device,
                )
                new_entry = dict(entry)
                
                if args.gt_position == 'first':
                    new_entry["intervention_first_answer"] = ans
                elif args.gt_position == 'last':
                    new_entry["intervention_last_answer"] = ans
                else:
                    new_entry["oracle_with_intervention"] = ans
                    
                updated_results.append(new_entry)
            except Exception:
                updated_results.append(entry)

        with out_path.open("w") as f: json.dump(updated_results, f, indent=2)
        return

    process_samples(
        data_dir=Path(args.data_dir), json_output_dir=args.json_output_dir,
        num_samples=None if args.find_all else args.num_samples, max_new_tokens=args.max_new_tokens,
        preferred_question=args.question, instruction=args.instruction, strong_instruction=args.strong_instruction
    )

if __name__ == "__main__":
    main()
