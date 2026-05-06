from __future__ import annotations

from typing import Sequence, Optional, Dict, Any
from PIL import Image
import torch
from pathlib import Path
import sys

from . import bair_efficient

from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, pipeline, logging
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

try:
    from .rag_core import RetrievedPassage
except Exception:
    # Optional for generation-only flows where retrieval indexing stack is unavailable.
    from typing import Any as RetrievedPassage

EARTHDIAL_AVAILABLE = False
InternVLChatModel = None
earthdial_build_transform = None
earthdial_dynamic_preprocess = None
_EARTHDIAL_IMPORT_ERROR = None
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EARTHDIAL_SRC = next(
    (
        p
        for p in (
            _REPO_ROOT / "EarthDial" / "src",
            _REPO_ROOT / "third_party" / "EarthDial" / "src",
        )
        if p.is_dir()
    ),
    _REPO_ROOT / "third_party" / "EarthDial" / "src",
)
if _EARTHDIAL_SRC.is_dir():
    import sys

    _earthdial_src_str = str(_EARTHDIAL_SRC)
    if _earthdial_src_str not in sys.path:
        sys.path.insert(0, _earthdial_src_str)
try:
    from earthdial.model.internvl_chat import InternVLChatModel as _InternVLChatModel
    from earthdial.train.dataset import (
        build_transform as _earthdial_build_transform,
        dynamic_preprocess as _earthdial_dynamic_preprocess,
    )

    InternVLChatModel = _InternVLChatModel
    earthdial_build_transform = _earthdial_build_transform
    earthdial_dynamic_preprocess = _earthdial_dynamic_preprocess
    EARTHDIAL_AVAILABLE = True
except Exception as _exc:
    _EARTHDIAL_IMPORT_ERROR = _exc

GEOPIX_AVAILABLE = False
GeoPixInferenceEngine = None
InferenceInputData = None
_GEOPIX_IMPORT_ERROR = None
_GEOPIX_ROOT = next(
    (
        p
        for p in (
            _REPO_ROOT / "GeoPix",
            _REPO_ROOT / "third_party" / "GeoPix",
        )
        if p.is_dir()
    ),
    _REPO_ROOT / "third_party" / "GeoPix",
)
if _GEOPIX_ROOT.is_dir():
    import sys

    _geopix_root_str = str(_GEOPIX_ROOT)
    if _geopix_root_str not in sys.path:
        sys.path.insert(0, _geopix_root_str)
try:
    from engine import GeoPixInferenceEngine as _GeoPixInferenceEngine
    from dataset.inference_input import InferenceInputData as _InferenceInputData

    GeoPixInferenceEngine = _GeoPixInferenceEngine
    InferenceInputData = _InferenceInputData
    GEOPIX_AVAILABLE = True
except Exception as _exc:
    _GEOPIX_IMPORT_ERROR = _exc

try:
    _GEOCHAT_ROOT = _REPO_ROOT / "third_party" / "GeoChat"
    if _GEOCHAT_ROOT.is_dir() and str(_GEOCHAT_ROOT) not in sys.path:
        sys.path.insert(0, str(_GEOCHAT_ROOT))
    from geochat.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from geochat.mm_utils import tokenizer_image_token as geochat_tokenizer_image_token
    from geochat.mm_utils import process_images as geochat_process_images
    from geochat.conversation import conv_templates as geochat_conv_templates
    from geochat.conversation import SeparatorStyle as GeoChatSeparatorStyle
    from geochat.mm_utils import KeywordsStoppingCriteria as GeoChatKeywordsStoppingCriteria
except Exception:
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200
    geochat_tokenizer_image_token = None
    geochat_process_images = None
    geochat_conv_templates = None
    GeoChatSeparatorStyle = None
    GeoChatKeywordsStoppingCriteria = None


def load_llm_model(model_name: str, gpu_id: int | str = 0, use_multi_gpu: bool = False) -> Dict[str, Any]:
    """Load LLM model once and return components.

    Args:
        model_name: Name of the model to load
        gpu_id: GPU ID to load the model on (default: 0; set -1 or "cpu" for CPU)
    """
    print(f"Loading LLM model: {model_name} on GPU/Device {gpu_id}")
    logging.set_verbosity_error()

    try:
        # Set active CUDA device early to avoid defaulting to cuda:0
        if torch.cuda.is_available() and gpu_id != "cpu" and int(gpu_id) >= 0:
            try:
                torch.cuda.set_device(int(gpu_id))
            except Exception:
                pass

        # Check if model requires trust_remote_code
        model_name_lower = model_name.lower()
        is_earthdial = "earthdial" in model_name_lower
        is_geopix = "geopix" in model_name_lower
        requires_trust_remote_code = (
            "llava" in model_name_lower
            or "onevision" in model_name_lower
            or "geochat" in model_name_lower
            or is_earthdial
            or is_geopix
        )
        prefer_slow_tokenizer = "geochat" in model_name_lower

        # Prefer bf16 when supported on the target GPU, otherwise use fp16.
        chosen_dtype = torch.float16
        try:
            if torch.cuda.is_available() and gpu_id != "cpu" and int(gpu_id) >= 0:
                with torch.cuda.device(int(gpu_id)):
                    if torch.cuda.is_bf16_supported():
                        chosen_dtype = torch.bfloat16
                        print("Using bfloat16 precision (reduces memory by ~50%)")
                    else:
                        print("Using float16 precision (reduces memory by ~50%)")
        except Exception:
            print("Using float16 precision (reduces memory by ~50%)")

        if is_geopix:
            if not GEOPIX_AVAILABLE:
                raise ImportError(
                    "GeoPix support is unavailable. Ensure `third_party/GeoPix` exists and "
                    "dependencies are installed. "
                    f"Import error: {_GEOPIX_IMPORT_ERROR}"
                )
            pretrained_path = model_name
            if not Path(pretrained_path).exists():
                from huggingface_hub import snapshot_download

                pretrained_path = snapshot_download(repo_id=model_name)
            engine = GeoPixInferenceEngine(
                pretrained_model_path=pretrained_path,
                pretrained_processor_path=pretrained_path,
            )
            tokenizer = engine.valid_tokenizer
            print(f"Loaded GeoPix model: {model_name} ({pretrained_path})")
            return {
                "model": engine.model,
                "tokenizer": tokenizer,
                "processor": None,
                "pipeline": None,
                "is_multimodal": True,
                "dtype": chosen_dtype,
                "model_name": model_name,
                "is_geopix": True,
                "geopix_engine": engine,
            }

        if is_earthdial:
            if not EARTHDIAL_AVAILABLE:
                raise ImportError(
                    "EarthDial support is unavailable. Ensure `third_party/EarthDial/src` "
                    "exists and dependencies are installed (pip install -e EarthDial). "
                    f"Import error: {_EARTHDIAL_IMPORT_ERROR}"
                )
            device_str = "cpu" if gpu_id == "cpu" or (not torch.cuda.is_available()) else f"cuda:{int(gpu_id)}"
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": chosen_dtype,
            }
            if use_multi_gpu:
                load_kwargs["device_map"] = "auto"
            model = InternVLChatModel.from_pretrained(model_name, **load_kwargs).eval()
            if not use_multi_gpu and device_str != "cpu":
                model = model.to(device_str)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            image_size = (
                getattr(getattr(model, "config", None), "force_image_size", None)
                or getattr(getattr(getattr(model, "config", None), "vision_config", None), "image_size", 224)
            )
            use_thumbnail = bool(getattr(getattr(model, "config", None), "use_thumbnail", False))
            print(f"Loaded EarthDial model: {model_name} on {device_str} with {chosen_dtype}")
            return {
                "model": model,
                "tokenizer": tokenizer,
                "processor": None,
                "pipeline": None,
                "is_multimodal": True,
                "dtype": chosen_dtype,
                "model_name": model_name,
                "is_earthdial": True,
                "earthdial_image_size": int(image_size),
                "earthdial_use_thumbnail": use_thumbnail,
                "earthdial_max_num": 6,
            }

        if "geochat" in model_name_lower or "skysense" in model_name_lower:
            # GeoChat registers custom AutoConfig/AutoModel mappings at import time.
            try:
                import geochat.model.language_model.geochat_llama  # noqa: F401
            except Exception:
                pass

        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=requires_trust_remote_code,
            use_fast=not prefer_slow_tokenizer,
        )
        if "geochat" in model_name_lower or "skysense" in model_name_lower:
            # Prefer GeoChat's official loader path for architecture/runtime compatibility.
            try:
                from geochat.model.builder import load_pretrained_model
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "SkySense / GeoChat checkpoints are loaded via `geochat.model.builder.load_pretrained_model`, "
                    "which requires the `geochat` package on PYTHONPATH (it is not published as a generic pip name "
                    "here). Install from the official GeoChat or SkySense-Chat repo (typically `pip install -e .` "
                    "in the repo that provides the `geochat/` package), then retry. "
                    "AdaptLLM / Qwen2.5-VL does not need this package."
                ) from exc
            device_str = "cpu" if gpu_id == "cpu" or (not torch.cuda.is_available()) else f"cuda:{int(gpu_id)}"
            device_map_cfg = "auto" if use_multi_gpu else None
            builder_model_name = model_name if "geochat" in model_name_lower else "geochat"
            tok_gc, model_gc, image_processor_gc, _ctx_len = load_pretrained_model(
                model_path=model_name,
                model_base=None,
                model_name=builder_model_name,
                load_8bit=False,
                load_4bit=False,
                device_map=device_map_cfg,
                device=device_str,
            )
            if device_str != "cpu" and not use_multi_gpu:
                model_gc = model_gc.to(device_str)
            print(f"Loaded GeoChat model via official builder on {device_str}")
            return {
                "model": model_gc,
                "tokenizer": tok_gc,
                "processor": image_processor_gc,
                "pipeline": None,
                "is_multimodal": True,
                "dtype": torch.float16,
                "model_name": model_name,
            }
        
        # Determine if this is a multimodal model based on model name
        is_multimodal_model = (
            "vl" in model_name_lower
            or "vision" in model_name_lower
            or "clip" in model_name_lower
            or "blip" in model_name_lower
            or "git" in model_name_lower
            or "llava" in model_name_lower
            or "geochat" in model_name_lower
        )
        
        # Check if this is LLaVA model
        is_llava = "llava" in model_name_lower or "geochat" in model_name_lower
        is_llava_onevision = "onevision" in model_name_lower
        
        # Load model and processor
        if is_multimodal_model:
            if "qwen2.5-vl" in model_name.lower():
                # Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL
                # Enable gradient checkpointing by default to save memory during training/gradient computation
                if use_multi_gpu:
                    # For multi-GPU, use "balanced" to distribute model more evenly across GPUs
                    # "balanced" distributes layers more evenly than "auto" which tends to put early layers on GPU 0
                    # This helps distribute backward pass across GPUs
                    device_map_config = "balanced"  # More even distribution than "auto"
                else:
                    device_map_config = {"": gpu_id} if gpu_id != "cpu" else "cpu"
                if Qwen2_5_VLForConditionalGeneration is None:
                    raise ImportError(
                        "Qwen2_5_VLForConditionalGeneration is unavailable in this transformers build."
                    )
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, 
                    device_map=device_map_config, 
                    torch_dtype=chosen_dtype, 
                    attn_implementation="eager",
                    low_cpu_mem_usage=True
                )
                # Enable gradient checkpointing to trade compute for memory
                if hasattr(model, "gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable()
                        print(f"  Gradient checkpointing enabled (trades compute for memory)")
                    except Exception:
                        pass
                processor = AutoProcessor.from_pretrained(model_name)
                device_info = "multiple GPUs (auto)" if use_multi_gpu else f"GPU {gpu_id}"
                print(f"Loaded Qwen2.5-VL model: {model_name} on {device_info} with {chosen_dtype}")
            elif is_llava:
                # For LLaVA models: try LlavaOnevisionForConditionalGeneration first (lmms-lab/llava-onevision-*),
                # then LlavaForConditionalGeneration, then AutoModelForCausalLM
                device_map_config = "auto" if use_multi_gpu else ("cpu" if gpu_id == "cpu" else {"": gpu_id})
                if "geochat" in model_name_lower:
                    device_map_config = None
                load_kwargs = {
                    "device_map": device_map_config,
                    "torch_dtype": chosen_dtype,
                    "attn_implementation": "eager",
                    "low_cpu_mem_usage": True,
                }
                if requires_trust_remote_code:
                    load_kwargs["trust_remote_code"] = True
                    print(f"Loading {model_name} with trust_remote_code=True")
                if "geochat" in model_name_lower:
                    load_kwargs["low_cpu_mem_usage"] = False
                    load_kwargs["ignore_mismatched_sizes"] = True
                    load_kwargs.pop("attn_implementation", None)
                
                model = None
                processor = None
                load_model_name = model_name  # may switch to HF-converted variant on failure (keeps model_name for output paths)
                if is_llava_onevision:
                    try:
                        from transformers import LlavaOnevisionForConditionalGeneration
                        model = LlavaOnevisionForConditionalGeneration.from_pretrained(load_model_name, **load_kwargs)
                        processor = AutoProcessor.from_pretrained(load_model_name, trust_remote_code=requires_trust_remote_code)
                        device_info = "multiple GPUs (auto)" if use_multi_gpu else f"Device {gpu_id}"
                        print(f"Loaded LLaVA-OneVision model: {load_model_name} on {device_info}")
                    except Exception as e:
                        print(f"Failed to load as LlavaOnevisionForConditionalGeneration: {e}")
                        # lmms-lab checkpoint often has arch mismatch (e.g. image_newline 3584 vs 4096); use HF-converted variant
                        if "lmms-lab" in model_name.lower() and "llava-onevision-qwen2-7b-ov" in model_name.lower():
                            load_model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
                            print(f"Retrying with HuggingFace-converted model: {load_model_name}")
                            try:
                                model = LlavaOnevisionForConditionalGeneration.from_pretrained(load_model_name, **load_kwargs)
                                processor = AutoProcessor.from_pretrained(load_model_name, trust_remote_code=False)
                                device_info = "multiple GPUs (auto)" if use_multi_gpu else f"Device {gpu_id}"
                                print(f"Loaded LLaVA-OneVision model: {load_model_name} on {device_info}")
                            except Exception as e2:
                                print(f"Fallback also failed: {e2}")
                                load_model_name = model_name
                
                if model is None:
                    try:
                        from transformers import LlavaForConditionalGeneration
                        model = LlavaForConditionalGeneration.from_pretrained(load_model_name, **load_kwargs)
                        processor = AutoProcessor.from_pretrained(load_model_name, trust_remote_code=requires_trust_remote_code)
                        device_info = "multiple GPUs (auto)" if use_multi_gpu else f"Device {gpu_id}"
                        print(f"Loaded LLaVA model: {load_model_name} on {device_info}")
                    except Exception as e:
                        print(f"Failed to load as LlavaForConditionalGeneration, trying AutoModelForCausalLM: {e}")
                        model = AutoModelForCausalLM.from_pretrained(load_model_name, **load_kwargs)
                        try:
                            processor = AutoProcessor.from_pretrained(load_model_name, trust_remote_code=requires_trust_remote_code)
                        except Exception:
                            processor = tok
                        if "geochat" in model_name_lower:
                            # GeoChat often has no AutoProcessor; use vision tower processor.
                            try:
                                vt = model.get_vision_tower()
                                processor = getattr(vt, "image_processor", processor)
                            except Exception:
                                pass
                        device_info = "multiple GPUs (auto)" if use_multi_gpu else f"Device {gpu_id}"
                        print(f"Loaded LLaVA model (as CausalLM): {load_model_name} on {device_info}")
            else:
                try:
                    # For other multimodal models, try AutoModel first
                    device_map_config = "auto" if use_multi_gpu else {"": gpu_id}
                    if "geochat" in model_name_lower:
                        # Avoid accelerate meta-init path that conflicts with GeoChat's
                        # nested CLIPVisionModel.from_pretrained call in __init__.
                        device_map_config = None
                    load_kwargs = {
                        "device_map": device_map_config,
                        "torch_dtype": chosen_dtype,
                        "attn_implementation": "eager",
                        "low_cpu_mem_usage": True,
                    }
                    if "geochat" in model_name_lower:
                        # GeoChat internally loads CLIPVisionModel during __init__;
                        # low_cpu_mem_usage/meta-init can break that nested load.
                        load_kwargs["low_cpu_mem_usage"] = False
                        load_kwargs["ignore_mismatched_sizes"] = True
                        load_kwargs.pop("attn_implementation", None)
                    if requires_trust_remote_code:
                        load_kwargs["trust_remote_code"] = True
                        print(f"Loading {model_name} with trust_remote_code=True")
                    
                    model = AutoModel.from_pretrained(model_name, **load_kwargs)
                    if "geochat" in model_name_lower and gpu_id != "cpu" and torch.cuda.is_available():
                        model = model.to(f"cuda:{int(gpu_id)}")
                    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=requires_trust_remote_code)
                    device_info = "multiple GPUs (auto)" if use_multi_gpu else f"GPU {gpu_id}"
                    print(f"Loaded multimodal model: {model_name} on {device_info}")
                except Exception as e:
                    print(f"Failed to load as multimodal model, trying CausalLM: {e}")
                    device_map_config = "auto" if use_multi_gpu else {"": gpu_id}
                    if "geochat" in model_name_lower:
                        device_map_config = None
                    load_kwargs = {
                        "device_map": device_map_config,
                        "torch_dtype": chosen_dtype,
                        "low_cpu_mem_usage": False if "geochat" in model_name_lower else True,
                    }
                    if "geochat" in model_name_lower:
                        load_kwargs["ignore_mismatched_sizes"] = True
                    if requires_trust_remote_code:
                        load_kwargs["trust_remote_code"] = True
                    
                    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                    if "geochat" in model_name_lower and gpu_id != "cpu" and torch.cuda.is_available():
                        model = model.to(f"cuda:{int(gpu_id)}")
                    processor = tok
        else:
            # For text-only models (e.g., judge models)
            device_map_config = "auto" if use_multi_gpu else {"": gpu_id}
            if "geochat" in model_name_lower:
                device_map_config = None
            load_kwargs = {
                "device_map": device_map_config,
                "torch_dtype": chosen_dtype,
                "attn_implementation": "eager",
                "low_cpu_mem_usage": True,
            }
            if "geochat" in model_name_lower:
                load_kwargs["low_cpu_mem_usage"] = False
                load_kwargs["ignore_mismatched_sizes"] = True
                load_kwargs.pop("attn_implementation", None)
            if requires_trust_remote_code:
                load_kwargs["trust_remote_code"] = True
                
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if "geochat" in model_name_lower and gpu_id != "cpu" and torch.cuda.is_available():
                model = model.to(f"cuda:{int(gpu_id)}")
            processor = tok
            device_info = "multiple GPUs (auto)" if use_multi_gpu else f"GPU {gpu_id}"
            print(f"Loaded text-only model: {model_name} on {device_info}")

        # Set attention implementation - use default (sdpa/flash) for better performance
        # Only use eager if there are CUDA kernel issues
        try:
            if hasattr(model, "config"):
                # For Qwen2.5-VL and LLaVA, prefer default attention (sdpa) for better generation
                # Eager attention is slower and uses more memory, only use if needed for debugging
                if "qwen2.5-vl" in model_name.lower() or "llava" in model_name.lower():
                    # Use default attention implementation for better performance
                    # model.config.attn_implementation = "eager"  # Uncomment only if debugging CUDA issues
                    pass
                else:
                    # For other models, try eager if available
                    try:
                        model.config.attn_implementation = "eager"
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Create appropriate pipeline
        pipe = None
        if is_multimodal_model:
            # For Qwen2.5-VL and LLaVA, use direct model inference without pipeline
            if "qwen2.5-vl" in model_name.lower() or is_llava:
                # Use the model and processor directly without pipeline
                pipe = None  # We'll handle this manually
                print(f"Using direct model inference for: {model_name}")
            else:
                # Try standard multimodal pipelines for other models
                try:
                    task = "visual-question-answering"
                    pipe = pipeline(task, model=model, tokenizer=tok, device_map="auto")
                    print(f"Using visual-question-answering pipeline for: {model_name}")
                except Exception:
                    try:
                        task = "image-to-text"
                        pipe = pipeline(task, model=model, tokenizer=tok, device_map="auto")
                        print(f"Using image-to-text pipeline for: {model_name}")
                    except Exception as e:
                        print(f"Warning: Could not create multimodal pipeline for {model_name}: {e}")
                        # Fall back to direct inference
                        pipe = None
                        print(f"Using direct model inference for: {model_name}")
        else:
            # Use text generation pipeline for text-only models
            try:
                task = "text-generation"
                pipe = pipeline(task, model=model, tokenizer=tok, device_map="auto")
            except Exception as e:
                print(f"Warning: Could not create text generation pipeline: {e}")
                pipe = None
        
        return {
            "model": model,
            "tokenizer": tok,
            "processor": processor,
            "pipeline": pipe,
            "is_multimodal": is_multimodal_model,
            "dtype": chosen_dtype,
            "model_name": model_name,
        }
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def build_context(passages: Sequence[RetrievedPassage], max_chars: int = 1500, max_tokens: int = None, tokenizer=None) -> str:
    """
    Build context string from retrieved passages.
    
    Args:
        passages: List of retrieved passages
        max_chars: Maximum characters (used if tokenizer is None)
        max_tokens: Maximum tokens (used if tokenizer is provided)
        tokenizer: Tokenizer to use for counting tokens (if None, uses character-based)
    """
    if not passages:
        return ""
    
    # Use token-based if tokenizer is provided
    if tokenizer is not None and max_tokens is not None:
        # For single document (oracle case), use full text
        if len(passages) == 1:
            p = passages[0]
            extract = p.extract.strip().replace("\n", " ")
            profs = ", ".join(p.professions) if p.professions else "unknown"
            full_block = f"Title: {p.title}\nProfessions: {profs}\nPassage: {extract}\n"
            
            # Count tokens
            try:
                tokens = tokenizer.encode(full_block, add_special_tokens=False)
                if len(tokens) <= max_tokens:
                    return full_block
                # If too long, truncate to max_tokens
                truncated_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
                return truncated_text + "..."
            except Exception:
                # Fallback to character-based if tokenization fails
                if len(full_block) <= max_tokens * 4:  # Rough estimate: 1 token ≈ 4 chars
                    return full_block
                return full_block[:max_tokens * 4] + "..."
        
        # For multiple passages, build context with token limits
        chunks = []
        total_tokens = 0
        
        def count_tokens(text: str) -> int:
            """Count tokens in text."""
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                # Fallback: rough estimate (1 token ≈ 4 characters)
                return len(text) // 4
        
        for p in passages:
            extract = p.extract.strip().replace("\n", " ")
            profs = ", ".join(p.professions) if p.professions else "unknown"
            block = f"Title: {p.title}\nProfessions: {profs}\nPassage: {extract}\n"
            
            block_tokens = count_tokens(block)
            
            # If adding this block would exceed limit, truncate the extract
            if total_tokens + block_tokens > max_tokens:
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Only add if we have meaningful space left
                    # Estimate how much extract we can include
                    header = f"Title: {p.title}\nProfessions: {profs}\nPassage: "
                    header_tokens = count_tokens(header)
                    extract_tokens = remaining_tokens - header_tokens - 10  # Reserve for "..."
                    
                    if extract_tokens > 0:
                        # Truncate extract to fit
                        try:
                            extract_tokens_list = tokenizer.encode(extract, add_special_tokens=False)
                            truncated_extract = tokenizer.decode(extract_tokens_list[:extract_tokens], skip_special_tokens=True)
                            block = f"{header}{truncated_extract}...\n"
                        except Exception:
                            # Fallback to character-based
                            truncated_extract = extract[:extract_tokens * 4]
                            block = f"{header}{truncated_extract}...\n"
                        chunks.append(block)
                break
            
            chunks.append(block)
            total_tokens += block_tokens
        
        return "\n\n".join(chunks)
    
    # Original character-based implementation (fallback)
    chunks = []
    total = 0
    for p in passages:
        snippet = p.extract.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        profs = ", ".join(p.professions) if p.professions else "unknown"
        block = f"Title: {p.title}\nProfessions: {profs}\nPassage: {snippet}\n"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks)


def generate_with_hf(
    model_components: Dict[str, Any],
    instruction: str,
    question: str,
    passages: Sequence[RetrievedPassage],
    image: Optional[str] = None,
    max_new_tokens: int = 128,
    max_context_tokens: int = 2000,
    multimodal_user_text: Optional[str] = None,
) -> str:
    """Generate answer using pre-loaded model components."""
    try:
        model = model_components["model"]
        tok = model_components["tokenizer"]
        processor = model_components["processor"]
        pipe = model_components["pipeline"]
        is_multimodal_model = model_components["is_multimodal"]
        
        # Determine if we should use multimodal processing
        is_multimodal = image is not None and is_multimodal_model
        
        # Use tokenizer for token-based context building if available
        context = build_context(passages, max_tokens=max_context_tokens, tokenizer=tok)
        titles = "\n".join([f"- {p.title} (profs: {', '.join(p.professions) or 'unknown'})" for p in passages])
        instr_str = f"Instruction: {instruction}\n\n" if instruction else ""
        # Check if we have retrieved passages or not
        has_retrieval = len(passages) > 0

        def _multimodal_user_message_text() -> str:
            if multimodal_user_text is not None:
                return multimodal_user_text
            if has_retrieval:
                return f"{instr_str}Context:\n{context}\n\n{question}"
            return f"{instr_str}{question}"

        # Suppressed verbose output - only show when mismatch is detected
        # print(f"Processing with {'multimodal' if is_multimodal else 'text-only'} model, retrieval: {has_retrieval}")
        if is_multimodal and image is not None:
            # LLaVA uses processor.apply_chat_template with list content; tokenizer template expects string and fails
            model_name_for_inference = model_components.get("model_name", "")
            is_llava_model = (
                "llava" in model_name_for_inference.lower()
                or "onevision" in model_name_for_inference.lower()
                or "geochat" in model_name_for_inference.lower()
                or "skysense" in model_name_for_inference.lower()
            )
            is_geochat_model = (
                "geochat" in model_name_for_inference.lower()
                or "skysense" in model_name_for_inference.lower()
            )
            is_qwen25_vl_model = "qwen2.5-vl" in model_name_for_inference.lower()
            is_earthdial_model = bool(model_components.get("is_earthdial"))
            is_geopix_model = bool(model_components.get("is_geopix"))
            # For multimodal models, create a combined prompt with image and text context
            if (
                hasattr(tok, "apply_chat_template")
                and not is_llava_model
                and not is_qwen25_vl_model
                and not is_earthdial_model
                and not is_geopix_model
            ):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(image)},
                            {"type": "text", "text": _multimodal_user_message_text()},
                        ],
                    },
                ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif not is_llava_model and not is_earthdial_model and not is_geopix_model:
                # Fallback for models without chat template
                prompt = _multimodal_user_message_text()
            
            # For multimodal, pass both image and text
            if pipe is None:
                image_obj = Image.open(image)
                if is_geochat_model:
                    # GeoChat follows LLaVA-style <image> token prompting and requires
                    # explicit image tensor preprocessing.
                    if geochat_tokenizer_image_token is None or geochat_process_images is None:
                        raise RuntimeError("GeoChat helpers are unavailable; ensure geochat package is installed.")
                    text_content = _multimodal_user_message_text()
                    if geochat_conv_templates is not None:
                        conv = geochat_conv_templates["llava_v1"].copy()
                        qs = f"{DEFAULT_IMAGE_TOKEN}\n{text_content}"
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                    else:
                        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{text_content}"
                        conv = None
                    input_ids = geochat_tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                    image_tensor = geochat_process_images([image_obj], processor, model.config)
                    model_device = next(model.parameters()).device
                    input_ids = input_ids.to(model_device)
                    image_tensor = image_tensor.to(model_device, dtype=getattr(model, "dtype", torch.float16))
                    stopping = None
                    if conv is not None and GeoChatKeywordsStoppingCriteria is not None and GeoChatSeparatorStyle is not None:
                        stop_str = conv.sep if conv.sep_style != GeoChatSeparatorStyle.TWO else conv.sep2
                        stopping = [GeoChatKeywordsStoppingCriteria([stop_str], tok, input_ids)]
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                            num_beams=1,
                            max_new_tokens=max_new_tokens,
                            length_penalty=2.0,
                            use_cache=True,
                            stopping_criteria=stopping,
                        )
                    input_len = input_ids.shape[1]
                    response = tok.decode(
                        generated_ids[0][input_len:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    if conv is not None and GeoChatSeparatorStyle is not None:
                        stop_str = conv.sep if conv.sep_style != GeoChatSeparatorStyle.TWO else conv.sep2
                        if response.endswith(stop_str):
                            response = response[: -len(stop_str)].strip()
                    out = [{"generated_text": response}]
                    del input_ids, image_tensor, generated_ids, image_obj
                    torch.cuda.empty_cache()
                elif is_earthdial_model:
                    text_content = _multimodal_user_message_text()
                    image_rgb = image_obj.convert("RGB")
                    image_size = int(model_components.get("earthdial_image_size", 224))
                    use_thumbnail = bool(model_components.get("earthdial_use_thumbnail", False))
                    max_num = int(model_components.get("earthdial_max_num", 6))
                    transform = earthdial_build_transform(is_train=False, input_size=image_size)
                    patches = [image_rgb]
                    try:
                        dyn = earthdial_dynamic_preprocess(
                            image_rgb,
                            image_size=image_size,
                            use_thumbnail=use_thumbnail,
                            max_num=max_num,
                        )
                        if isinstance(dyn, list) and len(dyn) > 0:
                            patches = dyn
                    except Exception:
                        pass
                    pixel_values = torch.stack([transform(p) for p in patches])
                    model_device = next(model.parameters()).device
                    pixel_values = pixel_values.to(device=model_device, dtype=model_components.get("dtype", torch.bfloat16))
                    generation_config = {
                        "num_beams": 1,
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "temperature": 0.0,
                    }
                    response = model.chat(
                        tokenizer=tok,
                        pixel_values=pixel_values,
                        question=text_content,
                        generation_config=generation_config,
                        verbose=False,
                    )
                    out = [{"generated_text": response}]
                    del pixel_values, image_obj
                    torch.cuda.empty_cache()
                elif is_geopix_model:
                    geopix_engine = model_components.get("geopix_engine")
                    if geopix_engine is None or InferenceInputData is None:
                        raise RuntimeError("GeoPix engine is unavailable in model components.")
                    input_data = InferenceInputData(
                        question=f"[Visual Question Answering] {_multimodal_user_message_text()}",
                        image_path=image,
                    )
                    input_batch = [input_data[0]]
                    input_batch = geopix_engine.valid_processor(input_batch)
                    response, _pred_masks = geopix_engine.inference_step(input_batch)
                    response = (response or "").replace("</s>", "").strip()
                    out = [{"generated_text": response}]
                elif is_llava_model:
                    # LLaVA / LLaVA-OneVision: use processor(images=..., text=...) API
                    text_content = _multimodal_user_message_text()
                    # LLaVA chat format: content is list of dicts with type "image" and "text"
                    conversation = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_content}
                        ]}
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    inputs = processor(images=image_obj, text=prompt, return_tensors="pt")
                    model_device = next(model.parameters()).device
                    inputs = inputs.to(model_device, dtype=model_components.get("dtype", torch.float16))
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                    # Decode only the new tokens (skip input prompt)
                    input_len = inputs.input_ids.shape[1]
                    response = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    out = [{"generated_text": response}]
                    del inputs, generated_ids, image_obj
                    torch.cuda.empty_cache()
                else:
                    # Direct model inference for Qwen2.5-VL
                    if process_vision_info is None:
                        raise ImportError(
                            "qwen_vl_utils is required for Qwen2.5-VL inference path. "
                            "Install it in the active environment or use a non-Qwen model."
                        )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_obj},
                                {"type": "text", "text": _multimodal_user_message_text()},
                            ],
                        }
                    ]
                    
                    # Use Qwen2.5-VL proper processing
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    # Move to the model's primary device (handles CUDA_VISIBLE_DEVICES remapping)
                    model_device = next(model.parameters()).device
                    try:
                        torch.cuda.set_device(model_device)
                    except Exception:
                        pass
                    inputs = inputs.to(model_device)
                    if "qwen2.5-vl" in model_name_for_inference.lower():
                        try:
                            from . import bottleneck_intervention
                            input_ids_list = inputs.input_ids[0].tolist()
                            
                            # 1. Exact Vision Token Boundaries
                            vis_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
                            vis_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
                            
                            v_start_idx = input_ids_list.index(vis_start_id) + 1
                            v_end_idx = input_ids_list.index(vis_end_id)
                            exact_nv = v_end_idx - v_start_idx
                            
                            # 2. Exact Question Suffix length (Avoiding Special Token Explosion)
                            # Encode ONLY the clean question string. Add 5 to safely cover the exact length 
                            # of Qwen's trailing generation headers (<|im_end|>\n<|im_start|>assistant\n)
                            q_tokens = tok.encode(question, add_special_tokens=False)
                            exact_qt = len(q_tokens) + 5
                            
                            # 3. Inject exact state before generation
                            bottleneck_intervention._intervention_state.visual_start_idx = v_start_idx
                            bottleneck_intervention._intervention_state.num_visual_tokens = exact_nv
                            bottleneck_intervention._intervention_state.question_tokens = exact_qt
                            
                        except Exception as e:
                            print(f"[BAIR Align Warning] Could not dynamically align tokens: {e}")
                    # -------------------------------------------------
                    # --------------------------------------------------
                    # Generate response (autocast to the chosen dtype if possible)
                    with torch.no_grad():
                        amp_dtype = model_components.get("dtype", torch.float16)
                        try:
                            with torch.cuda.amp.autocast(dtype=amp_dtype):
                                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                        except Exception:
                            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        response = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                
                    # Aggressively clear memory after generation
                    del inputs, generated_ids, generated_ids_trimmed, image_obj
                    if 'image_inputs' in locals():
                        del image_inputs
                    if 'video_inputs' in locals():
                        del video_inputs
                    torch.cuda.empty_cache()
                    
                    out = [{"generated_text": response}]
            elif hasattr(pipe, 'task') and pipe.task == "visual-question-answering":
                # For VQA pipeline, use different input format
                out = pipe(Image.open(image), question, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                # For image-to-text pipeline
                out = pipe(Image.open(image), prompt, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            # For text-only models, use the original logic
            if hasattr(tok, "apply_chat_template"):
                if has_retrieval:
                    # With retrieval: use retrieved context
                    messages = [
                        {"role": "user", "content": f"{question}\n\nTop retrieved titles:\n{titles}\n\nContext passages:\n{context}"},
                    ]
                else:
                    # Without retrieval: answer based on general knowledge
                    if image is not None:
                        # If we have an image but can't process it, indicate this limitation
                        messages = [
                            {"role": "user", "content": f"{question}\n\nNote: This model cannot process images, so the answer is based on general knowledge only."},
                        ]
                    else:
                        messages = [
                            {"role": "user", "content": question},
                        ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                if has_retrieval:
                    prompt = f"{question}\n\nTop retrieved titles:\n{titles}\n\nContext passages:\n{context}"
                else:
                    if image is not None:
                        # If we have an image but can't process it, indicate this limitation
                        prompt = f"{question}\n\nNote: This model cannot process images, so the answer is based on general knowledge only."
                    else:
                        prompt = question
            
            out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        
        if not out or not isinstance(out, list) or not out[0].get("generated_text"):
            print(f"Warning: Empty or invalid output from model")
            return "Error: No response generated from the model.", context
        
        text = out[0]["generated_text"]
        # If chat template appended prompt, strip it
        if 'prompt' in locals() and text.startswith(prompt):
            text = text[len(prompt):]
        
        result = text.strip()
        # Suppressed verbose output - only show when mismatch is detected
        # print(f"Generated response length: {len(result)} characters")
        # print(f"Generated response: {result[:200]}...")
        return result, context
        
    except Exception as e:
        print(f"Error in generate_with_hf: {e}")
        import traceback
        traceback.print_exc()
        # Return (str, str) so callers unpacking (answer, context) do not get "too many values to unpack"
        return f"Error generating response: {str(e)}", ""


def _geochat_build_input_ids_and_images(
    model_components: Dict[str, Any],
    image_obj: Image.Image,
    text_content: str,
):
    """Token + vision tensors for GeoChat / SkySense (same contract as generate_with_hf GeoChat branch)."""
    model = model_components["model"]
    tok = model_components["tokenizer"]
    processor = model_components["processor"]
    if geochat_tokenizer_image_token is None or geochat_process_images is None:
        raise RuntimeError("GeoChat helpers are unavailable; ensure geochat package is installed.")
    if geochat_conv_templates is not None:
        conv = geochat_conv_templates["llava_v1"].copy()
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{text_content}"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        conv_for_stop = conv
    else:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{text_content}"
        conv_for_stop = None
    input_ids = geochat_tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    image_tensor = geochat_process_images([image_obj], processor, model.config)
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    image_tensor = image_tensor.to(model_device, dtype=getattr(model, "dtype", torch.float16))
    return input_ids, image_tensor, tok, conv_for_stop


def _geochat_skysense_forward_prefill(model: torch.nn.Module, input_ids: torch.Tensor, image_tensor: torch.Tensor) -> None:
    """One prefill forward for BAIR calibration (GeoChat-style models). KV cache enabled for fair timing vs other suites."""
    with torch.no_grad():
        try:
            model(input_ids=input_ids, images=image_tensor, use_cache=True)
            return
        except TypeError:
            pass
        try:
            model(input_ids=input_ids, images=image_tensor)
            return
        except TypeError:
            pass
        attn = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
        model(input_ids=input_ids, attention_mask=attn, images=image_tensor, use_cache=True)


def generate_geochat_skysense_intervention(
    model_components: Dict[str, Any],
    image_path: str,
    question: str,
    user_text_no_context: str,
    user_text_with_context: str,
    max_new_tokens: int,
    alpha_v: float = 0.0,
    alpha_t: float = 0.0,
    gamma_s: float = 1.0,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
    use_madrag: bool = False,
    force_intervention_path: bool = False,
    max_retries: int = 2,
) -> str:
    """
    BAIR + optional Ms-PoE + MAD-RAG for GeoChat / SkySense (Llama attention patches).

    ``user_text_*`` must match oracle prompting (e.g. NWPU ``build_nwpu_user_text``).
    """
    from .bottleneck_intervention import (
        patch_llama_attention_for_bottleneck_intervention,
        set_bottleneck_intervention,
    )

    use_mspoe = abs(float(mspoe_scaling) - 1.0) > 1e-12
    if abs(alpha_v) < 1e-12 and abs(alpha_t) < 1e-12 and not use_mspoe and not use_madrag and not force_intervention_path:
        patch_llama_attention_for_bottleneck_intervention(False)
        text, _ctx = generate_with_hf(
            model_components=model_components,
            instruction="",
            question=question,
            passages=[],
            image=image_path,
            max_new_tokens=max_new_tokens,
            max_context_tokens=2000,
            multimodal_user_text=user_text_with_context,
        )
        return text

    from .vlm_geochat_helpers import (
        _force_eager_for_intervention,
        _is_degenerate_response,
        _llava_apply_mspoe_position_hook,
        _llava_count_visual_tokens,
    )

    model = model_components["model"]
    dtype = model_components.get("dtype", torch.float16)
    tok = model_components["tokenizer"]
    mspoe_handle = None
    _force_eager_for_intervention(model_components)
    patch_llama_attention_for_bottleneck_intervention(True)

    try:
        image_obj = Image.open(image_path).convert("RGB")
        clean_input_ids, clean_image_tensor, _tok_c, _ = _geochat_build_input_ids_and_images(
            model_components, image_obj, user_text_no_context
        )
        clean_batch = {"input_ids": clean_input_ids, "images": clean_image_tensor}
        try:
            num_visual_tokens = _llava_count_visual_tokens(model, clean_batch)
        except Exception:
            num_visual_tokens = max(1, int(clean_image_tensor.shape[0]) if clean_image_tensor.dim() > 0 else 256)
        if num_visual_tokens <= 0:
            patch_llama_attention_for_bottleneck_intervention(False)
            text, _ctx = generate_with_hf(
                model_components=model_components,
                instruction="",
                question=question,
                passages=[],
                image=image_path,
                max_new_tokens=max_new_tokens,
                max_context_tokens=2000,
                multimodal_user_text=user_text_with_context,
            )
            return text

        tail_text = f"\n\n{question}"
        _tid = tok.encode(tail_text, add_special_tokens=False)
        safe_tail_tokens = len(_tid) + 8

        if use_mspoe:
            mspoe_handle = _llava_apply_mspoe_position_hook(
                model, float(mspoe_scaling), mspoe_text_only, num_visual_tokens, visual_start_idx=0
            )

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
            with torch.amp.autocast("cuda", dtype=dtype):
                _geochat_skysense_forward_prefill(model, clean_input_ids, clean_image_tensor)

        del clean_input_ids, clean_image_tensor
        bair_efficient.optional_empty_cache_after_calibration()

        gen_input_ids, gen_image_tensor, tok_g, conv_for_stop = _geochat_build_input_ids_and_images(
            model_components, image_obj, user_text_with_context
        )
        gen_attention_mask = torch.ones_like(gen_input_ids, device=gen_input_ids.device, dtype=torch.long)

        stopping = None
        if conv_for_stop is not None and GeoChatKeywordsStoppingCriteria is not None and GeoChatSeparatorStyle is not None:
            stop_str = conv_for_stop.sep if conv_for_stop.sep_style != GeoChatSeparatorStyle.TWO else conv_for_stop.sep2
            stopping = [GeoChatKeywordsStoppingCriteria([stop_str], tok_g, gen_input_ids)]

        bair_active = abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
        current_alpha_v = alpha_v
        current_alpha_t = alpha_t
        retry_budget = max(1, int(max_retries)) if bair_active else 1
        response = ""

        for attempt in range(retry_budget):
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
                # Fast path first; fallback to no-cache only on OOM.
                use_cache_now = True
                try:
                    with torch.amp.autocast("cuda", dtype=dtype):
                        generated_ids = model.generate(
                            gen_input_ids,
                            attention_mask=gen_attention_mask,
                            images=gen_image_tensor,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=max_new_tokens,
                            length_penalty=2.0,
                            use_cache=use_cache_now,
                            stopping_criteria=stopping,
                        )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    use_cache_now = False
                    with torch.amp.autocast("cuda", dtype=dtype):
                        generated_ids = model.generate(
                            gen_input_ids,
                            attention_mask=gen_attention_mask,
                            images=gen_image_tensor,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=max_new_tokens,
                            length_penalty=2.0,
                            use_cache=use_cache_now,
                            stopping_criteria=stopping,
                        )
            input_len = gen_input_ids.shape[1]
            response = tok_g.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            del generated_ids
            bair_efficient.optional_empty_cache_after_generation_attempt()
            if conv_for_stop is not None and GeoChatSeparatorStyle is not None:
                stop_str = conv_for_stop.sep if conv_for_stop.sep_style != GeoChatSeparatorStyle.TWO else conv_for_stop.sep2
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)].strip()

            if not bair_active or not _is_degenerate_response(response):
                break
            current_alpha_v *= 0.8
            current_alpha_t *= 0.8

        set_bottleneck_intervention(False)
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
