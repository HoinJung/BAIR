"""
Shared helpers for GeoChat / SkySense / LLaVA-style intervention paths.

Extracted from experiments.gender_analysis so bair.llm_explainer does not import experiment scripts.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import torch


def _is_degenerate_response(response: str) -> bool:
    text = (response or "").strip().lower()
    if len(text) <= 5:
        return True
    # Catch contiguous substring loops such as "pazocalpazocalpazocal..."
    if re.search(r"([a-z]{3,12})\1{4,}", text):
        return True
    toks = re.findall(r"[a-z']+", text)
    if len(toks) < 8:
        # For long single-token gibberish, also treat as degenerate.
        if len(text) > 60 and " " not in text:
            return True
        return False
    max_run, cur = 1, 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    if max_run >= 4:
        return True
    unique_ratio = len(set(toks)) / max(1, len(toks))
    return len(toks) > 40 and unique_ratio < 0.2


def _force_eager_for_intervention(model_components: Dict[str, Any]) -> None:
    """
    Force eager attention for intervention runs across LLaVA/DeepSeek backbones.
    """
    model = model_components.get("model")
    if model is None:
        return
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "eager"
            tc = getattr(cfg, "text_config", None)
            if tc is not None:
                if hasattr(tc, "_attn_implementation"):
                    tc._attn_implementation = "eager"
                if hasattr(tc, "attn_implementation"):
                    tc.attn_implementation = "eager"
        lm = getattr(model, "language_model", None)
        if lm is not None:
            lcfg = getattr(lm, "config", None)
            if lcfg is not None:
                if hasattr(lcfg, "_attn_implementation"):
                    lcfg._attn_implementation = "eager"
                if hasattr(lcfg, "attn_implementation"):
                    lcfg.attn_implementation = "eager"
    except Exception:
        pass


def _llava_get_language_layers(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not locate language decoder layers for LLaVA BAIR / Ms-PoE.")


def _llava_get_position_hook_target(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model"):
        return model.model
    return model


def _llava_apply_mspoe_position_hook(
    model: torch.nn.Module,
    scaling_factor: float,
    text_only: bool,
    num_visual_tokens: int,
    visual_start_idx: int = 0,
):
    if abs(scaling_factor - 1.0) < 1e-12:
        return None
    target_module = _llava_get_position_hook_target(model)
    target_module._mspoe_delta = 0

    def pre_forward_hook(module, args, kwargs):
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            pos_ids = kwargs["position_ids"].clone()
            seq_len = pos_ids.shape[-1]
            if seq_len > 1:
                float_pos = pos_ids.float()
                if text_only:
                    v_start = float(visual_start_idx)
                    v_end = v_start + float(num_visual_tokens)
                    # Scale only text positions; keep visual-token positions unchanged.
                    text_mask = (float_pos < v_start) | (float_pos >= v_end)
                    float_pos[text_mask] = float_pos[text_mask] / scaling_factor
                else:
                    float_pos = float_pos / scaling_factor
                scaled_pos = float_pos.long()
                module._mspoe_delta = pos_ids[0, -1].item() - scaled_pos[0, -1].item()
                kwargs["position_ids"] = scaled_pos
            else:
                delta = getattr(module, "_mspoe_delta", 0)
                kwargs["position_ids"] = pos_ids - delta
        return args, kwargs

    return target_module.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)


def _llava_count_visual_tokens(model: torch.nn.Module, clean_inputs: Dict[str, Any]) -> int:
    total_seq_len = [0]

    def catch_seq_len(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        total_seq_len[0] = int(hidden_states.shape[1])
        raise RuntimeError("Caught Seq Len")

    layers = _llava_get_language_layers(model)
    handle = layers[0].register_forward_pre_hook(catch_seq_len, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**clean_inputs, use_cache=True)
    except RuntimeError:
        pass
    finally:
        handle.remove()

    text_len = int(clean_inputs["input_ids"].shape[1])
    num_visual = total_seq_len[0] - text_len
    if num_visual <= 0 and "input_ids" in clean_inputs:
        img_tok = getattr(getattr(model, "config", None), "image_token_index", None)
        if img_tok is not None:
            num_visual = int((clean_inputs["input_ids"] == img_tok).sum().item())
    return num_visual
