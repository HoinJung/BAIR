# coding=utf-8
"""
Bottleneck attention intervention for MedGemma/Gemma3, Qwen-VL, LLaVA-Med, and CheXagent.
Features True Pass-Through, Sharpness Boosting, and Standardized Soft Penalties.
"""

import math
import inspect
import threading
import warnings
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import transformers
from typing import List, Dict, Any, Optional, Tuple

# Thread-safe state manager
_intervention_state = threading.local()

# ---------------------------------------------------------------------------
# Visual: Exact Adaptive Restoration & Sharpness Boosting
# ---------------------------------------------------------------------------

def _visual_sharpness(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    N_v = probs.shape[-1]
    if N_v <= 1:
        return torch.ones(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    probs = probs.clamp(min=eps)
    ent = -(probs * probs.log()).sum(dim=-1)
    log_N = torch.log(torch.tensor(N_v, dtype=probs.dtype, device=probs.device))
    return 1.0 - ent / log_N

def _sharpness_at_T(G_v: torch.Tensor, T: float) -> torch.Tensor:
    x = G_v * T
    p = F.softmax(x.float(), dim=-1)
    return _visual_sharpness(p)

def _bisection_T(G_v: torch.Tensor, S_target: float, T_max: float = 100.0, tol: float = 1e-4, max_iter: int = 50) -> float:
    low, high = 0.0, T_max
    for _ in range(max_iter):
        T_mid = (low + high) / 2
        S_mid = _sharpness_at_T(G_v.unsqueeze(0), T_mid).item()
        if abs(S_mid - S_target) < tol:
            return T_mid
        if S_mid < S_target:
            low = T_mid
        else:
            high = T_mid
    return (low + high) / 2

def visual_exact_adaptive_restoration(
    E_v: torch.Tensor,
    E_t: torch.Tensor,
    M_target: torch.Tensor,
    S_target: torch.Tensor,
    alpha_v: float = 0.5,
    gamma_s: float = 1.0,
    eps: float = 1e-8,
    T_max: float = 100.0,
    bisection_tol: float = 1e-4,
) -> torch.Tensor:
    orig_shape = E_v.shape
    N_v = orig_shape[-1]
    
    if N_v == 0 or E_t.shape[-1] == 0:
        return E_v

    mu_v = E_v.mean(dim=-1, keepdim=True)
    sigma_v = E_v.std(dim=-1, keepdim=True).clamp(min=eps)
    Z_v = (E_v - mu_v) / sigma_v
    G_v = Z_v * torch.sigmoid(Z_v)
    
    G_flat = G_v.reshape(-1, N_v)
    E_v_flat = E_v.reshape(-1, N_v)
    E_t_flat = E_t.reshape(-1, E_t.shape[-1])
    
    M_flat = M_target.reshape(-1)
    S_flat = S_target.reshape(-1)
    
    out = torch.empty_like(G_flat)
    
    for i in range(G_flat.shape[0]):
        g = G_flat[i]
        if M_flat.numel() == 0 or S_flat.numel() == 0:
            out[i] = E_v_flat[i]
            continue
            
        idx = i % M_flat.shape[0] if i >= M_flat.shape[0] else i
        m_targ = M_flat[idx].item()
        s_targ = S_flat[idx].item()
        
        # APPLY SHARPNESS BOOST
        s_boosted = min(0.99, s_targ * gamma_s)
        
        M_safe = max(eps, min(1.0 - eps, m_targ))
        T_star = _bisection_T(g, s_boosted, T_max=T_max, tol=bisection_tol)
        E_tilde = g * T_star
        
        log_sum_exp_v = torch.logsumexp(E_tilde, dim=-1).item()
        log_sum_exp_t = torch.logsumexp(E_t_flat[i], dim=-1).item()
        
        alpha = math.log(M_safe / (1 - M_safe)) + log_sum_exp_t - log_sum_exp_v
        # target_E_v = E_tilde + alpha
        target_E_v = torch.nan_to_num(E_tilde + alpha, nan=0.0, posinf=10.0, neginf=-10.0)
        
        out[i] = E_v_flat[i] + alpha_v * (target_E_v - E_v_flat[i])
        
    return out.reshape(orig_shape).to(E_v.dtype)

def textual_asymmetric_penalty(E_t: torch.Tensor, alpha_t: float = 1.0, safe_tail_tokens: int = 0) -> torch.Tensor:
    L = E_t.shape[-1]
    if L == 0: return E_t
    
    device = E_t.device
    dtype = E_t.dtype
    
    mu_t = E_t.mean(dim=-1, keepdim=True)
    n_head = max(1, int(0.2 * L))
    n_tail = max(1, int(0.2 * L))
    
    mu_head = E_t[..., :n_head].mean(dim=-1, keepdim=True)
    mu_tail = E_t[..., L - n_tail : L].mean(dim=-1, keepdim=True)
    
    lam_prim = (mu_head - mu_t).clamp(min=0)
    lam_rec = (mu_tail - mu_t).clamp(min=0)
    
    j_one = torch.arange(1, L + 1, device=device, dtype=dtype)
    left = (1.0 - 2.0 * j_one / L).clamp(min=0).pow(2)
    right = (2.0 * j_one / L - 1.0).clamp(min=0).pow(2)
    
    penalty = lam_prim * left + lam_rec * right

    if safe_tail_tokens > 0:
        tail = min(L, int(safe_tail_tokens))
        if tail > 0:
            penalty = penalty.clone()
            penalty[..., L - tail :] = 0.0

    return E_t - (alpha_t * penalty)

def apply_bottleneck_intervention_adaptive(
    pre_softmax_row: torch.Tensor, num_visual_tokens: int, visual_start_idx: int,
    M_target: torch.Tensor, S_target: torch.Tensor,  
    alpha_v: float = 0.5, alpha_t: float = 1.0, gamma_s: float = 1.0,
    eps: float = 1e-8, T_max: float = 100.0, bisection_tol: float = 1e-4, safe_tail_tokens: int = 0,
) -> torch.Tensor:
    seq_len = pre_softmax_row.shape[-1]
    v_start = visual_start_idx
    v_end = min(visual_start_idx + num_visual_tokens, seq_len)
    
    E_v = pre_softmax_row[..., v_start:v_end]
    E_t_prefix = pre_softmax_row[..., :v_start]
    E_t_suffix = pre_softmax_row[..., v_end:]
    
    E_t = torch.cat([E_t_prefix, E_t_suffix], dim=-1) if E_t_prefix.shape[-1] > 0 else E_t_suffix
        
    E_v_hat = visual_exact_adaptive_restoration(
        E_v.float(), E_t.float(), M_target, S_target,
        alpha_v=alpha_v, gamma_s=gamma_s, eps=eps, T_max=T_max, bisection_tol=bisection_tol,
    )
    
    E_t_hat = textual_asymmetric_penalty(E_t.float(), alpha_t=alpha_t, safe_tail_tokens=safe_tail_tokens)
    
    if E_t_prefix.shape[-1] > 0:
        E_hat = torch.cat([E_t_hat[..., :v_start], E_v_hat, E_t_hat[..., v_start:]], dim=-1)
    else:
        E_hat = torch.cat([E_v_hat, E_t_hat], dim=-1)

    return E_hat.to(pre_softmax_row.dtype)

# ---------------------------------------------------------------------------
# Calibration & State Management
# ---------------------------------------------------------------------------
def compute_M_target_S_target_from_post_softmax_row(
    attn_row: torch.Tensor, num_visual_tokens: int, visual_start_idx: int = 0, eps: float = 1e-8,
) -> tuple:
    seq_len = attn_row.shape[-1]
    v_end = min(visual_start_idx + num_visual_tokens, seq_len)
    A_v = attn_row[..., visual_start_idx:v_end]
    
    N_v = A_v.shape[-1]
    if N_v == 0:
        batch_size, num_heads = attn_row.shape[0], attn_row.shape[1]
        dummy = torch.zeros((batch_size, num_heads), device=attn_row.device, dtype=attn_row.dtype)
        return dummy, dummy
        
    M_per_head = A_v.sum(dim=-1)
    A_v_sum = A_v.sum(dim=-1, keepdim=True).clamp(min=eps)
    A_v_norm = (A_v / A_v_sum).float().clamp(min=eps)
    ent = -(A_v_norm * A_v_norm.log()).sum(dim=-1)
    
    log_N = math.log(N_v) if N_v > 1 else 1.0
    S_per_head = 1.0 - ent.float() / log_N
    return M_per_head.float(), S_per_head


def apply_madrag_average_from_reference(
    probs_with_doc: torch.Tensor,
    probs_no_doc_ref: torch.Tensor,
    num_visual_tokens: int,
    visual_start_idx: int,
    question_tokens: int,
    bair_active: bool = False,  # NEW: Prevents double-penalizing the document
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MAD-RAG attention mixing rule.
    If BAIR is active, we skip halving the document tokens because BAIR's PATP
    has already penalized them optimally in the logit space.
    """
    if probs_with_doc.shape[:2] != probs_no_doc_ref.shape[:2]:
        return probs_with_doc

    out = probs_with_doc.clone()
    seq = int(out.shape[-1])
    ref_seq = int(probs_no_doc_ref.shape[-1])

    v_start = max(0, int(visual_start_idx))
    v_end = min(seq, v_start + max(0, int(num_visual_tokens)))
    ref_v_end = min(ref_seq, v_start + max(0, int(num_visual_tokens)))

    # Average visual-token attentions on overlapping visual span.
    if v_end > v_start and ref_v_end > v_start:
        n = min(v_end - v_start, ref_v_end - v_start)
        if n > 0:
            out[..., v_start : v_start + n] = 0.5 * (
                out[..., v_start : v_start + n] + probs_no_doc_ref[..., v_start : v_start + n].to(out.dtype)
            )

    # Average question-tail attentions by suffix alignment.
    qt = min(max(0, int(question_tokens)), seq, ref_seq)
    if qt > 0:
        out[..., seq - qt : seq] = 0.5 * (
            out[..., seq - qt : seq] + probs_no_doc_ref[..., ref_seq - qt : ref_seq].to(out.dtype)
        )

    # Document tokens are in the middle span for "with-doc" prompt.
    # FIX: ONLY halve the document if BAIR is NOT active.
    if not bair_active:
        doc_start = v_end
        doc_end = max(doc_start, seq - qt)
        if doc_end > doc_start:
            out[..., doc_start:doc_end] = 0.5 * out[..., doc_start:doc_end]

    z = out.sum(dim=-1, keepdim=True).clamp(min=eps)
    return out / z

def set_bottleneck_intervention(
    apply: bool,
    num_visual_tokens: int = 256,
    visual_start_idx: int = 0,
    calibration_run: bool = False,
    reset_layer: bool = False,
    alpha_v: float = 0.5,
    alpha_t: float = 1.0,
    gamma_s: float = 1.0,
    question_tokens: int = 0,
    use_madrag: bool = False,
    mspoe_scaling: float = 1.0,
    mspoe_text_only: bool = False,
):
    _intervention_state.apply = apply
    _intervention_state.num_visual_tokens = num_visual_tokens
    _intervention_state.visual_start_idx = visual_start_idx
    _intervention_state.calibration_run = calibration_run
    _intervention_state.alpha_v = alpha_v
    _intervention_state.alpha_t = alpha_t
    _intervention_state.gamma_s = gamma_s
    _intervention_state.question_tokens = question_tokens 
    _intervention_state.use_madrag = bool(use_madrag)
    _intervention_state.mspoe_scaling = float(mspoe_scaling)
    _intervention_state.mspoe_text_only = bool(mspoe_text_only)
    
    if reset_layer:
        _intervention_state.layer_idx = 0
        if calibration_run:
            _intervention_state.targets = {}
            _intervention_state.madrag_refs = {}

def get_bottleneck_intervention_params():
    apply = getattr(_intervention_state, "apply", False)
    nv = getattr(_intervention_state, "num_visual_tokens", 256)
    v_start = getattr(_intervention_state, "visual_start_idx", 0)
    calib = getattr(_intervention_state, "calibration_run", False)
    av = getattr(_intervention_state, "alpha_v", 0.5)
    at = getattr(_intervention_state, "alpha_t", 1.0)
    gs = getattr(_intervention_state, "gamma_s", 1.0)
    qt = getattr(_intervention_state, "question_tokens", 0) 
    um = getattr(_intervention_state, "use_madrag", False)
    return apply, nv, v_start, calib, av, at, gs, qt, um


def get_mspoe_intervention_params():
    ms = float(getattr(_intervention_state, "mspoe_scaling", 1.0))
    text_only = bool(getattr(_intervention_state, "mspoe_text_only", False))
    return ms, text_only


class NWPURAGInterventionManager:
    """
    Unified manager for NWPU experiments.

    This class centralizes:
    - BAIR intervention state and model patching
    - Ms-PoE position scaling hooks
    - LongLLMLingua context compression entrypoint
    """

    def __init__(self):
        self._mspoe_hook = None

    def patch_backbone(self, model_name: str, enable: bool = True) -> None:
        name = (model_name or "").lower()
        def _safe_patch(fn, flag: bool) -> None:
            try:
                fn(flag)
            except Exception:
                return
        # Always clear unrelated patches first to avoid stale hooks.
        _safe_patch(patch_gemma3_attention_for_bottleneck_intervention, False)
        _safe_patch(patch_qwen_vl_attention_for_bottleneck_intervention, False)
        _safe_patch(patch_llama_attention_for_bottleneck_intervention, False)
        _safe_patch(patch_phi3_attention_for_bottleneck_intervention, False)
        _safe_patch(patch_deepseek_attention_for_bottleneck_intervention, False)
        _safe_patch(patch_mistral_attention_for_bottleneck_intervention, False)

        if not enable:
            return
        if "qwen" in name:
            _safe_patch(patch_qwen_vl_attention_for_bottleneck_intervention, True)
        elif "llava" in name or "geochat" in name:
            _safe_patch(patch_llama_attention_for_bottleneck_intervention, True)
        elif "earthdial" in name or "phi3" in name:
            _safe_patch(patch_phi3_attention_for_bottleneck_intervention, True)
        elif "deepseek" in name:
            _safe_patch(patch_deepseek_attention_for_bottleneck_intervention, True)
        elif "gemma" in name:
            _safe_patch(patch_gemma3_attention_for_bottleneck_intervention, True)
        elif "mistral" in name:
            _safe_patch(patch_mistral_attention_for_bottleneck_intervention, True)

    def set_bair(
        self,
        enabled: bool,
        num_visual_tokens: int,
        visual_start_idx: int,
        calibration_run: bool,
        reset_layer: bool,
        alpha_v: float,
        alpha_t: float,
        gamma_s: float,
        question_tokens: int,
    ) -> None:
        set_bottleneck_intervention(
            enabled,
            num_visual_tokens=num_visual_tokens,
            visual_start_idx=visual_start_idx,
            calibration_run=calibration_run,
            reset_layer=reset_layer,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=gamma_s,
            question_tokens=question_tokens,
        )

    def enable_mspoe(
        self,
        model: Any,
        scaling_factor: float,
        text_only: bool = False,
        num_visual_tokens: int = 256,
    ) -> None:
        self.disable_mspoe()
        if abs(float(scaling_factor) - 1.0) < 1e-12:
            return

        def _pre_forward_hook(module, args, kwargs):
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                pos_ids = kwargs["position_ids"].float()
                if text_only:
                    mask = pos_ids >= num_visual_tokens
                    pos_ids[mask] = num_visual_tokens + (pos_ids[mask] - num_visual_tokens) / scaling_factor
                else:
                    pos_ids = pos_ids / scaling_factor
                kwargs["position_ids"] = pos_ids.long()
            return args, kwargs

        target = model
        if hasattr(model, "model"):
            target = model.model
        elif hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            target = model.language_model.model
        self._mspoe_hook = target.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    def disable_mspoe(self) -> None:
        if self._mspoe_hook is not None:
            self._mspoe_hook.remove()
            self._mspoe_hook = None

    def compress_longllmlingua(
        self,
        compressor: Any,
        context_docs: List[str],
        instruction: str,
        question: str,
        rate: float = 0.5,
    ) -> str:
        if not context_docs:
            return ""
        res = compressor.compress_prompt(
            context=context_docs,
            instruction=instruction or "",
            question=question,
            rate=rate,
            condition_in_question="after_condition",
            reorder_context="sort_based_on_metric",
            dynamic_context_compression_ratio=0.4,
            rank_method="longllmlingua",
        )
        return res["compressed_prompt"]

    @contextmanager
    def context(
        self,
        model_name: str,
        model: Any,
        use_bair: bool = False,
        use_mspoe: bool = False,
        mspoe_scaling: float = 1.0,
        mspoe_text_only: bool = False,
        num_visual_tokens: int = 256,
    ):
        self.patch_backbone(model_name, enable=use_bair)
        if use_mspoe:
            self.enable_mspoe(
                model=model,
                scaling_factor=mspoe_scaling,
                text_only=mspoe_text_only,
                num_visual_tokens=num_visual_tokens,
            )
        try:
            yield self
        finally:
            self.disable_mspoe()
            self.patch_backbone(model_name, enable=False)

# ---------------------------------------------------------------------------
# Patched attention forward (MedGemma, Qwen, CheXagent)
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# def eager_attention_forward_with_bottleneck_intervention(
#     module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#     attention_mask: torch.Tensor,
#     dropout: float = 0.0,
#     scaling: Optional[float] = None,
#     softcap: Optional[float] = None,
#     **kwargs,
# ):
#     apply, num_visual_tokens, v_start, calibration_run, alpha_v, alpha_t, gamma_s, question_tokens, use_madrag = get_bottleneck_intervention_params()
    
#     # =========================================================================
#     # TRUE PASS-THROUGH: Zero Interference for Baselines / (0,0) runs
#     # Gemma3: (..., mask, dropout, scaling, softcap, **kwargs)
#     # Qwen2-VL / Qwen2.5-VL / Mistral (recent transformers): (..., mask, scaling, dropout, **kwargs)
#     # =========================================================================
#     if not apply or ((alpha_v == 0.0 and alpha_t == 0.0 and not calibration_run) and not use_madrag):
#         mod_name = module.__class__.__name__
#         if "Gemma3" in mod_name:
#             import transformers.models.gemma3.modeling_gemma3 as m
#             sc = scaling if scaling is not None else module.head_dim**-0.5
#             return m._original_eager_attention_forward(module, query, key, value, attention_mask, dropout, sc, softcap, **kwargs)
#         elif "Qwen2_5" in mod_name:
#             import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m
#             sc = scaling if scaling is not None else module.head_dim**-0.5
#             return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
#         elif "Qwen2" in mod_name:
#             import transformers.models.qwen2_vl.modeling_qwen2_vl as m
#             sc = scaling if scaling is not None else module.head_dim**-0.5
#             return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
#         elif "Mistral" in mod_name:
#             import transformers.models.mistral.modeling_mistral as m
#             sc = scaling if scaling is not None else module.head_dim**-0.5
#             return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
#         elif "Deepseek" in mod_name or "DeepSeek" in mod_name:
#             import transformers.models.deepseek_v3.modeling_deepseek_v3 as m
#             sc = scaling if scaling is not None else getattr(module, "scaling", None)
#             if sc is None:
#                 # Last-resort fallback for unusual DeepSeek configs.
#                 hd = query.shape[-1]
#                 sc = hd ** -0.5
#             return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
#     # =========================================================================

#     if scaling is None:
#         scaling = module.head_dim**-0.5
#     key_states = _repeat_kv(key, module.num_key_value_groups)
#     value_states = _repeat_kv(value, module.num_key_value_groups)
#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

#     if softcap is not None:
#         attn_weights = (torch.tanh(attn_weights / softcap)) * softcap

#     if attention_mask is not None:
#         # Match recent eager_attention_forward: full add when shapes align; else slice last dim to key length.
#         if attention_mask.dim() == 4 and attention_mask.shape[-1] > key_states.shape[-2]:
#             attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
#         else:
#             attn_weights = attn_weights + attention_mask

#     seq_len = query.shape[2]
#     last_idx = seq_len - 1
#     current_layer = getattr(_intervention_state, "layer_idx", 0)
#     is_vision = "Vision" in module.__class__.__name__ or "vision" in module.__class__.__name__.lower()

#     if apply and seq_len > 1 and not is_vision:
#         row = attn_weights[:, :, last_idx, :].clone()
#         if calibration_run:
#             row_softmax = F.softmax(row.float(), dim=-1)
#             M_t, S_t = compute_M_target_S_target_from_post_softmax_row(row_softmax, num_visual_tokens, v_start)
#             _intervention_state.targets[current_layer] = {"M": M_t, "S": S_t}
#             if use_madrag:
#                 _intervention_state.madrag_refs[current_layer] = row_softmax.detach().to(torch.float32)
#             row_softmax = row_softmax.to(attn_weights.dtype)
#         else:
#             if hasattr(_intervention_state, "targets") and current_layer in _intervention_state.targets and (
#                 abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
#             ):
#                 targets = _intervention_state.targets[current_layer]
#                 row_hat = apply_bottleneck_intervention_adaptive(
#                     row, num_visual_tokens=num_visual_tokens, visual_start_idx=v_start,
#                     M_target=targets["M"], S_target=targets["S"],
#                     alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, safe_tail_tokens=question_tokens
#                 )
#                 row_softmax = F.softmax(row_hat.float(), dim=-1)
#             else:
#                 row_softmax = F.softmax(row.float(), dim=-1)

#             if use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
#                 row_softmax = apply_madrag_average_from_reference(
#                     row_softmax,
#                     _intervention_state.madrag_refs[current_layer],
#                     num_visual_tokens=num_visual_tokens,
#                     visual_start_idx=v_start,
#                     question_tokens=question_tokens,
#                 )
#             row_softmax = row_softmax.to(attn_weights.dtype)

#         attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
#         attn_weights[:, :, last_idx, :] = row_softmax
#     else:
#         attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
        
#     if not is_vision: _intervention_state.layer_idx = current_layer + 1
        
#     attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
#     return attn_output, attn_weights
def eager_attention_forward_with_bottleneck_intervention(
    module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
):
    apply, num_visual_tokens, v_start, calibration_run, alpha_v, alpha_t, gamma_s, question_tokens, use_madrag = get_bottleneck_intervention_params()
    single_token_decode = (
        apply
        and not calibration_run
        and query.shape[2] <= 1
    )
    
    # ... [Keep your existing True Pass-Through logic here] ...
    if not apply or single_token_decode or ((alpha_v == 0.0 and alpha_t == 0.0 and not calibration_run) and not use_madrag):
        mod_name = module.__class__.__name__
        if "Gemma3" in mod_name:
            import transformers.models.gemma3.modeling_gemma3 as m
            sc = scaling if scaling is not None else module.head_dim**-0.5
            return m._original_eager_attention_forward(module, query, key, value, attention_mask, dropout, sc, softcap, **kwargs)
        elif "Qwen2_5" in mod_name:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m
            sc = scaling if scaling is not None else module.head_dim**-0.5
            return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
        elif "Qwen2" in mod_name:
            import transformers.models.qwen2_vl.modeling_qwen2_vl as m
            sc = scaling if scaling is not None else module.head_dim**-0.5
            return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
        elif "Mistral" in mod_name:
            import transformers.models.mistral.modeling_mistral as m
            sc = scaling if scaling is not None else module.head_dim**-0.5
            return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)
        elif "Deepseek" in mod_name or "DeepSeek" in mod_name:
            import transformers.models.deepseek_v3.modeling_deepseek_v3 as m
            sc = scaling if scaling is not None else getattr(module, "scaling", None)
            if sc is None: sc = query.shape[-1] ** -0.5
            return m._original_eager_attention_forward(module, query, key, value, attention_mask, sc, dropout, **kwargs)

    if scaling is None: scaling = module.head_dim**-0.5
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None: attn_weights = (torch.tanh(attn_weights / softcap)) * softcap

    if attention_mask is not None:
        if attention_mask.dim() == 4 and attention_mask.shape[-1] > key_states.shape[-2]:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            attn_weights = attn_weights + attention_mask

    seq_len = query.shape[2]
    last_idx = seq_len - 1
    current_layer = getattr(_intervention_state, "layer_idx", 0)
    is_vision = "Vision" in module.__class__.__name__ or "vision" in module.__class__.__name__.lower()

    if apply and seq_len > 1 and not is_vision:
        row = attn_weights[:, :, last_idx, :].clone()
        bair_active_this_layer = hasattr(_intervention_state, "targets") and current_layer in _intervention_state.targets and (abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12)

        if calibration_run:
            row_softmax = F.softmax(row.float(), dim=-1)
            M_t, S_t = compute_M_target_S_target_from_post_softmax_row(row_softmax, num_visual_tokens, v_start)
            _intervention_state.targets[current_layer] = {"M": M_t, "S": S_t}
            if use_madrag:
                _intervention_state.madrag_refs[current_layer] = row_softmax.detach().to(torch.float32)
            row_softmax = row_softmax.to(attn_weights.dtype)
        else:
            if bair_active_this_layer:
                targets = _intervention_state.targets[current_layer]
                row_hat = apply_bottleneck_intervention_adaptive(
                    row, num_visual_tokens=num_visual_tokens, visual_start_idx=v_start,
                    M_target=targets["M"], S_target=targets["S"],
                    alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, safe_tail_tokens=question_tokens
                )
                row_softmax = F.softmax(row_hat.float(), dim=-1)
            else:
                row_softmax = F.softmax(row.float(), dim=-1)

            if use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
                row_softmax = apply_madrag_average_from_reference(
                    row_softmax, _intervention_state.madrag_refs[current_layer],
                    num_visual_tokens=num_visual_tokens, visual_start_idx=v_start, question_tokens=question_tokens,
                    bair_active=bair_active_this_layer
                )
            row_softmax = row_softmax.to(attn_weights.dtype)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
        attn_weights[:, :, last_idx, :] = row_softmax
    else:
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
        
    if not is_vision: _intervention_state.layer_idx = current_layer + 1
        
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights
def patch_gemma3_attention_for_bottleneck_intervention(use_intervention: bool = True):
    import transformers.models.gemma3.modeling_gemma3 as m
    if use_intervention:
        if not hasattr(m, "_original_eager_attention_forward"): m._original_eager_attention_forward = m.eager_attention_forward
        m.eager_attention_forward = eager_attention_forward_with_bottleneck_intervention
    else:
        if hasattr(m, "_original_eager_attention_forward"): m.eager_attention_forward = m._original_eager_attention_forward

def patch_qwen_vl_attention_for_bottleneck_intervention(use_intervention: bool = True):
    try:
        import transformers.models.qwen2_vl.modeling_qwen2_vl as q2
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q25
    except Exception:
        # Older/newer transformers builds may not ship Qwen2-VL modules.
        return
    for m in (q2, q25):
        if use_intervention:
            if not hasattr(m, "_original_eager_attention_forward"): m._original_eager_attention_forward = m.eager_attention_forward
            m.eager_attention_forward = eager_attention_forward_with_bottleneck_intervention
        elif hasattr(m, "_original_eager_attention_forward"): m.eager_attention_forward = m._original_eager_attention_forward

# def patch_deepseek_attention_for_bottleneck_intervention(use_intervention: bool = True):
#     """
#     Patch DeepSeek-V3 eager attention path for BAIR intervention.
#     Uses the same eager_attention_forward interception pattern as Qwen/Gemma.
#     """
#     candidates = []
#     try:
#         import transformers.models.deepseek_v3.modeling_deepseek_v3 as dsv3
#         candidates.append(dsv3)
#     except Exception:
#         pass
#     try:
#         import transformers.models.deepseek_v2.modeling_deepseek_v2 as dsv2
#         candidates.append(dsv2)
#     except Exception:
#         pass

#     if not candidates:
#         # No compatible DeepSeek eager-attention module in this transformers build.
#         return

#     for m in candidates:
#         if use_intervention:
#             if not hasattr(m, "_original_eager_attention_forward"):
#                 m._original_eager_attention_forward = m.eager_attention_forward
#             m.eager_attention_forward = eager_attention_forward_with_bottleneck_intervention
#         else:
#             if hasattr(m, "_original_eager_attention_forward"):
#                 m.eager_attention_forward = m._original_eager_attention_forward
import sys
_deepseek_patch_warned = False

def patch_deepseek_attention_for_bottleneck_intervention(use_intervention: bool = True):
    # Patch all loaded DeepSeek modules from trust_remote_code runs.
    ds_modules = []
    for module_name, module in sys.modules.items():
        if "modeling_deepseek" in module_name and hasattr(module, "DeepseekAttention"):
            ds_modules.append(module)

    global _deepseek_patch_warned
    if ds_modules:
        for ds_mod in ds_modules:
            # Force dictionary fallback just in case
            if hasattr(ds_mod, "DEEPSEEK_ATTENTION_CLASSES"):
                ds_mod.DEEPSEEK_ATTENTION_CLASSES["sdpa"] = ds_mod.DeepseekAttention
                ds_mod.DEEPSEEK_ATTENTION_CLASSES["flash_attention_2"] = ds_mod.DeepseekAttention

            if use_intervention:
                if not hasattr(ds_mod.DeepseekAttention, "_original_forward"):
                    ds_mod.DeepseekAttention._original_forward = ds_mod.DeepseekAttention.forward
                # DeepSeek decoder attention is LLaMA-like for Q/K/V + RoPE, so reuse patched forward.
                ds_mod.DeepseekAttention.forward = llama_attention_forward_with_bair
            else:
                if hasattr(ds_mod.DeepseekAttention, "_original_forward"):
                    ds_mod.DeepseekAttention.forward = ds_mod.DeepseekAttention._original_forward

    elif use_intervention and not _deepseek_patch_warned:
        print(
            "[DeepSeek BAIR] No DeepseekAttention module found; "
            "intervention hooks are not attached for this model runtime."
        )
        _deepseek_patch_warned = True
def patch_mistral_attention_for_bottleneck_intervention(use_intervention: bool = True):
    import transformers.models.mistral.modeling_mistral as mistral_mod

    def _legacy_forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Any] = None, output_attentions: bool = False, use_cache: bool = False, **kwargs):
        apply, num_visual_tokens, v_start, calibration_run, alpha_v, alpha_t, gamma_s, question_tokens, use_madrag = get_bottleneck_intervention_params()
        single_token_decode = (
            apply
            and not calibration_run
            and hidden_states.shape[1] <= 1
        )
        
        # PASS-THROUGH FOR LEGACY MISTRAL
        if not apply or single_token_decode or ((alpha_v == 0.0 and alpha_t == 0.0 and not calibration_run) and not use_madrag):
            return mistral_mod.MistralAttention._original_forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None: kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = mistral_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})

        key_states = mistral_mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = mistral_mod.repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None: attn_weights = attn_weights + attention_mask

        seq_len = q_len
        last_idx = seq_len - 1
        current_layer = getattr(_intervention_state, "layer_idx", 0)
        is_vision = "Vision" in self.__class__.__name__ or "vision" in self.__class__.__name__.lower()

        if apply and seq_len > 1 and not is_vision:
            row = attn_weights[:, :, last_idx, :].clone()
            if calibration_run:
                row_softmax = F.softmax(row.float(), dim=-1)
                M_t, S_t = compute_M_target_S_target_from_post_softmax_row(row_softmax, num_visual_tokens, v_start)
                _intervention_state.targets[current_layer] = {"M": M_t, "S": S_t}
                if use_madrag:
                    _intervention_state.madrag_refs[current_layer] = row_softmax.detach().to(torch.float32)
                row_softmax = row_softmax.to(attn_weights.dtype)
            else:
                if hasattr(_intervention_state, "targets") and current_layer in _intervention_state.targets and (
                    abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12
                ):
                    targets = _intervention_state.targets[current_layer]
                    row_hat = apply_bottleneck_intervention_adaptive(
                        row, num_visual_tokens=num_visual_tokens, visual_start_idx=v_start,
                        M_target=targets["M"], S_target=targets["S"],
                        alpha_v=alpha_v, alpha_t=alpha_t, gamma_s=gamma_s, safe_tail_tokens=question_tokens,
                    )
                    row_softmax = F.softmax(row_hat.float(), dim=-1)
                else:
                    row_softmax = F.softmax(row.float(), dim=-1)

                if use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
                    row_softmax = apply_madrag_average_from_reference(
                        row_softmax,
                        _intervention_state.madrag_refs[current_layer],
                        num_visual_tokens=num_visual_tokens,
                        visual_start_idx=v_start,
                        question_tokens=question_tokens,
                    )
                row_softmax = row_softmax.to(attn_weights.dtype)

            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
            attn_weights[:, :, last_idx, :] = row_softmax
        else:
            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)

        if not is_vision: _intervention_state.layer_idx = current_layer + 1

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), past_key_value

    if use_intervention:
        if hasattr(mistral_mod, "eager_attention_forward"):
            if not hasattr(mistral_mod, "_original_eager_attention_forward"): mistral_mod._original_eager_attention_forward = mistral_mod.eager_attention_forward
            mistral_mod.eager_attention_forward = eager_attention_forward_with_bottleneck_intervention
        else:
            if hasattr(mistral_mod, "MISTRAL_ATTENTION_CLASSES"):
                mistral_mod.MISTRAL_ATTENTION_CLASSES["sdpa"] = mistral_mod.MistralAttention
                mistral_mod.MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = mistral_mod.MistralAttention
            if not hasattr(mistral_mod.MistralAttention, "_original_forward"): mistral_mod.MistralAttention._original_forward = mistral_mod.MistralAttention.forward
            mistral_mod.MistralAttention.forward = _legacy_forward
    else:
        if hasattr(mistral_mod, "_original_eager_attention_forward"): mistral_mod.eager_attention_forward = mistral_mod._original_eager_attention_forward
        if hasattr(mistral_mod.MistralAttention, "_original_forward"): mistral_mod.MistralAttention.forward = mistral_mod.MistralAttention._original_forward

# # ---------------------------------------------------------------------------
# # LLaMA / Med-Flamingo Intervention (For Cross-Attention VLMs)
# # ---------------------------------------------------------------------------
# def llama_attention_forward_with_bair(self, hidden_states: torch.Tensor, *args, **kwargs):
#     """
#     Compatibility patch for LlamaAttention across old/new Transformers APIs.
#     Supports both:
#       - old: forward(hidden_states, attention_mask, position_ids, past_key_value, ...)
#       - new: forward(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, ...)
#     """
#     apply, nv, v_start, calib_run, alpha_v, alpha_t, gamma_s, qt, use_madrag = get_bottleneck_intervention_params()

#     # =========================================================================
#     # TRUE PASS-THROUGH
#     # =========================================================================
#     if not apply or ((alpha_v == 0.0 and alpha_t == 0.0 and not calib_run) and not use_madrag):
#         import transformers.models.llama.modeling_llama as llama_mod
#         return llama_mod.LlamaAttention._original_forward(self, hidden_states, *args, **kwargs)

#     # Parse both legacy and recent calling conventions.
#     position_embeddings = kwargs.pop("position_embeddings", None)
#     attention_mask = kwargs.pop("attention_mask", None)
#     position_ids = kwargs.pop("position_ids", None)
#     past_key_values = kwargs.pop("past_key_values", None)
#     past_key_value = kwargs.pop("past_key_value", None)
#     cache_position = kwargs.pop("cache_position", None)
#     output_attentions = kwargs.pop("output_attentions", False)
#     use_cache = kwargs.pop("use_cache", False)
#     if past_key_values is None:
#         past_key_values = past_key_value

#     new_api_call = False
#     if len(args) > 0:
#         first = args[0]
#         if isinstance(first, tuple) and len(first) == 2:
#             # New API positional form.
#             new_api_call = True
#             position_embeddings = first
#             if len(args) > 1:
#                 attention_mask = args[1]
#             if len(args) > 2:
#                 past_key_values = args[2]
#             if len(args) > 3:
#                 cache_position = args[3]
#         else:
#             # Legacy positional form.
#             attention_mask = first
#             if len(args) > 1:
#                 position_ids = args[1]
#             if len(args) > 2:
#                 past_key_values = args[2]
#             if len(args) > 3:
#                 output_attentions = args[3]
#             if len(args) > 4:
#                 use_cache = args[4]

#     cfg = getattr(self, "config", None)
#     head_dim = getattr(self, "head_dim", None)
#     if head_dim is None:
#         if cfg is None:
#             raise AttributeError("LlamaAttention patch could not resolve head_dim.")
#         head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

#     num_heads = getattr(self, "num_heads", None)
#     if num_heads is None:
#         if cfg is not None and hasattr(cfg, "num_attention_heads"):
#             num_heads = int(cfg.num_attention_heads)
#         elif hasattr(self, "q_proj") and hasattr(self.q_proj, "out_features"):
#             num_heads = int(self.q_proj.out_features // head_dim)
#         else:
#             raise AttributeError("LlamaAttention patch could not resolve num_heads.")

#     num_kv_heads = getattr(self, "num_key_value_heads", None)
#     if num_kv_heads is None:
#         if cfg is not None and hasattr(cfg, "num_key_value_heads"):
#             num_kv_heads = int(cfg.num_key_value_heads)
#         else:
#             num_kv_heads = num_heads

#     hidden_size = getattr(self, "hidden_size", None)
#     if hidden_size is None:
#         if cfg is not None and hasattr(cfg, "hidden_size"):
#             hidden_size = int(cfg.hidden_size)
#         else:
#             hidden_size = int(num_heads * head_dim)

#     num_kv_groups = getattr(self, "num_key_value_groups", None)
#     if num_kv_groups is None:
#         num_kv_groups = max(1, int(num_heads // max(1, num_kv_heads)))

#     bsz, q_len, _ = hidden_states.size()
#     query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
#     key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_values is not None and hasattr(past_key_values, "get_usable_length"):
#         kv_seq_len += past_key_values.get_usable_length(kv_seq_len, getattr(self, "layer_idx", 0))

#     import transformers.models.llama.modeling_llama as llama_mod
#     if position_embeddings is not None:
#         cos, sin = position_embeddings
#     else:
#         try:
#             cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         except TypeError:
#             cos, sin = self.rotary_emb(value_states, position_ids)

#     try:
#         query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
#     except TypeError:
#         query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#     if past_key_values is not None and hasattr(past_key_values, "update"):
#         cache_kwargs = {}
#         if cache_position is not None:
#             cache_kwargs["cache_position"] = cache_position
#         if "cache_kwargs" in kwargs and isinstance(kwargs["cache_kwargs"], dict):
#             cache_kwargs.update(kwargs["cache_kwargs"])
#         key_states, value_states = past_key_values.update(
#             key_states, value_states, getattr(self, "layer_idx", 0), cache_kwargs
#         )

#     from transformers.models.llama.modeling_llama import repeat_kv
#     key_states = repeat_kv(key_states, num_kv_groups)
#     value_states = repeat_kv(value_states, num_kv_groups)

#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
#     if attention_mask is not None:
#         if attention_mask.dim() == 4:
#             q_target = query_states.shape[-2]
#             k_target = key_states.shape[-2]
#             am = attention_mask
#             if am.shape[-2] != q_target or am.shape[-1] != k_target:
#                 if am.shape[-2] < q_target or am.shape[-1] < k_target:
#                     raise RuntimeError(
#                         f"Attention mask shape {tuple(am.shape)} is smaller than target "
#                         f"(q={q_target}, k={k_target}) in LLaMA BAIR patch."
#                     )
#                 am = am[:, :, :q_target, :k_target]
#             attn_weights = attn_weights + am
#         else:
#             attn_weights = attn_weights + attention_mask

#     current_layer = getattr(_intervention_state, "layer_idx", 0)

#     # [BAIR Intervention] RAG Context Suppression (alpha_t)
#     if apply and (abs(alpha_t) > 1e-12) and kv_seq_len > qt:
#         # For HF LLaVA/DeepSeek-LM backbones, visual tokens can be part of LM sequence.
#         # Do not penalize visual tokens; only penalize text-context span before question tail.
#         v_start = int(v_start)
#         nv = int(max(0, nv))
#         qt = int(max(0, qt))
#         ctx_start = max(0, v_start + nv)
#         ctx_end = max(ctx_start, kv_seq_len - qt)
#         if ctx_end > ctx_start:
#             attn_weights[:, :, :, ctx_start:ctx_end] -= (alpha_t * 0.5)
#     # =======================================================================

#     attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

#     if apply and q_len > 1:
#         row_probs = attn_probs[:, :, q_len - 1, :]
#         if calib_run:
#             # Keep layer-indexed no-doc reference for MAD-RAG.
#             if use_madrag:
#                 _intervention_state.madrag_refs[current_layer] = row_probs.detach().to(torch.float32)
#         elif use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
#             row_mixed = apply_madrag_average_from_reference(
#                 row_probs,
#                 _intervention_state.madrag_refs[current_layer],
#                 num_visual_tokens=nv,
#                 visual_start_idx=v_start,
#                 question_tokens=qt,
#             )
#             attn_probs = attn_probs.clone()
#             attn_probs[:, :, q_len - 1, :] = row_mixed

#     _intervention_state.layer_idx = current_layer + 1
#     attn_weights = attn_probs.to(query_states.dtype)

#     attn_weights = torch.nn.functional.dropout(
#         attn_weights, p=getattr(self, "attention_dropout", 0.0), training=self.training
#     )

#     attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
#     attn_output = self.o_proj(attn_output)

#     # Return shape follows caller API expectations.
#     # Older LLaMA decoder layers unpack 3 values; newer ones unpack 2.
#     try:
#         orig_sig = inspect.signature(llama_mod.LlamaAttention._original_forward)
#         expects_two = "position_embeddings" in orig_sig.parameters and "cache_position" in orig_sig.parameters
#     except Exception:
#         expects_two = bool(new_api_call or cache_position is not None)

#     if expects_two:
#         return attn_output, attn_weights
#     if output_attentions:
#         return attn_output, attn_weights, past_key_values
#     return attn_output, None, past_key_values
def llama_attention_forward_with_bair(self, hidden_states: torch.Tensor, *args, **kwargs):
    """
    BAIR on LlamaAttention. Must return the same tuple contract as Transformers:
    (attn_output, attn_weights_or_None, present_key_value_or_None).

    Bug fix: the previous version returned the *input* ``past_key_values`` handle (or
    unrelated objects) as the third value, so each decoder layer stored ``None`` /
    wrong entries in ``next_decoder_cache`` and ``LlamaModel`` then crashed with
    ``past_key_values[0][0]`` being ``None`` ('NoneType' is not subscriptable).
    """
    apply, nv, v_start, calib_run, alpha_v, alpha_t, gamma_s, qt, use_madrag = get_bottleneck_intervention_params()
    if not apply or ((alpha_v == 0.0 and alpha_t == 0.0 and not calib_run) and not use_madrag):
        import transformers.models.llama.modeling_llama as llama_mod
        return llama_mod.LlamaAttention._original_forward(self, hidden_states, *args, **kwargs)

    # --- LLaMA input parsing (positional + kwargs; match both old and new call styles) ---
    position_embeddings = kwargs.pop("position_embeddings", None)
    attention_mask = kwargs.pop("attention_mask", None)
    position_ids = kwargs.pop("position_ids", None)
    past_kv = kwargs.pop("past_key_values", kwargs.pop("past_key_value", None))
    cache_position = kwargs.pop("cache_position", None)
    output_attentions = kwargs.pop("output_attentions", False)
    use_cache = kwargs.pop("use_cache", False)
    new_api_call = False

    if len(args) > 0:
        first = args[0]
        if isinstance(first, tuple) and len(first) == 2:
            new_api_call = True
            position_embeddings = first
            if len(args) > 1:
                attention_mask = args[1]
            if len(args) > 2:
                past_kv = args[2]
            if len(args) > 3:
                cache_position = args[3]
        else:
            attention_mask = first
            if len(args) > 1:
                position_ids = args[1]
            if len(args) > 2:
                past_kv = args[2]
            if len(args) > 3:
                output_attentions = args[3]
            if len(args) > 4:
                use_cache = args[4]

    cfg = getattr(self, "config", None)
    head_dim = getattr(self, "head_dim", getattr(cfg, "head_dim", getattr(cfg, "hidden_size", 4096) // getattr(cfg, "num_attention_heads", 32)))
    num_heads = getattr(self, "num_heads", getattr(cfg, "num_attention_heads", 32))
    num_kv_heads = getattr(self, "num_key_value_heads", getattr(cfg, "num_key_value_heads", num_heads))
    hidden_size = getattr(self, "hidden_size", getattr(cfg, "hidden_size", num_heads * head_dim))
    num_kv_groups = getattr(self, "num_key_value_groups", max(1, int(num_heads // max(1, num_kv_heads))))
    layer_idx = getattr(self, "layer_idx", 0)

    import transformers.models.llama.modeling_llama as llama_mod
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_kv is not None and hasattr(past_kv, "get_usable_length"):
        kv_seq_len += past_kv.get_usable_length(kv_seq_len, layer_idx)
    elif past_kv is not None and isinstance(past_kv, tuple) and past_kv[0] is not None:
        kv_seq_len += past_kv[0].shape[-2]

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        try:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        except TypeError:
            cos, sin = self.rotary_emb(value_states, position_ids)

    try:
        query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # KV cache: exactly one path — HF either uses a Cache object (.update) or a legacy (k, v) tuple per layer.
    cache_object_in_use = False
    if past_kv is not None and hasattr(past_kv, "update"):
        cache_object_in_use = True
        cache_kwargs = {"cache_position": cache_position} if cache_position is not None else {}
        if "cache_kwargs" in kwargs and isinstance(kwargs["cache_kwargs"], dict):
            cache_kwargs.update(kwargs["cache_kwargs"])
        key_states, value_states = past_kv.update(key_states, value_states, layer_idx, cache_kwargs)
    elif past_kv is not None and isinstance(past_kv, tuple) and past_kv[0] is not None and past_kv[1] is not None:
        # Legacy: concat past K/V after RoPE (same order as HF LlamaAttention).
        key_states = torch.cat([past_kv[0], key_states], dim=2)
        value_states = torch.cat([past_kv[1], value_states], dim=2)

    if use_cache and cache_object_in_use:
        present_key_value = past_kv
    else:
        present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            (key_states, value_states) if use_cache else None
        )

    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)

    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            q_target, k_target = query_states.shape[-2], key_states.shape[-2]
            am = attention_mask
            if am.shape[-2] != q_target or am.shape[-1] != k_target:
                am = am[:, :, :q_target, :k_target]
            attn_scores = attn_scores + am
        else:
            attn_scores = attn_scores + attention_mask

    # --- BAIR INTERVENTION CORE ---
    current_layer = getattr(_intervention_state, "layer_idx", 0)
    is_vision = "Vision" in self.__class__.__name__ or "vision" in self.__class__.__name__.lower()

    if apply and q_len > 1 and not is_vision:
        row = attn_scores[:, :, q_len - 1, :].clone()
        bair_active_this_layer = (
            hasattr(_intervention_state, "targets")
            and current_layer in _intervention_state.targets
            and (abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12)
        )

        if calib_run:
            row_softmax = F.softmax(row.float(), dim=-1)
            M_t, S_t = compute_M_target_S_target_from_post_softmax_row(row_softmax, nv, v_start)
            _intervention_state.targets[current_layer] = {"M": M_t, "S": S_t}
            if use_madrag:
                _intervention_state.madrag_refs[current_layer] = row_softmax.detach().to(torch.float32)
            row_softmax = row_softmax.to(attn_scores.dtype)
        else:
            if bair_active_this_layer:
                targets = _intervention_state.targets[current_layer]
                row_hat = apply_bottleneck_intervention_adaptive(
                    row,
                    num_visual_tokens=nv,
                    visual_start_idx=v_start,
                    M_target=targets["M"],
                    S_target=targets["S"],
                    alpha_v=alpha_v,
                    alpha_t=alpha_t,
                    gamma_s=gamma_s,
                    safe_tail_tokens=qt,
                )
                row_softmax = F.softmax(row_hat.float(), dim=-1)
            else:
                row_softmax = F.softmax(row.float(), dim=-1)

            if use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
                row_softmax = apply_madrag_average_from_reference(
                    row_softmax,
                    _intervention_state.madrag_refs[current_layer],
                    num_visual_tokens=nv,
                    visual_start_idx=v_start,
                    question_tokens=qt,
                    bair_active=bair_active_this_layer,
                )
            row_softmax = row_softmax.to(attn_scores.dtype)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
        attn_probs[:, :, q_len - 1, :] = row_softmax
    else:
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)

    if not is_vision:
        _intervention_state.layer_idx = current_layer + 1

    attn_probs = torch.nn.functional.dropout(
        attn_probs, p=getattr(self, "attention_dropout", 0.0), training=self.training
    )

    attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)

    attn_weights_out: Optional[torch.Tensor] = attn_probs if output_attentions else None
    return attn_output, attn_weights_out, present_key_value
def patch_llama_attention_for_bottleneck_intervention(use_intervention: bool = True):
    import transformers.models.llama.modeling_llama as llama_mod
    if hasattr(llama_mod, "LLAMA_ATTENTION_CLASSES"):
        llama_mod.LLAMA_ATTENTION_CLASSES["sdpa"] = llama_mod.LlamaAttention
        llama_mod.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = llama_mod.LlamaAttention

    if use_intervention:
        if not hasattr(llama_mod.LlamaAttention, "_original_forward"): llama_mod.LlamaAttention._original_forward = llama_mod.LlamaAttention.forward
        llama_mod.LlamaAttention.forward = llama_attention_forward_with_bair
    else:
        if hasattr(llama_mod.LlamaAttention, "_original_forward"): llama_mod.LlamaAttention.forward = llama_mod.LlamaAttention._original_forward


def phi3_attention_forward_with_bair(self, hidden_states: torch.Tensor, *args, **kwargs):
    apply, nv, v_start, calib_run, alpha_v, alpha_t, gamma_s, qt, use_madrag = get_bottleneck_intervention_params()
    single_token_decode = (
        apply
        and not calib_run
        and hidden_states.shape[1] <= 1
    )
    if not apply or single_token_decode or ((alpha_v == 0.0 and alpha_t == 0.0 and not calib_run) and not use_madrag):
        return self.__class__._original_forward(self, hidden_states, *args, **kwargs)

    attention_mask = kwargs.pop("attention_mask", None)
    position_ids = kwargs.pop("position_ids", None)
    past_key_value = kwargs.pop("past_key_value", None)
    output_attentions = kwargs.pop("output_attentions", False)
    use_cache = kwargs.pop("use_cache", False)

    if len(args) > 0:
        attention_mask = args[0]
    if len(args) > 1:
        position_ids = args[1]
    if len(args) > 2:
        past_key_value = args[2]
    if len(args) > 3:
        output_attentions = args[3]
    if len(args) > 4:
        use_cache = args[4]

    import earthdial.model.phi3.modeling_phi3 as phi3_mod
    mspoe_scaling, mspoe_text_only = get_mspoe_intervention_params()

    bsz, q_len, _ = hidden_states.size()
    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos: query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim:]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is not None and hasattr(past_key_value, "get_usable_length"):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, layer_idx)

    position_ids_eff = position_ids
    if position_ids is not None and abs(float(mspoe_scaling) - 1.0) > 1e-12:
        pos = position_ids.float()
        if mspoe_text_only:
            mask = pos >= float(nv)
            pos[mask] = float(nv) + (pos[mask] - float(nv)) / float(mspoe_scaling)
        else:
            pos = pos / float(mspoe_scaling)
        position_ids_eff = pos.long()

    cos, sin = self.rotary_emb(value_states, position_ids_eff, seq_len=kv_seq_len)
    query_states, key_states = phi3_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_eff)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, getattr(self, "layer_idx", 0), cache_kwargs
        )

    key_states = phi3_mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = phi3_mod.repeat_kv(value_states, self.num_key_value_groups)

    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            q_target, k_target = query_states.shape[-2], key_states.shape[-2]
            am = attention_mask
            if am.shape[-2] != q_target or am.shape[-1] != k_target:
                am = am[:, :, :q_target, :k_target]
            attn_scores = attn_scores + am
        else:
            attn_scores = attn_scores + attention_mask

    current_layer = getattr(_intervention_state, "layer_idx", 0)
    if apply and q_len > 1:
        row = attn_scores[:, :, q_len - 1, :].clone()
        bair_active_this_layer = (
            hasattr(_intervention_state, "targets")
            and current_layer in _intervention_state.targets
            and (abs(alpha_v) > 1e-12 or abs(alpha_t) > 1e-12)
        )
        if calib_run:
            row_softmax = F.softmax(row.float(), dim=-1)
            M_t, S_t = compute_M_target_S_target_from_post_softmax_row(row_softmax, nv, v_start)
            _intervention_state.targets[current_layer] = {"M": M_t, "S": S_t}
            if use_madrag:
                _intervention_state.madrag_refs[current_layer] = row_softmax.detach().to(torch.float32)
            row_softmax = row_softmax.to(attn_scores.dtype)
        else:
            if bair_active_this_layer:
                targets = _intervention_state.targets[current_layer]
                row_hat = apply_bottleneck_intervention_adaptive(
                    row,
                    num_visual_tokens=nv,
                    visual_start_idx=v_start,
                    M_target=targets["M"],
                    S_target=targets["S"],
                    alpha_v=alpha_v,
                    alpha_t=alpha_t,
                    gamma_s=gamma_s,
                    safe_tail_tokens=qt,
                )
                row_softmax = F.softmax(row_hat.float(), dim=-1)
            else:
                row_softmax = F.softmax(row.float(), dim=-1)
            if use_madrag and hasattr(_intervention_state, "madrag_refs") and current_layer in _intervention_state.madrag_refs:
                row_softmax = apply_madrag_average_from_reference(
                    row_softmax,
                    _intervention_state.madrag_refs[current_layer],
                    num_visual_tokens=nv,
                    visual_start_idx=v_start,
                    question_tokens=qt,
                    bair_active=bair_active_this_layer,
                )
            row_softmax = row_softmax.to(attn_scores.dtype)
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
        attn_probs[:, :, q_len - 1, :] = row_softmax
    else:
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)

    _intervention_state.layer_idx = current_layer + 1
    attn_probs = F.dropout(attn_probs, p=getattr(self, "attention_dropout", 0.0), training=self.training)
    attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    attn_weights = attn_probs if output_attentions else None
    return attn_output, attn_weights, past_key_value


def patch_phi3_attention_for_bottleneck_intervention(use_intervention: bool = True):
    try:
        import earthdial.model.phi3.modeling_phi3 as phi3_mod
    except Exception:
        return

    classes = []
    if hasattr(phi3_mod, "Phi3Attention"):
        classes.append(phi3_mod.Phi3Attention)
    if hasattr(phi3_mod, "Phi3SdpaAttention"):
        classes.append(phi3_mod.Phi3SdpaAttention)

    for cls in classes:
        if use_intervention:
            if not hasattr(cls, "_original_forward"):
                cls._original_forward = cls.forward
            cls.forward = phi3_attention_forward_with_bair
        else:
            if hasattr(cls, "_original_forward"):
                cls.forward = cls._original_forward
