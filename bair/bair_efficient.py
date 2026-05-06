"""Experimental BAIR runtime tweaks (allocator + small tensor reuse).

Enable::

    export BAIR_EFFICIENT_MODE=1

When set:

- Skip ``torch.cuda.empty_cache()`` after calibration (and optionally after decode
  attempts) so pools stay warm between BAIR passes. Peak VRAM between passes may be higher.
- **MedGemma:** reuse the same ``pixel_values`` tensor from the calibration batch for the
  generation batch when shapes/dtypes match (same image). Vision encoder still runs twice;
  this only avoids a duplicate vision input buffer from the second ``processor`` call.

Full vision-forward dedup needs model-specific past-key plumbing; not implemented here.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch


def enabled() -> bool:
    v = os.environ.get("BAIR_EFFICIENT_MODE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def optional_empty_cache_after_calibration() -> None:
    if enabled():
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def optional_empty_cache_after_generation_attempt() -> None:
    """Use after a non-OOM decode step; OOM paths should still call empty_cache."""
    if enabled():
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reuse_medgemma_pixel_values_if_efficient(clean_inputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    if not enabled():
        return
    pv0 = clean_inputs.get("pixel_values")
    pv1 = inputs.get("pixel_values")
    if not isinstance(pv0, torch.Tensor) or not isinstance(pv1, torch.Tensor):
        return
    if pv0.shape != pv1.shape or pv0.dtype != pv1.dtype:
        return
    inputs["pixel_values"] = pv0
