# coding=utf-8
# Custom MedGemma model class for attention manipulation during generation.
# Subclasses the same class used by HuggingFace for google/medgemma-4b-it
# (Gemma3ForConditionalGeneration) so custom and original work identically.
# Add attention control by overriding forward() or language_model forward here.
# For bottleneck-only intervention (last input token): use bottleneck_intervention.py
# (patch_gemma3_attention_for_bottleneck_intervention + set_bottleneck_intervention).
#
# Usage (conda env cs577):
#   from custom_medgemma_model import MedGemmaForConditionalGenerationCustom
#   model = MedGemmaForConditionalGenerationCustom.from_pretrained(
#       "google/medgemma-4b-it", torch_dtype=torch.bfloat16, attn_implementation="eager"
#   )
#
# Original: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration

from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration


class MedGemmaForConditionalGenerationCustom(Gemma3ForConditionalGeneration):
    """
    Custom MedGemma class: same architecture as the original (Gemma3).
    Use this to load google/medgemma-4b-it and later add attention manipulation.
    """

    pass  # Override forward / attention in subclasses or here when needed


__all__ = ["MedGemmaForConditionalGenerationCustom"]
