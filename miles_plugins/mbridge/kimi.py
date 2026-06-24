# Copyright (c) 2026, Modal.
"""Kimi-K2.x HF->Megatron import bridge (mbridge).

Kimi-K2.5 / K2.6 ship as ``KimiK25ForConditionalGeneration`` (top-level
``model_type == "kimi_k25"``) — a vision-language wrapper whose *language model*
is a DeepSeek-V3 architecture (MLA + DeepSeek-MoE). For RL we bridge ONLY the
language model into a Megatron DeepSeek-V3 ``GPTModel``; ``vision_tower`` /
``mm_projector`` are not trained and are skipped.

Two deltas vs ``DeepseekV3Bridge``:
  1. the LM config is nested under ``hf_config.text_config`` (which uses the same
     field names ``DeepseekV3Bridge`` expects: q_lora_rank, kv_lora_rank,
     qk_nope_head_dim, n_routed_experts, rope_theta, rope_scaling[yarn], ...);
  2. HF tensor names are prefixed ``language_model.`` (e.g.
     ``language_model.model.layers.N.self_attn.q_a_proj.weight``).

mbridge's ``AutoBridge`` dispatches on the TOP-LEVEL ``model_type``, hence the
``@register_model("kimi_k25")``. K2.6 has no MTP (num_nextn_predict_layers == 0),
so the DeepseekV3 DIRECT/MLP/ATTENTION mappings cover every LM tensor (verified
against the K2.6 safetensors index: embed_tokens, model.norm, lm_head, per-layer
input_layernorm, MLA self_attn {q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
kv_a_layernorm,kv_b_proj,o_proj}, post_attention_layernorm, dense mlp.{gate,up,down}_proj
for the first FIRST_K_DENSE_REPLACE layer, and MoE mlp.{gate,experts.*,shared_experts.*}).
"""

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


def _prefix_language_model(mapping: dict) -> dict:
    """Prepend the VLM ``language_model.`` prefix to every HF weight name in a
    DeepseekV3Bridge name-mapping dict (values are ``str`` or ``list[str]``)."""
    out: dict = {}
    for mcore_name, hf in mapping.items():
        out[mcore_name] = f"language_model.{hf}" if isinstance(hf, str) else [f"language_model.{n}" for n in hf]
    return out


@register_model("kimi_k25")
class KimiBridge(DeepseekV3Bridge):
    """Bridge the Kimi-K2.x language model (DeepSeek-V3 arch) into Megatron-Core."""

    _DIRECT_MAPPING = _prefix_language_model(DeepseekV3Bridge._DIRECT_MAPPING)
    _MLP_MAPPING = _prefix_language_model(DeepseekV3Bridge._MLP_MAPPING)
    _ATTENTION_MAPPING = _prefix_language_model(DeepseekV3Bridge._ATTENTION_MAPPING)

    def __init__(self, hf_config, *args, **kwargs):
        # The DeepSeek-V3 language-model config (the fields DeepseekV3Bridge reads)
        # lives under text_config; bridge that sub-config, not the VLM wrapper.
        super().__init__(hf_config.text_config, *args, **kwargs)
