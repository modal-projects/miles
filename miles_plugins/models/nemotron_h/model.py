"""CUDA-only Nemotron Nano provider using Megatron-Core's hybrid stack."""

from dataclasses import replace

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.mbridge.nemotron_h import NemotronHBridge


class _AttentionOnlyLayer(TransformerLayer):
    def _forward_mlp(self, hidden_states, *args, **kwargs):
        return hidden_states


class _MoEOnlyLayer(MoETransformerLayer):
    def _forward_attention(self, hidden_states, *args, **kwargs):
        return hidden_states, None


class NemotronModel(HybridModel):
    def forward(self, *args, mtp_kwargs=None, **kwargs):
        assert (mtp_kwargs or {}).get("mtp_labels") is None, "native Nemotron Nano does not train MTP heads"
        return super().forward(*args, **kwargs).float()


def get_model_provider(args):
    hf = load_hf_config(args.hf_checkpoint)
    assert set(hf.hybrid_override_pattern) <= set(
        "ME*"
    ), "native Nemotron Nano supports Mamba-2, MoE and attention layers"
    converter = NemotronHBridge(hf)
    config = converter.config
    _apply_bridge_runtime_config(config, args)
    spec = replace(
        hybrid_stack_spec,
        submodules=replace(
            hybrid_stack_spec.submodules,
            attention_layer=replace(hybrid_stack_spec.submodules.attention_layer, module=_AttentionOnlyLayer),
            moe_layer=replace(hybrid_stack_spec.submodules.moe_layer, module=_MoEOnlyLayer),
        ),
    )

    def provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        assert config is None, "Nemotron Nano resolves its configuration from the HF checkpoint"
        return NemotronModel(
            config=converter.config,
            hybrid_stack_spec=spec,
            vocab_size=args.backend.padded_vocab_size,
            max_sequence_length=hf.max_position_embeddings,
            hybrid_layer_pattern=hf.hybrid_override_pattern,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

    return provider
