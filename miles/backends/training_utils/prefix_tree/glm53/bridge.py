"""Opt-in ModuleSpec substitution; the installed Bridge provider stays untouched.

Attach ``glm_tree_layout`` to PackedSeqParams for compact SFT execution. Ordinary
packed batches retain the original provider implementation. No parameter names,
checkpoint mappings, LoRA wrappers or hyperconnection modules are replaced.
"""

from copy import deepcopy

import torch
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.glm5_next.dsa import Glm5NextDSAAttention
from megatron.bridge.models.glm5_next.kda import Glm5NextKDAAttention, fused_kda_gate
from megatron.bridge.models.glm5_next.layer_specs import glm5_next_layer_spec
from megatron.core import parallel_state

from miles.backends.training_utils.prefix_tree.glm53.dsa import tree_kpool_select
from miles.backends.training_utils.prefix_tree.glm53.kda import tree_kda
from miles.backends.training_utils.prefix_tree.megatron import validate_megatron_tree_config


def glm53_tree_layer_spec(config, vp_stage=None):
    if config.context_parallel_size != 1:
        raise ValueError("GLM prefix SFT currently requires context_parallel_size=1")
    validate_megatron_tree_config(config)
    # Bridge dispatches direct-attribute checkpoint mappings by exact class
    # name. Preserve the provider's sharding rules for our execution subclasses.
    AutoMapping.register_module_type("TreeKDAAttention", "column")
    AutoMapping.register_module_type("TreeDSAAttention", "replicated")
    spec = deepcopy(glm5_next_layer_spec(config, vp_stage=vp_stage))
    for layer in spec.layer_specs:
        attention = deepcopy(layer.submodules.self_attention)
        replacements = {Glm5NextDSAAttention: TreeDSAAttention, Glm5NextKDAAttention: TreeKDAAttention}
        if attention.module not in replacements:
            raise TypeError(f"Unexpected GLM attention provider: {attention.module}")
        attention.module = replacements[attention.module]
        layer.submodules.self_attention = attention
    return spec


class TreeDSAAttention(Glm5NextDSAAttention):
    def _kpool_select(self, index_q, index_k, head_weights, gate_score, packed_seq_params):
        layout = getattr(packed_seq_params, "glm_tree_layout", None)
        if layout is None:
            return super()._kpool_select(index_q, index_k, head_weights, gate_score, packed_seq_params)
        if parallel_state.get_context_parallel_world_size() != 1:
            raise ValueError("Tree DSA does not support context parallelism")
        # Replay occurrence streams need their own tree mapping; refuse rather
        # than consume them in the segment indexer's different order.
        from megatron.bridge.models.glm5_next.kpool_indexer import _indexer_replay_manager

        replay = _indexer_replay_manager()
        if replay is not None and replay.enabled and replay.stage != "fallthrough":
            raise ValueError("GLM tree SFT does not yet support indexer replay")

        def select(q, k, weights, gates, *, query_start):
            return _select_segment(self, q, k, weights, gates, query_start=query_start)

        return tree_kpool_select(index_q, index_k, head_weights, gate_score, layout, select=select)


def _select_segment(module, q, k, weights, gates, *, query_start):
    # Reuse the provider's pool compression, scoring and token expansion kernels.
    # Unlike the flat selector, Q contains only the newly introduced segment.
    from megatron.bridge.models.glm5.tilelang.tilelang_indexer_fwd import indexer_fwd_interface
    from megatron.bridge.models.glm5_next.kpool_indexer import _select_expand_tail, build_pooled_keys

    cu = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32)
    pooled = build_pooled_keys(k, gates, module.index_kpool_compress_ape, cu, module.index_kpool)
    positions = torch.arange(query_start, query_start + q.shape[0], device=q.device, dtype=torch.int32)
    base = torch.zeros_like(positions)
    ends = (positions + 1) // module.index_kpool
    if pooled.shape[0]:
        logits = indexer_fwd_interface(q, pooled, weights, base, ends, clean_logits=True)
    else:
        logits = torch.full((q.shape[0], 1), -float("inf"), device=q.device, dtype=torch.float32)
    selected = _select_expand_tail(logits, base, base, positions, module.index_topk, module.index_kpool)
    return selected.unsqueeze(1)


class TreeKDAAttention(Glm5NextKDAAttention):
    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        layout = getattr(packed_seq_params, "glm_tree_layout", None)
        if layout is None:
            return super().forward(hidden_states, attention_mask, packed_seq_params=packed_seq_params, **kwargs)
        if kwargs.get("inference_context") is not None or kwargs.get("inference_params") is not None:
            raise ValueError("Tree KDA is a training-only execution path")
        # Preserve the provider's TP/SP projections and parameter ownership.
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        v, _ = self.linear_v(hidden_states)
        beta_logits, _ = self.linear_b(hidden_states)
        f_a, _ = self.linear_f_a(hidden_states)
        forget, _ = self.linear_f_b(f_a)
        g_a, _ = self.linear_g_a(hidden_states)
        norm_gate, _ = self.linear_g_b(g_a)
        g = fused_kda_gate(
            forget.transpose(0, 1).unflatten(-1, (self.local_num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )
        output = tree_kda(
            torch.cat((q, k, v), dim=-1).transpose(0, 1),
            g,
            torch.sigmoid(beta_logits.transpose(0, 1).float()),
            self.conv1d.weight.squeeze(1),
            layout,
            conv_bias=self.conv1d.bias,
        )
        shape = output.shape
        output = self.o_norm(output.reshape(-1, self.head_dim), norm_gate.transpose(0, 1).reshape(-1, self.head_dim))
        output = output.reshape(shape[0], shape[1], -1).transpose(0, 1)
        return self.linear_proj(output)
