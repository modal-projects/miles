"""CUDA Megatron adapter over Miles' native GLM KDA and DSA implementations."""

from copy import deepcopy

import torch
from megatron.core import parallel_state

from miles.backends.training_utils.prefix_tree.glm53.dsa import tree_kpool_select
from miles.backends.training_utils.prefix_tree.glm53.kda import tree_kda
from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.megatron import validate_megatron_tree_config
from miles.kernels.attention.dsa.kpool import build_pooled_keys, pool_boundaries
from miles_plugins.models.glm5_next.dsa import Glm5NextDSAAttention
from miles_plugins.models.glm5_next.glm5_next import get_glm5_next_spec
from miles_plugins.models.glm5_next.kda import Glm5NextKDAAttention, fused_kda_gate
from miles_plugins.models.glm5_next.ops.kpool_indexer import kpool_select_topk


def get_glm5_next_tree_spec(args, config, vp_stage=None):
    if config.context_parallel_size != 1:
        raise ValueError("Native GLM shared prefixes require context parallel size 1")
    validate_megatron_tree_config(config)
    spec = deepcopy(get_glm5_next_spec(args, config, vp_stage=vp_stage))
    replacements = {Glm5NextDSAAttention: TreeDSAAttention, Glm5NextKDAAttention: TreeKDAAttention}
    for layer in spec.layer_specs:
        attention = deepcopy(layer.submodules.self_attention)
        layer.submodules.self_attention = attention
        if attention.module not in replacements:
            raise ValueError(f"Unregistered native GLM attention: {attention.module}")
        attention.module = replacements[attention.module]
    config.prefix_tree_layout_type = GlmTreeLayout
    return spec


class TreeDSAAttention(Glm5NextDSAAttention):
    def _kpool_select(self, index_q, index_k, head_weights, gate_score, packed_seq_params):
        layout = getattr(
            packed_seq_params, "glm_tree_layout", None
        )  # config-access-exempt: ordinary packed batches have no tree metadata
        if layout is None:
            return super()._kpool_select(index_q, index_k, head_weights, gate_score, packed_seq_params)
        if parallel_state.get_context_parallel_world_size() != 1:
            raise ValueError("Tree DSA does not support context parallelism")

        def select(q, k, weights, gates, *, query_start):
            cu = torch.tensor([0, len(k)], device=k.device, dtype=torch.int32)
            return kpool_select_topk(
                index_q=q,
                pooled_k=build_pooled_keys(k, gates, self.index_kpool_compress_ape, cu, self.index_kpool),
                head_weights=weights,
                cu_seqlens=cu,
                pool_cu_seqlens=pool_boundaries(cu, self.index_kpool),
                index_topk=self.index_topk,
                kpool=self.index_kpool,
                query_token_ids=torch.arange(query_start, len(k), device=q.device),
                topk_backend=self.topk_backend,
            )

        return tree_kpool_select(index_q, index_k, head_weights, gate_score, layout, select=select)


class TreeKDAAttention(Glm5NextKDAAttention):
    def hf_forward(self, hidden_states, packed_seq_params):
        layout = getattr(
            packed_seq_params, "glm_tree_layout", None
        )  # config-access-exempt: ordinary packed batches have no tree metadata
        if layout is None:
            return super().hf_forward(hidden_states, packed_seq_params)
        kda = self.kda
        mixed = torch.cat((kda.q_proj(hidden_states), kda.k_proj(hidden_states), kda.v_proj(hidden_states)), dim=-1)
        beta = torch.sigmoid(kda.b_proj(hidden_states).float())
        forget = kda.f_b_proj(kda.f_a_proj(hidden_states))
        decay = fused_kda_gate(
            forget.unflatten(-1, (kda.num_heads, kda.head_dim)),
            kda.A_log,
            kda.dt_bias,
            lower_bound=kda.gate_lower_bound,
        )
        conv_weight = kda.conv1d.weight
        if conv_weight.ndim == 3:
            conv_weight = conv_weight.squeeze(1)
        output = tree_kda(mixed, decay, beta, conv_weight, layout, conv_bias=kda.conv1d.bias)
        norm_gate = kda.g_b_proj(kda.g_a_proj(hidden_states))
        shape = output.shape
        output = kda.o_norm(output.reshape(-1, kda.head_dim), norm_gate.reshape(-1, kda.head_dim))
        return kda.o_proj(output.reshape(shape[0], shape[1], -1))
