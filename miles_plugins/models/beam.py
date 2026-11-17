import copy
import dataclasses
import math
from functools import partial

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention, TENorm
from megatron.core.jit import jit_fuser
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


class BeamGateBias(MegatronModule):
    def __init__(self, config, num_heads, tp_group):
        super().__init__(config=config)
        self.tp_group = tp_group
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.bias = torch.nn.Parameter(torch.zeros(num_heads, dtype=config.params_dtype, device=device))
        set_tensor_model_parallel_attributes(self.bias, is_parallel=True, dim=0, stride=1)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        dp_cp_group = (metadata or {}).get("dp_cp_group")
        if dp_cp_group is None:
            dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            {"bias": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=dp_cp_group,
        )


class BeamSelfAttention(SelfAttention):
    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        self.output_gate_bias = BeamGateBias(config, self.num_attention_heads_per_partition, self.pg_collection.tp)
        # Preserve optimizer shard order: gate bias precedes Q/K normalization.
        for name in ("q_layernorm", "k_layernorm"):
            self.add_module(name, self._modules.pop(name))

    @jit_fuser
    def _apply_output_gate(self, x, gate):
        bias = self.output_gate_bias.bias
        gate = gate.contiguous().view(*x.shape).view(*x.shape[:-1], bias.numel(), -1).float()
        gate = gate + bias.float().view(*([1] * (x.dim() - 1)), bias.numel(), 1)
        return (x * torch.sigmoid(gate).view(*x.shape)).to(x.dtype)


def _ssmax_positions(query, packed_seq_params, cp_group):
    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = cp_group.rank() if cp_size > 1 else 0
    tokens = torch.arange(query.shape[0], device=query.device, dtype=torch.int64)
    if packed_seq_params is None:
        local_pos = tokens
        segment = query.shape[0] // 2
    else:
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        local_cu = cu.to(torch.int64) // cp_size
        seq_idx = (torch.searchsorted(local_cu, tokens, right=True) - 1).clamp(0, cu.numel() - 2)
        local_pos = tokens - local_cu[seq_idx]
        segment = (local_cu[1:] - local_cu[:-1])[seq_idx] // 2
    if cp_size > 1:
        local_pos = torch.where(
            local_pos >= segment,
            (2 * cp_size - 1 - cp_rank) * segment + local_pos - segment,
            cp_rank * segment + local_pos,
        )
    return local_pos.float()


class BeamDotProductAttention(TEDotProductAttention):
    def __init__(self, config, layer_number, *args, **kwargs):
        super().__init__(config, layer_number, *args, **kwargs)
        self.use_ssmax = bool(config.no_rope_freq[layer_number - 1])

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
    ):
        if self.use_ssmax:
            cp_group = self.cp_group
            if packed_seq_params is not None and packed_seq_params.cp_group is not None:
                cp_group = packed_seq_params.cp_group
            positions = _ssmax_positions(query, packed_seq_params, cp_group)
            scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048))
            query = query * scale.view(query.shape[0], *([1] * (query.dim() - 1))).to(query.dtype)
        return super().forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            num_splits=num_splits,
        )


class BeamTopKRouter(TopKRouter):
    """Keep the gate weight in FP32, as the checkpoint stores it and SGLang serves it.

    TopKRouter casts its weight to params_dtype; moe_router_dtype only sets the compute dtype.
    """

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data = self.weight.data.float()
        mark_keep_in_fp32(self.weight)


class BeamResidual:
    def __init__(self, layer_number, is_attention):
        index = 2 * layer_number + (0 if is_attention else 1)
        self.output_scale = 1.0 / math.sqrt(index)
        self.residual_scale = math.sqrt((index - 1) / index)

    def __call__(self, training, fused):
        return partial(self._add, training=training)

    def _add(self, output_with_bias, residual, dropout, *, training):
        output, bias = output_with_bias
        assert bias is None, "Beam residuals require bias-free projections"
        if training and dropout:
            output = F.dropout(output, p=dropout, training=True)
        # Round both scaled operands to the activation dtype before adding, as Reflection SGLang's
        # depth_scaled_residual and fused sandwich kernel do; torch.add(..., alpha=) skips one rounding.
        return residual * self.residual_scale + output.to(residual.dtype) * self.output_scale


def _with_fp32_router(mlp):
    # Megatron builds MLP specs as partial(module, submodules=...); only MoE layers carry a router.
    submodules = mlp.keywords.get("submodules")
    if not isinstance(submodules, MoESubmodules):
        return mlp
    submodules = dataclasses.replace(submodules, router=BeamTopKRouter)
    return partial(mlp.func, *mlp.args, **{**mlp.keywords, "submodules": submodules})


def beam_layer_spec(config, vp_stage=None):
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    layout = config.pipeline_model_parallel_layout
    if layout is None:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        layer_ids = range(offset, offset + len(block.layer_specs))
    else:
        layer_ids = layout.get_layer_id_list(layer_type=LayerType.decoder, vp_stage=vp_stage)
    block.layer_specs = [copy.deepcopy(spec) for spec in block.layer_specs]
    for layer_id, spec in zip(layer_ids, block.layer_specs, strict=True):
        spec.submodules.self_attention.module = BeamSelfAttention
        spec.submodules.self_attention.submodules.core_attention = BeamDotProductAttention
        spec.submodules.post_self_attn_layernorm = TENorm
        spec.submodules.post_mlp_layernorm = TENorm
        spec.submodules.self_attn_bda = ModuleSpec(
            BeamResidual, params={"layer_number": layer_id + 1, "is_attention": True}
        )
        spec.submodules.mlp_bda = ModuleSpec(
            BeamResidual, params={"layer_number": layer_id + 1, "is_attention": False}
        )
        spec.submodules.mlp = _with_fp32_router(spec.submodules.mlp)
    return block
