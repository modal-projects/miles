"""Multi-LoRA for DeepSeek-V4.1 on the Miles model definition (``--megatron-to-hf-mode raw``).

Adapters come from Megatron-Bridge's multi-LoRA layers, but target selection, base-weight
loading and HF export use the V4.1 checkpoint layout directly instead of AutoBridge.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_HF_TO_MEGATRON = {
    "layers.*.attn.wq_a": "decoder.layers.*.self_attention.linear_q_down_proj",
    "layers.*.attn.wkv": "decoder.layers.*.self_attention.linear_kv_proj",
    "layers.*.attn.wq_b": "decoder.layers.*.self_attention.linear_q_up_proj",
    "layers.*.attn.wo_b": "decoder.layers.*.self_attention.linear_proj",
    "layers.*.ffn.shared_experts.w1": "decoder.layers.*.mlp.shared_experts.linear_fc1",
    "layers.*.ffn.shared_experts.w3": "decoder.layers.*.mlp.shared_experts.linear_fc1",
    "layers.*.ffn.shared_experts.w2": "decoder.layers.*.mlp.shared_experts.linear_fc2",
}
_FUSED_TARGETS = {
    "decoder.layers.*.mlp.shared_experts.linear_fc1": (
        "layers.*.ffn.shared_experts.w1",
        "layers.*.ffn.shared_experts.w3",
    ),
}
_ROUTED_EXPERT_TARGETS = (
    "layers.*.ffn.experts.*.w1",
    "layers.*.ffn.experts.*.w2",
    "layers.*.ffn.experts.*.w3",
)

# (Megatron attribute on the attention module, HF module under ``layers.N.``)
_ATTENTION_ADAPTERS = (
    ("linear_q_down_proj", "attn.wq_a"),
    ("linear_kv_proj", "attn.wkv"),
    ("linear_q_up_proj", "attn.wq_b"),
    ("linear_proj", "attn.wo_b"),
)

SUPPORTED_TARGETS = tuple(_HF_TO_MEGATRON)


def resolve_deepseek_v41_adapter_targets(hf_targets: Sequence[str]) -> list[str]:
    targets = list(dict.fromkeys(hf_targets))
    routed = [target for target in targets if target in _ROUTED_EXPERT_TARGETS]
    if routed:
        raise NotImplementedError(
            f"DeepSeek-V4.1 multi-LoRA does not yet train routed experts ({routed}); "
            f"select from {list(SUPPORTED_TARGETS)}"
        )
    unsupported = [target for target in targets if target not in _HF_TO_MEGATRON]
    if unsupported:
        raise NotImplementedError(
            f"DeepSeek-V4.1 multi-LoRA does not support {unsupported}; select from {list(SUPPORTED_TARGETS)}"
        )
    for megatron_module, fused in _FUSED_TARGETS.items():
        selected = [target for target in fused if target in targets]
        if selected and len(selected) != len(fused):
            raise ValueError(
                f"{megatron_module} fuses {list(fused)} into one adapter; select all of them or none "
                f"(got {selected})"
            )
    return targets


def deepseek_v41_megatron_targets(hf_targets: Sequence[str]) -> list[str]:
    targets = resolve_deepseek_v41_adapter_targets(hf_targets)
    return list(dict.fromkeys(_HF_TO_MEGATRON[target] for target in targets))


def apply_deepseek_v41_multi_lora(model, args):
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear
    from megatron.core.transformer.moe.router import TopKRouter

    from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance, use_tp_exact_multi_lora_forward

    targets = deepseek_v41_megatron_targets(args.hf_lora_targets)
    lora = create_multi_lora_instance(args, target_modules=targets)
    model = lora(model, training=True)
    use_tp_exact_multi_lora_forward(model)
    # TE drops the gradient through a gathered layernorm output, which would cut the adapter's input gradient.
    gathered = [
        module.base_linear_name
        for module in model.modules()
        if isinstance(module, MultiLoRALinear) and getattr(module.to_wrap, "return_layernorm_output_gathered", False)
    ]
    assert not gathered, f"DeepSeek-V4.1 multi-LoRA does not support layernorm-fused targets: {gathered}"

    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.frozen_expert_bias = True

    for layer in model.decoder.layers:
        attention = layer.self_attention
        for attribute, _ in _ATTENTION_ADAPTERS:
            wrapper = getattr(attention, attribute)
            if not isinstance(wrapper, MultiLoRALinear):
                continue
            # V4.1 attention gathers its sequence-parallel input before every projection and
            # scatters only after linear_proj, so its adapters see and return full sequences.
            wrapper.disable_sequence_parallel_comm = True
            if wrapper.replicate_adapter:
                # Full-sequence inputs give every TP rank the complete gradient; do not TP-sum it.
                for param in wrapper.adapters.parameters():
                    param.sequence_parallel = False

    _check_layout(model, targets)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(
        "DeepSeek-V4.1 multi-LoRA applied: slots=%d max_rank=%d targets=%s trainable=%d total=%d",
        args.multi_lora_n_adapters,
        args.lora_rank,
        targets,
        trainable,
        total,
    )
    return model


def wrap_model_provider_with_deepseek_v41_multi_lora(provider_func, args):
    def wrapped(*provider_args, **provider_kwargs):
        return apply_deepseek_v41_multi_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


def _adapter_modules(layer) -> list[tuple[str, nn.Module | None]]:
    """``(HF module suffix, module)`` for every phase-1 target position in a decoder layer."""
    attention = layer.self_attention
    modules = [(hf_name, getattr(attention, attribute, None)) for attribute, hf_name in _ATTENTION_ADAPTERS]
    shared = getattr(layer.mlp, "shared_experts", None)
    modules.append(("ffn.shared_experts.fc1", getattr(shared, "linear_fc1", None)))
    modules.append(("ffn.shared_experts.w2", getattr(shared, "linear_fc2", None)))
    return modules


def _megatron_template(hf_suffix: str) -> str:
    if hf_suffix == "ffn.shared_experts.fc1":
        return "decoder.layers.*.mlp.shared_experts.linear_fc1"
    return _HF_TO_MEGATRON[f"layers.*.{hf_suffix}"]


def _check_layout(model, megatron_targets: Sequence[str]) -> None:
    missing = [
        f"layer {layer.layer_number - 1}: {template}"
        for layer in model.decoder.layers
        for hf_suffix, module in _adapter_modules(layer)
        if (template := _megatron_template(hf_suffix)) in megatron_targets and not hasattr(module, "adapters")
    ]
    if missing:
        raise RuntimeError(f"DeepSeek-V4.1 multi-LoRA did not wrap every target: {missing[:5]}")


def _unwrap(model_chunks):
    for chunk in model_chunks:
        while hasattr(chunk, "module"):
            chunk = chunk.module
        yield chunk


def _gather_tp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    from megatron.core import parallel_state

    world_size = parallel_state.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(
        parts, tensor.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
    )
    return torch.cat(parts, dim=dim)


def _full_weight(param: torch.Tensor, gather) -> torch.Tensor:
    if not getattr(param, "tensor_model_parallel", False):
        return param.detach()
    return gather(param.detach(), param.partition_dim)


def _export_module(hf_prefix: str, module, slot: int, gather) -> list[tuple[str, torch.Tensor]]:
    adapter = module.adapters[slot]
    lora_a = _full_weight(adapter.linear_in.weight, gather)
    lora_b_param = adapter.linear_out.weight
    if not hf_prefix.endswith("shared_experts.fc1"):
        return [
            (f"{hf_prefix}.lora_A.weight", lora_a),
            (f"{hf_prefix}.lora_B.weight", _full_weight(lora_b_param, gather)),
        ]
    # Each TP shard of linear_fc1 holds [w1 rows; w3 rows]; split before gathering.
    assert getattr(lora_b_param, "partition_dim", 0) == 0
    w1_b, w3_b = lora_b_param.detach().chunk(2, dim=0)
    if getattr(lora_b_param, "tensor_model_parallel", False):
        w1_b, w3_b = gather(w1_b, 0), gather(w3_b, 0)
    prefix = hf_prefix.removesuffix(".fc1")
    return [
        (f"{prefix}.w1.lora_A.weight", lora_a),
        (f"{prefix}.w1.lora_B.weight", w1_b),
        (f"{prefix}.w3.lora_A.weight", lora_a),
        (f"{prefix}.w3.lora_B.weight", w3_b),
    ]


def export_deepseek_v41_multi_lora(model_chunks, adapter, *, gather=_gather_tp) -> list[tuple[str, torch.Tensor]]:
    """One slot as HF PEFT tensors (``layers.N.attn.wq_a.lora_A.weight``, ...), TP-gathered and
    trimmed to the slot's rank. Collective over the TP group."""
    from miles.backends.megatron_utils.lora.slots import slice_lora_to_rank

    named = []
    for model in _unwrap(model_chunks):
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1
            for hf_suffix, module in _adapter_modules(layer):
                if not hasattr(module, "adapters"):
                    continue
                named.extend(_export_module(f"layers.{layer_id}.{hf_suffix}", module, adapter.slot, gather))
    if not named:
        raise RuntimeError("DeepSeek-V4.1 multi-LoRA export found no adapters")
    return [(name, slice_lora_to_rank(name, tensor, adapter.rank).contiguous()) for name, tensor in named]
