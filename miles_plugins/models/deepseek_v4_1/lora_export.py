"""Export one DeepSeek-V4.1 multi-LoRA slot as HF PEFT tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
from megatron.core import parallel_state

from miles.backends.megatron_utils.lora.slots import slice_lora_to_rank
from miles_plugins.models.deepseek_v4_1.lora_targets import HF_TO_MEGATRON

# (Megatron attribute on the attention module, HF module under ``layers.N.``)
ATTENTION_ADAPTERS = (
    ("linear_q_down_proj", "attn.wq_a"),
    ("linear_kv_proj", "attn.wkv"),
    ("linear_q_up_proj", "attn.wq_b"),
    ("linear_proj", "attn.wo_b"),
)


def adapter_modules(layer) -> list[tuple[str, nn.Module | None]]:
    """``(HF module suffix, module)`` for every supported target position in a decoder layer."""
    attention = layer.self_attention
    modules = [(hf_name, getattr(attention, attribute, None)) for attribute, hf_name in ATTENTION_ADAPTERS]
    shared = getattr(layer.mlp, "shared_experts", None)
    modules.append(("ffn.shared_experts.fc1", getattr(shared, "linear_fc1", None)))
    modules.append(("ffn.shared_experts.w2", getattr(shared, "linear_fc2", None)))
    return modules


def megatron_template(hf_suffix: str) -> str:
    if hf_suffix == "ffn.shared_experts.fc1":
        return "decoder.layers.*.mlp.shared_experts.linear_fc1"
    return HF_TO_MEGATRON[f"layers.*.{hf_suffix}"]


def _unwrap(model_chunks):
    for chunk in model_chunks:
        while hasattr(chunk, "module"):
            chunk = chunk.module
        yield chunk


def _gather_tp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(parts, tensor.contiguous(), group=parallel_state.get_tensor_model_parallel_group())
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
    named = []
    for model in _unwrap(model_chunks):
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1
            for hf_suffix, module in adapter_modules(layer):
                if not hasattr(module, "adapters"):
                    continue
                named.extend(_export_module(f"layers.{layer_id}.{hf_suffix}", module, adapter.slot, gather))
    if not named:
        raise RuntimeError("DeepSeek-V4.1 multi-LoRA export found no adapters")
    return [(name, slice_lora_to_rank(name, tensor, adapter.rank).contiguous()) for name, tensor in named]
