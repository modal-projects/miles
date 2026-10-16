"""DeepSeek-V4.1 LoRA targets: HF checkpoint names and the Megatron modules they map to."""

from __future__ import annotations

from collections.abc import Sequence

HF_TO_MEGATRON = {
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

SUPPORTED_TARGETS = tuple(HF_TO_MEGATRON)


def resolve_deepseek_v41_adapter_targets(hf_targets: Sequence[str]) -> list[str]:
    targets = list(dict.fromkeys(hf_targets))
    routed = [target for target in targets if target in _ROUTED_EXPERT_TARGETS]
    if routed:
        raise NotImplementedError(
            f"DeepSeek-V4.1 multi-LoRA does not train routed experts ({routed}); "
            f"select from {list(SUPPORTED_TARGETS)}"
        )
    unsupported = [target for target in targets if target not in HF_TO_MEGATRON]
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
    return list(dict.fromkeys(HF_TO_MEGATRON[target] for target in targets))
