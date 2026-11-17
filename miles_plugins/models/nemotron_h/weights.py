"""Nemotron-H HF names and Mamba component packing, shared by load and export."""

import re

import torch

_DIRECT = {
    "embedding.word_embeddings.weight": "backbone.embeddings.weight",
    "output_layer.weight": "lm_head.weight",
    "decoder.final_norm.weight": "backbone.norm_f.weight",
}
_LAYER = {
    "self_attention.linear_qkv.layer_norm_weight": "norm.weight",
    "self_attention.linear_proj.weight": "mixer.o_proj.weight",
    "mlp.linear_fc1.layer_norm_weight": "norm.weight",
    "mlp.linear_fc1.weight": "mixer.up_proj.weight",
    "mlp.linear_fc2.weight": "mixer.down_proj.weight",
    "mlp.shared_experts.linear_fc1.weight": "mixer.shared_experts.up_proj.weight",
    "mlp.shared_experts.linear_fc2.weight": "mixer.shared_experts.down_proj.weight",
    "mlp.router.weight": "mixer.gate.weight",
    "mlp.router.expert_bias": "mixer.gate.e_score_correction_bias",
    "pre_mlp_layernorm.weight": "norm.weight",
    "input_layernorm.weight": "norm.weight",
    "norm.weight": "norm.weight",
    "mixer.in_proj.layer_norm_weight": "norm.weight",
    "mixer.conv1d_weight": "mixer.conv1d.weight",
    "mixer.conv1d_bias": "mixer.conv1d.bias",
}


def hf_weight_names(name: str) -> list[str]:
    if name in _DIRECT:
        return [_DIRECT[name]]
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unsupported Nemotron-H parameter: {name}")
    layer, rest = match.groups()
    prefix = f"backbone.layers.{layer}."
    if rest in _LAYER:
        return [prefix + _LAYER[rest]]
    if rest == "self_attention.linear_qkv.weight":
        return [f"{prefix}mixer.{leaf}_proj.weight" for leaf in ("q", "k", "v")]
    expert = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if expert:
        fc, index = expert.groups()
        leaf = "up" if fc == "1" else "down"
        return [f"{prefix}mixer.experts.{index}.{leaf}_proj.weight"]
    if rest in {
        f"mixer.{leaf}" for leaf in ("A_log", "D", "dt_bias", "in_proj.weight", "out_proj.weight", "norm.weight")
    }:
        return [prefix + rest]
    raise ValueError(f"Unsupported Nemotron-H parameter: {name}")


def _component_sizes(config, name: str) -> tuple[int, ...] | None:
    inner = config.mamba_num_heads * config.mamba_head_dim
    state = config.n_groups * config.ssm_state_size
    if name.endswith("mixer.in_proj.weight"):
        return inner, inner, state, state, config.mamba_num_heads
    if name.endswith(("mixer.conv1d_weight", "mixer.conv1d_bias")):
        return inner, state, state
    return None


def pack_mamba_components(config, name: str, weight: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Arrange HF component-major rows into the contiguous TP shards Megatron consumes."""
    sizes = _component_sizes(config, name)
    if sizes is None:
        return weight
    assert all(size % tp_size == 0 for size in sizes), "Mamba components must divide tensor parallel size"
    shards = [part.reshape(tp_size, -1, *weight.shape[1:]) for part in weight.split(sizes)]
    return torch.cat(shards, dim=1).reshape_as(weight)


def unpack_mamba_components(config, name: str, weight: torch.Tensor, tp_size: int) -> torch.Tensor:
    sizes = _component_sizes(config, name)
    if sizes is None:
        return weight
    shards = weight.reshape(tp_size, -1, *weight.shape[1:])
    parts = shards.split([size // tp_size for size in sizes], dim=1)
    return torch.cat([part.reshape(-1, *weight.shape[1:]) for part in parts])
