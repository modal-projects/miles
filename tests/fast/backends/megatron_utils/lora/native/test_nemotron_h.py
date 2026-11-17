"""Nemotron Nano's ungated adapters and Mamba checkpoint component layout."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS, resolve_adapter_targets
from miles_plugins.models.nemotron_h.weights import hf_weight_names, pack_mamba_components, unpack_mamba_components


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "suffix, sizes, tail",
    [
        ("in_proj.weight", (16, 16, 8, 8, 8), (3,)),
        ("conv1d_weight", (16, 8, 8), (1, 4)),
        ("conv1d_bias", (16, 8, 8), ()),
    ],
)
def test_mamba_checkpoint_components_roundtrip_across_tensor_parallel_ranks(tp_size, suffix, sizes, tail):
    config = SimpleNamespace(mamba_num_heads=8, mamba_head_dim=2, n_groups=4, ssm_state_size=2)
    shape = (sum(sizes), *tail)
    count = 1
    for dim in shape:
        count *= dim
    hf = torch.arange(count).reshape(shape)
    name = f"decoder.layers.0.mixer.{suffix}"
    packed = pack_mamba_components(config, name, hf, tp_size)
    torch.testing.assert_close(unpack_mamba_components(config, name, packed, tp_size), hf)
    for rank, shard in enumerate(packed.chunk(tp_size)):
        expected = torch.cat([part.chunk(tp_size)[rank] for part in hf.split(sizes)])
        torch.testing.assert_close(shard, expected)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("decoder.layers.2.mlp.experts.linear_fc1.weight63", "backbone.layers.2.mixer.experts.63.up_proj.weight"),
        ("decoder.layers.2.mlp.experts.linear_fc2.weight63", "backbone.layers.2.mixer.experts.63.down_proj.weight"),
        ("decoder.layers.0.mixer.conv1d_weight", "backbone.layers.0.mixer.conv1d.weight"),
        ("decoder.layers.0.mixer.in_proj.layer_norm_weight", "backbone.layers.0.norm.weight"),
        ("decoder.final_norm.weight", "backbone.norm_f.weight"),
    ],
)
def test_nemotron_hf_names_cover_mamba_and_ungated_experts(name, expected):
    assert hf_weight_names(name) == [expected]


@pytest.mark.parametrize("shared_outer", [False, True])
def test_nemotron_targets_include_routed_and_shared_up_down_without_a_gate(shared_outer):
    prefix = "backbone.layers.*.mixer."
    targets = [
        f"{prefix}{leaf}"
        for leaf in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "experts.*.up_proj",
            "experts.*.down_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        )
    ]
    actual = resolve_adapter_targets(
        {"model_type": "nemotron_h"},
        targets,
        hf_modules=[],
        lora_type="canonical_lora",
        experts_shared_outer_loras=shared_outer,
    )
    assert set(actual) == {target.replace(".experts.*.", ".experts.") for target in targets}
    assert not MODEL_SPECS["nemotron_h"].attaches(f"{prefix}shared_experts.gate_proj")


@pytest.mark.parametrize(
    "model_type, block, leaves",
    [
        ("qwen3_moe", "mlp.experts.*", ("gate_proj", "up_proj", "down_proj")),
        ("deepseek_v3", "mlp.experts.*", ("gate_proj", "up_proj", "down_proj")),
        ("glm_moe_dsa", "mlp.experts.*", ("gate_proj", "up_proj", "down_proj")),
        ("kimi_k25", "mlp.experts.*", ("gate_proj", "up_proj", "down_proj")),
        ("glm5_next", "mlp.experts", ("gate_up_proj", "down_proj")),
        ("qwen3_5_moe", "mlp.experts", ("gate_up_proj", "down_proj")),
        ("deepseek_v41", "ffn.experts.*", ("w1", "w2", "w3")),
    ],
)
def test_moe_architectures_accept_their_routed_expert_targets(model_type, block, leaves):
    targets = [f"model.layers.*.{block}.{leaf}" for leaf in leaves]
    resolve_adapter_targets(
        {"model_type": model_type},
        targets,
        hf_modules=[],
        lora_type="canonical_lora",
        experts_shared_outer_loras=False,
    )
