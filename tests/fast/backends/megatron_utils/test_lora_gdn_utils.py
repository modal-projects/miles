"""Gated DeltaNet in_proj row layout helpers, without Megatron."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.gdn_utils import gdn_gate_rows


def _module(name, tp=1):
    config = SimpleNamespace(
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4
    )
    # Unsharded rows are q 8, k 8, v 16, z 16, b 4 and a 4
    adapter = SimpleNamespace(linear_out=SimpleNamespace(weight=torch.zeros(56 // tp, 8)))
    return SimpleNamespace(
        base_linear_name=f"decoder.layers.0.self_attention.{name}",
        to_wrap=SimpleNamespace(config=config),
        adapters=[adapter],
    )


@pytest.mark.parametrize(("tp", "expected"), [(1, slice(48, 56)), (2, slice(24, 28))])
def test_gate_rows_are_the_b_and_a_rows_of_each_tp_shard(tp, expected):
    assert gdn_gate_rows(_module("in_proj", tp)) == expected


def test_modules_other_than_gdn_in_proj_have_no_gate_rows():
    assert gdn_gate_rows(_module("out_proj")) is None
