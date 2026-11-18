"""Native adapter attachment retains deltas after installing base arithmetic controls."""

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native import lora as lifecycle
from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.modules.linear import attach_adapter_forward
from miles.backends.megatron_utils.lora.native.modules.moe import LoRAGroupedFC1
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.args.configs.lora import LoraConfig


@pytest.mark.parametrize("policy", ["native", "rowwise"])
def test_real_native_attach_keeps_expert_delta_and_every_gradient(monkeypatch, policy):
    class GroupedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_gemms, self.use_bias, self.tp_size = 2, False, 1
            for index in range(2):
                self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(3, 4, dtype=torch.float64)))

        def forward(self, value, splits):
            return (
                torch.cat([F.linear(part, getattr(self, f"weight{i}")) for i, part in enumerate(value.split(splits))]),
                None,
            )

    class UnusedNorm(nn.Module):
        pass

    te = ModuleType("transformer_engine.pytorch")
    te.GroupedLinear, te.RMSNorm, te.LayerNormLinear = GroupedLinear, UnusedNorm, UnusedNorm
    parent = ModuleType("transformer_engine")
    parent.pytorch = te
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, te.__name__, te)
    model = nn.Module()
    model.config = SimpleNamespace(sequence_parallel=False, hidden_size=4)
    model.decoder = nn.Module()
    layer = nn.Module()
    layer.layer_number = 1
    layer.self_attention = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    host = layer.mlp.experts.linear_fc1 = GroupedLinear()
    model.decoder.layers = nn.ModuleList([layer])
    context = AttachContext(
        LoRAConfig(2, 4, 0, None, kernel_policy=policy, experts_shared_outer_loras=False), model.config, 1, 0, None
    )

    def attach(mlp, _layer, ctx):
        adapter = LoRAGroupedFC1(
            hf_prefix="model.layers.0.mlp.experts.",
            hf_leaves=("up_proj",),
            reference=host.weight0,
            context=ctx,
            num_local_experts=2,
            moe_intermediate=3,
            width=4,
            is_ep=False,
        )
        mlp.experts.lora_fc1_adapter = adapter
        attach_adapter_forward(host, adapter)
        return 1

    spec = SimpleNamespace(
        name="test",
        validate=lambda ctx: None,
        has_mamba_layers=False,
        has_linear_attention_layers=False,
        attention=SimpleNamespace(applies_to=lambda attention: True, attach=lambda *args: 0),
        linear_attention=None,
        mlp=SimpleNamespace(attach=lambda *args: 0),
        experts=SimpleNamespace(attach=attach),
        lm_head=None,
    )
    monkeypatch.setattr(lifecycle, "resolve_checkpoint_spec", lambda path: ("test", spec))
    monkeypatch.setattr(lifecycle, "_resolve_attach_context", lambda *args: context)
    lifecycle.apply_native_lora(model, SimpleNamespace(hf_checkpoint="/unused"))
    adapter = layer.mlp.experts.lora_fc1_adapter
    with torch.no_grad():
        adapter.w1_B.normal_()
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    actual, _ = host(x, [2, 3])
    expected = torch.cat(
        [
            F.linear(part, getattr(host, f"weight{i}"))
            + 2 * F.linear(F.linear(part, adapter.w1_A[i]), adapter.w1_B[i])
            for i, part in enumerate(x.split([2, 3]))
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = (x, *adapter.parameters())
    for got, reference in zip(
        torch.autograd.grad(actual.square().sum(), parameters),
        torch.autograd.grad(expected.square().sum(), parameters),
        strict=True,
    ):
        torch.testing.assert_close(got, reference, atol=1e-11, rtol=1e-11)
        assert torch.count_nonzero(got)


def test_kernel_policy_cli_defaults_native_and_rejects_unknown_values():
    parser = argparse.ArgumentParser()
    LoraConfig.add_arguments(parser)
    assert parser.parse_args([]).lora_kernel_policy == "native"
    assert parser.parse_args(["--lora-kernel-policy", "rowwise"]).lora_kernel_policy == "rowwise"
    with pytest.raises(SystemExit):
        parser.parse_args(["--lora-kernel-policy", "unknown"])
