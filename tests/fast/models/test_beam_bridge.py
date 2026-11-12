import math
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

BeamConfig = pytest.importorskip("sglang.srt.configs.beam").BeamConfig
AutoBridge = pytest.importorskip("megatron.bridge").AutoBridge
pytest.importorskip("miles_plugins.megatron_bridge.beam")


@pytest.mark.parametrize("entry", ["hf_loader", "bridge"])
def test_config_registration_in_fresh_process(tmp_path, entry):
    BeamConfig().save_pretrained(tmp_path)
    if entry == "hf_loader":
        code = """
import sys
from miles.utils.hf_utils.config import load_hf_config
assert load_hf_config(sys.argv[1]).model_type == "beam"
assert "megatron.bridge" not in sys.modules
"""
    else:
        code = """
import sys
import miles_plugins.megatron_bridge
from megatron.bridge import AutoBridge
from miles.utils.hf_utils.config import load_hf_config
config = load_hf_config(sys.argv[1])
provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
assert type(provider).__name__ == "BeamModelProvider"
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"quantization_config": {"quant_method": "fp8"}}, "BF16 HF checkpoint"),
        ({"attention_dropout": 0.1}, "attention_dropout=0"),
        ({"num_experts_per_tok": 4}, "grouped routing"),
    ],
)
def test_bridge_rejects_unsupported_checkpoint(overrides, message):
    config = BeamConfig(**overrides)
    before = deepcopy(config.to_dict())
    with pytest.raises(ValueError, match=message):
        AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
    assert config.to_dict() == before


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tensor_model_parallel_size": 16}, "TP must divide"),
        ({"fp8": "hybrid"}, "requires BF16"),
        ({"bf16": False, "fp16": True}, "requires BF16"),
    ],
)
def test_provider_rejects_unsupported_config_before_build(overrides, message):
    provider = AutoBridge.from_hf_config(BeamConfig()).to_megatron_provider(load_weights=False)
    for name, value in overrides.items():
        setattr(provider, name, value)
    before = vars(provider).copy()
    with pytest.raises(AssertionError, match=message):
        provider.provide()
    assert vars(provider) == before


def _softcap_head(cap):
    from megatron.bridge.models.gemma.modules import extend_instance

    from miles_plugins.megatron_bridge.beam import BeamOutputLayer

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(final_logit_softcapping=cap)

        def forward(self, logits):
            return logits, None

    head = Head()
    extend_instance(head, BeamOutputLayer)
    return head


def test_output_softcap_runs_in_fp32():
    logits = (torch.randn(4, 4096) * 20).bfloat16().requires_grad_()
    capped, bias = _softcap_head(30.0)(logits)
    expected = torch.tanh(logits.detach().double() / 30.0)
    assert capped.dtype == torch.float32 and bias is None
    torch.testing.assert_close(capped.double(), 30.0 * expected, rtol=0, atol=1e-5)
    capped.sum().backward()
    assert logits.grad.dtype == torch.bfloat16
    torch.testing.assert_close(logits.grad, (1 - expected.square()).bfloat16(), rtol=2**-7, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang softcap kernel requires CUDA")
def test_output_softcap_matches_sglang_kernel():
    softcap = pytest.importorskip("sglang.kernels.ops.activation.softcap").softcap_inplace_logits
    logits = (torch.randn(8, 151936, device="cuda") * 20).bfloat16()
    capped, _ = _softcap_head(30.0)(logits)
    expected = logits.float()
    softcap(expected, 30.0)
    torch.testing.assert_close(capped, expected, rtol=0, atol=1e-5)


def _residual_case(layer_number, is_attention, device="cpu"):
    from miles_plugins.models.beam import BeamResidual

    generator = torch.Generator(device="cpu").manual_seed(layer_number * 2 + is_attention)
    branch = (torch.randn(64, 512, generator=generator) * 3).bfloat16().to(device)
    residual = (torch.randn(64, 512, generator=generator) * 8).bfloat16().to(device)
    depth = 2 * layer_number + (0 if is_attention else 1)
    scales = (1.0 / math.sqrt(depth), math.sqrt(depth - 1) / math.sqrt(depth))
    return BeamResidual(layer_number, is_attention), branch, residual, scales


@pytest.mark.parametrize("layer_number", [1, 2, 6, 52])
@pytest.mark.parametrize("is_attention", [True, False])
def test_residual_matches_sglang_depth_scaled_residual(layer_number, is_attention):
    depth_scaled_residual = pytest.importorskip("sglang.srt.models.beam").depth_scaled_residual
    module, branch, residual, scales = _residual_case(layer_number, is_attention)
    actual = module(training=False, fused=False)((branch, None), residual, 0.0)
    assert torch.equal(actual, depth_scaled_residual(branch, residual, scales))


def test_residual_gradients_are_the_depth_scales():
    module, branch, residual, scales = _residual_case(3, False)
    branch.requires_grad_()
    residual.requires_grad_()
    module(training=True, fused=False)((branch, None), residual, 0.0).float().sum().backward()
    assert torch.equal(branch.grad, torch.full_like(branch, scales[0]))
    assert torch.equal(residual.grad, torch.full_like(residual, scales[1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang fused sandwich kernel requires CUDA")
@pytest.mark.parametrize("is_attention", [True, False])
def test_residual_matches_sglang_fused_sandwich_kernel(is_attention):
    fused = pytest.importorskip("sglang.kernels.ops.layernorm.beam_fused").fused_sandwich_residual_rmsnorm
    module, branch, residual, scales = _residual_case(5, is_attention, device="cuda")
    actual = module(training=False, fused=False)((branch, None), residual, 0.0)
    _, combined = fused(branch, residual, None, None, 1e-6, *scales)
    assert torch.equal(actual, combined)


def test_lora_targets_cover_every_beam_projection(tmp_path):
    from sglang.test.beam_fixture import make_beam_fixture

    from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
    from miles.utils.lora.utils import matches_lora_target

    config, state = make_beam_fixture(tmp_path)
    modules = [name.removesuffix(".weight") for name in state]
    targets = resolve_hf_lora_targets(config.to_dict())
    adapted = {module for module in modules if any(matches_lora_target(module, target) for target in targets)}
    # Every linear projection, including the attention output gate and each routed expert; not the router.
    assert adapted == {module for module in modules if module.endswith("_proj")}


def _qkvg_rows(groups=2, heads=6, head_dim=4, rank=3):
    names = ("q_proj", "gate_proj", "k_proj", "v_proj")
    sizes = (heads * head_dim, heads * head_dim, groups * head_dim, groups * head_dim)
    rows = {name: torch.randn(size, rank) for name, size in zip(names, sizes, strict=True)}
    # linear_qkv stores each query group's rows as [q, g, k, v].
    fused = torch.cat([rows[name].view(groups, -1, rank) for name in names], dim=1).reshape(-1, rank)
    config = SimpleNamespace(num_query_groups=groups, num_attention_heads=heads, kv_channels=head_dim)
    return rows, fused, config


def test_lora_export_splits_the_fused_qkvg_adapter():
    from miles_plugins.megatron_bridge.beam import BeamBridge

    rows, fused, config = _qkvg_rows()
    names = [f"model.layers.0.self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "gate_proj")]
    slices = BeamBridge.__new__(BeamBridge)._get_fused_adapter_linear_out_slices(
        [SimpleNamespace(config=config)], names, fused
    )
    assert slices.keys() == set(names)
    for name, expected in rows.items():
        assert torch.equal(slices[f"model.layers.0.self_attn.{name}.weight"], expected)
