from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["precision"],
    hardware=["hopper"],
)

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
    _quantize_param,
)
from tools.convert_hf_to_fp8 import block_fp8


@pytest.mark.parametrize("shape", [(128, 128), (384, 256), (192, 384)])
def test_live_fp8_export_matches_canonical_checkpoint_encoding(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
) -> None:
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    torch.manual_seed(42)
    weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    live = dict(_quantize_param("model.layers.0.self_attn.q_proj.weight", weight, [128, 128]))
    expected_weight, expected_scale = block_fp8(weight, [128, 128])

    torch.testing.assert_close(
        live["model.layers.0.self_attn.q_proj.weight"].view(torch.uint8),
        expected_weight.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        live["model.layers.0.self_attn.q_proj.weight_scale_inv"],
        expected_scale,
        rtol=0,
        atol=0,
    )
