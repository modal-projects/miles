from unittest.mock import MagicMock, patch

import torch

from tools.convert_hf_to_fp8 import ConversionResult, process_file


def test_offline_fp8_keeps_unsupported_weights_in_high_precision() -> None:
    weights = {
        "model.layers.0.proj.weight": torch.ones((256, 128)),
        "model.layers.0.linear_attn.in_proj_ba.weight": torch.ones((32, 2048)),
        "model.layers.0.fused_experts.weight": torch.ones((2, 2, 2)),
        "model.layers.0.proj.bias": torch.ones(128),
        "model.visual.blocks.0.attn.proj.weight": torch.ones((128, 128)),
    }
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.keys.return_value = list(weights)
    reader.get_tensor.side_effect = weights.__getitem__
    encoded = (torch.zeros((256, 128)), torch.ones((2, 1)))

    with (
        patch("tools.convert_hf_to_fp8.torch.cuda.memory_allocated", return_value=0),
        patch("tools.convert_hf_to_fp8.safetensors.safe_open", return_value=reader),
        patch("tools.convert_hf_to_fp8.safetensors.torch.save_file") as save_file,
        patch("tools.convert_hf_to_fp8.quant_fp8", return_value=encoded) as quant_fp8,
    ):
        process_file("input", "output", "model.safetensors", "block", [128, 128], ConversionResult())

    quant_fp8.assert_called_once_with(weights["model.layers.0.proj.weight"], "block", [128, 128])
    saved = save_file.call_args.args[0]
    assert saved["model.layers.0.proj.weight"] is encoded[0]
    assert saved["model.layers.0.proj.weight_scale_inv"] is encoded[1]
    for name in weights.keys() - {"model.layers.0.proj.weight"}:
        assert saved[name] is weights[name]
