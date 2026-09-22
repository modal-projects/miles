import json
from types import SimpleNamespace

import safetensors
import safetensors.torch
import torch
from tools.convert_hf_to_nvfp4 import convert_nvfp4

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_nvfp4 import quantize_params_nvfp4
from miles.utils.nvfp4 import NVFP4_GROUP_SIZE


def test_converter_keeps_nested_decoder_tail_in_bf16(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"text_config":{"num_hidden_layers":1}}')

    prefix = "model.language_model.layers.0"
    weights = {
        f"{prefix}.mlp.experts.0.{projection}.weight": torch.ones((1, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    safetensors.torch.save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})

    convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        num_layers_at_end_in_bf16=1,
    )

    config = json.loads((save_dir / "config.json").read_text())
    assert config["quantization_config"]["ignore"] == [
        f"{prefix}.",
        f"{prefix}.mlp.experts",
        "model.layers.0.",
    ]
    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        assert all("weight_scale" not in name for name in checkpoint.keys())
        assert checkpoint.get_tensor(f"{prefix}.mlp.experts.0.down_proj.weight").dtype == torch.bfloat16


def test_online_export_keeps_the_same_nested_decoder_tail_in_bf16():
    weights = [
        (
            f"model.language_model.layers.39.mlp.experts.0.{projection}.weight",
            torch.ones((1, NVFP4_GROUP_SIZE), dtype=torch.bfloat16),
        )
        for projection in ("gate_proj", "up_proj")
    ]

    actual = quantize_params_nvfp4(
        args=SimpleNamespace(
            fp4_param=False,
            fp4_param_gather=False,
            extra_high_precision_layers_megatron=[],
            first_last_layers_bf16=True,
            num_layers=40,
            num_layers_at_start_in_bf16=0,
            num_layers_at_end_in_bf16=6,
        ),
        megatron_name="decoder.layers.39.mlp.experts.linear_fc1.weight0",
        converted_named_params=weights,
        quantization_config={"quant_method": "nvfp4"},
    )

    assert actual is weights
