import json

import pytest
import safetensors
import safetensors.torch
import torch
from tools.convert_hf_to_nvfp4 import convert_nvfp4

from miles.utils.nvfp4 import NVFP4_GROUP_SIZE


@pytest.mark.parametrize(
    ("layer_root", "config"),
    [
        ("model.layers", {"num_hidden_layers": 1}),
        ("model.language_model.layers", {"text_config": {"num_hidden_layers": 1}}),
    ],
)
def test_converter_preserves_bf16_layers_at_the_checkpoint_decoder_root(tmp_path, layer_root, config):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))

    layer_prefix = f"{layer_root}.0"
    weights = {
        f"{layer_prefix}.mlp.experts.0.{projection}.weight": torch.ones((1, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    safetensors.torch.save_file(weights, model_dir / "model.safetensors", metadata={"format": "pt"})

    convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        num_layers_at_end_in_bf16=1,
    )

    output_config = json.loads((save_dir / "config.json").read_text())
    assert output_config["quantization_config"]["ignore"] == [
        f"{layer_prefix}.",
        f"{layer_prefix}.mlp.experts",
    ]
    with safetensors.safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        assert all("weight_scale" not in name for name in checkpoint.keys())
        assert checkpoint.get_tensor(f"{layer_prefix}.mlp.experts.0.down_proj.weight").dtype == torch.bfloat16


def test_converter_rejects_bf16_carveout_without_a_decoder_layer_root(tmp_path):
    model_dir = tmp_path / "model"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')
    safetensors.torch.save_file(
        {"model.embed_tokens.weight": torch.ones((1, NVFP4_GROUP_SIZE), dtype=torch.bfloat16)},
        model_dir / "model.safetensors",
        metadata={"format": "pt"},
    )

    with pytest.raises(ValueError, match="Could not find decoder layers containing routed experts"):
        convert_nvfp4(
            str(model_dir),
            str(save_dir),
            device="cpu",
            num_layers_at_end_in_bf16=1,
        )
