import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import safetensors.torch
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import (
    quantize_params_fp8,
)
from miles.backends.training_utils.weight_update.views import (
    get_quantized_weight_basenames,
    load_weight_views,
)
from tools.convert_hf_to_fp8 import should_quantize


def test_offline_fp8_quantizes_only_matrix_weights() -> None:
    assert should_quantize("model.layers.0.proj.weight", torch.ones((2, 2)))
    assert not should_quantize("model.layers.0.fused_experts.weight", torch.ones((2, 2, 2)))
    assert not should_quantize("model.layers.0.proj.bias", torch.ones(2))
    assert not should_quantize(
        "model.layers.0.proj.weight",
        torch.ones((129, 128)),
        (128, 128),
    )


def test_fp8_scope_is_derived_from_the_canonical_checkpoint(tmp_path: Path) -> None:
    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(
        {
            "model.layers.0.in_proj_a.weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            "model.layers.0.in_proj_a.weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            "model.layers.0.norm.weight": torch.ones(2),
        },
        tmp_path / shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.in_proj_a.weight": shard,
                    "model.layers.0.in_proj_a.weight_scale_inv": shard,
                    "model.layers.0.norm.weight": shard,
                }
            }
        )
    )

    assert get_quantized_weight_basenames(str(tmp_path), {"quant_method": "fp8"}) == {"model.layers.0.in_proj_a"}


def test_fp8_scope_supports_single_file_checkpoints(tmp_path: Path) -> None:
    safetensors.torch.save_file(
        {
            "model.layers.0.in_proj_a.weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
            "model.layers.0.in_proj_a.weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            "model.layers.0.norm.weight": torch.ones(2),
        },
        tmp_path / "model.safetensors",
    )

    assert get_quantized_weight_basenames(str(tmp_path), {"quant_method": "fp8"}) == {"model.layers.0.in_proj_a"}


def test_weight_views_preserve_quantizer_specific_module_scope(tmp_path: Path) -> None:
    fp8 = tmp_path / "fp8"
    nvfp4 = tmp_path / "nvfp4"
    for checkpoint in (fp8, nvfp4):
        checkpoint.mkdir()
        safetensors.torch.save_file(
            {
                "model.layers.0.proj.weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
                "model.layers.0.proj.weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
            },
            checkpoint / "model.safetensors",
        )

    configs = {
        str(fp8): {"quant_method": "fp8"},
        str(nvfp4): {"quant_method": "nvfp4"},
    }
    with patch(
        "miles.backends.training_utils.weight_update.views.load_hf_config",
        side_effect=lambda checkpoint: SimpleNamespace(quantization_config=configs[checkpoint]),
    ):
        views = load_weight_views({"fp8": str(fp8), "nvfp4": str(nvfp4)})

    assert views[0].quantized_weight_basenames == frozenset({"model.layers.0.proj"})
    assert views[1].quantized_weight_basenames is None


def test_weight_view_names_are_opaque_safe_path_components(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bf16"
    checkpoint.mkdir()
    with patch(
        "miles.backends.training_utils.weight_update.views.load_hf_config",
        return_value=SimpleNamespace(quantization_config=None),
    ):
        [view] = load_weight_views({"bf16-reference.v2": str(checkpoint)})
        assert view.name == "bf16-reference.v2"

        with pytest.raises(ValueError, match="invalid weight view name"):
            load_weight_views({"../bf16": str(checkpoint)})


def test_fp8_live_export_follows_the_canonical_module_set() -> None:
    tensors = [
        ("model.layers.0.in_proj_a.weight", torch.ones((2, 2))),
        ("model.layers.0.norm.weight", torch.ones(2)),
        ("model.layers.0.in_proj_a.weight_scale", torch.ones(1)),
    ]
    quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }

    with patch(
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8._quantize_param",
        return_value=[("model.layers.0.in_proj_a.weight", torch.zeros((2, 2)))],
    ) as quantize:
        output = quantize_params_fp8(
            Namespace(),
            "an.unrecognized.megatron.parameter",
            tensors,
            quantization_config,
            {"model.layers.0.in_proj_a"},
        )

    quantize.assert_called_once()
    assert [name for name, _tensor in output] == [
        "model.layers.0.in_proj_a.weight",
        "model.layers.0.norm.weight",
    ]


def test_fp8_view_uses_canonical_checkpoint_storage() -> None:
    tensors = [("model.layers.0.proj.weight", torch.ones((2, 2)))]
    config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 1],
    }
    encoded = (
        torch.zeros((2, 2), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2), dtype=torch.float32),
    )

    with (
        patch(
            "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8.blockwise_cast_to_fp8_triton",
            return_value=encoded,
        ) as canonical,
        patch(
            "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8._get_scale_format",
            return_value="ue8m0",
        ),
        patch("miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8.quant_weight_ue8m0") as runtime,
    ):
        output = quantize_params_fp8(
            Namespace(),
            "module.module.decoder.layers.0.self_attention.linear_proj.weight",
            tensors,
            config,
            {"model.layers.0.proj"},
            canonical_storage=True,
        )

    canonical.assert_called_once()
    runtime.assert_not_called()
    assert [name for name, _tensor in output] == [
        "model.layers.0.proj.weight",
        "model.layers.0.proj.weight_scale_inv",
    ]
