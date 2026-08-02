from argparse import Namespace

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import (
    UpdateWeightFromDiskDelta,
)


def test_external_resume_uses_loaded_actor_as_delta_baseline(tmp_path, monkeypatch) -> None:
    existing = tmp_path / "weight_v000239" / "model.safetensors.index.json"
    existing.parent.mkdir()
    existing.write_text("existing")
    monkeypatch.setattr(UpdateWeightFromDiskDelta, "_init_lora", lambda *_args, **_kwargs: None)
    updater = UpdateWeightFromDiskDelta(
        Namespace(
            update_weight_disk_dir=str(tmp_path),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3-128",
            custom_update_weight_post_write_path=None,
            hf_checkpoint="/original-hf",
            check_weight_update_equal=False,
            rollout_endpoint_url="https://rollout.example",
        ),
        [],
        lambda: {},
        model_name="test",
        quantization_config=None,
        initial_weight_version=239,
    )
    updater.rollout_engines = []
    current = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    updater._for_each_hf_bucket = lambda callback: callback([("weight", current)])
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.barrier", lambda **_kwargs: None)
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.get_gloo_group",
        lambda: None,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.make_tensor_reader",
        lambda _path: pytest.fail("external resume must not read the original HF checkpoint"),
    )

    updater._capture_baseline()

    assert existing.read_text() == "existing"
    expected = current.view(torch.uint8).numpy().reshape(-1)
    np.testing.assert_array_equal(updater._snapshot["weight"], expected)
