import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, base: Path):
        self.weight_views = tuple(
            SimpleNamespace(
                name=name,
                checkpoint=str(base / name),
                source_only_suffixes=((".input_scale",) if name == "nvfp4" else ()),
            )
            for name in ("fp8", "nvfp4")
        )

    @staticmethod
    def iter_hf_weight_views(_weights):
        yield {
            "fp8": [
                ("weight", torch.arange(4, dtype=torch.uint8)),
                ("weight_scale", torch.ones(1, dtype=torch.float32)),
            ],
            "nvfp4": [
                ("weight", torch.arange(2, dtype=torch.uint8)),
                ("weight_scale", torch.ones(1, dtype=torch.float8_e4m3fn)),
            ],
        }


def _write_base_checkpoints(base: Path, *, include_unexpected: bool = False) -> None:
    for view in ("fp8", "nvfp4"):
        view_dir = base / view
        view_dir.mkdir(parents=True)
        dynamic = {
            "weight": torch.arange(4 if view == "fp8" else 2, dtype=torch.uint8),
            "weight_scale": torch.ones(
                1,
                dtype=(torch.float32 if view == "fp8" else torch.float8_e4m3fn),
            ),
        }
        if include_unexpected:
            dynamic["missing.weight"] = torch.ones(1)
        safetensors.torch.save_file(dynamic, view_dir / "dynamic.safetensors")
        weight_map = {name: "dynamic.safetensors" for name in dynamic}

        frozen = {"frozen.vision.weight": torch.ones(2)}
        safetensors.torch.save_file(frozen, view_dir / "frozen.safetensors")
        weight_map["frozen.vision.weight"] = "frozen.safetensors"
        if view == "nvfp4":
            static = {"expert.input_scale": torch.ones((), dtype=torch.float32)}
            safetensors.torch.save_file(static, view_dir / "static.safetensors")
            weight_map["expert.input_scale"] = "static.safetensors"
        (view_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_snapshot_publisher_writes_one_complete_checkpoint_per_view(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_base_checkpoints(base)

    output = tmp_path / "output"
    publisher = SnapshotPublisher(_Iterator(base))
    with (
        patch(
            "miles.backends.training_utils.weight_update.snapshot_publisher.dist.get_rank",
            return_value=0,
        ),
        patch.object(SnapshotPublisher, "_copy_metadata"),
    ):
        publisher.write_model(
            output,
            weights={},
            hf_checkpoint="/unused",
            source_tensor_prefixes=("frozen.",),
        )
        publisher.mark_weight_views_complete(output)

    assert {path.name for path in output.iterdir()} == {"fp8", "nvfp4"}
    fp8 = safetensors.torch.load_file(output / "fp8" / "model-00001.safetensors")
    nvfp4 = safetensors.torch.load_file(output / "nvfp4" / "model-00001.safetensors")
    assert fp8["weight"].numel() == 4
    assert nvfp4["weight"].numel() == 2
    fp8_index = json.loads((output / "fp8" / "model.safetensors.index.json").read_text())
    assert "input_scale" not in fp8_index["weight_map"]
    for view in ("fp8", "nvfp4"):
        assert (output / view / ".complete").is_file()
        index = json.loads((output / view / "model.safetensors.index.json").read_text())
        frozen = safetensors.torch.load_file(output / view / index["weight_map"]["frozen.vision.weight"])
        torch.testing.assert_close(frozen["frozen.vision.weight"], torch.ones(2))
    nvfp4_index = json.loads((output / "nvfp4" / "model.safetensors.index.json").read_text())
    static = safetensors.torch.load_file(output / "nvfp4" / nvfp4_index["weight_map"]["expert.input_scale"])
    assert static["expert.input_scale"].item() == 1


def test_snapshot_publisher_rejects_an_omitted_dynamic_tensor(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_base_checkpoints(base, include_unexpected=True)

    publisher = SnapshotPublisher(_Iterator(base))
    with (
        patch(
            "miles.backends.training_utils.weight_update.snapshot_publisher.dist.get_rank",
            return_value=0,
        ),
        patch.object(SnapshotPublisher, "_copy_metadata"),
        pytest.raises(ValueError, match="omitted canonical tensors"),
    ):
        publisher.write_model(tmp_path / "output", weights={}, hf_checkpoint="/unused")


class _Float32Iterator(_Iterator):
    """Emits a plain-float tensor in float32 that the canonical checkpoints store in bf16."""

    @staticmethod
    def iter_hf_weight_views(_weights):
        (bucket,) = _Iterator.iter_hf_weight_views(_weights)
        a_log = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        yield {view: [*tensors, ("A_log", a_log)] for view, tensors in bucket.items()}


def test_snapshot_publisher_writes_each_view_in_its_canonical_layout(tmp_path: Path) -> None:
    """The export must hold the bytes its view's deltas build on, not the emitted dtype."""
    base = tmp_path / "base"
    _write_base_checkpoints(base)
    for view in ("fp8", "nvfp4"):
        safetensors.torch.save_file({"A_log": torch.zeros(3, dtype=torch.bfloat16)}, base / view / "a_log.safetensors")
        index_path = base / view / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["A_log"] = "a_log.safetensors"
        index_path.write_text(json.dumps(index))

    output = tmp_path / "output"
    with (
        patch(
            "miles.backends.training_utils.weight_update.snapshot_publisher.dist.get_rank",
            return_value=0,
        ),
        patch.object(SnapshotPublisher, "_copy_metadata"),
    ):
        SnapshotPublisher(_Float32Iterator(base)).write_model(
            output, weights={}, hf_checkpoint="/unused", source_tensor_prefixes=("frozen.",)
        )

    for view in ("fp8", "nvfp4"):
        index = json.loads((output / view / "model.safetensors.index.json").read_text())
        exported = safetensors.torch.load_file(output / view / index["weight_map"]["A_log"])["A_log"]
        assert exported.dtype == torch.bfloat16
        assert torch.equal(exported, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).to(torch.bfloat16))
