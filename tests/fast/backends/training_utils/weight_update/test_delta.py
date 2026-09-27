import json
from argparse import Namespace
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RejectingApiClient:
    def __init__(self, calls: list[tuple[str, dict]], failing_method: str) -> None:
        self._calls = calls
        self._failing_method = failing_method

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            if name == self._failing_method:
                return {"success": False, "error_message": "engine rejected the weights"}
            return {"success": True}

        return method


class TestPostWriteHookConstruction:
    def test_configured_post_write_hook_is_loaded_from_function_registry(self, tmp_path: Path) -> None:
        """A configured post-write path becomes the hook, resolved through the shared function registry."""
        hook = object()
        args = Namespace(
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3",
            custom_update_weight_post_write_path="miles_plugins.example:upload_delta",
        )

        with patch("miles.utils.function_registry.load_function", return_value=hook) as load_function:
            protocol = UpdateWeightFromDiskDelta(args)

        load_function.assert_called_once_with("miles_plugins.example:upload_delta")
        assert protocol._post_write_hook is hook


class TestCanonicalCheckpointLayout:
    @staticmethod
    def _protocol(checkpoint: Path) -> UpdateWeightFromDiskDelta:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(hf_checkpoint=str(checkpoint))
        return protocol

    def test_casts_only_between_plain_float_storage_dtypes(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"router": torch.zeros((2, 3), dtype=torch.bfloat16)},
            tmp_path / "model.safetensors",
        )

        emitted = torch.ones((2, 3), dtype=torch.float32)
        matched = self._protocol(tmp_path)._match_checkpoint_layout("router", emitted)

        assert matched.dtype is torch.bfloat16
        torch.testing.assert_close(matched.float(), emitted)

    def test_preserves_an_exact_nvfp4_layout(self, tmp_path: Path) -> None:
        tensors = {
            "expert.weight": torch.zeros((2, 3), dtype=torch.uint8),
            "expert.weight_scale": torch.zeros((2, 1), dtype=torch.float8_e4m3fn),
            "expert.weight_scale_2": torch.zeros((), dtype=torch.float32),
        }
        safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
        protocol = self._protocol(tmp_path)

        for name, emitted in tensors.items():
            assert protocol._match_checkpoint_layout(name, emitted) is emitted

    def test_rejects_a_missing_quantization_step(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"expert.weight": torch.zeros((2, 3), dtype=torch.uint8)},
            tmp_path / "model.safetensors",
        )

        with pytest.raises(ValueError, match="must be produced by the model's weight converter"):
            self._protocol(tmp_path)._match_checkpoint_layout("expert.weight", torch.ones((2, 3), dtype=torch.bfloat16))

    def test_rejects_shape_and_name_mismatches(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"weight": torch.zeros((2, 3), dtype=torch.bfloat16)},
            tmp_path / "model.safetensors",
        )
        protocol = self._protocol(tmp_path)

        with pytest.raises(ValueError, match="has shape"):
            protocol._match_checkpoint_layout("weight", torch.ones((3, 2), dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="absent from the canonical checkpoint"):
            protocol._match_checkpoint_layout("missing", torch.ones((2, 3), dtype=torch.bfloat16))


def test_send_bucket_encodes_a_scalar_tensor(tmp_path: Path) -> None:
    safetensors.torch.save_file(
        {"weight_scale": torch.ones((), dtype=torch.float32)},
        tmp_path / "model.safetensors",
    )
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(hf_checkpoint=str(tmp_path))
    protocol._use_pinned = False
    protocol._view_protocols = {}
    protocol._pool = MagicMock()
    protocol._inflight = deque()
    protocol.total_bytes = 0

    protocol.send_bucket([("weight_scale", torch.ones((), dtype=torch.float32))])

    _, name, payload, nbytes, pinned = protocol._pool.submit.call_args.args
    assert name == "weight_scale"
    assert payload.shape == (torch.float32.itemsize,)
    assert nbytes == torch.float32.itemsize
    assert not pinned


class TestReloadEnginesFailureTransitions:
    @staticmethod
    def _make_protocol(calls: list[tuple[str, dict]], failing_method: str) -> UpdateWeightFromDiskDelta:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_disk_dir="/shared/delta",
            pause_generation_mode="retract",
            check_weight_update_equal=False,
        )
        protocol.rollout_engines = [_RejectingApiClient(calls, failing_method)]
        protocol._post_write_hook = None
        protocol._version_dir = "/shared/delta/v7"
        return protocol

    @pytest.mark.parametrize(
        ("failing_method", "expected_calls"),
        [
            ("pull_weights", ["pull_weights"]),
            (
                "update_weights_from_disk",
                ["pull_weights", "pause_generation", "flush_cache", "update_weights_from_disk"],
            ),
        ],
    )
    def test_reload_engine_failure_stops_before_the_next_lifecycle_phase(self, failing_method: str, expected_calls: list[str]) -> None:
        """A rejected pull never pauses the engine, and a rejected disk reload never resumes it."""
        calls: list[tuple[str, dict]] = []
        protocol = self._make_protocol(calls, failing_method)

        with (
            patch(f"{_DELTA_MODULE}.dist") as dist_mock,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
        ):
            dist_mock.get_rank.return_value = 0
            with pytest.raises(RuntimeError, match="engine rejected the weights"):
                protocol._reload_engines(7)

        assert [name for name, _kwargs in calls] == expected_calls


def test_artifact_only_sync_publishes_without_engine_calls() -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace()
    protocol.rollout_engines = []
    protocol._post_write_hook = MagicMock()
    protocol._version_dir = "/shared/delta/weight_v000007"

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        protocol._reload_engines(7)

    protocol._post_write_hook.assert_called_once_with(protocol.args, protocol._version_dir, [])
    dist_mock.barrier.assert_called_once()


def test_multiview_parent_manifest_references_every_child_file(tmp_path: Path) -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._version_dir = str(tmp_path / "weight_v000007")
    protocol.delta_encoding = "xor"
    protocol.checksum_algorithm = "xxh3"
    protocol._view_protocols = {}
    for name, files in {
        "fp8": ("model.safetensors.index.json", "model-00000-of-00001.safetensors"),
        "nvfp4": ("model.safetensors.index.json", "model-00000-of-00001.safetensors"),
    }.items():
        child = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        child._published_files = files
        protocol._view_protocols[name] = child
    Path(protocol._version_dir).mkdir()

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = 0
        protocol._write_view_manifest(7)

    index = json.loads((Path(protocol._version_dir) / "model.safetensors.index.json").read_text())
    assert index["metadata"]["weight_views"] == ["fp8", "nvfp4"]
    assert set(index["weight_map"].values()) == {
        "fp8/model.safetensors.index.json",
        "fp8/model-00000-of-00001.safetensors",
        "nvfp4/model.safetensors.index.json",
        "nvfp4/model-00000-of-00001.safetensors",
    }


def test_multiview_baseline_validates_after_capturing_live_tensors(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    fp8_checkpoint = checkpoints / "fp8"
    nvfp4_checkpoint = checkpoints / "nvfp4"
    fp8_checkpoint.mkdir(parents=True)
    nvfp4_checkpoint.mkdir(parents=True)
    safetensors.torch.save_file(
        {"weight": torch.ones((2, 2), dtype=torch.bfloat16)},
        fp8_checkpoint / "model.safetensors",
    )
    safetensors.torch.save_file(
        {
            "weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "expert.input_scale": torch.ones((), dtype=torch.float32),
        },
        nvfp4_checkpoint / "model.safetensors",
    )

    parent = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    parent.delta_dir = str(tmp_path / "deltas")
    parent._post_write_hook = None
    parent.is_sender = True
    parent.args = Namespace()
    parent._view_protocols = {}
    for name, checkpoint, source_only_suffixes in (
        ("fp8", fp8_checkpoint, ()),
        ("nvfp4", nvfp4_checkpoint, (".input_scale",)),
    ):
        child = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        child.args = Namespace(hf_checkpoint=str(checkpoint))
        child._snapshot = {}
        child._source_only_suffixes = source_only_suffixes
        child._source_tensor_prefixes = ()
        parent._view_protocols[name] = child

    def iter_buckets(*, materialize: bool):
        assert materialize
        yield {
            "fp8": [("weight", torch.ones((2, 2), dtype=torch.bfloat16))],
            "nvfp4": [("weight", torch.ones((2, 2), dtype=torch.bfloat16))],
        }

    def gather_messages(output, message, **_kwargs):
        output[0] = message

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = gather_messages
        parent._capture_view_baselines(iter_buckets)

    assert set(parent._view_protocols["fp8"]._snapshot) == {"weight"}
    assert set(parent._view_protocols["nvfp4"]._snapshot) == {"weight"}


def test_multiview_finalize_does_not_publish_parent_when_a_child_fails(
    tmp_path: Path,
) -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._version_dir = str(tmp_path / "weight_v000007")
    Path(protocol._version_dir).mkdir()
    fp8 = MagicMock()
    nvfp4 = MagicMock()
    nvfp4._write_delta_files.side_effect = RuntimeError("failed child")
    protocol._view_protocols = {"fp8": fp8, "nvfp4": nvfp4}

    with pytest.raises(RuntimeError, match="failed child"):
        protocol.finalize(7)

    assert not (Path(protocol._version_dir) / "model.safetensors.index.json").exists()
