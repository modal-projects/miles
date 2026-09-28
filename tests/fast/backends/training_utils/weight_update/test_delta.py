import json
from argparse import Namespace
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightView
from miles.utils.disk_delta import checksum, load_delta_lineage_checksums

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


def _write_delta(root: Path, version: int, checksums: dict[str, str], *, algorithm: str = "adler32") -> None:
    version_dir = root / f"weight_v{version:06d}"
    version_dir.mkdir(parents=True)
    shard = "model-00000-of-00001.safetensors"
    safetensors.torch.save_file(
        {name: torch.zeros(1, dtype=torch.uint8) for name in checksums},
        version_dir / shard,
        metadata=checksums,
    )
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": f"{version:06d}",
                    "base_version": f"{version - 1:06d}",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": algorithm,
                },
                "weight_map": {name: shard for name in checksums},
            }
        )
    )


def test_load_delta_lineage_checksums_tracks_latest_tensor_state(
    tmp_path: Path,
) -> None:
    _write_delta(tmp_path, 1, {"a": "a-v1", "b": "b-v1"})
    _write_delta(tmp_path, 2, {"a": "a-v2"})

    assert load_delta_lineage_checksums(str(tmp_path), 2) == {
        "a": ("adler32", "a-v2"),
        "b": ("adler32", "b-v1"),
    }


def test_load_delta_lineage_checksums_requires_every_transition(
    tmp_path: Path,
) -> None:
    _write_delta(tmp_path, 2, {"a": "a-v2"})

    with pytest.raises(ValueError, match="version 1"):
        load_delta_lineage_checksums(str(tmp_path), 2)


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

    def test_each_weight_view_keeps_the_normal_hook_and_owns_its_directory(self, tmp_path: Path) -> None:
        hook = object()
        views = (
            WeightView("bf16", "/checkpoints/bf16", None),
            WeightView("fp8-e4m3", "/checkpoints/fp8", {"quant_method": "fp8"}),
        )
        args = Namespace(
            update_weight_disk_dir=str(tmp_path / "updates"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3",
            update_weight_views={view.name: view.checkpoint for view in views},
            custom_update_weight_post_write_path="plugins:publish",
        )

        with (
            patch(
                f"{_DELTA_MODULE}.load_weight_views",
                side_effect=lambda config: views if config is not None else (),
            ),
            patch("miles.utils.function_registry.load_function", return_value=hook),
        ):
            protocol = UpdateWeightFromDiskDelta(args)

        assert protocol._post_write_hook is None
        assert set(protocol._view_protocols) == {"bf16", "fp8-e4m3"}
        for name, child in protocol._view_protocols.items():
            assert child.delta_dir == str(tmp_path / "updates" / name)
            assert child.args.update_weight_view == name
            assert child._post_write_hook is hook


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
    parent.is_sender = True
    parent.args = Namespace()
    parent._view_protocols = {}
    for name, checkpoint in (
        ("fp8", fp8_checkpoint),
        ("nvfp4", nvfp4_checkpoint),
    ):
        child = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        child.args = Namespace(hf_checkpoint=str(checkpoint))
        child.delta_dir = str(tmp_path / "deltas" / name)
        child._post_write_hook = None
        child._snapshot = {}
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


def test_multiview_resume_seeds_verified_live_state_without_deleting_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    canonical = torch.zeros((2, 2), dtype=torch.bfloat16)
    resumed = torch.ones((2, 2), dtype=torch.bfloat16)
    safetensors.torch.save_file({"weight": canonical}, checkpoint / "model.safetensors")

    delta_dir = tmp_path / "updates" / "bf16"
    resumed_bytes = resumed.contiguous().reshape(-1).view(torch.uint8).numpy()
    _write_delta(
        delta_dir,
        1,
        {"weight": checksum("adler32", resumed_bytes)},
    )

    parent = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    parent.is_sender = True
    parent.args = Namespace(update_weight_initial_version=1)
    child = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    child.args = Namespace(hf_checkpoint=str(checkpoint))
    child.delta_dir = str(delta_dir)
    child.checksum_algorithm = "adler32"
    child._post_write_hook = MagicMock()
    child._snapshot = {}
    parent._view_protocols = {"bf16": child}

    def iter_buckets(*, materialize: bool):
        assert materialize
        yield {"bf16": [("weight", resumed)]}

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

    assert (delta_dir / "weight_v000001/model.safetensors.index.json").exists()
    child._post_write_hook.assert_not_called()
    assert child._snapshot["weight"].tobytes() == resumed_bytes.tobytes()


def test_multiview_resume_rejects_live_state_that_disagrees_with_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    canonical = torch.zeros((2, 2), dtype=torch.bfloat16)
    safetensors.torch.save_file({"weight": canonical}, checkpoint / "model.safetensors")
    delta_dir = tmp_path / "updates" / "bf16"
    _write_delta(delta_dir, 1, {"weight": "not-the-live-checksum"})

    parent = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    parent.is_sender = True
    parent.args = Namespace(update_weight_initial_version=1)
    child = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    child.args = Namespace(hf_checkpoint=str(checkpoint))
    child.delta_dir = str(delta_dir)
    child.checksum_algorithm = "adler32"
    child._post_write_hook = None
    child._snapshot = {}
    parent._view_protocols = {"bf16": child}

    def iter_buckets(*, materialize: bool):
        yield {"bf16": [("weight", torch.ones_like(canonical))]}

    def gather_messages(output, message, **_kwargs):
        output[0] = message

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = gather_messages
        with pytest.raises(RuntimeError, match="does not match v1"):
            parent._capture_view_baselines(iter_buckets)


def test_multiview_finalize_publishes_each_view_independently() -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    fp8 = MagicMock()
    nvfp4 = MagicMock()
    fp8._inflight = []
    nvfp4._inflight = []
    fp8.update_weight_metrics = {"perf/update_weights_density": 0.5}
    nvfp4.update_weight_metrics = {"perf/update_weights_density": 0.25}
    protocol._view_protocols = {"fp8": fp8, "nvfp4": nvfp4}

    def gather_one_rank(output, value, **_kwargs):
        output[0] = value

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = gather_one_rank
        protocol.finalize(7)

    fp8.after_base_weights.assert_called_once_with()
    fp8.finalize.assert_called_once_with(7)
    nvfp4.after_base_weights.assert_called_once_with()
    nvfp4.finalize.assert_called_once_with(7)
    assert protocol.update_weight_metrics == {
        "perf/update_weights_density/fp8": 0.5,
        "perf/update_weights_density/nvfp4": 0.25,
    }


def test_multiview_failure_does_not_roll_back_a_published_sibling() -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    fp8 = MagicMock()
    fp8._inflight = []
    fp8.update_weight_metrics = {}
    nvfp4 = MagicMock()
    nvfp4._inflight = []
    nvfp4.after_base_weights.side_effect = RuntimeError("failed child")
    protocol._view_protocols = {"fp8": fp8, "nvfp4": nvfp4}

    def gather_one_rank(output, value, **_kwargs):
        output[0] = value

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = gather_one_rank
        with pytest.raises(RuntimeError, match="failed child"):
            protocol.finalize(7)

    fp8.finalize.assert_called_once_with(7)
    nvfp4.finalize.assert_not_called()


def test_multiview_finalize_does_not_wait_for_an_unready_sibling() -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    publish_order = []
    fp8_future = Future()
    fp8 = MagicMock()
    fp8._inflight = [fp8_future]
    fp8.update_weight_metrics = {}
    fp8.finalize.side_effect = lambda _version: publish_order.append("fp8")
    nvfp4 = MagicMock()
    nvfp4._inflight = []
    nvfp4.update_weight_metrics = {}

    def publish_nvfp4(_version):
        publish_order.append("nvfp4")
        fp8_future.set_result(None)

    nvfp4.finalize.side_effect = publish_nvfp4
    protocol._view_protocols = {"fp8": fp8, "nvfp4": nvfp4}

    def gather_one_rank(output, value, **_kwargs):
        output[0] = value

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = gather_one_rank
        protocol.finalize(7)

    nvfp4.finalize.assert_called_once_with(7)
    fp8.finalize.assert_called_once_with(7)
    assert publish_order == ["nvfp4", "fp8"]
