from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import (
    UpdateWeightFromDiskDelta,
)


@pytest.mark.parametrize(
    ("initial_version", "published_version"),
    [(0, 1), (119, 120)],
)
def test_disk_delta_continues_from_initial_version(
    tmp_path, monkeypatch, initial_version: int, published_version: int
) -> None:
    monkeypatch.setattr(UpdateWeightFromDiskDelta, "_init_lora", lambda *_args, **_kwargs: None)
    updater = UpdateWeightFromDiskDelta(
        Namespace(
            update_weight_disk_dir=str(tmp_path),
            update_weight_delta_encoding="zstd",
            update_weight_delta_checksum="xxh64",
            rollout_endpoint_url=None,
            custom_update_weight_post_write_path=None,
        ),
        [],
        lambda: {},
        model_name="test",
        quantization_config=None,
        initial_weight_version=initial_version,
    )
    updater._capture_baseline = Mock()
    updater._publish = Mock()
    updater._reload_engines = Mock()
    updater._record_metrics = Mock()

    updater.update_weights()
    assert updater.weight_version == initial_version
    updater.update_weights()
    assert updater.weight_version == published_version


def test_disk_delta_rejects_a_negative_initial_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(UpdateWeightFromDiskDelta, "_init_lora", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="must be non-negative"):
        UpdateWeightFromDiskDelta(
            Namespace(
                update_weight_disk_dir=str(tmp_path),
                update_weight_delta_encoding="xor",
                update_weight_delta_checksum="xxh3-128",
                rollout_endpoint_url=None,
                custom_update_weight_post_write_path=None,
            ),
            [],
            lambda: {},
            model_name="test",
            quantization_config=None,
            initial_weight_version=-1,
        )


def test_resumed_disk_delta_preserves_history_and_replaces_abandoned_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(UpdateWeightFromDiskDelta, "_init_lora", lambda *_args, **_kwargs: None)
    history = tmp_path / "weight_v000120"
    abandoned = tmp_path / "weight_v000121"
    history.mkdir()
    abandoned.mkdir()
    (history / "history").write_text("keep")
    (abandoned / "stale").write_text("remove")
    updater = UpdateWeightFromDiskDelta(
        Namespace(
            update_weight_disk_dir=str(tmp_path),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3-128",
            rollout_endpoint_url=None,
            custom_update_weight_post_write_path=None,
        ),
        [],
        lambda: {},
        model_name="test",
        quantization_config=None,
        initial_weight_version=120,
    )
    updater.weight_version = 121
    updater._snapshot = {}
    updater._for_each_hf_bucket = Mock()
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.barrier", lambda **_kwargs: None)
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.get_gloo_group",
        lambda: None,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.get_parallel_state",
        lambda: SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=1), tp=SimpleNamespace(rank=0)),
    )

    updater._encode_delta()

    assert (history / "history").read_text() == "keep"
    assert not abandoned.exists()


@pytest.mark.parametrize("initial_version", [0, 120])
def test_baseline_clears_only_a_fresh_version_stream(tmp_path, monkeypatch, initial_version: int) -> None:
    monkeypatch.setattr(UpdateWeightFromDiskDelta, "_init_lora", lambda *_args, **_kwargs: None)
    history = tmp_path / "weight_v000001"
    history.mkdir()
    (history / "history").write_text("old")
    updater = UpdateWeightFromDiskDelta(
        Namespace(
            update_weight_disk_dir=str(tmp_path),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3-128",
            rollout_endpoint_url=None,
            custom_update_weight_post_write_path=None,
            hf_checkpoint="/checkpoint",
            check_weight_update_equal=False,
        ),
        [],
        lambda: {},
        model_name="test",
        quantization_config=None,
        initial_weight_version=initial_version,
    )
    updater.rollout_engines = []
    updater._for_each_hf_bucket = Mock()
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.barrier", lambda **_kwargs: None)
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.get_gloo_group",
        lambda: None,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.make_tensor_reader",
        lambda _path: Mock(),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta.ray.get",
        lambda value: value,
    )

    updater._capture_baseline()

    assert history.exists() is (initial_version > 0)
