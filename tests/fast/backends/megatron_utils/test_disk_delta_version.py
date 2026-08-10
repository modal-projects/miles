from argparse import Namespace
from unittest.mock import Mock

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import (
    UpdateWeightFromDiskDelta,
)


@pytest.mark.parametrize(
    ("initial_version", "published_version"),
    [(0, 1), (119, 120)],
)
def test_disk_delta_continues_from_initial_version(tmp_path, monkeypatch, initial_version: int, published_version: int) -> None:
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
