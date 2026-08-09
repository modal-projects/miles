from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import (
    can_overlap_external_weight_sync,
    uses_external_disk_deltas,
)


@pytest.mark.parametrize(
    ("transfer_mode", "endpoint", "pause_mode", "uses_deltas", "can_overlap"),
    [
        ("disk-delta", "https://fleet.example", "in_place", True, True),
        ("disk-delta", "https://fleet.example", "retract", True, False),
        ("disk-delta", None, "in_place", False, False),
        ("broadcast", "https://fleet.example", "in_place", False, False),
    ],
)
def test_external_weight_sync_capabilities(
    transfer_mode,
    endpoint,
    pause_mode,
    uses_deltas,
    can_overlap,
):
    args = SimpleNamespace(
        update_weight_transfer_mode=transfer_mode,
        rollout_endpoint_url=endpoint,
        pause_generation_mode=pause_mode,
    )

    assert uses_external_disk_deltas(args) is uses_deltas
    assert can_overlap_external_weight_sync(args) is can_overlap
