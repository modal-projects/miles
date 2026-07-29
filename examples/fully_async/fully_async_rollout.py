"""Compatibility entry points for the fully-async rollout implementation.

New configurations should use
``miles.rollout.fully_async_rollout.FullyAsyncRolloutFn`` directly.
"""

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.data_source import DataSource
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.utils.async_utils import run

__all__ = ["FullyAsyncRolloutFn", "generate_rollout_fully_async"]

_legacy_rollout: FullyAsyncRolloutFn | None = None


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    """Serve the former function-based configuration path without duplicating the pool."""
    if evaluation:
        raise ValueError("FullyAsyncRolloutFn does not serve evaluation")

    global _legacy_rollout
    if _legacy_rollout is None:
        _legacy_rollout = FullyAsyncRolloutFn(
            RolloutFnConstructorInput(args=args, data_source=data_buffer)
        )
    output = run(_legacy_rollout(RolloutFnTrainInput(rollout_id=rollout_id)))
    return output.samples
