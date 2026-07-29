"""Shared rollout failure metadata.

Environment adapters decide whether a terminal outcome is caused by policy
behavior or infrastructure. Schedulers consume only this normalized marker.
"""

from miles.utils.types import Sample


INFRASTRUCTURE_FAILURE_KEY = "_miles_infrastructure_failure"


def mark_infrastructure_failure(sample: Sample) -> None:
    sample.metadata[INFRASTRUCTURE_FAILURE_KEY] = True


def clear_infrastructure_failure(sample: Sample) -> None:
    """Clear attempt-local failure state before re-running a sample."""
    sample.metadata.pop(INFRASTRUCTURE_FAILURE_KEY, None)


def is_infrastructure_failure(sample: Sample) -> bool:
    return bool(sample.metadata.get(INFRASTRUCTURE_FAILURE_KEY))
