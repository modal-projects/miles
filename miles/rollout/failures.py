"""Shared rollout failure metadata consumed by rollout schedulers."""

from miles.utils.types import Sample


INFRASTRUCTURE_FAILURE_KEY = "_miles_infrastructure_failure"
NON_RETRYABLE_FAILURE_KEY = "_miles_non_retryable_failure"
LOSS_MASKED_FAILURE_KEY = "_miles_loss_masked_failure"


def mark_non_retryable_failure(sample: Sample) -> None:
    """Classify an aborted attempt that repeating cannot repair."""
    sample.metadata[NON_RETRYABLE_FAILURE_KEY] = True


def mark_infrastructure_failure(sample: Sample) -> None:
    mark_non_retryable_failure(sample)
    sample.metadata[INFRASTRUCTURE_FAILURE_KEY] = True


def clear_failure_classification(sample: Sample) -> None:
    """Clear attempt-local failure state before re-running a sample."""
    sample.metadata.pop(INFRASTRUCTURE_FAILURE_KEY, None)
    sample.metadata.pop(NON_RETRYABLE_FAILURE_KEY, None)
    sample.metadata.pop(LOSS_MASKED_FAILURE_KEY, None)


def is_infrastructure_failure(sample: Sample) -> bool:
    return bool(sample.metadata.get(INFRASTRUCTURE_FAILURE_KEY))


def is_non_retryable_failure(sample: Sample) -> bool:
    return bool(sample.metadata.get(NON_RETRYABLE_FAILURE_KEY))


def mark_loss_masked_failure(sample: Sample) -> None:
    """Mark a fixed-shape placeholder that must not affect rewards or loss scaling."""
    sample.metadata[LOSS_MASKED_FAILURE_KEY] = True


def is_loss_masked_failure(sample: Sample) -> bool:
    return bool(sample.metadata.get(LOSS_MASKED_FAILURE_KEY))
