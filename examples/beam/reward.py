"""Boxed integer accuracy and the DAPO soft overlong penalty."""

from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.rm_hub.math_dapo_utils import compute_score


def overlong_penalty(length: int, maximum: int, buffer: int, factor: float = 1.0) -> float:
    if maximum < 1 or not 0 < buffer <= maximum or factor < 0:
        raise ValueError("Require maximum >= buffer > 0 and a nonnegative penalty factor")
    if not 0 <= length <= maximum:
        raise ValueError(f"Response length {length} is outside [0, {maximum}]")
    return min(-(length - (maximum - buffer)) / buffer * factor, 0.0)


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await reward(args, item, **kwargs) for item in sample]
    result = dict(compute_score(sample.response, sample.label, strict_box_verify=True))
    penalty = overlong_penalty(
        sample.response_length,
        args.rollout_max_response_len,
        max(1, args.rollout_max_response_len // 5),
    )
    result.update(raw_score=result["score"], overlong_penalty=penalty)
    result["score"] += penalty
    return result


def filter_group(args, samples, **kwargs):
    if len(samples) < 2:
        raise ValueError("DAPO requires at least two completions per prompt")
    # Length penalties must not make an all-wrong group eligible for training.
    accuracy = [sample.reward["acc"] for sample in samples]
    keep = min(accuracy) != max(accuracy)
    return FilterOutput(keep=keep, reason=None if keep else "uniform_accuracy")
