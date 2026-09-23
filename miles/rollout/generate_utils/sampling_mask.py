from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.utils.sampling_mask import RolloutSamplingMask, sampling_support_replay_enabled
from miles.utils.types import Sample


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
    *,
    evaluation: bool = False,
) -> bool:
    """Validate whether a training request must return its realized sampling support."""
    if evaluation:
        return False

    params = sampling_params or {}
    configured_top_p = float(getattr(args, "rollout_top_p", 1.0))
    request_top_p = float(configured_top_p if params.get("top_p") is None else params["top_p"])
    if not 0.0 < request_top_p <= 1.0:
        raise ValueError(f"training request top_p must be in (0, 1], got {request_top_p}")

    if not sampling_support_replay_enabled(args):
        raw_top_k = params.get("top_k")
        request_top_k = -1 if raw_top_k is None else int(raw_top_k)
        if request_top_p < 1.0 or request_top_k > 0:
            raise ValueError("bounded training-request sampling requires bounded rollout sampling")
        return False

    missing_params = [name for name in ("top_p", "top_k", "temperature") if params.get(name) is None]
    if missing_params:
        raise ValueError(f"sampling-support replay requires explicit request parameters: {', '.join(missing_params)}")

    request_top_k = int(params["top_k"])
    if request_top_k <= 0:
        raise ValueError(f"training request top_k must be positive, got {request_top_k}")

    configured_temperature = float(getattr(args, "rollout_temperature", 1.0))
    request_temperature = float(params["temperature"])
    if request_temperature != configured_temperature:
        raise ValueError(
            f"request temperature {request_temperature} does not match --rollout-temperature {configured_temperature}"
        )

    unsupported = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
    }
    for name, allowed_values in unsupported.items():
        if params.get(name) not in allowed_values:
            raise ValueError(
                f"{name} is not supported with sampling-support replay because "
                "the trainer cannot reproduce its logit transformation"
            )
    return True


def _sampling_mask_from_supports(
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
    support_logprobs: Sequence[Sequence[float]] | None = None,
) -> RolloutSamplingMask:
    if len(token_ids) != len(supports):
        raise ValueError(f"sampling support length {len(supports)} != token length {len(token_ids)}")
    if support_logprobs is not None and len(support_logprobs) != len(supports):
        raise ValueError("sampling support logprob rows must align with support rows")

    for row_index, (token_id, support) in enumerate(zip(token_ids, supports, strict=True)):
        if not support:
            raise ValueError("sampling support must contain at least one token")
        if int(token_id) not in support:
            raise ValueError(f"sampled token {token_id} is absent from its sampling support")
        if len(set(int(value) for value in support)) != len(support):
            raise ValueError("sampling support must not contain duplicate token ids")
        if support_logprobs is not None and len(support_logprobs[row_index]) != len(support):
            raise ValueError("sampling support logprobs must align with support ids")
    return RolloutSamplingMask.from_mask_list(supports, support_logprobs)


def append_sampling_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: dict,
    *,
    aborted: bool = False,
    require_support_logprobs: bool = False,
) -> list[float]:
    """Append SGLang's realized support and return its normalized log-probs."""
    supports = meta_info.get("output_token_sampling_mask")
    support_logprobs = meta_info.get("output_token_sampling_mask_logprobs")
    log_probs = meta_info.get("output_token_sampling_logprobs")
    if supports is None or log_probs is None:
        finish_reason = meta_info.get("finish_reason") or {}
        if (aborted or finish_reason.get("type") == "abort") and not output_token_ids:
            _append_sampling_mask(
                sample,
                RolloutSamplingMask.from_mask_list([], [] if require_support_logprobs else None),
            )
            return []
        raise ValueError(
            "SGLang response is missing output_token_sampling_mask or "
            "output_token_sampling_logprobs; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    if len(log_probs) != len(output_token_ids):
        raise ValueError(f"sampling log-prob length {len(log_probs)} != output token length {len(output_token_ids)}")
    if require_support_logprobs and support_logprobs is None:
        raise ValueError(
            "SGLang response is missing output_token_sampling_mask_logprobs; score centering with sampling replay requires normalized probabilities for the complete sampling support"
        )

    _append_sampling_mask(
        sample,
        _sampling_mask_from_supports(output_token_ids, supports, support_logprobs),
    )
    return [float(value) for value in log_probs]


def append_forced_sampling_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    """Record singleton support for non-sampled tokens inserted by the environment."""
    _, _, existing_logprobs = sample.rollout_sampling_mask._as_distribution_tensors()
    logprobs = [[0.0] for _ in token_ids] if existing_logprobs is not None else None
    sampling_mask = RolloutSamplingMask.from_mask_list(
        [[int(token_id)] for token_id in token_ids],
        logprobs,
    )
    _append_sampling_mask(sample, sampling_mask)


def merge_sampling_masks(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> RolloutSamplingMask | None:
    """Merge two per-response ragged masks with forced observation tokens between them."""
    first_mask = first.rollout_sampling_mask
    second_mask = second.rollout_sampling_mask
    if first_mask is None or second_mask is None:
        if first_mask is None and second_mask is None:
            return None
        raise ValueError("cannot merge samples unless both turns carry a complete rollout sampling mask")

    _, _, first_logprobs = first_mask._as_distribution_tensors()
    _, _, second_logprobs = second_mask._as_distribution_tensors()
    if (first_logprobs is None) != (second_logprobs is None):
        raise ValueError("cannot merge sampling masks with incomplete support logprobs")
    observation_logprobs = [[0.0] for _ in observation_token_ids] if first_logprobs is not None else None
    observation_mask = RolloutSamplingMask.from_mask_list(
        [[int(token_id)] for token_id in observation_token_ids],
        observation_logprobs,
    )
    return RolloutSamplingMask.concatenate((first_mask, observation_mask, second_mask))


def _append_sampling_mask(sample: Sample, sampling_mask: RolloutSamplingMask) -> None:
    if sample.rollout_sampling_mask is None:
        if sample.response_length != 0:
            raise ValueError("cannot initialize a sampling mask after response tokens have already been appended")
        sample.rollout_sampling_mask = sampling_mask
        return

    if len(sample.rollout_sampling_mask) != sample.response_length:
        raise ValueError(
            f"sampling mask length {len(sample.rollout_sampling_mask)} is not aligned with "
            f"response_length {sample.response_length} before appending"
        )
    sample.rollout_sampling_mask = RolloutSamplingMask.concatenate((sample.rollout_sampling_mask, sampling_mask))
