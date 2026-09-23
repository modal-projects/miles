import math
import os
from collections.abc import Mapping, Sequence

import numpy as np
import pybase64
import torch

from miles.utils.sampling_mask import sampling_support_replay_enabled
from miles.utils.score_centering import RolloutScoreCenteringHead, score_centering_enabled
from miles.utils.types import Sample


def score_centering_request_fields(
    args,
    sampling_params: Mapping[str, object],
    *,
    evaluation: bool,
    openai: bool = False,
) -> dict[str, object]:
    """Return opt-in SGLang fields after validating the behavior policy."""
    if evaluation or not score_centering_enabled(args):
        return {}

    temperature = float(_value_or_default(sampling_params, "temperature", args.rollout_temperature))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("score centering requires stochastic sampling with temperature > 0")
    if temperature != float(args.rollout_temperature):
        raise ValueError(
            f"request temperature {temperature} does not match --rollout-temperature {args.rollout_temperature}"
        )
    if int(_value_or_default(sampling_params, "beam_width", 1)) != 1:
        raise ValueError("score centering requires independently sampled actions and does not support beam search")

    if sampling_support_replay_enabled(args):
        return {"return_sampling_mask_logprobs": True}

    _validate_unreplayed_logit_transforms(sampling_params)
    top_p = float(_value_or_default(sampling_params, "top_p", 1.0))
    top_k = int(_value_or_default(sampling_params, "top_k", -1))
    min_p = float(_value_or_default(sampling_params, "min_p", 0.0))
    if (top_p, top_k, min_p) != (1.0, -1, 0.0):
        raise ValueError("score centering without sampling replay requires top_p=1, top_k=-1, and min_p=0")
    if temperature != 1.0 and os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError(
            "score centering requires temperature-scaled sampler logprobs; unset SGLANG_RETURN_ORIGINAL_LOGPROB"
        )

    top_k_field = "top_logprobs" if openai else "top_logprobs_num"
    return {
        top_k_field: int(args.score_centering_top_k),
        "return_flat_raw_output_top_logprobs": True,
        "return_flat_raw_output_top_logprobs_b64": True,
    }


def append_score_centering_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: Mapping[str, object],
    *,
    top_k: int,
) -> None:
    """Append a validated raw sampler head from one SGLang response."""
    count = len(output_token_ids)
    if count == 0:
        _append_head(sample, RolloutScoreCenteringHead(ids=[], offsets=[0], logprobs=[]))
        return

    if meta_info.get("output_top_logprobs_temperature_scaled") is not True:
        raise ValueError(
            "score centering requires temperature-scaled output top logprobs; "
            "use an SGLang build that reports output_top_logprobs_temperature_scaled=true"
        )
    shape = meta_info.get("output_top_logprobs_shape")
    if shape != [count, top_k]:
        raise ValueError(f"score-centering sampler head must have shape {[count, top_k]}, got {shape}")

    ids = _decode_flat_array(
        meta_info,
        plain_key="output_top_logprobs_idx_flat",
        encoded_key="output_top_logprobs_idx_flat_b64",
        dtype=np.dtype("<i4"),
    )
    logprobs = _decode_flat_array(
        meta_info,
        plain_key="output_top_logprobs_val_flat",
        encoded_key="output_top_logprobs_val_flat_b64",
        dtype=np.dtype("<f4"),
    )
    expected = count * top_k
    if ids.size != expected or logprobs.size != expected:
        raise ValueError(
            f"score-centering sampler head has {ids.size} ids and {logprobs.size} logprobs; expected {expected}"
        )
    offsets = torch.arange(0, expected + 1, top_k, dtype=torch.long)
    _append_head(
        sample,
        RolloutScoreCenteringHead(
            ids=torch.from_numpy(ids),
            offsets=offsets,
            logprobs=torch.from_numpy(logprobs),
        ),
    )


def append_forced_score_centering_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    head = RolloutScoreCenteringHead.from_rows(
        [[int(token_id)] for token_id in token_ids],
        [[0.0] for _ in token_ids],
    )
    _append_head(sample, head)


def merge_score_centering_heads(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> RolloutScoreCenteringHead | None:
    first_head = first.rollout_score_centering_head
    second_head = second.rollout_score_centering_head
    if first_head is None or second_head is None:
        if first_head is None and second_head is None:
            return None
        raise ValueError("cannot merge samples unless both turns carry complete score-centering heads")
    observation_head = RolloutScoreCenteringHead.from_rows(
        [[int(token_id)] for token_id in observation_token_ids],
        [[0.0] for _ in observation_token_ids],
    )
    return RolloutScoreCenteringHead.concatenate((first_head, observation_head, second_head))


def _validate_unreplayed_logit_transforms(sampling_params: Mapping[str, object]) -> None:
    defaults = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
    }
    for name, allowed in defaults.items():
        if sampling_params.get(name) not in allowed:
            raise ValueError(f"{name} is not supported with score centering")
    for name in (
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "response_format",
        "tools",
        "min_new_tokens",
        "min_tokens",
        "custom_logit_processor",
        "custom_params",
    ):
        if sampling_params.get(name):
            raise ValueError(f"constrained sampling ({name}) is not supported with score centering")


def _value_or_default(sampling_params: Mapping[str, object], name: str, default: object) -> object:
    value = sampling_params.get(name)
    return default if value is None else value


def _decode_flat_array(
    meta_info: Mapping[str, object],
    *,
    plain_key: str,
    encoded_key: str,
    dtype: np.dtype,
) -> np.ndarray:
    if encoded_key in meta_info:
        declared_dtype = meta_info.get(f"{encoded_key}_dtype")
        expected_dtype = "int32" if np.issubdtype(dtype, np.integer) else "float32"
        if declared_dtype != expected_dtype:
            raise ValueError(f"{encoded_key} declares dtype {declared_dtype!r}, expected {expected_dtype!r}")
        return np.frombuffer(pybase64.b64decode(meta_info[encoded_key]), dtype=dtype).copy()
    if plain_key in meta_info:
        return np.asarray(meta_info[plain_key], dtype=dtype)
    raise ValueError(f"SGLang response is missing {plain_key} or {encoded_key}")


def _append_head(sample: Sample, head: RolloutScoreCenteringHead) -> None:
    if sample.rollout_score_centering_head is None:
        if sample.response_length != 0:
            raise ValueError("cannot initialize score-centering data after response tokens were appended")
        sample.rollout_score_centering_head = head
        return
    if len(sample.rollout_score_centering_head) != sample.response_length:
        raise ValueError("score-centering head is not aligned with response_length before appending")
    sample.rollout_score_centering_head = RolloutScoreCenteringHead.concatenate(
        (sample.rollout_score_centering_head, head)
    )
