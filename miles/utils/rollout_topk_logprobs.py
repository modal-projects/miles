"""Sampling constraints for recording rollout top-k logprobs candidates."""

import math
from collections.abc import Mapping
from typing import Any


def validate_rollout_topk_logprobs_sampling(
    sampling: Mapping[str, Any], *, temperature: float, candidate_count: int
) -> None:
    """Require a bounded support covered by the recorded candidates.

    Filtered generation requests SGLang's post-filter support probabilities.
    The candidate count must cover the complete realized support.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Rollout top-k logprobs collection requires the same positive rollout temperature on every generation call"
        )
    top_p = sampling.get("top_p", 1.0)
    top_k = sampling.get("top_k", -1)
    if not 0 < top_p <= 1 or (top_k != -1 and not 0 < top_k <= candidate_count):
        raise ValueError("Rollout top-k logprobs collection requires top_p in (0, 1] and top_k=-1 or 1..k")
    if top_p < 1 and top_k == -1:
        raise ValueError(
            "Rollout top-k logprobs collection requires positive top_k with top_p filtering to bound the support"
        )
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Rollout top-k logprobs collection requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Rollout top-k logprobs collection does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Rollout top-k logprobs collection does not support strict tool schemas")
