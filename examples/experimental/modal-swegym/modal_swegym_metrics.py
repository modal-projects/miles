"""SWE-Gym-specific rollout metrics.

The fully-async scheduler reports only scheduling and staleness. This hook
aggregates environment/tool/verifier fields owned by the Modal adapter.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from miles.utils.types import Sample

_AGENT_MEAN_ONLY_METRICS = {
    "agent_tool_input_over_64k_count",
    "agent_tool_input_over_64k_ratio",
    "agent_tool_output_hard_limit_count",
    "agent_tool_output_hard_limit_ratio",
    "agent_tool_output_truncated_count",
    "agent_tool_output_truncated_ratio",
    "context_limit_exceeded",
    "generation_bound",
    "infra_error",
    "policy_failure",
    "model_request_count",
    "tool_calls",
    "tool_timeout_count",
    "turns",
    "verifier_reward_missing",
    "verifier_timeout",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _summary(metrics: dict[str, Any], prefix: str, values: list[float]) -> None:
    if not values:
        return
    metrics[f"{prefix}_mean"] = sum(values) / len(values)
    metrics[f"{prefix}_p90"] = _percentile(values, 0.90)
    metrics[f"{prefix}_max"] = max(values)


def _numeric_agent_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    agent_metrics = [sample.metadata.get("agent_metrics") or {} for sample in samples]
    keys = sorted(
        {
            key
            for metrics in agent_metrics
            for key, value in metrics.items()
            if key != "agent_worker_index"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    for key in keys:
        values = [float(metrics[key]) for metrics in agent_metrics if isinstance(metrics.get(key), (int, float))]
        if key in _AGENT_MEAN_ONLY_METRICS:
            output[f"rollout_agent/{key}_mean"] = sum(values) / len(values)
        else:
            _summary(output, f"rollout_agent/{key}", values)


def _summarize_sample_metadata(
    samples: list[Sample],
    output: dict[str, Any],
    *,
    metadata_prefix: str,
    metric_prefix: str,
) -> None:
    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.metadata.items()
            if key.startswith(metadata_prefix)
            and isinstance(value, (int, float))
        }
    )
    for key in keys:
        values = [
            float(sample.metadata[key])
            for sample in samples
            if isinstance(sample.metadata.get(key), (int, float))
        ]
        _summary(
            output,
            f"{metric_prefix}/{key.removeprefix(metadata_prefix)}",
            values,
        )


def _request_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    backend = [
        float(duration)
        for sample in samples
        for duration in sample.metadata.get("model_request/durations_seconds", [])
        if isinstance(duration, (int, float))
    ]
    client = [
        float(duration)
        for sample in samples
        for duration in (sample.metadata.get("agent_metrics") or {}).get(
            "client_model_request_durations_seconds",
            [],
        )
        if isinstance(duration, (int, float))
    ]
    _summary(output, "rollout_model/request_latency_seconds", backend)
    _summary(output, "rollout_model/client_request_latency_seconds", client)

    backend_seconds = sum(backend)
    client_seconds = sum(client)
    completion_tokens = sum(int(sample.metadata.get("model_request/completion_tokens", 0)) for sample in samples)
    prompt_tokens = sum(int(sample.metadata.get("model_request/prompt_tokens", 0)) for sample in samples)
    output.update(
        {
            "rollout_model/request_count": len(backend),
            "rollout_model/request_total_seconds": backend_seconds,
            "rollout_model/client_request_count": len(client),
            "rollout_model/client_request_total_seconds": client_seconds,
            "rollout_model/client_minus_backend_seconds_signed": client_seconds - backend_seconds,
            "rollout_model/client_minus_backend_request_count": len(client) - len(backend),
            "rollout_model/prompt_tokens": prompt_tokens,
            "rollout_model/completion_tokens": completion_tokens,
            "rollout_model/completion_tokens_per_backend_request_second": (completion_tokens / backend_seconds if backend_seconds else 0.0),
            "rollout_model/non_200_count": sum(int(sample.metadata.get("model_request/non_200_count", 0)) for sample in samples),
        }
    )


def add_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    samples = [sample for sample in samples if not sample.remove_sample]
    if not samples:
        return
    _numeric_agent_metrics(samples, output)
    _summarize_sample_metadata(
        samples,
        output,
        metadata_prefix="session_collect/",
        metric_prefix="rollout_session",
    )
    _request_metrics(samples, output)

    known_statuses = {
        "Submitted",
        "LimitsExceeded",
        "completed",
        "command_timeout",
        "verifier_timeout",
        "verifier_infra_error",
        "sandbox_not_found",
        "agent_error",
        "sandbox_infra_error",
        "session_record_timeout",
        "session_record_request_error",
        "session_sample_collection_error",
        "session_create_error",
        "agent_function_exception",
        "no_model_calls",
        "prompt_exceeds_max_seq_len",
        "unknown",
    }
    statuses = Counter(
        status if status in known_statuses else "other"
        for sample in samples
        for status in [str(sample.metadata.get("exit_status", "unknown"))]
    )
    for status, count in statuses.items():
        safe_status = status.replace("/", "_").replace(" ", "_")
        output[f"rollout_agent/exit_status/{safe_status}_ratio"] = count / len(samples)

    agent_metrics = [sample.metadata.get("agent_metrics") or {} for sample in samples]
    verifier_return_codes = [
        int(metrics["verifier_return_code"])
        for metrics in agent_metrics
        if isinstance(metrics.get("verifier_return_code"), (int, float))
    ]
    if verifier_return_codes:
        output["rollout_agent/verifier_nonzero_return_code_ratio"] = (
            sum(code != 0 for code in verifier_return_codes)
            / len(verifier_return_codes)
        )
    context_limits = sum(bool(metrics.get("context_limit_exceeded")) for metrics in agent_metrics)
    output["rollout_agent/context_limit_exit_ratio"] = context_limits / len(samples)
    output["rollout_agent/step_limit_exit_ratio"] = max(
        0,
        statuses["LimitsExceeded"] - context_limits,
    ) / len(samples)


def log_rollout_data(
    rollout_id: int,
    args,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Extend the standard Miles rollout log; returning False preserves it."""
    del rollout_id, args, rollout_time
    if rollout_extra_metrics is not None and samples:
        add_metrics(samples, rollout_extra_metrics)
        # Exact vectors are needed only for the aggregate above. Do not carry
        # hundreds of per-turn floats per trajectory into trainer object-store
        # payloads after their p50/p90/max/count totals have been recorded.
        for sample in samples:
            sample.metadata.pop("model_request/durations_seconds", None)
            agent_metrics = sample.metadata.get("agent_metrics")
            if isinstance(agent_metrics, dict):
                agent_metrics.pop(
                    "client_model_request_durations_seconds",
                    None,
                )
    return False
