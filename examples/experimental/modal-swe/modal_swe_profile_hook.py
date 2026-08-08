"""Request hook used by the standalone Modal SWE throughput profiler."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


async def profile_rollout_request_hook(
    args: Namespace,
    context: Any,
    request: dict[str, Any],
) -> None:
    """Route a trajectory stickily without requiring a training weight version."""
    request["payload"]["weight_version"] = {
        "min_version": None,
        "exact_version": 0,
    }
    headers = dict(request.get("headers") or {})
    session_id = getattr(context, "session_id", None)
    if session_id is not None:
        headers[str(args.rollout_session_affinity_header)] = str(session_id)
    request["headers"] = headers
    request["max_retries"] = int(args.rollout_request_retry_attempts)
    request["retry_sleep"] = float(args.rollout_request_retry_sleep)
