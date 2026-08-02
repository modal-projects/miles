import inspect
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from miles.utils.misc import load_function


@dataclass(frozen=True)
class RolloutRequestContext:
    session_id: str | None = None


async def prepare_rollout_request(
    args: Namespace,
    context: RolloutRequestContext,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = {
        "url": url,
        "payload": payload,
        "headers": headers,
        "max_retries": 60,
        "retry_sleep": 1.0,
    }
    hook_path = getattr(args, "custom_rollout_request_hook_path", None)
    if hook_path is None:
        return request

    result = load_function(hook_path)(args, context, request)
    if inspect.isawaitable(result):
        result = await result
    if result is not None:
        if not isinstance(result, dict):
            raise TypeError(f"{hook_path} must return None or a dict, got {type(result).__name__}")
        request.update(result)
    request["max_retries"] = int(request["max_retries"])
    request["retry_sleep"] = float(request["retry_sleep"])
    if request["max_retries"] < 1:
        raise ValueError("max_retries must be at least 1")
    if request["retry_sleep"] < 0:
        raise ValueError("retry_sleep must be non-negative")
    return request
