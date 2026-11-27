"""What a session's turns asked of the inference engines, for an external rollout service sizing them."""

from collections.abc import Iterable

from miles.rollout.session.types import SessionRecord

SESSION_LOAD_KEY = "session_load"


def session_load(turns: Iterable[tuple[SessionRecord, SessionRecord | None]]) -> dict[str, float]:
    """Each turn comes with the turn it continues, None for a session's first."""
    turn_seconds = engine_seconds = 0.0
    prefix_tokens = uncached_prefix_tokens = 0
    for record, previous in turns:
        meta_info = record.response["choices"][0]["meta_info"]
        if record.request_timestamp is not None:
            turn_seconds += record.timestamp - record.request_timestamp
            # SGLang's own time for the request, from its tokenizer receiving it to the finished output.
            engine_seconds += meta_info.get("e2e_latency", 0.0)
        if previous is None:
            continue
        prefix = _common_prefix(_token_ids(previous), record.request["input_ids"])
        prefix_tokens += prefix
        uncached_prefix_tokens += max(0, prefix - meta_info.get("cached_tokens", 0))
    return {
        "turn_seconds": turn_seconds,
        "engine_seconds": engine_seconds,
        "prefix_tokens": prefix_tokens,
        "uncached_prefix_tokens": uncached_prefix_tokens,
    }


def _token_ids(record: SessionRecord) -> list[int]:
    output = record.response["choices"][0]["meta_info"]["output_token_logprobs"]
    return record.request["input_ids"] + [item[1] for item in output]


def _common_prefix(a: list[int], b: list[int]) -> int:
    # Slices compare in C. A turn usually starts with the whole earlier sequence; else a binary search finds the split.
    shorter = min(len(a), len(b))
    if a[:shorter] == b[:shorter]:
        return shorter
    low, high = 0, shorter - 1
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low
