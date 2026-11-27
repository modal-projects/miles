from miles.rollout.session.session_load import session_load
from miles.rollout.session.types import SessionRecord


def _turn(
    start: float, end: float, input_ids: list[int], output_ids: list[int], cached: int, engine: float = 0.0
) -> SessionRecord:
    meta_info = {
        "cached_tokens": cached,
        "output_token_logprobs": [(-0.1, token) for token in output_ids],
        "e2e_latency": engine,
    }
    return SessionRecord(
        timestamp=end,
        request_timestamp=start,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"input_ids": input_ids},
        response={"choices": [{"meta_info": meta_info}]},
    )


def test_a_turn_expects_the_whole_sequence_it_continues_cached():
    first = _turn(0.0, 2.0, [1, 2, 3], [4, 5], cached=2)
    second = _turn(5.0, 6.5, [1, 2, 3, 4, 5, 6, 7], [8], cached=3)

    assert session_load([(first, None), (second, first)]) == {
        "turn_seconds": 3.5,
        "engine_seconds": 0.0,
        "prefix_tokens": 5,
        "uncached_prefix_tokens": 2,
    }


def test_a_turn_that_parts_from_the_earlier_sequence_expects_only_what_they_share():
    earlier = _turn(0.0, 1.0, [1, 2, 3], [4, 5], cached=0)
    retried = _turn(1.0, 2.0, [1, 2, 3, 9], [6], cached=3)
    parted = _turn(2.0, 3.0, [1, 2, 7, 8, 9, 10], [6], cached=4)

    assert session_load([(retried, earlier), (parted, earlier)]) == {
        "turn_seconds": 2.0,
        "engine_seconds": 0.0,
        "prefix_tokens": 5,
        "uncached_prefix_tokens": 0,
    }


def test_a_turns_engine_time_is_what_sglang_timed_the_request_at():
    """The rest of a turn's time is spent between the session server and the engine's scheduler."""
    first = _turn(0.0, 2.0, [1, 2, 3], [4, 5], cached=0, engine=0.5)
    second = _turn(5.0, 6.5, [1, 2, 3, 4, 5, 6], [7], cached=5, engine=1.0)

    load = session_load([(first, None), (second, first)])

    assert (load["turn_seconds"], load["engine_seconds"]) == (3.5, 1.5)
