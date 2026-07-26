import time
from types import SimpleNamespace

import pytest
import requests


@pytest.mark.parametrize(
    ("target_version", "expected"),
    [
        (
            0,
            {
                "local_checkpoint_dir": "/local",
                "base_checkpoint_dir": "/base",
                "target_version": 0,
                "destination": "disk",
            },
        ),
        (
            3,
            {
                "local_checkpoint_dir": "/local",
                "base_checkpoint_dir": "/base",
                "checkpoint_source_dir": "/deltas",
                "target_version": 3,
                "destination": "disk",
            },
        ),
    ],
)
def test_stage_weight_update_request(target_version, expected):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = SimpleNamespace(
        hf_checkpoint="/base",
        update_weight_disk_dir="/deltas",
        update_weight_local_checkpoint_dir="/local",
    )
    calls = []
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.stage_weight_update(target_version)

    assert calls == [("stage_weight_update", expected)]


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )
