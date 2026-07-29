"""Tests for the session-server OpenAI endpoint tracer."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils import openai_endpoint_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import encode_samples
from miles.utils.http_utils import post_bytes_no_retry
from miles.utils.types import Sample


async def _wait_for_deletes() -> None:
    if openai_endpoint_utils._PENDING_DELETES:
        await asyncio.gather(*list(openai_endpoint_utils._PENDING_DELETES))


@pytest.mark.asyncio
async def test_create_reads_session_server_instance_id_from_args(monkeypatch):
    calls = []

    async def fake_post(url, payload, action="post", **kwargs):
        calls.append((action, url, kwargs))
        return {"session_id": "session-123"}

    monkeypatch.setattr(openai_endpoint_utils, "post", fake_post)
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        session_server_instance_ids={12345: "server-instance-123"},
    )

    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.base_url == "http://127.0.0.1:12345/sessions/session-123"
    assert tracer.session_server_id == "127.0.0.1:12345"
    assert tracer.session_server_instance_id == "server-instance-123"
    assert calls == [
        (
            "post",
            "http://127.0.0.1:12345/sessions",
            {"max_retries": 1},
        )
    ]


@pytest.mark.asyncio
async def test_create_without_instance_id_on_args(monkeypatch):
    async def fake_post(url, payload, action="post", **kwargs):
        return {"session_id": "session-123"}

    monkeypatch.setattr(openai_endpoint_utils, "post", fake_post)
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
    )

    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.session_server_instance_id is None


@pytest.mark.asyncio
async def test_create_distributes_sessions_with_affinity(monkeypatch):
    calls = []

    async def fake_post(url, payload, action="post", **kwargs):
        calls.append((action, url))
        if action == "post":
            return {"session_id": f"session-{len(calls)}"}
        return {}

    async def fake_post_bytes(url, payload, *, timeout):
        calls.append(("post_bytes", url))
        return encode_samples([], {}, "no_records")

    monkeypatch.setattr(openai_endpoint_utils, "post", fake_post)
    monkeypatch.setattr(
        openai_endpoint_utils,
        "post_bytes_no_retry",
        fake_post_bytes,
    )
    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=ports,
    )
    chosen_ports = Counter()

    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        chosen_ports[port] += 1

        await tracer.collect_samples(Sample(), max_seq_len=None)
        await _wait_for_deletes()

        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            f"{tracer.base_url}/samples",
            tracer.base_url,
        ]

    assert chosen_ports == Counter({port: 8 for port in ports})


def _tracer() -> OpenAIEndpointTracer:
    return OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345",
        session_id="sid-1",
    )


def _computed_reply_payload() -> bytes:
    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    return encode_samples([sample], {"max_trim_tokens": 1})


class _CollectCalls:
    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        self.calls: list[str] = []

        async def fake_post_bytes(url, payload, *, timeout):
            self.calls.append(f"POST {url}")
            assert payload == {"max_seq_len": 7}
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        async def fake_post(url, payload, action="post", **kwargs):
            self.calls.append(f"DELETE {url}")
            if isinstance(delete_outcome, Exception):
                raise delete_outcome
            return {}

        monkeypatch.setattr(
            openai_endpoint_utils,
            "post_bytes_no_retry",
            fake_post_bytes,
        )
        monkeypatch.setattr(openai_endpoint_utils, "post", fake_post)


@pytest.mark.asyncio
async def test_collect_samples_reports_timing_and_deletes(monkeypatch):
    calls = _CollectCalls(
        monkeypatch,
        post_outcome=_computed_reply_payload(),
    )

    result = await _tracer().collect_samples(Sample(), max_seq_len=7)
    await _wait_for_deletes()

    assert calls.calls == [
        "POST http://127.0.0.1:12345/sessions/sid-1/samples",
        "DELETE http://127.0.0.1:12345/sessions/sid-1",
    ]
    (sample,) = result.samples
    assert sample.tokens == [1, 2, 10]
    assert sample.status == Sample.Status.COMPLETED
    assert result.session_metadata["max_trim_tokens"] == 1
    assert result.session_metadata["session_collect/response_bytes"] > 0
    assert result.session_metadata["session_collect/request_seconds"] >= 0
    assert result.session_metadata["session_collect/decode_seconds"] >= 0
    assert result.session_metadata["session_collect/total_seconds"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("422: trim_count 2 exceeds allowed=1"),
        asyncio.TimeoutError(),
    ],
)
async def test_collect_samples_error_still_deletes(monkeypatch, error):
    calls = _CollectCalls(monkeypatch, post_outcome=error)

    with pytest.raises(type(error)):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    await _wait_for_deletes()

    assert calls.calls[-1] == (
        "DELETE http://127.0.0.1:12345/sessions/sid-1"
    )


@pytest.mark.asyncio
async def test_discard_session_deletes_without_collecting(monkeypatch):
    calls = []

    async def fake_post(url, payload, action="post", **kwargs):
        calls.append((action, url, kwargs))
        return {}

    monkeypatch.setattr(openai_endpoint_utils, "post", fake_post)
    tracer = _tracer()

    await tracer.discard_session()

    assert calls == [
        ("delete", tracer.base_url, {"max_retries": 1}),
    ]


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        text: str = "",
    ):
        self.status_code = status_code
        self.content = content
        self.text = text


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_count = 0

    async def post(self, url, json=None):
        self.post_count += 1
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_post_bytes_no_retry_returns_raw_bytes(monkeypatch):
    client = _FakeClient([_FakeResponse(200, content=b"\x00\x01binary")])
    monkeypatch.setattr(http_utils, "_http_client", client)

    result = await post_bytes_no_retry("http://x/samples", {}, timeout=5)

    assert result == b"\x00\x01binary"
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_does_not_retry(monkeypatch):
    client = _FakeClient(
        [
            _FakeResponse(422, text="cursor 3 != len(accumulated) 4"),
            RuntimeError("late"),
        ]
    )
    monkeypatch.setattr(http_utils, "_http_client", client)

    with pytest.raises(RuntimeError, match="422.*cursor 3"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)

    assert client.post_count == 1
