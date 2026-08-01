"""Tests for the session-server OpenAI endpoint tracer."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import openai_endpoint_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import encode_samples
from miles.utils.types import Sample


async def _wait_for_deletes() -> None:
    if openai_endpoint_utils._PENDING_DELETES:
        await asyncio.gather(*list(openai_endpoint_utils._PENDING_DELETES))


class _FakeClient:
    def __init__(self):
        self.is_closed = False

    async def aclose(self):
        self.is_closed = True


@pytest.mark.asyncio
async def test_create_reads_session_server_instance_id_from_args(monkeypatch):
    calls = []
    client = _FakeClient()

    async def fake_request(client_arg, method, url, *, payload, timeout):
        assert client_arg is client
        calls.append((method, url, payload, timeout))
        return b'{"session_id":"session-123"}'

    monkeypatch.setattr(openai_endpoint_utils, "_new_session_client", lambda: client)
    monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)
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
            "POST",
            "http://127.0.0.1:12345/sessions",
            {},
            openai_endpoint_utils._SESSION_CREATE_TIMEOUT,
        )
    ]
    assert not client.is_closed


@pytest.mark.asyncio
async def test_create_without_instance_id_on_args(monkeypatch):
    async def fake_request(client, method, url, *, payload, timeout):
        return b'{"session_id":"session-123"}'

    monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
    )

    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.session_server_instance_id is None


@pytest.mark.asyncio
async def test_create_distributes_sessions_with_affinity(monkeypatch):
    calls = []

    async def fake_request(client, method, url, *, payload, timeout):
        calls.append((method, url))
        if url.endswith("/sessions"):
            return b'{"session_id":"session-id"}'
        if url.endswith("/samples"):
            return encode_samples([], {}, "no_records")
        return b""

    monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)
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
        assert calls == [
            ("POST", f"{prefix}/sessions"),
            ("POST", f"{tracer.base_url}/samples"),
            ("DELETE", tracer.base_url),
        ]
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
        self.client = _FakeClient()

        async def fake_request(client, method, url, *, payload, timeout):
            assert client is self.client
            self.calls.append(f"{method} {url}")
            if method == "DELETE":
                if isinstance(delete_outcome, Exception):
                    raise delete_outcome
                return b""
            assert payload == {"max_seq_len": 7}
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)


@pytest.mark.asyncio
async def test_collect_samples_reports_timing_and_deletes(monkeypatch):
    calls = _CollectCalls(
        monkeypatch,
        post_outcome=_computed_reply_payload(),
    )
    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345",
        session_id="sid-1",
        client=calls.client,
    )

    result = await tracer.collect_samples(Sample(), max_seq_len=7)
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
    assert calls.client.is_closed


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
    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345",
        session_id="sid-1",
        client=calls.client,
    )

    with pytest.raises(type(error)):
        await tracer.collect_samples(Sample(), max_seq_len=7)
    await _wait_for_deletes()

    assert calls.calls[-1] == (
        "DELETE http://127.0.0.1:12345/sessions/sid-1"
    )
    assert calls.client.is_closed


@pytest.mark.asyncio
async def test_discard_session_deletes_without_collecting(monkeypatch):
    calls = []
    client = _FakeClient()

    async def fake_request(client_arg, method, url, *, payload, timeout):
        assert client_arg is client
        calls.append((method, url, payload, timeout))
        return b""

    monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)
    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345",
        session_id="sid-1",
        client=client,
    )

    await tracer.discard_session()

    assert calls == [
        (
            "DELETE",
            tracer.base_url,
            None,
            openai_endpoint_utils._SESSION_DELETE_TIMEOUT,
        ),
    ]
    assert client.is_closed


@pytest.mark.asyncio
async def test_concurrent_request_timeouts_close_every_socket():
    request_count = 16
    disconnected_count = 0
    all_disconnected = asyncio.Event()

    async def hold_connection(reader, writer):
        nonlocal disconnected_count
        try:
            await reader.readuntil(b"\r\n\r\n")
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected_count += 1
            if disconnected_count == request_count:
                all_disconnected.set()

    server = await asyncio.start_server(hold_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    clients = [
        openai_endpoint_utils._new_session_client()
        for _ in range(request_count)
    ]
    try:
        requests = [
            openai_endpoint_utils._request_bytes(
                client,
                "POST",
                f"http://127.0.0.1:{port}/sessions/sid/samples",
                payload={},
                timeout=0.2,
            )
            for client in clients
        ]
        results = await asyncio.gather(*requests, return_exceptions=True)
        assert all(isinstance(result, TimeoutError) for result in results)
        await asyncio.wait_for(all_disconnected.wait(), timeout=1)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))
        server.close()
        await server.wait_closed()
