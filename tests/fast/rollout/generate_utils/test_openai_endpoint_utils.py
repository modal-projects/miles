"""Tests for session-server client transport and wire-version selection."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest

import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils import openai_endpoint_utils
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    SessionInfrastructureError,
)
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, COMPUTED_FIELDS_V2, encode_samples
from miles.utils.http_utils import post_bytes_no_retry
from miles.utils.types import Sample


async def _wait_for_deletes() -> None:
    if openai_endpoint_utils._PENDING_DELETES:
        await asyncio.gather(*list(openai_endpoint_utils._PENDING_DELETES))


class _SessionClient:
    def __init__(self):
        self.is_closed = False

    async def aclose(self):
        self.is_closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
async def test_request_bytes_classifies_transient_status_as_infrastructure(status_code):
    transport = httpx.MockTransport(lambda _request: httpx.Response(status_code, content=b"unavailable"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SessionInfrastructureError, match=str(status_code)):
            await openai_endpoint_utils._request_bytes(
                client,
                "POST",
                "http://session-server/sessions/sid/samples",
                payload={},
                timeout=1,
            )


@pytest.mark.asyncio
async def test_request_bytes_classifies_lost_existing_session_as_infrastructure():
    transport = httpx.MockTransport(lambda _request: httpx.Response(404, content=b"missing"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SessionInfrastructureError, match="404"):
            await openai_endpoint_utils._request_bytes(
                client,
                "POST",
                "http://session-server/sessions/sid/samples",
                payload={},
                timeout=1,
            )


@pytest.mark.asyncio
async def test_request_bytes_does_not_hide_protocol_failure():
    transport = httpx.MockTransport(lambda _request: httpx.Response(422, content=b"bad sample cursor"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="422.*bad sample cursor"):
            await openai_endpoint_utils._request_bytes(
                client,
                "POST",
                "http://session-server/sessions/sid/samples",
                payload={},
                timeout=1,
            )


@pytest.mark.asyncio
async def test_request_bytes_wraps_transport_failure():
    def fail(_request):
        raise httpx.ReadError("connection closed")

    transport = httpx.MockTransport(fail)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SessionInfrastructureError, match="connection closed"):
            await openai_endpoint_utils._request_bytes(
                client,
                "POST",
                "http://session-server/sessions/sid/samples",
                payload={},
                timeout=1,
            )


@pytest.mark.asyncio
async def test_create_reads_instance_id_and_selects_v2_wire_fields(monkeypatch):
    calls = []
    client = _SessionClient()

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
        use_session_server="v2",
    )

    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.base_url == "http://127.0.0.1:12345/sessions/session-123"
    assert tracer.session_server_instance_id == "server-instance-123"
    assert tracer.samples_wire_fields == COMPUTED_FIELDS_V2
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:12345/sessions",
            {},
            openai_endpoint_utils._SESSION_CREATE_TIMEOUT,
        )
    ]
    assert not client.is_closed
    await tracer.discard_session()
    await _wait_for_deletes()
    assert client.is_closed


@pytest.mark.asyncio
async def test_create_selects_v1_wire_fields(monkeypatch):
    async def fake_request(client, method, url, *, payload, timeout):
        return b'{"session_id":"session-123"}'

    monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        use_session_server=True,
    )

    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.samples_wire_fields == COMPUTED_FIELDS
    await tracer.discard_session()
    await _wait_for_deletes()


@pytest.mark.asyncio
async def test_create_distributes_sessions_round_robin_with_affinity(monkeypatch):
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
    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=ports)
    chosen = Counter()

    for _ in range(8):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        chosen[port] += 1
        await tracer.collect_samples(Sample(), max_seq_len=None)
        await _wait_for_deletes()

        prefix = f"http://127.0.0.1:{port}"
        assert calls == [
            ("POST", f"{prefix}/sessions"),
            ("POST", f"{tracer.base_url}/samples"),
            ("DELETE", tracer.base_url),
        ]

    assert chosen == Counter({port: 2 for port in ports})


def _computed_reply_payload(*, fields=COMPUTED_FIELDS) -> bytes:
    sample = Sample(
        tokens=[1, 2, 10],
        response="r",
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.5],
        status=Sample.Status.COMPLETED,
    )
    return encode_samples([sample], {"max_trim_tokens": 1}, None, fields=fields)


class _CollectCalls:
    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        self.calls = []
        self.client = _SessionClient()

        async def fake_request(client, method, url, *, payload, timeout):
            assert client is self.client
            self.calls.append((method, url, payload))
            if method == "DELETE":
                if isinstance(delete_outcome, Exception):
                    raise delete_outcome
                return b""
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        monkeypatch.setattr(openai_endpoint_utils, "_request_bytes", fake_request)

    def tracer(self, *, fields=COMPUTED_FIELDS):
        return OpenAIEndpointTracer(
            "http://127.0.0.1:12345",
            "sid-1",
            samples_wire_fields=fields,
            client=self.client,
        )


@pytest.mark.asyncio
async def test_collect_reports_timing_and_retires_session(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())

    result = await calls.tracer().collect_samples(Sample(), max_seq_len=7)
    await _wait_for_deletes()

    assert [call[0] for call in calls.calls] == ["POST", "DELETE"]
    assert calls.calls[0][2] == {"max_seq_len": 7}
    assert len(result.samples) == 1
    assert result.session_metadata["session_collect/response_bytes"] > 0
    assert result.session_metadata["session_collect/request_seconds"] >= 0
    assert result.session_metadata["session_collect/decode_seconds"] >= 0
    assert result.session_metadata["session_collect/total_seconds"] >= 0
    assert calls.client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("bad samples"), asyncio.TimeoutError()])
async def test_collect_error_still_retires_session(monkeypatch, error):
    calls = _CollectCalls(monkeypatch, post_outcome=error)

    with pytest.raises(type(error)):
        await calls.tracer().collect_samples(Sample(), max_seq_len=7)
    await _wait_for_deletes()

    assert calls.calls[-1][0] == "DELETE"
    assert calls.client.is_closed


@pytest.mark.asyncio
async def test_collect_v2_carries_agent_metadata_and_decodes_extras(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 10],
        response="r",
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.5],
        status=Sample.Status.COMPLETED,
        reward=0.75,
        metadata={"leaf": {"node_id": 1}},
    )
    payload = encode_samples([sample], {}, None, fields=COMPUTED_FIELDS_V2)
    calls = _CollectCalls(monkeypatch, post_outcome=payload)
    input_sample = Sample(metadata={"env": "keep-me"})

    result = await calls.tracer(fields=COMPUTED_FIELDS_V2).collect_samples(
        input_sample,
        max_seq_len=7,
        agent_metadata={"reward": 0.75},
    )
    await _wait_for_deletes()

    assert calls.calls[0][2] == {"max_seq_len": 7, "metadata": {"reward": 0.75}}
    [decoded] = result.samples
    assert decoded.reward == 0.75
    assert decoded.metadata == {"env": "keep-me", "leaf": {"node_id": 1}}


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


class _FakeGlobalClient:
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
    client = _FakeGlobalClient([_FakeResponse(200, content=b"\x00\x01binary")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    assert await post_bytes_no_retry("http://x/samples", {}, timeout=5) == b"\x00\x01binary"
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_does_not_retry_and_carries_body(monkeypatch):
    client = _FakeGlobalClient([_FakeResponse(422, text="cursor mismatch"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(RuntimeError, match="422.*cursor mismatch"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_transport_error_propagates_once(monkeypatch):
    client = _FakeGlobalClient([ConnectionError("boom"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(ConnectionError, match="boom"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1
