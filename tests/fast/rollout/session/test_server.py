import json
from types import SimpleNamespace

import httpx
import pytest

from miles.rollout.generate_utils import rollout_request
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.server import SessionServer


@pytest.mark.asyncio
async def test_proxy_applies_request_hook_and_retries(monkeypatch):
    async def hook(_args, context, request):
        assert context.session_id == "session-1"
        request["payload"]["weight_version"] = {"min_version": 3}
        request["headers"] = {**request["headers"], "Modal-Session-ID": context.session_id}
        request["max_retries"] = 2
        request["retry_sleep"] = 0

    monkeypatch.setattr(rollout_request, "load_function", lambda _path: hook)
    requests = []

    async def handler(request):
        requests.append(request)
        status = 409 if len(requests) == 1 else 200
        return httpx.Response(status, json={"ok": True})

    server = SessionServer.__new__(SessionServer)
    server.args = SimpleNamespace(custom_rollout_request_hook_path="test.hook")
    server.backend_url = "https://rollout.example"
    server.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await server.do_proxy(
            ProxyRequest(method="POST", session_id="session-1"),
            "v1/chat/completions",
            body=json.dumps({"messages": []}).encode(),
            headers={"content-type": "application/json"},
        )
    finally:
        await server.client.aclose()

    assert result["status_code"] == 200
    assert len(requests) == 2
    assert json.loads(requests[1].content)["weight_version"] == {"min_version": 3}
    assert requests[1].headers["Modal-Session-ID"] == "session-1"


@pytest.mark.asyncio
async def test_proxy_does_not_retry_permanent_client_error(monkeypatch):
    def hook(_args, _context, request):
        request["max_retries"] = 10
        request["retry_sleep"] = 0

    monkeypatch.setattr(rollout_request, "load_function", lambda _path: hook)
    request_count = 0

    async def handler(_request):
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, json={"error": "invalid request"})

    server = SessionServer.__new__(SessionServer)
    server.args = SimpleNamespace(custom_rollout_request_hook_path="test.hook")
    server.backend_url = "https://rollout.example"
    server.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await server.do_proxy(
            ProxyRequest(method="POST", session_id="session-1"),
            "v1/chat/completions",
            body=b"{}",
            headers={"content-type": "application/json"},
        )
    finally:
        await server.client.aclose()

    assert result["status_code"] == 400
    assert request_count == 1


@pytest.mark.asyncio
async def test_proxy_does_not_retry_ambiguous_read_error(monkeypatch):
    def hook(_args, _context, request):
        request["max_retries"] = 10
        request["retry_sleep"] = 0

    monkeypatch.setattr(rollout_request, "load_function", lambda _path: hook)
    request_count = 0

    async def request(*_args, **_kwargs):
        nonlocal request_count
        request_count += 1
        raise httpx.ReadError("response was lost after request dispatch")

    server = SessionServer.__new__(SessionServer)
    server.args = SimpleNamespace(custom_rollout_request_hook_path="test.hook")
    server.backend_url = "https://rollout.example"
    server.client = SimpleNamespace(request=request)

    result = await server.do_proxy(
        ProxyRequest(method="POST", session_id="session-1"),
        "v1/chat/completions",
        body=b"{}",
        headers={"content-type": "application/json"},
    )

    assert result["status_code"] == 502
    assert request_count == 1
