import ast
import asyncio
import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def proxy():
    path = Path(__file__).with_name("server.py")
    tree = ast.parse(path.read_text())
    namespace = {
        "asyncio": asyncio,
        "inspect": inspect,
        "json": json,
        "httpx": httpx,
        "logger": logging.getLogger(__name__),
        "ProxyRequest": object,
        "RolloutRequestContext": SimpleNamespace,
        "_DROP_REQUEST_HEADERS": ("content-length", "transfer-encoding", "host"),
    }
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    server = namespace["SessionServer"].__new__(namespace["SessionServer"])
    server.args = SimpleNamespace(custom_rollout_request_hook_path="test.hook", rollout_request_timeout_secs=1)
    server.backend_url = "http://backend.invalid"
    options = {"max_retries": 3, "retry_sleep": 0}

    async def prepare(*args, **kwargs):
        return {**kwargs, **options}

    namespace["prepare_rollout_request"] = prepare
    request = SimpleNamespace(method="POST", query="mode=test", session_id="session-a")

    async def run(transport):
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            server.client = client
            return await server.do_proxy(
                request, "generate", body=b'{"input_ids":[1,2,3]}', headers={"x-session": "session-a"}
            )

    return SimpleNamespace(server=server, options=options, run=run)


@pytest.mark.parametrize("async_classifier", [False, True])
def test_retry_classifier_preserves_request_and_attempt_bound(proxy, async_classifier):
    requests = []

    def accepted(response):
        return response.status_code == 503 and response.text == "rejected before dispatch"

    async def accepted_async(response):
        return accepted(response)

    proxy.options["retry_response"] = accepted_async if async_classifier else accepted

    async def transport(request):
        requests.append((str(request.url), request.content, dict(request.headers)))
        return httpx.Response(503, text="rejected before dispatch")

    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == 503
    assert len(requests) == 3
    assert requests == [requests[0]] * 3


@pytest.mark.parametrize("status", [502, 503])
def test_async_false_does_not_retry_an_ambiguous_response(proxy, status):
    calls = []

    async def rejected(response):
        return False

    async def transport(request):
        calls.append(request)
        return httpx.Response(status, text="generation failed")

    proxy.options["retry_response"] = rejected
    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == status
    assert len(calls) == 1


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ReadError, httpx.WriteError])
def test_never_replays_ambiguous_transport_failure(proxy, error):
    calls = []

    async def transport(request):
        calls.append(request)
        raise error("request may have been dispatched")

    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == 502
    assert len(calls) == 1


@pytest.mark.parametrize("timeout", [0.01, 0, -1])
def test_total_deadline_cancels_dispatch_without_retry(proxy, timeout):
    proxy.server.args.rollout_request_timeout_secs = timeout
    events = []

    async def transport(request):
        events.append("dispatch")
        try:
            await asyncio.sleep(10)
        finally:
            events.append("cancelled")

    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == 502
    assert events == ["dispatch", "cancelled"]


def test_retries_share_the_total_deadline(proxy):
    proxy.server.args.rollout_request_timeout_secs = 0.01
    proxy.options.update(retry_sleep=1, retry_response=lambda response: True)
    calls = []

    async def transport(request):
        calls.append(request)
        return httpx.Response(503, text="rejected before dispatch")

    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == 502
    assert len(calls) == 1


def test_non_boolean_classifier_fails_instead_of_replaying(proxy):
    proxy.options["retry_response"] = lambda response: "false"
    with pytest.raises(TypeError, match="bool"):
        asyncio.run(proxy.run(lambda request: httpx.Response(502)))


@pytest.mark.parametrize("async_classifier", [False, True])
def test_classifier_timeout_is_not_reported_as_the_request_deadline(proxy, async_classifier):
    def classify(response):
        raise TimeoutError("classifier timed out")

    async def classify_async(response):
        return classify(response)

    proxy.options["retry_response"] = classify_async if async_classifier else classify
    with pytest.raises(TimeoutError, match="classifier timed out"):
        asyncio.run(proxy.run(lambda request: httpx.Response(503)))


def test_outer_deadline_cancels_without_becoming_a_proxy_response(proxy):
    events = []

    async def transport(request):
        events.append("dispatch")
        try:
            await asyncio.sleep(10)
        finally:
            events.append("cancelled")

    async def run():
        async with asyncio.timeout(0.01):
            return await proxy.run(transport)

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    assert events == ["dispatch", "cancelled"]


def test_none_disables_the_total_deadline(proxy):
    proxy.server.args.rollout_request_timeout_secs = None

    async def transport(request):
        await asyncio.sleep(0.01)
        return httpx.Response(200, text="completed")

    result = asyncio.run(proxy.run(transport))
    assert result["status_code"] == 200
    assert result["response_body"] == b"completed"
