from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["weight-update"])

import asyncio
import functools
import json

import httpx
import pytest

from miles.ray.rollout import external_rollout
from miles.ray.rollout.external_rollout import ExternalRollout
from miles.utils import http_utils


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):
    """Replies scripted per path, as (status, body or Location); records each request."""
    replies: dict[str, list] = {}
    requests: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body, request.headers.get("authorization")))
        status, reply = replies[request.url.path].pop(0)
        if status == 303:
            return httpx.Response(303, headers={"location": reply})
        return httpx.Response(status, json=reply)

    async def no_wait(_):
        pass

    monkeypatch.setattr(
        external_rollout.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(http_utils.asyncio, "sleep", no_wait)
    monkeypatch.setattr(external_rollout, "_STATUS_POLL_S", 0)
    return replies, requests


def _serve(rollout: ExternalRollout, *versions: int) -> dict:
    async def serve_each():
        for version in versions:
            metrics = await rollout.serve(version)
        return metrics

    return asyncio.run(serve_each())


def test_the_first_version_starts_the_run_and_waits_until_an_engine_serves_it(service, monkeypatch):
    replies, requests = service
    monkeypatch.setenv("MILES_ROLLOUT_EXTERNAL_API_KEY", "key")
    replies.update(
        {
            "/start": [(200, {"start_id": "s1"})],
            "/status": [(200, {"serving": {}}), (200, {"serving": {"2": 0}}), (200, {"serving": {"3": 1}})],
        }
    )

    assert _serve(ExternalRollout("https://control/"), 3) == {}

    assert (
        requests == [("POST", "/start", {"version": 3}, "Bearer key")] + [("GET", "/status", None, "Bearer key")] * 3
    )


def test_each_later_version_names_the_start_and_reports_the_wait_and_the_slowest_gpus(service):
    replies, requests = service
    engines = {
        "e0": {"stats": {"prepare": [{"read_s": 1.0}, {"read_s": 2.0}], "apply": [{"gpu_s": 0.1}]}},
        "e1": {"stats": None},
    }
    replies.update(
        {
            "/start": [(200, {"start_id": "s1"})],
            "/status": [(200, {"serving": {"3": 1}})],
            "/versions": [(200, {"seconds": 2.5, "engines": engines})],
        }
    )

    assert _serve(ExternalRollout("https://control"), 3, 4) == {
        "perf/update_weights_external_s": 2.5,
        "perf/update_weights_engine_prepare_read_s": 2.0,
        "perf/update_weights_engine_apply_gpu_s": 0.1,
    }
    assert requests[-1][:3] == ("POST", "/versions", {"version": 4, "base_version": 3, "start": 3, "start_id": "s1"})


def test_a_call_the_platform_lost_is_sent_again_and_a_refusal_stops_the_run(service):
    replies, requests = service
    replies.update(
        {
            "/start": [(200, {"start_id": "s1"})],
            "/status": [(200, {"serving": {"0": 1}})],
            "/versions": [(503, {}), (200, {"seconds": 1, "engines": {}}), (409, {})],
        }
    )

    with pytest.raises(httpx.HTTPStatusError, match="409"):
        _serve(ExternalRollout("https://control"), 0, 1, 2)
    assert [path for _, path, _, _ in requests] == ["/start", "/status", "/versions", "/versions", "/versions"]
    # The call sent again names the same base; the next version follows the one served.
    assert [(body["version"], body["base_version"]) for _, _, body, _ in requests[2:]] == [(1, 0), (1, 0), (2, 1)]


def test_a_reply_slower_than_its_endpoint_allows_is_followed_through_its_redirect(service):
    replies, requests = service
    replies.update(
        {
            "/start": [(200, {"start_id": "s1"})],
            "/status": [(200, {"serving": {"0": 1}})],
            "/versions": [(303, "/result")],
            "/result": [(200, {"seconds": 160, "engines": {}})],
        }
    )

    assert _serve(ExternalRollout("https://control"), 0, 1)["perf/update_weights_external_s"] == 160
    assert [(method, path) for method, path, _, _ in requests][2:] == [("POST", "/versions"), ("GET", "/result")]
