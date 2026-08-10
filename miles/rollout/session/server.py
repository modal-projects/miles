"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared httpx client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import asyncio
import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI

from miles.rollout.generate_utils.rollout_request import RolloutRequestContext, prepare_rollout_request
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

# Request headers that must not be forwarded verbatim to the upstream backend.
_DROP_REQUEST_HEADERS = ("content-length", "transfer-encoding", "host")


def _transport_error_is_safe_to_retry(exc: httpx.TransportError) -> bool:
    """Whether httpx failed before a stateful request could reach the server."""
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.args = args
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "rollout_request_timeout_secs", None)
        if timeout is None:
            timeout = getattr(args, "miles_router_timeout", 600.0)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        # `in_place` weight updates keep the active request and its KV-backed
        # prefix, so the session may request only additional R3 rows.
        self.use_addition_r3 = getattr(args, "pause_generation_mode", None) == "in_place"
        setup_session_routes(self.app, self, args, use_addition_r3=self.use_addition_r3)

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}

        max_retries = 1
        retry_sleep = 1.0
        if request.session_id is not None and getattr(self.args, "custom_rollout_request_hook_path", None):
            prepared = await prepare_rollout_request(
                self.args,
                RolloutRequestContext(session_id=request.session_id),
                url=url,
                payload=json.loads(body),
                headers=headers,
            )
            url = prepared["url"]
            headers = prepared["headers"] or {}
            body = json.dumps(
                prepared["payload"], ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            max_retries = prepared["max_retries"]
            retry_sleep = prepared["retry_sleep"]

        response = None
        for attempt in range(max_retries):
            try:
                response = await self.client.request(request.method, url, content=body, headers=headers)
            except httpx.TransportError as exc:
                if not _transport_error_is_safe_to_retry(exc) or attempt + 1 == max_retries:
                    logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
                    error_body = json.dumps(
                        {"error": f"backend transport error: {type(exc).__name__}: {exc}"}
                    ).encode()
                    return {
                        "request_body": body,
                        "response_body": error_body,
                        "status_code": 502,
                        "headers": {"content-type": "application/json"},
                    }
            else:
                # Only retry responses that explicitly reject the request
                # before generation. A generic 5xx may be returned after the
                # stateful request was dispatched, so replaying it could
                # advance the same session twice.
                retryable = response.status_code in (409, 429)
                if not retryable or attempt + 1 == max_retries:
                    break
                await response.aread()
            await asyncio.sleep(retry_sleep)

        assert response is not None
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("session_server")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    # A long-horizon agent sends one HTTP request per model turn. Uvicorn's
    # default access logger therefore emits thousands of successful 200 lines,
    # obscuring lifecycle warnings and errors while duplicating the aggregate
    # request metrics collected by the rollout hooks. Keep application logging
    # at info, but disable per-request transport access logs.
    uvicorn.run(
        server.app,
        host=args.session_server_ip,
        port=args.session_server_port,
        log_level="info",
        access_log=False,
    )
