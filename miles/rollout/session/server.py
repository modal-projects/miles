"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared httpx client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import asyncio
import json
import logging
import time

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


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.args = args
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = (
            getattr(args, "rollout_request_timeout_secs", None)
            if getattr(args, "rollout_endpoint_url", None)
            else None
        )
        if timeout is None:
            timeout = getattr(args, "miles_router_timeout", 600.0)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(timeout),
        )
        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        setup_session_routes(self.app, self, args)

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
                prepared["payload"],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            max_retries = prepared["max_retries"]
            retry_sleep = prepared["retry_sleep"]

        backend_request_started = time.monotonic()
        response = None
        for attempt in range(max_retries):
            try:
                response = await self.client.request(request.method, url, content=body, headers=headers)
            except httpx.TransportError as exc:
                if attempt + 1 == max_retries:
                    backend_request_seconds = time.monotonic() - backend_request_started
                    logger.warning(
                        "Proxy transport error for %s %s: %s",
                        request.method,
                        path,
                        exc,
                    )
                    error_body = json.dumps(
                        {"error": f"backend transport error: {type(exc).__name__}: {exc}"}
                    ).encode()
                    return {
                        "request_body": body,
                        "response_body": error_body,
                        "status_code": 502,
                        "headers": {"content-type": "application/json"},
                        "backend_request_seconds": backend_request_seconds,
                    }
            else:
                retryable = response.status_code in (409, 429) or response.status_code >= 500
                if not retryable or attempt + 1 == max_retries:
                    break
                await response.aread()
                logger.info(
                    "Proxy request returned %s; retrying (%d/%d): %s",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    url,
                )
            await asyncio.sleep(retry_sleep)

        assert response is not None
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "backend_request_seconds": time.monotonic() - backend_request_started,
        }


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("session_server")
    # At agentic rollout rates, httpx's INFO record for every successful
    # upstream request overwhelms useful session/TITO diagnostics. Transport
    # failures are logged explicitly by ``do_proxy`` and remain visible.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(
        server.app,
        host=args.session_server_ip,
        port=args.session_server_port,
        log_level="info",
        access_log=False,
    )
