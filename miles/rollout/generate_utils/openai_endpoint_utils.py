"""Utilities for tracing session-server OpenAI requests."""

import asyncio
import itertools
import json
import logging
import time
from argparse import Namespace

import httpx

from miles.rollout.session.samples.codec import SamplesReply, decode_samples_and_merge_input_sample
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_CREATE_TIMEOUT = 30
_SESSION_REQUEST_TIMEOUT = 120
_SESSION_DELETE_TIMEOUT = 10
_PENDING_DELETES: set[asyncio.Task] = set()
_SESSION_SERVER_ROUND_ROBIN = itertools.count()


def _new_session_client() -> httpx.AsyncClient:
    """Return one session-owned client with no persistent idle connection."""
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=1,
            max_keepalive_connections=0,
        ),
        timeout=httpx.Timeout(
            connect=10,
            read=None,
            write=10,
            pool=10,
        ),
    )


async def _request_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    payload: dict | None,
    timeout: float,
) -> bytes:
    """Run one bounded session-control request and always close its stream."""
    async with asyncio.timeout(timeout):
        async with client.stream(method, url, json=payload) as response:
            content = await response.aread()
            if not response.is_success:
                detail = content.decode(errors="replace")[:1000]
                raise RuntimeError(
                    f"{method} {url} failed with {response.status_code}: {detail}"
                )
            return content


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        self._client = client or _new_session_client()
        self._retirement_task: asyncio.Task | None = None

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace):
        session_ip = getattr(args, "session_server_ip", None)
        session_ports = getattr(args, "session_server_ports", None)
        if not session_ip or not session_ports:
            raise RuntimeError("session_server_ip/session_server_ports are not set. Pass --use-session-server to start the session server.")
        # Choose the owner once. Every later request for this session reuses
        # the same URL, providing session affinity without another router.
        # Evenly spread long-lived sessions across the process pool. Random
        # choice produced persistent 2x shard skew at the experiment's
        # hundreds-of-session concurrency.
        session_port = session_ports[next(_SESSION_SERVER_ROUND_ROBIN) % len(session_ports)]
        session_url = f"http://{session_ip}:{session_port}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        client = _new_session_client()
        try:
            payload = await _request_bytes(
                client,
                "POST",
                f"{session_url}/sessions",
                payload={},
                timeout=_SESSION_CREATE_TIMEOUT,
            )
            response = json.loads(payload)
            session_id = response["session_id"]
            if not isinstance(session_id, str):
                raise TypeError("session_id must be a string")
            return OpenAIEndpointTracer(
                router_url=session_url,
                session_id=session_id,
                session_server_instance_id=instance_ids.get(session_port),
                client=client,
            )
        except BaseException:
            await client.aclose()
            raise

    async def collect_samples(self, input_sample: Sample, *, max_seq_len: int | None) -> SamplesReply:
        """Fetch server-assembled training samples for this session."""
        collect_started = time.monotonic()
        request_started = time.monotonic()
        try:
            payload = await _request_bytes(
                self._client,
                "POST",
                f"{self.base_url}/samples",
                payload={"max_seq_len": max_seq_len},
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            self._schedule_delete()

        request_seconds = time.monotonic() - request_started
        decode_started = time.monotonic()
        reply = await asyncio.to_thread(
            decode_samples_and_merge_input_sample,
            payload,
            input_sample,
        )
        reply.session_metadata.update(
            {
                "session_collect/response_bytes": len(payload),
                "session_collect/request_seconds": request_seconds,
                "session_collect/decode_seconds": time.monotonic() - decode_started,
                "session_collect/total_seconds": time.monotonic() - collect_started,
            }
        )
        return reply

    def _schedule_delete(self) -> asyncio.Task:
        if self._retirement_task is not None:
            return self._retirement_task

        task = asyncio.create_task(self._delete_and_close())
        self._retirement_task = task
        _PENDING_DELETES.add(task)

        def _done(completed: asyncio.Task) -> None:
            _PENDING_DELETES.discard(completed)

        task.add_done_callback(_done)
        return task

    async def discard_session(self) -> None:
        """Delete a cancelled session without collecting its sample."""
        await asyncio.shield(self._schedule_delete())

    async def _delete_and_close(self) -> bool:
        try:
            await _request_bytes(
                self._client,
                "DELETE",
                self.base_url,
                payload=None,
                timeout=_SESSION_DELETE_TIMEOUT,
            )
            return True
        except Exception as exc:
            logger.debug(
                "Failed to delete session %s: %s",
                self.session_id,
                exc,
            )
            return False
        finally:
            await self._client.aclose()
