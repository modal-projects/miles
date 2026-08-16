"""Utilities for tracing session-server OpenAI requests."""

import asyncio
import itertools
import json
import logging
from argparse import Namespace

import httpx

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_CREATE_TIMEOUT = 30
_SESSION_REQUEST_TIMEOUT = 120
_SESSION_DELETE_TIMEOUT = 10
_PENDING_DELETES: set[asyncio.Task] = set()
_SESSION_SERVER_ROUND_ROBIN = itertools.count()


def _new_session_client() -> httpx.AsyncClient:
    """Create a session-control client whose socket cannot leak into a pool."""
    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10),
    )


async def _request_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    payload: dict | None,
    timeout: float,
) -> bytes:
    """Run one bounded control request and always close its response stream."""
    async with asyncio.timeout(timeout):
        async with client.stream(method, url, json=payload) as response:
            content = await response.aread()
            if not response.is_success:
                detail = content.decode(errors="replace")[:1000]
                raise RuntimeError(f"{method} {url} failed with {response.status_code}: {detail}")
            return content


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        self.samples_wire_fields = samples_wire_fields
        self._client = client or _new_session_client()
        self._retirement_task: asyncio.Task | None = None

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance that owns this session."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace):
        session_ip = getattr(args, "session_server_ip", None)
        session_ports = getattr(args, "session_server_ports", None)
        if not session_ip or not session_ports:
            raise RuntimeError(
                "session_server_ip/session_server_ports are not set. "
                "Pass --use-session-server to start the session server."
            )

        # Select the owner once and use its URL for the complete session. Round
        # robin avoids persistent shard skew when episodes remain alive for an
        # hour or more.
        session_port = session_ports[next(_SESSION_SERVER_ROUND_ROBIN) % len(session_ports)]
        session_url = f"http://{session_ip}:{session_port}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        client = _new_session_client()
        try:
            payload = await _request_bytes(
                client,
                "POST",
                f"{session_url}/sessions",
                payload={},
                timeout=_SESSION_CREATE_TIMEOUT,
            )
            session_id = json.loads(payload)["session_id"]
            if not isinstance(session_id, str):
                raise TypeError("session_id must be a string")
            return OpenAIEndpointTracer(
                router_url=session_url,
                session_id=session_id,
                session_server_instance_id=instance_ids.get(session_port),
                samples_wire_fields=COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS,
                client=client,
            )
        except BaseException:
            await client.aclose()
            raise

    async def collect_samples(
        self,
        input_sample: Sample,
        *,
        max_seq_len: int | None,
        agent_metadata: dict | None = None,
    ) -> SamplesReply:
        """Fetch and decode server-assembled samples, then retire the session."""
        body: dict = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata

        try:
            payload = await _request_bytes(
                self._client,
                "POST",
                f"{self.base_url}/samples",
                payload=body,
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            self._schedule_delete()

        return decode_samples_and_merge_input_sample(
            payload,
            input_sample,
            fields=self.samples_wire_fields,
        )

    def _schedule_delete(self) -> asyncio.Task:
        if self._retirement_task is not None:
            return self._retirement_task
        task = asyncio.create_task(self._delete_and_close())
        self._retirement_task = task
        _PENDING_DELETES.add(task)
        task.add_done_callback(_PENDING_DELETES.discard)
        return task

    async def discard_session(self) -> None:
        """Delete a cancelled session without collecting samples."""
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
        except Exception as error:
            logger.debug("Failed to delete session %s: %s", self.session_id, error)
            return False
        finally:
            await self._client.aclose()
