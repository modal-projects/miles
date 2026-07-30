"""Utilities for tracing session-server OpenAI requests."""

import asyncio
import itertools
import logging
import time
from argparse import Namespace

from miles.rollout.session.samples.codec import SamplesReply, decode_samples_and_merge_input_sample
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120
_SESSION_DELETE_TIMEOUT = 10
_PENDING_DELETES: set[asyncio.Task] = set()
_SESSION_SERVER_ROUND_ROBIN = itertools.count()


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str, session_server_instance_id: str | None = None):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id

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
        response = await post(
            f"{session_url}/sessions",
            {},
            # Session creation is not idempotent: a retry after a lost response
            # would leave an unreachable session behind.
            max_retries=1,
            action="post",
        )
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=response["session_id"],
            session_server_instance_id=instance_ids.get(session_port),
        )

    async def collect_samples(self, input_sample: Sample, *, max_seq_len: int | None) -> SamplesReply:
        """Fetch server-assembled training samples for this session."""
        collect_started = time.monotonic()
        request_started = time.monotonic()
        try:
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                {"max_seq_len": max_seq_len},
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

    def _schedule_delete(self) -> None:
        task = asyncio.create_task(self._best_effort_delete())
        _PENDING_DELETES.add(task)

        def _done(completed: asyncio.Task) -> None:
            _PENDING_DELETES.discard(completed)

        task.add_done_callback(_done)

    async def discard_session(self) -> None:
        """Delete a cancelled session without collecting its sample."""
        await self._best_effort_delete()

    async def _best_effort_delete(self) -> bool:
        try:
            await asyncio.wait_for(
                post(
                    self.base_url,
                    {},
                    max_retries=3,
                    action="delete",
                ),
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
