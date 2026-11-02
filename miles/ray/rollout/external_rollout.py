"""An external rollout service's weight API: /start, /versions and /status on --rollout-external-control-url."""

import asyncio
import logging

import httpx

from miles.backends.training_utils.weight_update.utils import engine_stat_metrics
from miles.utils.http_utils import _post, rollout_external_headers

logger = logging.getLogger(__name__)

_STATUS_POLL_S = 5


class ExternalRollout:
    def __init__(self, control_url: str) -> None:
        self._url = control_url.rstrip("/")
        self._start: dict | None = None

    async def serve(self, version: int) -> dict[str, float]:
        """The first version starts the run and waits for one engine; later ones, for every engine taking traffic."""
        if self._start is None:
            reply = await self._call("/start", {"version": version})
            self._start = {"start": version, "start_id": reply["start_id"]}
            logger.info(f"waiting for the external rollout to serve version {version}")
            while not (await self._call("/status", None))["serving"].get(str(version)):
                await asyncio.sleep(_STATUS_POLL_S)
            logger.info(f"the external rollout serves version {version}")
            return {}
        reply = await self._call("/versions", {"version": version, **self._start})
        stats = [engine["stats"] or {} for engine in reply["engines"].values()]
        metrics = {
            "perf/update_weights_external_s": reply["seconds"],
            **engine_stat_metrics([{"stats": s.get("prepare")} for s in stats], "prepare"),
            **engine_stat_metrics([{"stats": s.get("update")} for s in stats], "apply"),
        }
        logger.info(f"the external rollout serves version {version}: {metrics}")
        return metrics

    async def _call(self, path: str, body: dict | None) -> dict:
        # A reply slower than its web endpoint allows may come back as a redirect to where it will appear.
        async with httpx.AsyncClient(
            headers=rollout_external_headers(), timeout=httpx.Timeout(None, connect=10.0), follow_redirects=True
        ) as client:
            if body is None:
                return await _post(client, f"{self._url}{path}", None, action="get")
            return await _post(client, f"{self._url}{path}", body)
