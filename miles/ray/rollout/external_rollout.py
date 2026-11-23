"""An external rollout service's control API on --rollout-external-control-url: weights, adapters, steps and load."""

import asyncio
import json
import logging
from urllib.parse import urlencode

import httpx

from miles.backends.training_utils.weight_update.utils import engine_stat_metrics
from miles.utils.http_utils import _post, rollout_external_headers

logger = logging.getLogger(__name__)

_STATUS_POLL_S = 5


class ExternalRollout:
    def __init__(self, control_url: str) -> None:
        self._url = control_url.rstrip("/")
        self._start: dict | None = None
        self._served: int | None = None

    async def serve(self, version: int, publication: dict | None = None) -> dict[str, float]:
        """The first version starts the run and waits for one engine; later ones, for every engine taking traffic.

        ``publication`` names the files of a version the service cannot find by its number alone (gpu-delta).
        """
        publication_field = {} if publication is None else {"publication": publication}
        if self._start is None:
            reply = await self._call("/start", {"version": version, **publication_field})
            self._start = {"start": version, "start_id": reply["start_id"]}
            logger.info(f"waiting for the external rollout to serve version {version}")
            # A control service restarted before any version learns the run from this poll, the publication with it.
            run = {**self._start, "head": version}
            if publication is not None:
                run["publication"] = json.dumps(publication)
            status = f"/status?{urlencode(run)}"
            while not (await self._call(status, None))["serving"].get(str(version)):
                await asyncio.sleep(_STATUS_POLL_S)
            logger.info(f"the external rollout serves version {version}")
            self._served = version
            return {}
        reply = await self._call(
            "/versions", {"version": version, "base_version": self._served, **self._start, **publication_field}
        )
        self._served = version
        stats = [engine["stats"] or {} for engine in reply["engines"].values()]
        metrics = {
            "perf/update_weights_external_s": reply["seconds"],
            **engine_stat_metrics([{"stats": s.get("prepare")} for s in stats], "prepare"),
            **engine_stat_metrics([{"stats": s.get("apply")} for s in stats], "apply"),
        }
        logger.info(f"the external rollout serves version {version}: {metrics}")
        return metrics

    async def publish_adapter(self, path: str, withdrawn: bool = False) -> None:
        """Returns once every serving engine holds the adapter at `path`, or no longer does if `withdrawn`."""
        await self._call("/adapters", {"path": path, "withdrawn": withdrawn, **self._start})

    async def report_step(self, step: dict[str, float]) -> None:
        """One training step, from which the service sizes its engines; a failed report loses only that step."""
        try:
            await self._call("/steps", step)
        except Exception as error:
            logger.warning(f"the external rollout did not take a training step: {error!r}")

    async def report_load(self, load: dict[str, float | None]) -> int | None:
        """The rollout's load, from which the service sizes its engines; it replies the requests they run at once."""
        try:
            reply = await self._call("/load", load)
        except Exception as error:
            logger.warning(f"the external rollout did not take a load report: {error!r}")
            return None
        return reply.get("capacity")

    async def _call(self, path: str, body: dict | None) -> dict:
        # A reply slower than its web endpoint allows may come back as a redirect to where it will appear.
        async with httpx.AsyncClient(
            headers=rollout_external_headers(), timeout=httpx.Timeout(None, connect=10.0), follow_redirects=True
        ) as client:
            if body is None:
                return await _post(client, f"{self._url}{path}", None, action="get")
            return await _post(client, f"{self._url}{path}", body)
