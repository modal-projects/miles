"""Tinker's in-place sampler backend on Miles' own engines: every engine serves each model's latest adapter."""

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from miles.tinker.core.utils import IN_PLACE_SAMPLER_NAME, IN_PLACE_VERSION_FILENAME

logger = logging.getLogger(__name__)

# A cell that comes up waits at most this long for the adapters; arbitrary.
_WAITING_CELLS_POLL_S = 5


class InPlaceAdapters:
    """Each save switches the model's adapter on every engine; a cell that comes up first gets every model's."""

    def __init__(self, controller, root: str) -> None:
        self.controller, self.root = controller, os.path.abspath(root)
        # Each model's adapter name: (its latest directory, that save's count).
        self.latest: dict[str, tuple[str, int]] = {}
        # Each engine cell's workers hash when it last held every model's latest adapter.
        self.synced: dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.latest = await asyncio.to_thread(self._on_disk)
        failure = await self._deliver({})
        if failure is not None:
            raise RuntimeError(failure["error"])

    async def update(self, link: str) -> dict | None:
        name, latest = await asyncio.to_thread(self._read, link)
        self.latest[name] = latest
        return await self._deliver({name: latest})

    async def remove(self, link: str) -> dict | None:
        name = self._name(link)
        self.latest.pop(name, None)
        return await self._deliver({name: None})

    async def update_waiting_cells(self) -> None:
        while True:
            await asyncio.sleep(_WAITING_CELLS_POLL_S)
            if await self.controller.has_pending_weights():
                await self._deliver({})

    async def _deliver(self, changes: dict[str, tuple[str, int] | None]) -> dict | None:
        async with self.lock:
            info = await self.controller.start_update_weights()
            try:
                hashes = info.snapshot_cell_id_to_hashes
                results = await asyncio.gather(
                    *[
                        self._deliver_to(engine, self.latest if self.synced.get(cell) != hashes[cell] else changes)
                        for engine, cell in zip(info.rollout_engines, info.engine_cell_ids, strict=True)
                    ],
                    return_exceptions=True,
                )
            except BaseException:
                await self.controller.abort_update_weights()
                raise
            failed = [cell for cell, result in zip(info.engine_cell_ids, results, strict=True) if result is not None]
            for cell, result in zip(info.engine_cell_ids, results, strict=True):
                if result is not None:
                    logger.error(f"engine cell {cell} did not take the in-place adapters: {result!r}")
            await self.controller.end_update_weights(hashes, failed)
            await self.controller.resume_health_monitoring()
            self.synced = {cell: workers for cell, workers in hashes.items() if cell not in failed}
        if failed and len(failed) == len(info.engine_cell_ids):
            return {"error": "no engine took the in-place adapters", "error_category": "server"}
        return None

    @staticmethod
    async def _deliver_to(engine, changes: Mapping[str, tuple[str, int] | None]) -> None:
        for name, latest in changes.items():
            if latest is None:
                await engine.unload_lora_adapter(name)
            else:
                directory, version = latest
                await engine.update_lora_adapter(name, directory, weight_version=str(version), pause_mode="in_place")

    def _on_disk(self) -> dict[str, tuple[str, int]]:
        links = Path(self.root).glob(f"*/sampler_weights/{IN_PLACE_SAMPLER_NAME}")
        return dict(self._read(str(link)) for link in links if link.is_symlink())

    def _read(self, link: str) -> tuple[str, tuple[str, int]]:
        directory = os.path.join(os.path.dirname(link), os.readlink(link))
        version = json.loads(Path(directory, IN_PLACE_VERSION_FILENAME).read_text())["version"]
        return self._name(link), (directory, version)

    def _name(self, link: str) -> str:
        return os.path.relpath(link, self.root)
