"""Tinker's in-place backend on Miles' own engines: what each engine cell gets from a save, a delete or a restart."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import miles.tinker.in_place_adapters as in_place_adapters
from miles.tinker.in_place_adapters import InPlaceAdapters


class FakeEngine:
    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = [], fail

    async def update_lora_adapter(self, name, path, weight_version, pause_mode):
        if self.fail:
            raise RuntimeError("engine down")
        self.calls.append(("update", name, Path(path).name, weight_version))

    async def unload_lora_adapter(self, name):
        self.calls.append(("unload", name))


class FakeController:
    """Engine cells by id: (engine, workers hash); a new hash is a restarted cell."""

    def __init__(self, cells: dict) -> None:
        self.cells, self.failed, self.pending, self.monitoring = cells, [], False, True

    async def start_update_weights(self):
        self.monitoring = False
        ids = list(self.cells)
        return SimpleNamespace(
            rollout_engines=[self.cells[cell][0] for cell in ids],
            engine_cell_ids=ids,
            snapshot_cell_id_to_hashes={cell: self.cells[cell][1] for cell in ids},
        )

    async def end_update_weights(self, hashes, failed):
        self.failed.append(sorted(failed))

    async def abort_update_weights(self):
        pass

    async def resume_health_monitoring(self):
        self.monitoring = True

    async def has_pending_weights(self):
        return self.pending


def _save(root: Path, model: str, commit: str, version: int) -> str:
    """Miles' commit: a new directory, then the model's link switched to it."""
    sampler = root / model / "sampler_weights"
    directory = sampler / f".latest.{commit}"
    directory.mkdir(parents=True)
    (directory / "version.json").write_text(json.dumps({"version": version}))
    (sampler / "staging").symlink_to(directory.name)
    os.replace(sampler / "staging", sampler / "latest")
    return str(sampler / "latest")


async def test_start_defines_every_models_latest_adapter_on_every_engine(tmp_path):
    _save(tmp_path, "m1", "c1", 1)
    _save(tmp_path, "m1", "c2", 2)
    _save(tmp_path, "m2", "c1", 1)
    engines = [FakeEngine(), FakeEngine()]
    adapters = InPlaceAdapters(FakeController({"a": (engines[0], "h"), "b": (engines[1], "h")}), str(tmp_path))

    await adapters.start()

    for engine in engines:
        assert sorted(engine.calls) == [
            ("update", "m1/sampler_weights/latest", ".latest.c2", "2"),
            ("update", "m2/sampler_weights/latest", ".latest.c1", "1"),
        ]


async def test_a_save_switches_its_adapter_and_a_restarted_cell_gets_every_adapter(tmp_path):
    old, restarted = FakeEngine(), FakeEngine()
    controller = FakeController({"a": (old, "h"), "b": (restarted, "h")})
    adapters = InPlaceAdapters(controller, str(tmp_path))
    await adapters.start()
    await adapters.update(_save(tmp_path, "m1", "c1", 1))
    old.calls.clear()
    controller.cells["b"] = (restarted := FakeEngine(), "h2")

    assert await adapters.update(_save(tmp_path, "m2", "c1", 1)) is None

    assert old.calls == [("update", "m2/sampler_weights/latest", ".latest.c1", "1")]
    assert sorted(restarted.calls) == [
        ("update", "m1/sampler_weights/latest", ".latest.c1", "1"),
        ("update", "m2/sampler_weights/latest", ".latest.c1", "1"),
    ]


async def test_a_cell_that_fails_a_save_is_reported_and_given_every_adapter_next_time(tmp_path):
    good, bad = FakeEngine(), FakeEngine(fail=True)
    controller = FakeController({"a": (good, "h"), "b": (bad, "h")})
    adapters = InPlaceAdapters(controller, str(tmp_path))
    await adapters.update(_save(tmp_path, "m1", "c1", 1))
    assert controller.failed == [["b"]]

    bad.fail = False
    await adapters.update(_save(tmp_path, "m2", "c1", 1))

    assert controller.failed[-1] == []
    assert sorted(call[1] for call in bad.calls) == ["m1/sampler_weights/latest", "m2/sampler_weights/latest"]


async def test_a_delete_unloads_the_adapter_on_every_engine(tmp_path):
    engine = FakeEngine()
    adapters = InPlaceAdapters(FakeController({"a": (engine, "h")}), str(tmp_path))
    link = _save(tmp_path, "m1", "c1", 1)
    await adapters.update(link)

    await adapters.remove(link)

    assert engine.calls[-1] == ("unload", "m1/sampler_weights/latest")
    assert adapters.latest == {}


async def test_a_cell_waiting_for_weights_gets_every_adapter_without_a_save(tmp_path, monkeypatch):
    monkeypatch.setattr(in_place_adapters, "_WAITING_CELLS_POLL_S", 0.01)
    _save(tmp_path, "m1", "c1", 1)
    engine = FakeEngine()
    controller = FakeController({"a": (engine, "h")})
    adapters = InPlaceAdapters(controller, str(tmp_path))
    await adapters.start()
    controller.cells["a"] = (restarted := FakeEngine(), "h2")
    controller.pending = True

    task = asyncio.create_task(adapters.update_waiting_cells())
    await asyncio.sleep(0.1)
    task.cancel()

    assert restarted.calls[0] == ("update", "m1/sampler_weights/latest", ".latest.c1", "1")


async def test_engines_are_health_checked_again_after_each_delivery(tmp_path):
    controller = FakeController({"a": (FakeEngine(), "h")})
    adapters = InPlaceAdapters(controller, str(tmp_path))

    await adapters.update(_save(tmp_path, "m1", "c1", 1))

    assert controller.monitoring
