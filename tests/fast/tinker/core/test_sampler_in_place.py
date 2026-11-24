"""Sampler saves under their own root, and in place: one adapter per model, which every save replaces."""

import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path

import pytest
from tests.fast.tinker.harness import await_settled, created_model, make_service

from miles.tinker.core.future import DONE


@pytest.fixture
async def in_place(tmp_path):
    gateway = make_service(tmp_path / "checkpoints", sampler_root=str(tmp_path / "sampler"), sampler_in_place=True)
    run_task = asyncio.create_task(gateway.run())
    try:
        yield gateway
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task


async def _save(service, model_id: str, seq_id: int, sampler_path: str | None = None, ttl_seconds=None) -> str:
    request_id = service.submit(
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": seq_id, "sampler_path": sampler_path, "ttl_seconds": ttl_seconds},
    )
    future = await await_settled(service, "tenant", request_id)
    assert future.state == DONE, future.result
    return future.result["path"]


def _link(service, model_id: str) -> Path:
    return Path(service.config.sampler_root).resolve() / model_id / "sampler_weights" / "latest"


async def test_each_save_replaces_the_models_adapter_under_the_next_count(in_place):
    model_id = await created_model(in_place)
    link = _link(in_place, model_id)

    first = await _save(in_place, model_id, 1)
    first_dir = link.resolve()
    second = await _save(in_place, model_id, 2, sampler_path="ckpt")

    assert (first, second) == (f"tinker://{model_id}/sampler_weights/1", f"tinker://{model_id}/sampler_weights/ckpt")
    assert link.resolve() != first_dir
    assert json.loads((link / "version.json").read_text()) == {"version": 2}
    assert in_place.backend.named("publish_sampler") == [{"path": str(link)}] * 2


async def test_any_sampler_path_of_the_model_samples_its_adapter_by_name(in_place):
    model_id = await created_model(in_place)
    path = await _save(in_place, model_id, 1)

    sample_id, _ = in_place.submit_sample(
        "tenant",
        {
            "model_path": path,
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(in_place, "tenant", sample_id)

    [sample] = in_place.backend.named("sample")
    assert (sample["lora_name"], sample["lora_path"]) == (f"{model_id}/sampler_weights/latest", None)


async def test_a_ttl_never_ends_the_models_adapter(in_place):
    model_id = await created_model(in_place)
    path = await _save(in_place, model_id, 1, ttl_seconds=1)

    await in_place.set_checkpoint_ttl("tenant", model_id, "sampler_weights/1", 1)

    assert in_place.expiry.expired(float("inf")) == []
    assert os.path.islink(_link(in_place, model_id)) and path.endswith("/1")


async def test_deleting_a_sampler_path_of_the_model_removes_its_adapter(in_place):
    model_id = await created_model(in_place)
    await _save(in_place, model_id, 1)
    link = _link(in_place, model_id)

    await in_place.delete_checkpoint("tenant", model_id, "sampler_weights/1")

    assert not os.path.lexists(link)
    assert in_place.backend.named("withdraw_sampler") == [{"path": str(link)}]


async def test_snapshot_saves_go_under_the_sampler_root(tmp_path):
    service = make_service(tmp_path / "checkpoints", sampler_root=str(tmp_path / "sampler"))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        await _save(service, model_id, 1)
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task

    [published] = service.backend.named("publish_sampler")
    assert Path(published["path"]).is_relative_to((tmp_path / "sampler").resolve() / model_id / "sampler_weights")
