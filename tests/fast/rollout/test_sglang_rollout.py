import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.rollout import sglang_rollout
from miles.rollout.sglang_rollout import GenerateState, get_model_url
from miles.utils.misc import SingletonMeta


def test_external_rollout_uses_endpoint_concurrency(monkeypatch) -> None:
    monkeypatch.setattr("miles.rollout.sglang_rollout.load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr("miles.rollout.sglang_rollout.load_processor", lambda *args, **kwargs: None)
    SingletonMeta.clear_all_instances()
    args = SimpleNamespace(
        hf_checkpoint="unused",
        chat_template_path=None,
        rollout_endpoint_url="https://rollout.example",
        rollout_num_gpus=0,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=7,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=64,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=None,
    )

    state = GenerateState(args)

    assert state.semaphore._value == 7


def test_external_rollout_uses_endpoint_url() -> None:
    args = SimpleNamespace(
        rollout_endpoint_url="https://rollout.example",
        sglang_model_routers={"default": ("127.0.0.1", 30000)},
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30001,
    )

    assert get_model_url(args, "default") == "https://rollout.example/generate"


@pytest.mark.asyncio
async def test_external_rollout_cancels_surplus_requests_without_worker_discovery(monkeypatch) -> None:
    pending = asyncio.create_task(asyncio.sleep(60))
    state = SimpleNamespace(aborted=False, pendings={pending})
    discover_workers = AsyncMock(side_effect=AssertionError("external fleets are opaque"))
    abort_agents = AsyncMock()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "get", discover_workers)
    monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", abort_agents)

    aborted = await sglang_rollout.abort(
        SimpleNamespace(rollout_endpoint_url="https://rollout.example", partial_rollout=False),
        rollout_id=1,
    )

    assert aborted == []
    assert pending.cancelled()
    assert state.pendings == set()
    discover_workers.assert_not_awaited()
    abort_agents.assert_awaited_once()
