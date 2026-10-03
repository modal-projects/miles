from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell
from tests.fast.utils.test_score_centering_speculative import _server_info

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StateInitializing, StatePendingWeights
from miles.ray.rollout.server_cell import ABORT_REQUEST_TIMEOUT_SECONDS, ServerCell, ServerCellMetadata

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


def _admission_cell(monkeypatch, *, update_weights, info, for_evaluation=False, loss_type="score_centering"):
    client = SimpleNamespace(get_server_info=AsyncMock(return_value=info))
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", lambda **kwargs: client)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", AsyncMock(return_value=True))
    router = SimpleNamespace(add_worker=AsyncMock(), remove_worker=AsyncMock())
    cell = track_server_cell(
        ServerCell(
            args=make_args(loss_type=loss_type),
            meta=ServerCellMetadata(
                model_id="default",
                worker_type="regular",
                cell_id="inference-engine-0-0-0",
                num_gpus_per_engine=1,
                gpu_offset=0,
                sglang_api_key=None,
                worker_name="inference-engine-0-0-0-0",
                needs_offload=False,
                update_weights=update_weights,
                workers_hash="pseudo-hash-0",
            ),
            router_api_client=router,
            provider=None,
            for_evaluation=for_evaluation,
        )
    )
    addr_info = CellAddrInfo(server_url="http://engine:30000", bootstrap_port=None, gate_url=None)
    cell._state = (
        StatePendingWeights(addr_info=addr_info)
        if update_weights
        else StateInitializing(addr_info=addr_info, start_time=time.monotonic())
    )
    return cell, client, router


@pytest.mark.parametrize("update_weights", [False, True], ids=["initializing-frozen", "replacement-after-sync"])
async def test_unsafe_score_centering_cells_never_enter_the_router(monkeypatch, update_weights):
    cell, client, router = _admission_cell(
        monkeypatch,
        update_weights=update_weights,
        info=_server_info(dflash_sampling_verify_available=False),
    )
    with pytest.raises(ValueError, match="http://engine:30000.*dflash_sampling_verify_available"):
        await (cell.mark_weights_ready() if update_weights else cell.tick())
    client.get_server_info.assert_awaited_once()
    router.add_worker.assert_not_awaited()
    assert not cell.is_serving


@pytest.mark.parametrize("update_weights", [False, True], ids=["initializing-frozen", "replacement-after-sync"])
async def test_safe_score_centering_cells_are_validated_before_admission(monkeypatch, update_weights):
    cell, client, router = _admission_cell(monkeypatch, update_weights=update_weights, info=_server_info())

    async def check_admission(**kwargs):
        client.get_server_info.assert_awaited_once()
        assert not cell.is_serving

    router.add_worker.side_effect = check_admission
    await (cell.mark_weights_ready() if update_weights else cell.tick())
    router.add_worker.assert_awaited_once()
    assert cell.is_serving


@pytest.mark.parametrize("for_evaluation,loss_type", [(True, "score_centering"), (False, "policy_loss")])
async def test_eval_and_other_losses_keep_existing_admission_behavior(monkeypatch, for_evaluation, loss_type):
    cell, client, router = _admission_cell(
        monkeypatch,
        update_weights=True,
        info=_server_info(speculative_algorithm="EAGLE"),
        for_evaluation=for_evaluation,
        loss_type=loss_type,
    )
    await cell.mark_weights_ready()
    client.get_server_info.assert_not_awaited()
    router.add_worker.assert_awaited_once()
    assert cell.is_serving


class TestServerCellAbortAll:
    async def test_abort_all_forwards_the_bounded_request_to_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An addressable cell forwards the abort budget and returns its engine's result."""
        result = object()
        timeouts: list[float | None] = []

        class _RecordingApiClient:
            def __init__(self, server_url: str, api_key: str | None = None) -> None:
                self.server_url = server_url
                self.api_key = api_key

            async def abort_all_requests(self, timeout: float | None = None) -> object:
                timeouts.append(timeout)
                return result

        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
        cell = track_server_cell(
            ServerCell(
                args=make_args(),
                meta=ServerCellMetadata(
                    model_id="default",
                    worker_type="regular",
                    cell_id="inference-engine-0-0-0",
                    num_gpus_per_engine=1,
                    gpu_offset=0,
                    sglang_api_key="secret",
                    worker_name="inference-engine-0-0-0-0",
                    needs_offload=False,
                    update_weights=True,
                    workers_hash="pseudo-hash-0",
                ),
                router_api_client=None,
                provider=None,
            )
        )
        cell._state = StatePendingWeights(
            addr_info=CellAddrInfo(
                server_url="http://10.0.0.1:30000",
                bootstrap_port=None,
                gate_url=None,
            )
        )

        actual = await cell.abort_all()

        assert actual is result
        assert timeouts == [ABORT_REQUEST_TIMEOUT_SECONDS]
