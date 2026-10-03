"""The rollout gate validates effective worker settings before training traffic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.test_inference_controller import _make_controller, _RecordingServer
from tests.fast.utils.test_score_centering_speculative import _server_info


def _cell(*, info=None, ready=True):
    return SimpleNamespace(
        server_url="http://engine:30000",
        is_pending_weights_or_serving=ready,
        api_client=SimpleNamespace(get_server_info=AsyncMock(return_value=info or _server_info())),
    )


def _controller(servers, *, loss_type="score_centering", eval_num_gpus=0):
    controller = _make_controller(servers)
    controller.args.loss_type = loss_type
    controller.args.eval_num_gpus = eval_num_gpus
    return controller


@pytest.mark.asyncio
async def test_external_engine_configuration_overrides_controller_defaults():
    cell = _cell(info=_server_info(speculative_accept_threshold_acc=0.5))
    server = _RecordingServer({"external": cell})
    server.health_checker_activeness.bump_active(False)
    controller = _controller({"actor": server})
    controller.args.sglang_speculative_algorithm = None
    controller.args.rollout_external = True
    with pytest.raises(ValueError, match="http://engine:30000.*exact DFlash sampling"):
        await controller.prepare_rollout(rollout_id=0)
    assert not server.health_checker_activeness.get().active


@pytest.mark.asyncio
async def test_changed_worker_settings_are_checked_on_the_next_rollout():
    cell = _cell()
    controller = _controller({"actor": _RecordingServer({"cell": cell})})
    await controller.prepare_rollout(rollout_id=0)
    cell.api_client.get_server_info.return_value = _server_info(speculative_accept_threshold_single=0.5)
    with pytest.raises(ValueError, match="exact DFlash sampling"):
        await controller.prepare_rollout(rollout_id=1)
    assert cell.api_client.get_server_info.await_count == 2


@pytest.mark.asyncio
async def test_replacement_worker_must_report_its_own_sampling_capability():
    original = _cell()
    server = _RecordingServer({"cell": original})
    controller = _controller({"actor": server})
    await controller.prepare_rollout(rollout_id=0)
    replacement = _cell(info=_server_info(dflash_sampling_verify_available=False))
    server.server_cells["cell"] = replacement
    with pytest.raises(ValueError, match="dflash_sampling_verify_available"):
        await controller.prepare_rollout(rollout_id=1)
    replacement.api_client.get_server_info.assert_awaited_once()


@pytest.mark.asyncio
async def test_only_selected_training_workers_are_checked():
    actor, other, evaluation, initializing = _cell(), _cell(), _cell(), _cell(ready=False)
    controller = _controller(
        {
            "actor": _RecordingServer({"actor": actor, "initializing": initializing}, model_name="actor"),
            "other": _RecordingServer({"other": other}, model_name="other"),
            "eval": _RecordingServer({"eval": evaluation}, model_name="eval"),
        },
        eval_num_gpus=1,
    )
    await controller.prepare_rollout(rollout_id=0, model_id="actor")
    actor.api_client.get_server_info.assert_awaited_once()
    other.api_client.get_server_info.assert_not_awaited()
    evaluation.api_client.get_server_info.assert_not_awaited()
    initializing.api_client.get_server_info.assert_not_awaited()
    await controller.prepare_rollout(rollout_id=1)
    other.api_client.get_server_info.assert_awaited_once()
    evaluation.api_client.get_server_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_eval_and_other_losses_keep_existing_rollout_behavior():
    cell = _cell(info=_server_info(dflash_sampling_verify_available=False))
    controller = _controller({"actor": _RecordingServer({"cell": cell})})
    await controller.prepare_eval()
    controller.args.loss_type = "policy_loss"
    await controller.prepare_rollout(rollout_id=0)
    cell.api_client.get_server_info.assert_not_awaited()
