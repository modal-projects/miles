from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import rollout_request


@pytest.mark.asyncio
async def test_prepare_rollout_request_applies_async_hook(monkeypatch):
    async def hook(args, context, request):
        assert args.name == "test"
        assert context.session_id == "session-1"
        request["payload"]["weight_version"] = {"min_version": 7}
        return {"max_retries": 9, "retry_sleep": 0.25}

    monkeypatch.setattr(rollout_request, "load_function", lambda _path: hook)
    request = await rollout_request.prepare_rollout_request(
        SimpleNamespace(
            name="test",
            custom_rollout_request_hook_path="test.hook",
        ),
        rollout_request.RolloutRequestContext(session_id="session-1"),
        url="https://rollout.example/v1/chat/completions",
        payload={},
    )

    assert request["payload"]["weight_version"] == {"min_version": 7}
    assert request["max_retries"] == 9
    assert request["retry_sleep"] == 0.25


@pytest.mark.asyncio
async def test_prepare_rollout_request_rejects_invalid_hook_result(monkeypatch):
    monkeypatch.setattr(rollout_request, "load_function", lambda _path: lambda *_args: "invalid")

    with pytest.raises(TypeError, match="must return None or a dict"):
        await rollout_request.prepare_rollout_request(
            SimpleNamespace(custom_rollout_request_hook_path="test.hook"),
            SimpleNamespace(),
            url="https://rollout.example/generate",
            payload={},
        )
