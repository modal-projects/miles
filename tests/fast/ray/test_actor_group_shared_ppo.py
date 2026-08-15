from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


class _RemoteTrain:
    def __init__(self, rank, calls):
        self.rank = rank
        self.calls = calls

    def remote(self, rollout_id, rollout_data_ref, **kwargs):
        self.calls.append((self.rank, rollout_id, rollout_data_ref, kwargs))

        async def result():
            return {"rank": self.rank}

        return result()


class _Handle:
    def __init__(self, rank, calls):
        self.train = _RemoteTrain(rank, calls)


async def test_train_routes_each_critic_payload_to_matching_actor_rank():
    from miles.ray.actor_group import RayTrainGroup

    calls = []
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, calls), _Handle(1, calls)]
    payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

    result = await group.train(5, {"data_ref": "rollout"}, external_data=payloads)

    assert result == [{"rank": 0}, {"rank": 1}]
    assert calls == [
        (0, 5, "rollout", {"witness_info": None, "attempt": 0, "external_data": payloads[0]}),
        (1, 5, "rollout", {"witness_info": None, "attempt": 0, "external_data": payloads[1]}),
    ]


async def test_train_broadcasts_without_lifecycle_options():
    from miles.ray.actor_group import RayTrainGroup

    calls = []
    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, calls), _Handle(1, calls)]

    await group.train(7, {"data_ref": "rollout"})

    assert calls == [
        (0, 7, "rollout", {"witness_info": None, "attempt": 0}),
        (1, 7, "rollout", {"witness_info": None, "attempt": 0}),
    ]


async def test_train_rejects_wrong_number_of_rank_payloads():
    import pytest

    from miles.ray.actor_group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group._actor_handles = [_Handle(0, []), _Handle(1, [])]

    with pytest.raises(ValueError, match="one payload per train worker"):
        await group.train(5, {"data_ref": "rollout"}, external_data=[{"values": []}])


async def test_update_weights_resumes_health_monitor_after_failure():
    import pytest

    from miles.ray.actor_group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, use_fault_tolerance=False)
    group.rollout_manager = MagicMock()
    group.rollout_manager.get_updatable_engines_and_lock.remote = AsyncMock(return_value={"engine": "info"})
    group.rollout_manager.health_monitoring_pause.remote = AsyncMock()
    group.rollout_manager.health_monitoring_resume.remote = AsyncMock()
    group._broadcast = AsyncMock(side_effect=RuntimeError("weight sync failed"))

    with pytest.raises(RuntimeError, match="weight sync failed"):
        await group.update_weights(rollout_id=3)

    group.rollout_manager.health_monitoring_pause.remote.assert_awaited_once_with()
    group.rollout_manager.health_monitoring_resume.remote.assert_awaited_once_with()
