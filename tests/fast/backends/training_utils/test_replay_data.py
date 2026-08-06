from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import replay_data as replay_data_module
from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, replay_data)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # PP case: 2 local modules at global streams 1 and 3 out of 4.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]

    register_replay_list_sequential(replays, replay_data)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, replay_data)


@pytest.mark.parametrize("consume", [False, True])
def test_fill_replay_data_optionally_consumes_source(monkeypatch, consume):
    class _Iterator:
        def reset(self):
            pass

    monkeypatch.setattr(
        replay_data_module,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0, size=1)),
    )
    rollout_data = {
        "rollout_routed_experts": [],
        "rollout_routed_experts_layer_indices": [3, 4],
    }

    fill_replay_data(
        args=SimpleNamespace(qkv_format="thd", sequence_parallel=False, data_pad_size_multiplier=1),
        models=[],
        data_iterator=[_Iterator()],
        num_microbatches=[],
        rollout_data=rollout_data,
        data_key="rollout_routed_experts",
        replay_list=[],
        register_replay_list_func=lambda *_args, **_kwargs: None,
        global_stream_indices_key="rollout_routed_experts_layer_indices",
        consume=consume,
    )

    assert ("rollout_routed_experts" in rollout_data) is not consume
    assert ("rollout_routed_experts_layer_indices" in rollout_data) is not consume
