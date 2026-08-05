from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import replay_utils


class _Replay:
    def __init__(self):
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def _models():
    config = SimpleNamespace(
        moe_layer_freq=[0, 1, 0, 1, 1, 0],
        num_layers=6,
    )
    return [SimpleNamespace(module=SimpleNamespace(config=config))]


@pytest.fixture(autouse=True)
def _model_layout(monkeypatch):
    monkeypatch.setattr(
        replay_utils,
        "get_num_layers_to_build",
        lambda _config, vp_stage: 6,
    )
    monkeypatch.setattr(
        replay_utils,
        "get_transformer_layer_offset",
        lambda _config, vp_stage: 0,
    )


def test_get_local_moe_layer_indices_uses_model_layout():
    assert replay_utils.get_local_moe_layer_indices(_models()) == [1, 3, 4]


def test_register_local_routing_shard():
    replays = [_Replay(), _Replay(), _Replay()]
    replay_data = torch.arange(2 * 3 * 2).reshape(2, 3, 2)

    replay_utils.register_replay_list_moe(
        replays,
        replay_data,
        models=_models(),
        global_layer_indices=[1, 3, 4],
    )

    for stream, replay in enumerate(replays):
        torch.testing.assert_close(replay.recorded[0], replay_data[:, stream])


def test_register_local_routing_shard_rejects_wrong_layers():
    with pytest.raises(ValueError, match="does not match"):
        replay_utils.register_replay_list_moe(
            [_Replay(), _Replay(), _Replay()],
            torch.zeros((2, 3, 2)),
            models=_models(),
            global_layer_indices=[1, 2, 4],
        )
