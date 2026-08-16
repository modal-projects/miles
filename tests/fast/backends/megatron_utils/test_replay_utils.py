import importlib
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

import pytest
import torch


class _Replay:
    def __init__(self):
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def _models():
    config = SimpleNamespace(moe_layer_freq=[0, 1, 0, 1, 1, 0], num_layers=6)
    return [SimpleNamespace(module=SimpleNamespace(config=config))]


@pytest.fixture(scope="module")
def replay_utils():
    block = ModuleType("megatron.core.transformer.transformer_block")
    block.get_num_layers_to_build = lambda _config, vp_stage: 6
    layer = ModuleType("megatron.core.transformer.transformer_layer")
    layer.get_transformer_layer_offset = lambda _config, vp_stage: 0
    modules = {
        "megatron": ModuleType("megatron"),
        "megatron.core": ModuleType("megatron.core"),
        "megatron.core.transformer": ModuleType("megatron.core.transformer"),
        block.__name__: block,
        layer.__name__: layer,
    }
    module_name = "miles.backends.megatron_utils.replay_utils"
    previous = sys.modules.pop(module_name, None)
    try:
        with patch.dict(sys.modules, modules):
            yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


def test_get_local_moe_layer_indices_uses_model_layout(replay_utils):
    assert replay_utils.get_local_moe_layer_indices(_models()) == [1, 3, 4]


def test_register_pp_local_routing_shard(replay_utils):
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


def test_register_pp_local_routing_shard_rejects_wrong_layers(replay_utils):
    with pytest.raises(ValueError, match="does not match"):
        replay_utils.register_replay_list_moe(
            [_Replay(), _Replay(), _Replay()],
            torch.zeros((2, 3, 2)),
            models=_models(),
            global_layer_indices=[1, 2, 4],
        )


def test_register_routing_rejects_registration_count_mismatch(replay_utils):
    with pytest.raises(ValueError, match="2 routing replay streams for 3 local MoE layers"):
        replay_utils.register_replay_list_moe(
            [_Replay(), _Replay()],
            torch.zeros((2, 6, 2)),
            models=_models(),
        )
