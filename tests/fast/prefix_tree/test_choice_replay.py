import importlib.util
import sys
from pathlib import Path

import torch

_MANUAL = Path(__file__).parents[2] / "manual/prefix_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DATA = _load("_choice_math_data", _MANUAL / "math_sft_data.py")
_REPLAY = _load("_choice_replay", _MANUAL / "glm53_choice_replay.py")


def _batch(slots=(0, 0)):
    return _DATA.MathMicrobatch(
        sequences=((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 7, 8)),
        targets=((2, 3, 4, 5, 6, 9), (2, 3, 4, 7, 8, 9)),
        weights=((0, 0, 0, 1, 1, 1),) * 2,
        slots=slots,
        computation_keys=((None,) * 4 + (0,) * 2, (None,) * 4 + (1,) * 2),
        prompt_indices=(0,),
        real_input_tokens=12,
        supervised_tokens=6,
        padding_tokens=0,
    )


def test_equal_replayed_choices_preserve_nontrivial_sharing():
    data = _batch()
    result, report = _REPLAY.compatible_data(data, {"routes": torch.zeros(12, 2, dtype=torch.long)})
    assert result == data
    assert report == {
        "incompatible_original_rows": 0,
        "forked_samples": 0,
        "compact_rows_before": 8,
        "compact_rows_after": 8,
    }


def test_incompatible_experts_and_keys_fork_before_the_token_on_the_sp_grid():
    data = _batch()
    routes = torch.zeros(12, 2, dtype=torch.long)
    routes[9] = 1  # Sample one, logical position three: fork at position two.
    keys = torch.zeros(12, 1, 4, dtype=torch.long)
    keys[10] = 2
    result, report = _REPLAY.compatible_data(data, {"routes": routes, "keys": keys})
    expanded, compact = _REPLAY.plans(result)
    first, mapping = _REPLAY.occurrence_maps(expanded, compact, device="cpu")
    assert report["forked_samples"] == 1
    assert report["compact_rows_after"] == 10
    assert compact.sample_nodes[0][:2] == compact.sample_nodes[1][:2]
    assert compact.sample_nodes[0][2] != compact.sample_nodes[1][2]
    for value in (routes, keys):
        torch.testing.assert_close(value[first][mapping], value, atol=0, rtol=0)
    assert result.sequences == data.sequences
    assert result.targets == data.targets
    assert result.weights == data.weights


def test_replay_choices_never_merge_different_adapter_slots():
    data = _batch(slots=(0, 1))
    routes = torch.arange(24).reshape(12, 2)
    result, report = _REPLAY.compatible_data(data, {"routes": routes})
    assert result == data
    assert report["incompatible_original_rows"] == 0
    assert report["compact_rows_after"] == 12
