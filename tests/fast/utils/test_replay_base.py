from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from unittest.mock import patch

import pytest
import torch

from miles.utils.replay_base import BaseReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_fills_all_invalid_rows_with_arange():
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)


@pytest.mark.parametrize("stage", ["replay_forward", "replay_backward"])
@pytest.mark.parametrize("padding", [False, True])
def test_replay_probabilities_and_gradients_without_host_scalar_reads(stage, padding):
    scores = torch.arange(10, dtype=torch.float32).reshape(2, 5).requires_grad_()
    indices = torch.tensor([[3, 1], [-1, -1] if padding else [4, 2]])
    expected_indices = torch.tensor([[3, 1], [0, 1] if padding else [4, 2]])
    manager = _make_replay_manager(indices)
    manager.stage = stage
    topk = manager.get_topk_fn(_topk, return_probs=True)

    with patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host scalar read")):
        probabilities, actual_indices = topk(scores, 2)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(probabilities, scores.gather(1, expected_indices))
    probabilities.sum().backward()
    expected_gradient = torch.zeros_like(scores).scatter_(1, expected_indices, 1)
    torch.testing.assert_close(scores.grad, expected_gradient)
