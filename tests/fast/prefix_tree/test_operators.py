"""The row-wise kernel control preserves the linear operation and input fan-in."""

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.operators import rowwise_frozen_linear


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_frozen_projection_outputs_and_shared_input_gradients_match_fp64_expansion(noncontiguous):
    torch.manual_seed(71)
    value = torch.randn(5, 13, dtype=torch.float64)
    weight = torch.randn(7, 13, dtype=torch.float64)
    if noncontiguous:
        value = value.t().contiguous().t()
        weight = weight.t().contiguous().t()
    rows = torch.tensor([0, 1, 0, 2, 4, 0, 3, 4])
    upstream = torch.randn(8, 7, dtype=torch.float64)
    reference = value.detach().requires_grad_()
    compact = value.detach().requires_grad_()
    expected = F.linear(reference[rows], weight)
    actual = rowwise_frozen_linear(compact, weight)[rows]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(compact.grad, reference.grad, atol=1e-12, rtol=1e-12)


def test_outer_autocast_does_not_change_the_requested_projection_precision():
    value = torch.randn(5, 13, requires_grad=True)
    weight = torch.randn(7, 13)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = rowwise_frozen_linear(value, weight)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, F.linear(value, weight))


def test_trainable_weights_are_rejected_before_materializing_batched_weight_gradients():
    with pytest.raises(ValueError, match="frozen weights"):
        rowwise_frozen_linear(torch.randn(2, 3), torch.randn(4, 3, requires_grad=True))


def test_trainable_rowwise_linear_every_gradient_and_double_backward_match_dense():
    from miles.backends.training_utils.prefix_tree.operators import rowwise_linear

    value = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rowwise_linear, (value, weight))
    assert torch.autograd.gradgradcheck(rowwise_linear, (value, weight))
    upstream = torch.randn(4, 3, dtype=torch.float64)
    actual = rowwise_linear(value, weight)
    expected = F.linear(value, weight)
    a = torch.autograd.grad((actual * upstream).sum(), (value, weight))
    b = torch.autograd.grad((expected * upstream).sum(), (value, weight))
    for got, reference in zip(a, b, strict=True):
        torch.testing.assert_close(got, reference, atol=1e-12, rtol=1e-12)


def test_trainable_rowwise_empty_groups_keep_zero_gradients():
    from miles.backends.training_utils.prefix_tree.operators import rowwise_linear

    value = torch.randn(0, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    rowwise_linear(value, weight).sum().backward()
    assert value.grad.shape == value.shape
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
