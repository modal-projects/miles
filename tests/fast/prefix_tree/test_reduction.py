import pytest
import torch

from miles.backends.training_utils.prefix_tree import reduction
from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather


def test_ordered_gather_fp64_gradient_and_empty_rows():
    plan = OrderedRowGather.from_indices([2, 0, 2, 0, 2], 4, device="cpu")
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(plan.gather, (values,))
    plan.gather(values).sum().backward()
    torch.testing.assert_close(values.grad[:, 0], torch.tensor([2, 0, 3, 0], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_order_accumulates_before_casting_back(dtype):
    plan = OrderedRowGather.from_indices([0] * 32, 1, device="cpu")
    value = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    coefficients = torch.tensor([[256.0, 2.0]] + [[1.0, 0.01]] * 30 + [[-256.0, -2.0]], dtype=dtype)
    plan.gather(value).backward(coefficients)
    torch.testing.assert_close(value.grad, coefficients.float().sum(dim=0, keepdim=True).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_wide_noncontiguous_adjoint_includes_empty_rows(monkeypatch, dtype):
    monkeypatch.setattr(reduction, "_MAX_ORDERED_ELEMENTS", 8)
    indices = [2, 0, 2, 0, 2]
    plan = OrderedRowGather.from_indices(indices, 4, device="cpu")
    values = torch.zeros(4, 11, dtype=dtype, requires_grad=True)
    gradient = torch.arange(55, dtype=dtype).reshape(11, 5).t()
    assert not gradient.is_contiguous()
    plan.gather(values).backward(gradient)
    expected = torch.zeros_like(values)
    for row in range(4):
        positions = [i for i, owner in enumerate(indices) if owner == row]
        expected[row] = gradient[positions].double().sum(0).to(dtype)
    torch.testing.assert_close(values.grad, expected, rtol=0, atol=0)
