"""The common kernel policy retains grouped ownership and every gradient."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.kernel_policy import frozen_expert_forward, grouped_rowwise
from miles.backends.training_utils.prefix_tree.normalization import _projection


@pytest.mark.parametrize("counts", [(2, 0, 3), (0, 4, 0), (1, 1, 1)])
def test_grouped_policy_matches_each_weight_and_input_gradient(counts):
    torch.manual_seed(17)
    x = torch.randn(sum(counts), 7, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(len(counts), 7, 5, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor(counts).cumsum(0)
    actual = grouped_rowwise(x, weights, offsets)
    expected = torch.cat([part @ weight for part, weight in zip(x.split(counts), weights, strict=True)])
    upstream = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad((actual * upstream).sum(), (x, weights))
    b = torch.autograd.grad((expected * upstream).sum(), (x, weights))
    for got, reference in zip(a, b, strict=True):
        torch.testing.assert_close(got, reference, atol=1e-12, rtol=1e-12)


def test_grouped_policy_rejects_incomplete_row_inventory():
    with pytest.raises(ValueError, match="offsets"):
        grouped_rowwise(torch.randn(5, 3), torch.randn(2, 3, 4), torch.tensor([2, 4]))


def test_frozen_experts_keep_input_gradient_with_empty_experts():
    module = SimpleNamespace(num_gemms=3, use_bias=False, tp_size=1)
    for i in range(3):
        setattr(module, f"weight{i}", torch.randn(5, 7, dtype=torch.float64))
    x = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    actual, bias = frozen_expert_forward(module, x, [2, 0, 2])
    expected = torch.cat([F.linear(part, getattr(module, f"weight{i}")) for i, part in enumerate(x.split([2, 0, 2]))])
    assert bias is None
    torch.testing.assert_close(actual, expected)
    (a,) = torch.autograd.grad(actual.square().sum(), (x,))
    (b,) = torch.autograd.grad(expected.square().sum(), (x,))
    torch.testing.assert_close(a, b)


def test_projection_preserves_provider_reciprocal_and_its_derivative():
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 7, dtype=torch.float64)

    # Deliberately nonstandard RMS epsilon placement: the adapter must delegate.
    def native(value, matrix, eps):
        return F.linear(value, matrix), (value.square().sum(-1, keepdim=True).sqrt() + eps).reciprocal()

    got, rms = _projection(native)(x, weight, 0.17)
    expected, expected_rms = native(x, weight, 0.17)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(rms, expected_rms, atol=0, rtol=0)
    (a,) = torch.autograd.grad((got * rms).sum(), (x,))
    (b,) = torch.autograd.grad((expected * expected_rms).sum(), (x,))
    torch.testing.assert_close(a, b)
