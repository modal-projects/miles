import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/glm53_projection_control.py"
_SPEC = importlib.util.spec_from_file_location("_projection_control", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
projection_control = _MODULE.projection_control


@pytest.mark.parametrize("inside_sqrt", [True, False])
@pytest.mark.parametrize("mode", ["bmm", "fp64"])
def test_projection_oracle_preserves_provider_epsilon_and_normalization_gradient(inside_sqrt, mode):
    torch.manual_seed(17)
    # Low activation variance makes moving epsilon across sqrt a large error,
    # unlike the approximately unit-variance reduced random-model case.
    x = (torch.randn(23, 17, dtype=torch.float64) * 1e-4).requires_grad_()
    weight = torch.randn(5, 17, dtype=torch.float64)
    eps = 1e-5

    def native(value, matrix, epsilon):
        mean_square = value.square().mean(-1, keepdim=True)
        reciprocal = torch.rsqrt(mean_square + epsilon) if inside_sqrt else (mean_square.sqrt() + epsilon).reciprocal()
        return value @ matrix.t(), reciprocal

    actual_p, actual_r = projection_control(native, mode)(x, weight, eps)
    expected_p, expected_r = native(x, weight, eps)
    torch.testing.assert_close(actual_r, expected_r, atol=0, rtol=0)
    actual_norm_gradient = torch.autograd.grad(actual_r.sum(), x, retain_graph=True)[0]
    expected_norm_gradient = torch.autograd.grad(expected_r.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(actual_norm_gradient, expected_norm_gradient, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(actual_p.square().sum() + actual_r.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected_p.square().sum() + expected_r.sum(), x)[0]
    torch.testing.assert_close(actual_p, expected_p, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)


def test_rowwise_control_refuses_trainable_projection_weights():
    x = torch.ones(2, 3)
    weight = torch.ones(4, 3, requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        projection_control(lambda *args: None, "bmm")(x, weight, 1e-5)
