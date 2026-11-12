import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/glm53_lora_kernel_control.py"
_SPEC = importlib.util.spec_from_file_location("_lora_kernel_control", _PATH)
_CONTROL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTROL)


@pytest.mark.parametrize("implementation", ["grouped_reference", "grouped_rowwise"])
def test_grouped_reference_preserves_rows_and_every_gradient_with_an_empty_group(implementation):
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    a, b = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    coefficients = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    expected = torch.bmm(a[:, None], b[torch.tensor([0, 0, 2, 2, 2])]).squeeze(1)
    actual = getattr(_CONTROL, implementation)(x, w, torch.tensor([2, 2, 5]))
    (expected * coefficients).sum().backward()
    (actual * coefficients).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad, a.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(w.grad, b.grad, rtol=1e-12, atol=1e-12)
    assert not w.grad[1].any()


def test_frozen_expert_control_preserves_input_adjoint_with_empty_group():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    weights = [torch.randn(4, 3, generator=generator, dtype=torch.float64) for _ in range(3)]
    module = SimpleNamespace(
        num_gemms=3, use_bias=False, tp_size=1, **{f"weight{i}": w for i, w in enumerate(weights)}
    )
    actual, bias = _CONTROL.frozen_expert_forward(module, x, [2, 0, 3])
    expected = torch.cat([part @ w.t() for part, w in zip(reference.split([2, 0, 3]), weights, strict=True)])
    coefficients = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    assert bias is None
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad, reference.grad, rtol=1e-12, atol=1e-12)
