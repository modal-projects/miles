"""Explicit row-wise math controls shared by architecture execution adapters."""

import torch


class _RowwiseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight):
        ctx.save_for_backward(value, weight)
        return torch.bmm(value.unsqueeze(1), weight.t().unsqueeze(0).expand(len(value), -1, -1)).squeeze(1)

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        with torch.autocast(value.device.type, enabled=False):
            # Avoid materializing [rows, output, input] weight gradients from a
            # broadcasted BMM. This is the ordinary linear adjoint; shared-row
            # fan-in still has the caller's dtype/rounding semantics.
            dx = (
                torch.bmm(gradient.unsqueeze(1), weight.unsqueeze(0).expand(len(value), -1, -1)).squeeze(1)
                if ctx.needs_input_grad[0]
                else None
            )
            dw = gradient.t().matmul(value) if ctx.needs_input_grad[1] else None
        return dx, dw


def rowwise_linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Same per-row forward expression for frozen or trainable weights.

    The custom adjoint avoids allocating one full weight gradient per row.
    It does not claim expanded-BF16 bitwise gradient equivalence: shared
    activation adjoints can merge before a low-precision cast.
    """
    if value.ndim != 2 or weight.ndim != 2 or value.shape[1] != weight.shape[1]:
        raise ValueError("Expected [rows, input] values and [output, input] weights")
    if value.dtype != weight.dtype or value.device != weight.device:
        raise ValueError("Row-wise values and weights must have the same dtype and device")
    with torch.autocast(value.device.type, enabled=False):
        return _RowwiseLinear.apply(value, weight)


def rowwise_frozen_linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project independent rows without a row-count-dependent ordinary GEMM.

    This opt-in operation expresses each row as the same 1×K batched product.
    Unlike ordinary GEMM, changing the batch need not switch the M tiling or
    split-K reduction. The CUDA diagnostic verifies dispatch for tested shapes;
    this function does not promise bitwise invariance on every device/library.

    Frozen weights avoid the large per-row weight-gradient intermediates of an
    expanded batched operand. Input gradients remain ordinary PyTorch autograd.
    Preserve the supplied precision even inside an outer autocast context.
    """
    if value.ndim != 2 or weight.ndim != 2 or value.shape[1] != weight.shape[1]:
        raise ValueError("Expected [rows, input] values and [output, input] weights")
    if weight.requires_grad:
        raise ValueError("Row-wise projection requires frozen weights")
    if value.dtype != weight.dtype or value.device != weight.device:
        raise ValueError("Row-wise values and weights must have the same dtype and device")
    with torch.autocast(value.device.type, enabled=False):
        return torch.bmm(value.unsqueeze(1), weight.t().unsqueeze(0).expand(len(value), -1, -1)).squeeze(1)
