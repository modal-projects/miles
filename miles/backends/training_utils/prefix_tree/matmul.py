"""Fixed CUDA tiling for row-count-independent low-precision projections.

Import only on CUDA: the planner and CPU correctness tests do not require Triton.
No autotuning or row-count specialization changes the reduction at shared rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "inner"])
def _row_linear_kernel(
    x,
    w,
    out,
    rows,
    columns: tl.constexpr,
    inner,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sw0: tl.constexpr,
    sw1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    c = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(inner, BK)):
        indices = (block * BK + k).to(tl.int64)
        a = tl.load(
            x + r[:, None] * sx0 + indices[None, :] * sx1,
            mask=(r[:, None] < rows) & (indices[None, :] < inner),
            other=0,
        )
        b = tl.load(
            w + indices[:, None] * sw1 + c[None, :] * sw0,
            mask=(indices[:, None] < inner) & (c[None, :] < columns),
            other=0,
        )
        accumulator = tl.dot(a, b, accumulator)
    tl.store(
        out + r[:, None] * columns + c[None, :],
        accumulator.to(out.dtype.element_ty),
        mask=(r[:, None] < rows) & (c[None, :] < columns),
    )


def fixed_row_linear(value, weight):
    """A CUDA forward primitive; autograd ownership stays with rowwise_linear."""
    rows, inner = value.shape
    columns = weight.shape[0]
    result = torch.empty((rows, columns), dtype=value.dtype, device=value.device)
    if rows and columns:
        _row_linear_kernel[(triton.cdiv(rows, 16), triton.cdiv(columns, 64))](
            value,
            weight,
            result,
            rows,
            columns,
            inner,
            *value.stride(),
            *weight.stride(),
            BM=16,
            BN=64,
            BK=32,
            num_warps=4,
            num_stages=2
        )
    return result
