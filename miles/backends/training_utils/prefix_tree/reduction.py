"""Fixed-order shared-row gradient reduction, without CUDA scatter atomics."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


_MAX_ORDERED_ELEMENTS = 1 << 24


class _OrderedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices, order, lengths):
        ctx.save_for_backward(order, lengths)
        ctx.input_dtype = value.dtype
        return value.index_select(0, indices)

    @staticmethod
    def backward(ctx, gradient):
        order, lengths = ctx.saved_tensors
        dtype = torch.float64 if ctx.input_dtype == torch.float64 else torch.float32
        if gradient.ndim != 2 or gradient.numel() <= _MAX_ORDERED_ELEMENTS or torch.is_grad_enabled():
            ordered = gradient.index_select(0, order).to(dtype)
            reduced = torch.segment_reduce(ordered, reduce="sum", lengths=lengths)
            return reduced.to(ctx.input_dtype), None, None, None
        result = gradient.new_empty((lengths.numel(), gradient.shape[1]), dtype=ctx.input_dtype)
        columns = max(1, _MAX_ORDERED_ELEMENTS // max(gradient.shape[0], 1))
        # Bound the FP32 reorder workspace independently of vocabulary width.
        for start in range(0, gradient.shape[1], columns):
            end = start + columns
            ordered = gradient[:, start:end].index_select(0, order).to(dtype)
            reduced = torch.segment_reduce(ordered, reduce="sum", lengths=lengths)
            result[:, start:end].copy_(reduced.to(ctx.input_dtype))
            del ordered, reduced
        return result, None, None, None


@dataclass(frozen=True)
class OrderedRowGather:
    """Prepare once on CPU, then reuse on device for forward/backward.

    Deterministic for a fixed occurrence layout. Changing the tree can change
    the contributions themselves (kernel shapes/rounding), not just their order.
    """

    indices: torch.Tensor
    order: torch.Tensor
    lengths: torch.Tensor

    @classmethod
    def from_indices(cls, indices: Sequence[int], input_rows: int, *, device) -> "OrderedRowGather":
        if input_rows < 1 or any(index < 0 or index >= input_rows for index in indices):
            raise ValueError("row indices must address the input tensor")
        # Stable counting sort: row IDs are dense integers, so preparation is
        # O(gathered rows + input rows), rather than a comparison sort.
        occurrences = [[] for _ in range(input_rows)]
        for position, index in enumerate(indices):
            occurrences[index].append(position)
        lengths = [len(row) for row in occurrences]
        order = [position for row in occurrences for position in row]
        return cls(
            indices=torch.tensor(indices, dtype=torch.long, device=device),
            order=torch.tensor(order, dtype=torch.long, device=device),
            lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        )

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return _OrderedGather.apply(value, self.indices, self.order, self.lengths)
