"""Fixed per-adapter GEMM shapes for dense single-rank multi-LoRA."""

from __future__ import annotations

import inspect
import os
import threading
from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, wraps
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

OPTION = "--lora-batch-invariant"
_MARKER = "_miles_batch_invariant_lora"
_unit_lock = threading.Lock()


@dataclass(frozen=True)
class Span:
    start: int
    rows: int
    padded_rows: int


@dataclass(frozen=True)
class Partition:
    spans: tuple[Span, ...]
    counts: tuple[int, ...]
    real_rows: int
    total_rows: int

    @cached_property
    def tail_slot(self) -> int:
        return next((i for i, (span, n) in enumerate(zip(self.spans, self.counts, strict=True)) if n > span.rows), -1)


@dataclass
class _WorkUnit:
    partition: Partition | None = None
    head_calls: int = 0
    bad_padding_grad: torch.Tensor | None = None


# Autograd recomputation can run on a different Python thread.
_active_unit: _WorkUnit | None = None


def validate_args(args: Namespace | SimpleNamespace) -> None:
    for name in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
    ):
        if getattr(args, name, None) != 1:
            raise ValueError(f"{OPTION} requires {name}=1")
    if not args.deterministic_mode or not args.bf16:
        raise ValueError(f"{OPTION} requires deterministic_mode and bf16")
    if args.qkv_format != "thd" or getattr(args, "allgather_cp", False):
        raise ValueError(f"{OPTION} requires THD without all-gather CP")
    for name in ("lora_dropout", "attention_dropout", "hidden_dropout"):
        if getattr(args, name, None) != 0:
            raise ValueError(f"{OPTION} requires {name}=0")
    if args.enable_mtp_training or getattr(args, "num_experts", None):
        raise ValueError(f"{OPTION} does not support MTP or MoE")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError(f"{OPTION} requires CUBLAS_WORKSPACE_CONFIG")


@contextmanager
def work_unit(active: bool) -> Generator[None, None, None]:
    global _active_unit
    if not active:
        yield
        return
    if not _unit_lock.acquire(blocking=False):
        raise RuntimeError("Batch-invariant LoRA requires serialized work units")
    unit = _WorkUnit()
    _active_unit = unit
    try:
        yield
        if unit.bad_padding_grad is not None and bool(unit.bad_padding_grad):
            raise RuntimeError("Batch padding carries a nonzero upstream gradient")
        if unit.partition is None or unit.head_calls == 0:
            raise RuntimeError("Batch-invariant LoRA did not observe the frozen head")
    finally:
        _active_unit = None
        _unit_lock.release()


def partition(
    slots: Sequence[int],
    lengths: Sequence[int],
    counts: Sequence[int],
    multiple: int,
    total_rows: int,
) -> Partition:
    slots, lengths, counts = tuple(slots), tuple(lengths), tuple(counts)
    if not slots or len(slots) != len(lengths) or slots != tuple(sorted(slots)):
        raise ValueError("Expected sorted adapter slots and matching token lengths")
    if multiple < 1 or any(n < 1 for n in lengths):
        raise ValueError("Expected positive token lengths and padding multiple")
    if any(s < 0 or s >= len(counts) for s in slots):
        raise ValueError("Adapter slot outside the resident table")
    real = [0] * len(counts)
    for slot, length in zip(slots, lengths, strict=True):
        real[slot] += length
    n_real = sum(real)
    expected = real.copy()
    expected[slots[-1]] += -n_real % multiple
    if tuple(expected) != counts or sum(expected) != total_rows:
        raise ValueError("Adapter counts disagree with THD padding")
    start = 0
    spans = []
    for n in real:
        spans.append(Span(start, n, n + (-n % multiple) if n else 0))
        start += n
    return Partition(tuple(spans), counts, n_real, total_rows)


_LIB = "miles_bi_lora"


def _layout_args(layout: Partition) -> tuple[list[int], list[int], list[int], int, int]:
    return (
        [span.start for span in layout.spans],
        [span.rows for span in layout.spans],
        [span.padded_rows for span in layout.spans],
        layout.real_rows,
        layout.tail_slot,
    )


def _offsets(padded: Sequence[int]) -> list[int]:
    offsets = [0]
    for n in padded:
        offsets.append(offsets[-1] + n)
    return offsets


def _slack(rows: Sequence[int], padded: Sequence[int]) -> int:
    return max(p - n for n, p in zip(rows, padded, strict=True))


def _record_padding_grad(bad: torch.Tensor) -> None:
    unit = _active_unit
    if unit is None:
        raise RuntimeError("Padding gradient has no active work unit")
    unit.bad_padding_grad = bad if unit.bad_padding_grad is None else unit.bad_padding_grad | bad


def _any_nonzero(grad: torch.Tensor) -> torch.Tensor:
    return grad.ne(0).any() if grad.numel() else grad.new_zeros((), dtype=torch.bool)


def _mm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    # Keep cuBLAS pointer alignment identical to a fresh allocation.
    if out.storage_offset() * out.element_size() % 256:
        out.copy_(torch.mm(a, b))
    else:
        torch.mm(a, b, out=out)


def _pack(x: torch.Tensor, starts: Sequence[int], rows: Sequence[int], padded: Sequence[int]) -> torch.Tensor:
    offsets = _offsets(padded)
    packed = x.new_empty((offsets[-1], x.shape[1]))
    for start, n, p, offset in zip(starts, rows, padded, offsets, strict=False):
        packed[offset : offset + n].copy_(x[start : start + n])
        packed[offset + n : offset + p].zero_()
    return packed


@torch.library.custom_op(f"{_LIB}::dense_mm", mutates_args=())
def _dense_mm_op(
    x: torch.Tensor,
    weights: torch.Tensor,
    starts: list[int],
    rows: list[int],
    padded: list[int],
    real_rows: int,
    tail_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = _offsets(padded)
    xp = _pack(x, starts, rows, padded)
    # Padded rows spill into the next span, which overwrites them afterwards.
    y = x.new_empty((len(x) + _slack(rows, padded), weights.shape[1]))
    for slot, (start, n, p, offset) in enumerate(zip(starts, rows, padded, offsets, strict=False)):
        if n:
            _mm_out(xp[offset : offset + p], weights[slot].t(), y[start : start + p])
    if real_rows < len(x):
        _mm_out(x[real_rows:], weights[tail_slot].t(), y[real_rows : len(x)])
    return y[: len(x)], xp


@_dense_mm_op.register_fake
def _(x, weights, starts, rows, padded, real_rows, tail_slot):
    return x.new_empty((len(x), weights.shape[1])), x.new_empty((sum(padded), x.shape[1]))


@torch.library.custom_op(f"{_LIB}::dense_mm_backward", mutates_args=())
def _dense_mm_backward_op(
    gy: torch.Tensor,
    xp: torch.Tensor,
    weights: torch.Tensor,
    starts: list[int],
    rows: list[int],
    padded: list[int],
    real_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = _offsets(padded)
    total = len(gy)
    gyp = _pack(gy, starts, rows, padded)
    gx = xp.new_empty((total + _slack(rows, padded), xp.shape[1]))
    gw = torch.zeros_like(weights)
    for slot, (start, n, p, offset) in enumerate(zip(starts, rows, padded, offsets, strict=False)):
        if n:
            g = gyp[offset : offset + p]
            _mm_out(g, weights[slot], gx[start : start + p])
            torch.mm(g.t(), xp[offset : offset + p], out=gw[slot])
    gx[real_rows:total].zero_()
    return gx[:total], gw, _any_nonzero(gy[real_rows:])


@_dense_mm_backward_op.register_fake
def _(gy, xp, weights, starts, rows, padded, real_rows):
    return gy.new_empty((len(gy), xp.shape[1])), torch.empty_like(weights), gy.new_empty((), dtype=torch.bool)


def _dense_setup(ctx, inputs, output) -> None:
    _, weights, starts, rows, padded, real_rows, _ = inputs
    ctx.save_for_backward(output[1], weights)
    ctx.layout = (starts, rows, padded, real_rows)


def _dense_backward(ctx, gy: torch.Tensor, _gxp: torch.Tensor | None):
    xp, weights = ctx.saved_tensors
    gx, gw, bad = _dense_mm_backward_op(gy.contiguous(), xp, weights, *ctx.layout)
    _record_padding_grad(bad)
    return gx, gw, None, None, None, None, None


_dense_mm_op.register_autograd(_dense_backward, setup_context=_dense_setup)


@torch.library.custom_op(f"{_LIB}::frozen_head", mutates_args=())
def _frozen_head_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    starts: list[int],
    rows: list[int],
    padded: list[int],
    real_rows: int,
    tail_slot: int,
) -> torch.Tensor:
    offsets = _offsets(padded)
    xp = _pack(x, starts, rows, padded)
    # Padded rows spill into the next span, which overwrites them afterwards.
    y = x.new_empty((len(x) + _slack(rows, padded), weight.shape[0]))
    for start, n, p, offset in zip(starts, rows, padded, offsets, strict=False):
        if n:
            _mm_out(xp[offset : offset + p], weight.t(), y[start : start + p])
    if real_rows < len(x):
        _mm_out(x[real_rows:], weight.t(), y[real_rows : len(x)])
    return y[: len(x)]


@_frozen_head_op.register_fake
def _(x, weight, starts, rows, padded, real_rows, tail_slot):
    return x.new_empty((len(x), weight.shape[0]))


@torch.library.custom_op(f"{_LIB}::frozen_head_backward", mutates_args=())
def _frozen_head_backward_op(
    gy: torch.Tensor,
    weight: torch.Tensor,
    starts: list[int],
    rows: list[int],
    padded: list[int],
    real_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = len(gy)
    gx = gy.new_empty((total + _slack(rows, padded), weight.shape[1]))
    for start, n, p in zip(starts, rows, padded, strict=True):
        if not n:
            continue
        # dgrad rows depend only on their own upstream row, so a view suffices.
        grad = gy[start : start + p]
        if len(grad) < p:
            grad = torch.cat((grad, gy.new_zeros((p - len(grad), gy.shape[1]))))
        _mm_out(grad, weight, gx[start : start + p])
    gx[real_rows:total].zero_()
    return gx[:total], _any_nonzero(gy[real_rows:])


@_frozen_head_backward_op.register_fake
def _(gy, weight, starts, rows, padded, real_rows):
    return gy.new_empty((len(gy), weight.shape[1])), gy.new_empty((), dtype=torch.bool)


def _head_setup(ctx, inputs, output) -> None:
    _, weight, starts, rows, padded, real_rows, _ = inputs
    ctx.save_for_backward(weight)
    ctx.layout = (starts, rows, padded, real_rows)


def _head_backward(ctx, gy: torch.Tensor):
    (weight,) = ctx.saved_tensors
    gx, bad = _frozen_head_backward_op(gy.contiguous(), weight, *ctx.layout)
    _record_padding_grad(bad)
    return gx, None, None, None, None, None, None


_frozen_head_op.register_autograd(_head_backward, setup_context=_head_setup)


def dense_mm(x: torch.Tensor, weights: torch.Tensor, layout: Partition) -> torch.Tensor:
    if x.ndim != 2 or weights.ndim != 3:
        raise ValueError("Expected flat THD inputs and stacked dense LoRA weights")
    if len(x) != layout.total_rows or len(weights) != len(layout.spans):
        raise ValueError("LoRA shapes disagree with the batch partition")
    return _dense_mm_op(x.contiguous(), weights.contiguous(), *_layout_args(layout))[0]


def frozen_head_mm(x: torch.Tensor, weight: torch.Tensor, layout: Partition) -> torch.Tensor:
    return _frozen_head_op(x.contiguous(), weight, *_layout_args(layout))


def _head_forward(head: torch.nn.Module) -> Callable[..., tuple[torch.Tensor, None]]:
    original = head.forward

    @wraps(original)
    def forward(inputs: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        unit = _active_unit
        if unit is None or unit.partition is None:
            raise RuntimeError("Frozen head has no active adapter partition")
        layout = unit.partition
        weight = kwargs.get("weight")
        if weight is None and args:
            weight = args[0]
        if weight is None:
            weight = head.weight
        if weight is None or weight.requires_grad:
            raise ValueError("Batch-invariant LoRA requires frozen head weights")
        if inputs.ndim != 3 or inputs.shape[1] != 1 or inputs.shape[0] != layout.total_rows:
            raise ValueError("Expected a single-rank THD frozen head")
        if getattr(head, "bias", None) is not None:
            raise ValueError("Batch-invariant LoRA requires a bias-free head")
        output = frozen_head_mm(inputs[:, 0], weight, layout)
        unit.head_calls += 1
        return output.unsqueeze(1), None

    return forward


def _batch_reader(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(original)
    def get_batch(
        data_iterator: object,
        keys: Sequence[str],
        pad_multiplier: int = 128,
        qkv_format: str = "thd",
        get_position_ids: bool = False,
        allgather_cp: bool = False,
    ) -> dict[str, Any]:
        batch = original(
            data_iterator,
            keys,
            pad_multiplier=pad_multiplier,
            qkv_format=qkv_format,
            get_position_ids=get_position_ids,
            allgather_cp=allgather_cp,
        )
        unit = _active_unit
        if unit is not None:
            if qkv_format != "thd" or allgather_cp:
                raise ValueError("Batch-invariant LoRA requires single-rank THD")
            unit.partition = partition(
                batch["adapter_slots"],
                [t.numel() for t in batch["unconcat_tokens"]],
                batch["adapter_token_counts"].tolist(),
                pad_multiplier,
                batch["tokens"].numel(),
            )
        return batch

    return get_batch


def install(
    megatron_model: ModuleType | SimpleNamespace,
    models: Sequence[torch.nn.Module],
    *,
    layers: ModuleType | SimpleNamespace | None = None,
) -> None:
    if not dist.is_initialized() or dist.get_world_size() != 1:
        raise ValueError("Batch-invariant LoRA requires one initialized rank")
    if layers is None:
        # Megatron is optional in the CPU control-plane environment.
        from megatron.bridge.peft import multi_lora_layers as layers
    expected = ("x", "stacked_weights", "token_splits", "offsets")
    if tuple(inspect.signature(layers._dense_multi_lora_mm).parameters) != expected:
        raise RuntimeError("Unsupported Megatron-Bridge dense LoRA signature")
    heads = [m for chunk in models for n, m in chunk.named_modules() if n.rsplit(".", 1)[-1] == "output_layer"]
    if len(heads) != 1 or any(p.requires_grad for p in heads[0].parameters()):
        raise ValueError("Batch-invariant LoRA requires one frozen output layer")
    if not getattr(megatron_model.get_batch, _MARKER, False):
        reader = _batch_reader(megatron_model.get_batch)
        setattr(reader, _MARKER, True)
        megatron_model.get_batch = reader
    if not getattr(layers._dense_multi_lora_mm, _MARKER, False):
        original = layers._dense_multi_lora_mm

        @wraps(original)
        def mm(
            x: torch.Tensor,
            stacked_weights: torch.Tensor,
            *,
            token_splits: Sequence[int],
            offsets: torch.Tensor | None,
        ) -> torch.Tensor:
            unit = _active_unit
            if unit is None:
                return original(x, stacked_weights, token_splits=token_splits, offsets=offsets)
            if unit.partition is None or tuple(token_splits) != unit.partition.counts:
                raise ValueError("LoRA rows disagree with the active partition")
            return dense_mm(x, stacked_weights, unit.partition)

        setattr(mm, _MARKER, True)
        layers._dense_multi_lora_mm = mm
    head = heads[0]
    if not getattr(head.forward, _MARKER, False):
        forward = _head_forward(head)
        setattr(forward, _MARKER, True)
        head.forward = forward
