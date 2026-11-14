"""Fixed per-adapter GEMM shapes for dense single-rank multi-LoRA."""

from __future__ import annotations

import inspect
import os
import threading
from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

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


@dataclass
class _WorkUnit:
    partition: Partition | None = None
    head_calls: int = 0


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


class _UnusedRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor) -> None:
        (grad,) = grad_outputs
        if bool(torch.count_nonzero(grad)):
            raise RuntimeError("Batch padding carries a nonzero upstream gradient")
        return None


def _padded_input(x: torch.Tensor, span: Span) -> torch.Tensor:
    return torch.cat(
        (
            x[span.start : span.start + span.rows],
            x.new_zeros((span.padded_rows - span.rows, *x.shape[1:])),
        ),
        dim=0,
    )


def dense_mm(x: torch.Tensor, weights: torch.Tensor, layout: Partition) -> torch.Tensor:
    if x.ndim != 2 or weights.ndim != 3:
        raise ValueError("Expected flat THD inputs and stacked dense LoRA weights")
    if len(x) != layout.total_rows or len(weights) != len(layout.spans):
        raise ValueError("LoRA shapes disagree with the batch partition")
    outputs = []
    for slot, (span, weight) in enumerate(zip(layout.spans, weights, strict=True)):
        if span.rows:
            out = F.linear(_padded_input(x, span), weight)[: span.rows]
        else:
            out = x.new_empty((0, weight.shape[0]))
        extra = layout.counts[slot] - span.rows
        if extra:
            tail = x[layout.real_rows :]
            out = torch.cat((out, _UnusedRows.apply(F.linear(tail, weight))), dim=0)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


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
        outputs = []
        for span in layout.spans:
            if span.rows:
                output, bias = original(_padded_input(inputs, span), *args, **kwargs)
                if bias is not None:
                    raise ValueError("Batch-invariant LoRA requires a bias-free head")
                outputs.append(output[: span.rows])
        if layout.real_rows < layout.total_rows:
            tail, bias = original(inputs[layout.real_rows :], *args, **kwargs)
            if bias is not None:
                raise ValueError("Batch-invariant LoRA requires a bias-free head")
            outputs.append(_UnusedRows.apply(tail))
        unit.head_calls += 1
        return torch.cat(outputs, dim=0), None

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
