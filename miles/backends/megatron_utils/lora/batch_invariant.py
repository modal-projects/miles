"""Batch-invariant GEMMs for dense single-rank multi-LoRA."""

from __future__ import annotations

import inspect
import os
import threading
from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

try:
    # Registers torch.ops.miles_grouped_gemm, built from megatron_utils/kernels/grouped_gemm.
    import miles_grouped_gemm
except ImportError:
    miles_grouped_gemm = None

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
    if not args.deterministic_mode:
        raise ValueError(f"{OPTION} requires deterministic_mode")
    if args.qkv_format != "thd" or getattr(args, "allgather_cp", False):
        raise ValueError(f"{OPTION} requires THD without all-gather CP")
    for name in ("lora_dropout", "attention_dropout", "hidden_dropout"):
        if getattr(args, name, None) != 0:
            raise ValueError(f"{OPTION} requires {name}=0")
    # TE's flash backend is invariant on sm90 and sm100; its cuDNN backend cannot run deterministically on sm90.
    if getattr(args.attention_backend, "name", args.attention_backend) != "flash":
        raise ValueError(f"{OPTION} requires --attention-backend flash")
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


def _layout_args(layout: Partition) -> tuple[list[int], list[int], list[int], int]:
    return (
        [span.start for span in layout.spans],
        [span.rows for span in layout.spans],
        [span.padded_rows for span in layout.spans],
        layout.real_rows,
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


@torch.library.custom_op(f"{_LIB}::frozen_head", mutates_args=())
def _frozen_head_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    starts: list[int],
    rows: list[int],
    padded: list[int],
    real_rows: int,
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
def _(x, weight, starts, rows, padded, real_rows):
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
    _, weight, starts, rows, padded, real_rows = inputs
    ctx.save_for_backward(weight)
    ctx.layout = (starts, rows, padded, real_rows)


def _head_backward(ctx, gy: torch.Tensor):
    (weight,) = ctx.saved_tensors
    gx, bad = _frozen_head_backward_op(gy.contiguous(), weight, *ctx.layout)
    _record_padding_grad(bad)
    return gx, None, None, None, None, None


_frozen_head_op.register_autograd(_head_backward, setup_context=_head_setup)


_verified_gemms: set[tuple[object, ...]] = set()


def _column_major(t: torch.Tensor) -> bool:
    return t.stride(-2) == 1 and t.stride(-1) == t.size(-2)


def _cutlass_setup(ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    mat_a, mat_b, offs, allow_tf32 = inputs
    ctx.save_for_backward(mat_a, mat_b, offs)
    ctx.allow_tf32 = allow_tf32
    ctx.column_major = (_column_major(mat_a), _column_major(mat_b))


def _cutlass_backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """torch._grouped_mm's backward formulas, so bf16 gradients match it bitwise."""
    mat_a, mat_b, offs = ctx.saved_tensors
    a_column_major, b_column_major = ctx.column_major
    op = partial(torch.ops.miles_grouped_gemm.grouped_mm, allow_tf32=ctx.allow_tf32)
    # The kernel rejects broadcast (stride-0) gradients such as ones_like(...).expand.
    grad = grad.contiguous()
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = op(mat_b, grad.mT, offs).mT if a_column_major else op(grad, mat_b.mT, offs)
    if ctx.needs_input_grad[1]:
        grad_b = op(grad.mT, mat_a, offs).mT if b_column_major else op(mat_a.mT, grad, offs)
    return grad_a, grad_b, None, None


def _cutlass_fake(
    mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor, allow_tf32: bool = False
) -> torch.Tensor:
    if mat_a.ndim == 2 and mat_b.ndim == 2:
        return mat_a.new_empty(len(offs), mat_a.shape[0], mat_b.shape[1])
    if mat_a.ndim == 2:
        return mat_a.new_empty(mat_a.shape[0], mat_b.shape[-1])
    return mat_a.new_empty(mat_a.shape[1], mat_b.shape[1])


if miles_grouped_gemm is not None:
    torch.library.register_fake("miles_grouped_gemm::grouped_mm", _cutlass_fake)
    torch.library.register_autograd("miles_grouped_gemm::grouped_mm", _cutlass_backward, setup_context=_cutlass_setup)


def _grouped_mm(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    # torch._grouped_mm runs CUTLASS only for bf16 and otherwise loops cuBLAS per group, which is not invariant.
    if x.is_cuda:
        return torch.ops.miles_grouped_gemm.grouped_mm(x, weights.transpose(-2, -1), offsets, _allow_tf32())
    return torch._grouped_mm(x, weights.transpose(-2, -1), offsets)


def _allow_tf32() -> bool:
    return torch.get_float32_matmul_precision() != "highest"


def _group_offsets(sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(sizes, device=device).cumsum(0, dtype=torch.int32)


def _verify_grouped_mm(x: torch.Tensor, weights: torch.Tensor) -> None:
    """Fail closed unless one group's fwd, dgrad and wgrad ignore the other groups."""
    out_features, in_features = weights.shape[1:]
    key = (x.device, x.dtype, out_features, in_features, x.dtype == torch.float32 and _allow_tf32())
    if key in _verified_gemms:
        return
    if x.is_cuda and (
        miles_grouped_gemm is None
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or torch.cuda.get_device_capability(x.device)[0] not in (8, 9, 10, 12)
    ):
        raise RuntimeError(f"{OPTION} needs the miles_grouped_gemm extension on sm80, sm90, sm100 or sm120")
    generator = torch.Generator(device=x.device).manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=x.device, dtype=x.dtype)

    rows = 257
    own_x, own_w, own_gy = randn(rows, in_features), randn(out_features, in_features), randn(rows, out_features)

    def run(before: tuple[int, ...], after: tuple[int, ...], slack: int) -> tuple[torch.Tensor, ...]:
        sizes = (*before, rows + slack, *after)
        start, slot = sum(before), len(before)
        inputs = torch.cat((randn(start, in_features), own_x, randn(slack + sum(after), in_features)))
        weights = torch.stack([own_w if i == slot else randn(out_features, in_features) for i in range(len(sizes))])
        # Padding rows carry zero upstream gradient, as the frozen head enforces.
        grad = torch.cat(
            (
                randn(start, out_features),
                own_gy,
                own_gy.new_zeros(slack, out_features),
                randn(sum(after), out_features),
            )
        )
        with torch.enable_grad():
            inputs.requires_grad_(True)
            weights.requires_grad_(True)
            output = _grouped_mm(inputs, weights, _group_offsets(sizes, x.device))
            grad_x, grad_w = torch.autograd.grad(output, (inputs, weights), grad)
        own = slice(start, start + rows)
        return output[own], grad_x[own], grad_w[slot]

    solo = run((), (), 0)
    # The kernel picks smaller tiles when rows per group <= 128; the empty groups cover that switch.
    for before, after, slack in (
        ((3000, 0), (64,), 127),
        ((1,), (0, 6000), 0),
        ((), (511,), 31),
        ((0,) * 4, (0,) * 3, 0),
    ):
        if not all(torch.equal(a, b) for a, b in zip(solo, run(before, after, slack), strict=True)):
            raise RuntimeError(
                f"grouped_mm is not batch-invariant for {in_features}->{out_features} {x.dtype} "
                f"on {x.device}; disable {OPTION}"
            )
    _verified_gemms.add(key)


def _check_dense_grad(grad: torch.Tensor, *, real_rows: int) -> torch.Tensor:
    _record_padding_grad(_any_nonzero(grad[real_rows:]))
    # The grouped-GEMM backward rejects broadcast gradients.
    return grad.contiguous()


def dense_mm(
    x: torch.Tensor,
    weights: torch.Tensor,
    layout: Partition,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 2 or weights.ndim != 3:
        raise ValueError("Expected flat THD inputs and stacked dense LoRA weights")
    if len(x) != layout.total_rows or len(weights) != len(layout.spans):
        raise ValueError("LoRA shapes disagree with the batch partition")
    _verify_grouped_mm(x, weights)
    if offsets is None:
        offsets = _group_offsets(layout.counts, x.device)
    # Per-group grouped-GEMM tiling does not depend on the other groups, so no solo padding is needed.
    output = _grouped_mm(x, weights, offsets)
    if output.requires_grad:
        output.register_hook(partial(_check_dense_grad, real_rows=layout.real_rows))
    return output


def frozen_head_mm(x: torch.Tensor, weight: torch.Tensor, layout: Partition) -> torch.Tensor:
    return _frozen_head_op(x.contiguous(), weight, *_layout_args(layout))


# Rejects Transformer Engine's quantized tensor subclasses.
_PLAIN_TENSORS = (torch.Tensor, torch.nn.Parameter)
_TE_GEMM_PARAMS = (
    "A",
    "B",
    "out_dtype",
    "quantization_params",
    "gelu",
    "alpha",
    "accumulate",
    "layout",
    "out",
    "bias",
    "ub",
)


def _te_gemm_modules() -> tuple[ModuleType, ...]:
    try:
        from transformer_engine.pytorch.module import layernorm_linear, linear
    except ImportError:
        return ()
    return linear, layernorm_linear


def _frozen_gemm(x: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    # One group: the grouped kernel never splits K and its tile choice leaves every row's sum unchanged, while
    # cuBLAS picks split-K kernels by M, so a small batch's rows change bits once other rows are packed beside it.
    offsets = torch.tensor([len(x)], device=x.device, dtype=torch.int32)
    return torch.ops.miles_grouped_gemm.grouped_mm(x, mat_b, offsets, False).contiguous()


def _te_gemm(original: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(original)
    if not set(_TE_GEMM_PARAMS) <= set(signature.parameters):
        raise RuntimeError("Unsupported Transformer Engine general_gemm signature")

    @wraps(original)
    def general_gemm(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        # Wgrad ("NT") only runs for trainable weights, which the frozen base does not have.
        if _active_unit is None or p["layout"] == "NT":
            return original(*args, **kwargs)
        a, b = p["A"], p["B"]
        if (
            type(a) not in _PLAIN_TENSORS
            or type(b) not in _PLAIN_TENSORS
            or a.dtype != b.dtype
            or a.dtype not in (torch.bfloat16, torch.float16)
            or p["out_dtype"] not in (None, a.dtype)
            or p["quantization_params"] is not None
            or p["gelu"]
            or p["accumulate"]
            or p["bias"] is not None
            or p["ub"] is not None
            or p["alpha"] != 1.0
        ):
            raise RuntimeError(f"{OPTION} supports only plain BF16/FP16 frozen Transformer Engine GEMMs")
        # TE computes B @ op(A)^T: "TN" is fprop with A the [out, in] weight, "NN" is dgrad.
        mat_b = a.t() if p["layout"] == "TN" else a
        out = _frozen_gemm(b.reshape(-1, b.shape[-1]), mat_b.unsqueeze(0)).view(*b.shape[:-1], mat_b.shape[1])
        if p["out"] is not None:
            p["out"].copy_(out)
            out = p["out"]
        return out, None, None, None

    return general_gemm


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
    te_gemm_modules: Sequence[ModuleType | SimpleNamespace] | None = None,
) -> None:
    if not dist.is_initialized() or dist.get_world_size() != 1:
        raise ValueError("Batch-invariant LoRA requires one initialized rank")
    if layers is None:
        # Megatron is optional in the CPU control-plane environment.
        from megatron.bridge.peft import multi_lora_layers as layers
    expected = ("x", "stacked_weights", "token_splits", "offsets")
    if tuple(inspect.signature(layers._dense_multi_lora_mm).parameters) != expected:
        raise RuntimeError("Unsupported Megatron-Bridge dense LoRA signature")
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("Batch-invariant LoRA requires torch._grouped_mm")
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
            return dense_mm(x, stacked_weights, unit.partition, offsets)

        setattr(mm, _MARKER, True)
        layers._dense_multi_lora_mm = mm
    # Frozen base linears run at the packed row count, which picks cuBLAS kernels by M.
    for module in _te_gemm_modules() if te_gemm_modules is None else te_gemm_modules:
        if not getattr(module.general_gemm, _MARKER, False):
            gemm = _te_gemm(module.general_gemm)
            setattr(gemm, _MARKER, True)
            module.general_gemm = gemm
    head = heads[0]
    if not getattr(head.forward, _MARKER, False):
        forward = _head_forward(head)
        setattr(forward, _MARKER, True)
        head.forward = forward
