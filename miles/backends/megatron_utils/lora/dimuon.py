"""Dimuon-lite: a per-factor Muon-type optimizer for multi-LoRA slots.

Each LoRA factor (A: ``[rank, in]``, B: ``[out, rank]``; packed experts add a
leading ``[num_local_experts]`` axis) is updated as

    m      <- beta1 * m + (1 - beta1) * g                      (EMA momentum)
    O      <- polar_express(m, steps=5)                         (per matrix block)
    O      <- O * sqrt(rank) / ||O||_F                          (per matrix block)
    param  <- param - lr * (rank / reference_rank)^exponent * O

which matches the behavior measured on Tinker's Dimuon for each factor's first
update: Frobenius norm ``sqrt(rank)`` per factor, the ``(r/32)^0.175`` rank
correction, no shape scale and no weight decay. Tinker's later updates add a
damping-dependent correction that is not reproduced here.

Fused projections are split into per-projection blocks before the matrix sign,
so one projection's gradient never mixes into another's direction: B of a fused
``linear_qkv`` is split into q/k/v row blocks per query group, and B of a gated
``linear_fc1`` into gate/up row blocks. A of a fused module is a single matrix
shared by its projections and is orthogonalized whole.

Tensor-parallel shards are all-gathered, orthogonalized whole on every TP rank,
and sliced back, so each rank applies the exact update of the unsharded matrix.

Padded rank rows of A and columns of B stay exactly zero: their gradients are
zero, and the polynomial iteration and scaling preserve zero rows and columns.

The math below is plain torch so it can be tested without Megatron.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

# Polar Express (arXiv 2505.16932) coefficients, including its 1.01 safety factor,
# as shipped in emerging_optimizers v0.3.0 ("polar_express"). Steps past the list
# repeat the last tuple.
POLAR_EXPRESS_COEFFICIENTS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)
NUM_POLAR_STEPS = 5
REFERENCE_RANK = 32
RANK_LR_CORRECTION_EXPONENT = 0.175
_EPS = 1e-7


def polar_express(x: torch.Tensor, steps: int = NUM_POLAR_STEPS, eps: float = _EPS) -> torch.Tensor:
    """Approximate the orthogonal polar factor of each trailing 2-D matrix of ``x`` (fp32)."""
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    for i in range(steps):
        a, b, c = POLAR_EXPRESS_COEFFICIENTS[min(i, len(POLAR_EXPRESS_COEFFICIENTS) - 1)]
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transpose else x


@dataclass(frozen=True)
class FactorSpec:
    """How to turn one LoRA factor parameter into Dimuon matrix blocks.

    ``rank`` is the slot's actual adapter rank (not the padded maximum).
    ``row_split`` splits rows of the *unsharded* matrix into projection blocks:
    the rows are viewed as ``(groups, sum(row_split))`` and block ``j`` collects
    the ``row_split[j]`` rows of every group. ``None`` means one block.
    ``partition_dim`` is the tensor-parallel shard axis of the trailing 2-D
    matrix (0 or 1), or ``None`` when the parameter is not TP-sharded.
    ``batched`` marks packed expert tensors with a leading expert axis.
    """

    rank: int
    row_split: tuple[int, ...] | None = None
    partition_dim: int | None = None
    tp_group: object | None = None
    batched: bool = False


def rank_scale(rank: int, exponent: float = RANK_LR_CORRECTION_EXPONENT) -> float:
    return (rank / REFERENCE_RANK) ** exponent


def _blocks(full: torch.Tensor, row_split: tuple[int, ...] | None) -> list[torch.Tensor]:
    """Split ``[..., rows, cols]`` into per-projection blocks of shape ``[..., rows_j, cols]``."""
    if row_split is None:
        return [full]
    *lead, rows, cols = full.shape
    group = sum(row_split)
    assert rows % group == 0, f"rows {rows} are not a multiple of the projection group {row_split}"
    grouped = full.reshape(*lead, rows // group, group, cols)
    return [b.reshape(*lead, -1, cols) for b in torch.split(grouped, row_split, dim=-2)]


def _unblocks(blocks: list[torch.Tensor], row_split: tuple[int, ...] | None, like: torch.Tensor) -> torch.Tensor:
    if row_split is None:
        return blocks[0]
    *lead, rows, cols = like.shape
    groups = rows // sum(row_split)
    parts = [b.reshape(*lead, groups, size, cols) for b, size in zip(blocks, row_split, strict=True)]
    return torch.cat(parts, dim=-2).reshape(like.shape)


def normalized_sign(full: torch.Tensor, rank: int, row_split: tuple[int, ...] | None) -> torch.Tensor:
    """Per-block polar factor scaled to Frobenius norm ``sqrt(rank)`` per matrix.

    ``full`` is the unsharded momentum, ``[rows, cols]`` or packed ``[E, rows, cols]``.
    An all-zero block (e.g. A before B has moved) yields a zero update."""
    out = []
    for block in _blocks(full.float(), row_split):
        sign = polar_express(block)
        norm = sign.norm(dim=(-2, -1), keepdim=True)
        out.append(torch.where(norm > _EPS, sign * (math.sqrt(rank) / norm.clamp_min(_EPS)), torch.zeros_like(sign)))
    return _unblocks(out, row_split, full)


GatherFn = Callable[[torch.Tensor, int, object], torch.Tensor]
SliceFn = Callable[[torch.Tensor, int, object], torch.Tensor]


def _dist_gather(local: torch.Tensor, dim: int, group) -> torch.Tensor:
    import torch.distributed as dist

    world = dist.get_world_size(group)
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def _dist_slice(full: torch.Tensor, dim: int, group) -> torch.Tensor:
    import torch.distributed as dist

    world, rank = dist.get_world_size(group), dist.get_rank(group)
    return full.chunk(world, dim=dim)[rank]


def factor_direction(
    momentum: torch.Tensor,
    spec: FactorSpec,
    *,
    gather: GatherFn = _dist_gather,
    slice_: SliceFn = _dist_slice,
) -> torch.Tensor:
    """Dimuon direction (before ``lr * rank_scale``) for this rank's shard of one factor."""
    dim = None
    if spec.partition_dim is not None:
        dim = spec.partition_dim + (1 if spec.batched else 0)
    full = momentum if dim is None else gather(momentum, dim, spec.tp_group)
    direction = normalized_sign(full, spec.rank, spec.row_split)
    return direction if dim is None else slice_(direction, dim, spec.tp_group)


class Dimuon(torch.optim.Optimizer):
    """Dimuon-lite over LoRA factor parameters; every parameter needs a :class:`FactorSpec`.

    Specs are keyed by ``id(param)`` of the tensors this optimizer steps (the FP32
    masters once Megatron wraps it) and are registered with :meth:`set_specs`."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = False,
        rank_lr_correction_exponent: float = RANK_LR_CORRECTION_EXPONENT,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            rank_lr_correction_exponent=rank_lr_correction_exponent,
        )
        super().__init__(params, defaults)
        self.specs: dict[int, FactorSpec] = {}

    def set_specs(self, specs: dict[int, FactorSpec]) -> None:
        self.specs = dict(specs)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "Dimuon does not support closures"
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                spec = self.specs.get(id(p))
                if spec is None:
                    raise KeyError(f"Dimuon has no FactorSpec for a parameter of shape {tuple(p.shape)}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad.float(), 1 - beta)
                m = p.grad.float().lerp(buf, beta) if group["nesterov"] else buf
                direction = factor_direction(m, spec)
                scale = group["lr"] * rank_scale(spec.rank, group["rank_lr_correction_exponent"])
                p.add_(direction.to(p.dtype), alpha=-scale)
        return None


def dimuon_init_state_fn(opt, config=None) -> None:
    """Lazily create Dimuon state (LayerWise checkpointing needs it before the first step)."""
    for group in opt.param_groups:
        for p in group["params"]:
            if len(opt.state[p]) == 0:
                opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.float32)


# --------------------------------------------------------------------------- Megatron wiring


def fused_row_split(module) -> tuple[int, ...] | None:
    """Projection row split for B of a fused ``linear_qkv`` / gated ``linear_fc1``, else ``None``.

    q/k/v rows are interleaved per query group (Megatron's layout), so the split is
    per group. Gate/up rows of a TP shard are ``[gate_shard; up_shard]``, so after
    gathering the shards the rows read ``[g0; u0; g1; u1; ...]`` and the split is
    ``(ffn_shard, ffn_shard)`` per shard."""
    name = module.base_linear_name.rsplit(".", 1)[-1]
    config = module.to_wrap.config
    if name == "linear_qkv":
        kv = config.kv_channels
        q = config.num_attention_heads // config.num_query_groups * kv
        if getattr(config, "attention_output_gate", False):
            return (q, q, kv, kv)
        return (q, kv, kv)
    if name == "linear_fc1" and getattr(config, "gated_linear_unit", False):
        out = module.adapters[0].linear_out.weight.shape[-2]  # local rows: 2 * ffn_shard
        assert out % 2 == 0, f"gated linear_fc1 adapter rows {out} are not even"
        return (out // 2, out // 2)
    return None


def build_factor_specs(model_chunks, slot: int) -> dict[int, FactorSpec]:
    """FactorSpecs for slot ``slot``'s adapter params, keyed by ``id`` of the model param
    and of its FP32 ``main_param`` when one exists."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRAGroupedExpertLinear, MultiLoRALinear
    from megatron.core import parallel_state as ps

    tp_group = ps.get_tensor_model_parallel_group()
    tp_size = ps.get_tensor_model_parallel_world_size()
    specs: dict[int, FactorSpec] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, MultiLoRALinear):
                continue
            batched = isinstance(module, MultiLoRAGroupedExpertLinear)
            rank = int(module.rank_values[slot].item())
            adapter = module.adapters[slot]
            for param, row_split in (
                (adapter.linear_in.weight, None),
                (adapter.linear_out.weight, fused_row_split(module)),
            ):
                partition_dim = None
                if not batched and tp_size > 1 and getattr(param, "tensor_model_parallel", False):
                    partition_dim = int(param.partition_dim)
                spec = FactorSpec(
                    rank=rank,
                    row_split=row_split,
                    partition_dim=partition_dim,
                    tp_group=tp_group if partition_dim is not None else None,
                    batched=batched,
                )
                specs[id(param)] = spec
                main = getattr(param, "main_param", None)
                if main is not None:
                    specs[id(main)] = spec
    return specs
