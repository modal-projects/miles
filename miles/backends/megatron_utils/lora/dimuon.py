"""Dimuon, a per-factor Muon-type optimizer for multi-LoRA slots.

Each LoRA factor P with gradient G keeps one BF16 momentum matrix M and steps as

    L, R  <- diag((mean-square rows / columns of M, max-normalized) + damping) ** -1/4
    work  <- beta1 * M + (1 - beta1) * G                       (FP32)
    D     <- L @ polar_express(L @ work @ R) @ R               (per matrix block)
    D     <- D * sqrt(rank) / ||D||_F
    P     <- P - lr * (rank / 32) ** 0.175 * D
    M     <- BF16(work)

The row and column scales come from the momentum stored by the previous step, so a
fresh factor starts with scalar scales and the optimizer stays stateful at beta1 = 0.
Polar Express runs five steps in FP16 Horner form after FP32 normalization.

B of a fused ``linear_qkv`` splits into q, k and v row blocks per query group, B of a
gated ``linear_fc1`` into gate and up row blocks, and B of a Gated DeltaNet ``in_proj``
into q, k, v, z, b and a blocks, so projections never share a direction or scales.
A of a fused module is one shared matrix and is treated whole.

TP-sharded factors are all-gathered, transformed whole on every TP rank and sliced back.
Padded rank rows of A and columns of B stay zero because their gradients are zero and
every step preserves zero rows and columns.

The math is plain torch so it runs without Megatron.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from miles.backends.megatron_utils.lora.gdn_utils import gdn_in_proj_split, is_gdn_in_proj

# First five Polar Express iterations (arXiv 2505.16932), with the 1.01 safety factor
# applied as a / 1.01, b / 1.01**3 and c / 1.01**5.
_PAPER_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.4884880243148474, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_COEFFICIENTS = tuple((a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _PAPER_COEFFICIENTS)
REFERENCE_RANK = 32
RANK_LR_CORRECTION_EXPONENT = 0.175
DAMPING = 1e-4
_TINY = 1e-30


def polar_express(x: torch.Tensor) -> torch.Tensor:
    """Approximate the orthogonal polar factor of each trailing 2-D matrix of ``x``.

    Each matrix is normalized in FP32 by 1.01 times its Frobenius norm, then iterated in
    FP16 Horner form on its smaller Gram matrix. Returns FP32."""
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x.float()
    x = (x / (1.01 * x.norm(dim=(-2, -1), keepdim=True)).clamp_min(_TINY)).half()
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = x @ x.mT
        polynomial = c * gram
        polynomial.diagonal(dim1=-2, dim2=-1).add_(b)
        polynomial = polynomial @ gram
        polynomial.diagonal(dim1=-2, dim2=-1).add_(a)
        x = polynomial @ x
    x = x.float()
    return x.mT if transpose else x


def diagonal_scales(previous: torch.Tensor, damping: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Row and column scales ``(L, R)`` from mean squares of the previous momentum.

    Each statistic is normalized by its maximum with a zero guard, so an all-zero momentum
    gives scalar scales. Returns FP32 tensors shaped ``[..., rows, 1]`` and ``[..., 1, cols]``."""
    squares = previous.float().square()
    rows = squares.mean(dim=-1, keepdim=True)
    cols = squares.mean(dim=-2, keepdim=True)
    rows = rows / rows.amax(dim=(-2, -1), keepdim=True).clamp_min(_TINY)
    cols = cols / cols.amax(dim=(-2, -1), keepdim=True).clamp_min(_TINY)
    return (rows + damping).pow(-0.25), (cols + damping).pow(-0.25)


@dataclass(frozen=True)
class FactorSpec:
    """How one LoRA factor parameter maps onto Dimuon matrix blocks.

    ``rank`` is the slot's actual adapter rank, not the padded maximum.
    ``row_split`` gives the per-group row sizes of the projection blocks in the
    unsharded matrix, whose rows read as ``(groups, sum(row_split))``. ``None`` means
    one block. ``partition_dim`` is the TP shard axis of the trailing 2-D matrix, or
    ``None`` when the parameter is not TP-sharded. ``batched`` marks packed expert
    tensors with a leading expert axis.
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


def block_direction(work: torch.Tensor, previous: torch.Tensor, rank: int, damping: float) -> torch.Tensor:
    """Dimuon direction of one matrix block, with Frobenius norm ``sqrt(rank)`` per matrix.

    ``work`` is the FP32 working momentum and ``previous`` the stored momentum it was built
    from. An all-zero ``work`` yields a zero direction."""
    left, right = diagonal_scales(previous, damping)
    direction = left * polar_express(left * work.float() * right) * right
    norm = direction.norm(dim=(-2, -1), keepdim=True).clamp_min(_TINY)
    return direction * (math.sqrt(rank) / norm)


def dimuon_direction(
    work: torch.Tensor,
    previous: torch.Tensor,
    rank: int,
    row_split: tuple[int, ...] | None,
    damping: float = DAMPING,
) -> torch.Tensor:
    """Per-projection-block Dimuon direction for an unsharded factor, ``[rows, cols]`` or ``[E, rows, cols]``."""
    blocks = [
        block_direction(w, p, rank, damping)
        for w, p in zip(_blocks(work, row_split), _blocks(previous, row_split), strict=True)
    ]
    return _unblocks(blocks, row_split, work)


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
    work: torch.Tensor,
    previous: torch.Tensor,
    spec: FactorSpec,
    damping: float = DAMPING,
    *,
    gather: GatherFn = _dist_gather,
    slice_: SliceFn = _dist_slice,
) -> torch.Tensor:
    """Dimuon direction (before ``lr * rank_scale``) for this rank's shard of one factor."""
    if spec.partition_dim is None:
        return dimuon_direction(work, previous, spec.rank, spec.row_split, damping)
    dim = spec.partition_dim + (1 if spec.batched else 0)
    full_work = gather(work, dim, spec.tp_group)
    full_previous = gather(previous, dim, spec.tp_group)
    direction = dimuon_direction(full_work, full_previous, spec.rank, spec.row_split, damping)
    return slice_(direction, dim, spec.tp_group)


def advance_momentum(previous: torch.Tensor, gradient: torch.Tensor, beta: float) -> torch.Tensor:
    """FP32 working momentum ``beta * previous + (1 - beta) * gradient``, multiplied then fused-added."""
    work = previous.float().mul(beta)
    return work.add_(gradient.float(), alpha=1 - beta)


class Dimuon(torch.optim.Optimizer):
    """Dimuon over LoRA factor parameters, each of which needs a ``FactorSpec``.

    Specs are keyed by ``id(param)`` of the tensors this optimizer steps, which are the
    FP32 masters once Megatron wraps it, and are registered with ``set_specs``."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-2,
        momentum: float = 0.9,
        damping: float = DAMPING,
        rank_lr_correction_exponent: float = RANK_LR_CORRECTION_EXPONENT,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            damping=damping,
            rank_lr_correction_exponent=rank_lr_correction_exponent,
        )
        super().__init__(params, defaults)
        self.specs: dict[int, FactorSpec] = {}

    def set_specs(self, specs: dict[int, FactorSpec]) -> None:
        self.specs = dict(specs)

    def load_state_dict(self, state_dict) -> None:
        # Restore momentum as BF16 because torch casts loaded state to each parameter's dtype
        super().load_state_dict(state_dict)
        for state in self.state.values():
            if "momentum_buffer" in state:
                state["momentum_buffer"] = state["momentum_buffer"].to(torch.bfloat16)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "Dimuon does not support closures"
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                spec = self.specs.get(id(p))
                if spec is None:
                    raise KeyError(f"Dimuon has no FactorSpec for a parameter of shape {tuple(p.shape)}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                previous = state["momentum_buffer"]
                work = advance_momentum(previous, p.grad, group["momentum"])
                direction = factor_direction(work, previous, spec, group["damping"])
                # Store the rounded momentum only after the scales and direction have read the old one
                previous.copy_(work)
                scale = group["lr"] * rank_scale(spec.rank, group["rank_lr_correction_exponent"])
                p.add_(direction.to(p.dtype), alpha=-scale)
        return None


def dimuon_init_state_fn(opt, config=None) -> None:
    """Lazily create Dimuon state (LayerWise checkpointing needs it before the first step)."""
    for group in opt.param_groups:
        for p in group["params"]:
            if len(opt.state[p]) == 0:
                opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.bfloat16)


# --------------------------------------------------------------------------- Megatron wiring


def fused_row_split(module) -> tuple[int, ...] | None:
    """Projection row split for B of a fused ``linear_qkv``, gated ``linear_fc1`` or GDN ``in_proj``, else ``None``.

    q/k/v rows are interleaved per query group (Megatron's layout), so the split is
    per group. With an attention output gate each group's rows are query then gate,
    and both stay in one block because HF ``q_proj`` holds query and gate together.
    Gate/up rows of a TP shard are ``[gate_shard; up_shard]``, so after gathering the
    shards the rows read ``[g0; u0; g1; u1; ...]`` and the split is
    ``(ffn_shard, ffn_shard)`` per shard."""
    name = module.base_linear_name.rsplit(".", 1)[-1]
    config = module.to_wrap.config
    if name == "linear_qkv":
        kv = config.kv_channels
        q = config.num_attention_heads // config.num_query_groups * kv
        if getattr(config, "attention_output_gate", False):
            return (2 * q, kv, kv)
        return (q, kv, kv)
    if is_gdn_in_proj(module):
        return gdn_in_proj_split(config, module.adapters[0].linear_out.weight.shape[-2])
    if name == "linear_fc1" and getattr(config, "gated_linear_unit", False):
        out = module.adapters[0].linear_out.weight.shape[-2]  # local rows, twice the ffn shard
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
