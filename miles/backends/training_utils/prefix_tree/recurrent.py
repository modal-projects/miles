"""Common tree levels and native-chunk-aligned differentiable state execution."""

from dataclasses import dataclass

import torch

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan, build_prefix_tree

# The validated FLA KDA/GDN adapters use 64-token chunks. Each adapter owns its
# chunk-size contract: arbitrary restart positions change low-precision math.
NATIVE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class TreeLevel:
    segments: tuple[int, ...]
    parents: tuple[int, ...]
    rows: torch.Tensor
    cu_seqlens: torch.Tensor


@dataclass(frozen=True)
class ConvolutionTreeLayout:
    kernel_size: int
    rows: torch.Tensor
    sequence_ids: torch.Tensor
    restore_rows: torch.Tensor

    @classmethod
    def from_plan(cls, plan, *, device, kernel_size):
        if kernel_size < 1:
            raise ValueError("Convolution kernel size must be positive")
        rows, sequence_ids, restore = [], [], []
        for segment, (start, end) in enumerate(zip(plan.kv_offsets[:-1], plan.kv_offsets[1:], strict=True)):
            length = plan.segment_offsets[segment + 1] - plan.segment_offsets[segment]
            path = plan.kv_indices[start:end]
            context = path[-min(len(path), length + kernel_size - 1) :]
            halo = len(context) - length
            restore.extend(range(len(rows) + halo, len(rows) + len(context)))
            rows.extend(context)
            sequence_ids.extend([segment] * len(context))
        return cls(
            kernel_size,
            torch.tensor(rows, device=device, dtype=torch.long),
            torch.tensor([sequence_ids], device=device, dtype=torch.int32),
            torch.tensor(restore, device=device, dtype=torch.long),
        )


@dataclass(frozen=True)
class RecurrentTreeLayout:
    plan: PrefixTreePlan
    levels: tuple[TreeLevel, ...]
    restore_rows: torch.Tensor
    paths: tuple[torch.Tensor, ...]
    recurrent_levels: tuple[TreeLevel, ...]
    recurrent_restore_rows: torch.Tensor
    chunk_size: int = NATIVE_CHUNK_SIZE
    convolution: ConvolutionTreeLayout | None = None

    @classmethod
    def from_plan(cls, plan: PrefixTreePlan, *, device, chunk_size=NATIVE_CHUNK_SIZE, convolution_kernel_size=None):
        segments = tuple(
            tuple(range(a, b)) for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True)
        )
        levels, restore = _levels(segments, plan.segment_parents, len(plan.tokens), device=device)
        recurrent_segments, recurrent_parents = _recurrent_segments(plan, chunk_size)
        recurrent_levels, recurrent_restore = _levels(
            recurrent_segments, recurrent_parents, len(plan.tokens), device=device
        )
        paths = tuple(
            torch.tensor(plan.kv_indices[a:b], device=device, dtype=torch.long)
            for a, b in zip(plan.kv_offsets[:-1], plan.kv_offsets[1:], strict=True)
        )
        convolution = (
            ConvolutionTreeLayout.from_plan(plan, device=device, kernel_size=convolution_kernel_size)
            if convolution_kernel_size is not None
            else None
        )
        return cls(plan, levels, restore, paths, recurrent_levels, recurrent_restore, chunk_size, convolution)


def _recurrent_segments(plan, chunk_size):
    """Radix tree of complete native recurrence chunks, collapsing unary runs.

    Forks inside a chunk duplicate only that chunk's shared tail. Projections
    and convolution remain unique. Full chunks and their differentiable states
    are shared; no per-chunk Python/kernel loop is introduced for long prefixes.
    """
    if chunk_size < 1:
        raise ValueError("Native recurrence chunk size must be positive")
    chunk_ids, chunks, sequences = {}, [], []
    for path in plan.sample_nodes:
        sequence = []
        for start in range(0, len(path), chunk_size):
            chunk = path[start : start + chunk_size]
            if chunk not in chunk_ids:
                chunk_ids[chunk] = len(chunks)
                chunks.append(chunk)
            sequence.append(chunk_ids[chunk])
        sequences.append(sequence)
    chunk_plan = build_prefix_tree(sequences)
    segments = tuple(
        tuple(row for token in chunk_plan.tokens[a:b] for row in chunks[token])
        for a, b in zip(chunk_plan.segment_offsets[:-1], chunk_plan.segment_offsets[1:], strict=True)
    )
    return segments, chunk_plan.segment_parents


def _levels(segments, parents, output_rows, *, device):
    depths, groups = [], []
    for segment, parent in enumerate(parents):
        depth = depths[parent] + 1 if parent >= 0 else 0
        depths.append(depth)
        if depth == len(groups):
            groups.append([])
        groups[depth].append(segment)
    levels, execution_rows = [], []
    for group in groups:
        rows, offsets = [], [0]
        for segment in group:
            rows.extend(segments[segment])
            offsets.append(len(rows))
        execution_rows.extend(rows)
        levels.append(
            TreeLevel(
                tuple(group),
                tuple(parents[s] for s in group),
                torch.tensor(rows, device=device, dtype=torch.long),
                torch.tensor(offsets, device=device, dtype=torch.int32),
            )
        )
    # Some partial-chunk rows appear in several branches. Choose their first
    # occurrence consistently; suffix/state gradients still sum through gathers.
    restore = [-1] * output_rows
    for index, row in enumerate(execution_rows):
        if restore[row] < 0:
            restore[row] = index
    return tuple(levels), torch.tensor(restore, device=device, dtype=torch.long)


def execute_tree_levels(values, levels, restore_rows, *, step):
    """Run a native stateful operator on levels, summing child adjoints naturally.

    Values and outputs use [1, tokens, ...]; native states use [segments, ...].
    The caller owns the operator equation, precision, and chunk-size contract.
    """
    states, outputs = {}, []
    for level in levels:
        initial = None
        if level.parents[0] >= 0:
            initial = torch.cat([states[parent] for parent in level.parents], dim=0)
        output, final = step(
            *(value.index_select(1, level.rows) for value in values),
            initial_state=initial,
            cu_seqlens=level.cu_seqlens,
        )
        if final is None:
            raise ValueError("A tree stateful operator must return final states")
        for index, segment in enumerate(level.segments):
            states[segment] = final[index : index + 1]
        outputs.append(output)
    return torch.cat(outputs, dim=1).index_select(1, restore_rows)
