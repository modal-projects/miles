"""Immutable GLM execution maps; positions are logical depths, never DFS offsets."""

from dataclasses import dataclass

import torch

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan, build_prefix_tree

# The pinned FLA chunk_kda uses 64-token chunks. State checkpoints must remain
# on that grid: restarting a chunk at an arbitrary fork changes BF16 rounding.
KDA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class TreeLevel:
    segments: tuple[int, ...]
    parents: tuple[int, ...]
    rows: torch.Tensor
    cu_seqlens: torch.Tensor


@dataclass(frozen=True)
class GlmTreeLayout:
    plan: PrefixTreePlan
    levels: tuple[TreeLevel, ...]
    restore_rows: torch.Tensor
    paths: tuple[torch.Tensor, ...]
    recurrent_levels: tuple[TreeLevel, ...]
    recurrent_restore_rows: torch.Tensor

    def prepare_packed(self, model, packed) -> None:
        # Bridge/Megatron and replay are optional GPU dependencies.
        from megatron.core.utils import get_attr_wrapped_model

        from miles.backends.training_utils.prefix_tree.glm53.bridge import glm53_tree_layer_spec
        from miles.utils.replay_base import all_replay_managers

        config = get_attr_wrapped_model(model, "config", allow_none=False)
        if getattr(config, "transformer_layer_spec", None) is not glm53_tree_layer_spec:
            raise ValueError("Build the GLM provider with transformer_layer_spec=glm53_tree_layer_spec")
        if any(manager.enabled and manager.stage != "fallthrough" for manager in all_replay_managers):
            raise ValueError("GLM tree SFT does not yet map router/indexer replay occurrence streams")
        packed.glm_tree_layout = self

    @classmethod
    def from_plan(cls, plan: PrefixTreePlan, *, device):
        segments = tuple(
            tuple(range(a, b)) for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True)
        )
        levels, restore = _levels(segments, plan.segment_parents, len(plan.tokens), device=device)
        recurrent_segments, recurrent_parents = _recurrent_segments(plan)
        recurrent_levels, recurrent_restore = _levels(
            recurrent_segments, recurrent_parents, len(plan.tokens), device=device
        )
        paths = tuple(
            torch.tensor(plan.kv_indices[a:b], device=device, dtype=torch.long)
            for a, b in zip(plan.kv_offsets[:-1], plan.kv_offsets[1:], strict=True)
        )
        return cls(plan, levels, restore, paths, recurrent_levels, recurrent_restore)


def _recurrent_segments(plan):
    """Radix tree of complete native KDA chunks, collapsing all unary runs.

    Forks inside a chunk duplicate only that chunk's shared tail. Projections
    and convolution remain unique. Full chunks and their differentiable states
    are shared; no per-chunk Python/kernel loop is introduced for long prefixes.
    """
    chunk_ids, chunks, sequences = {}, [], []
    for path in plan.sample_nodes:
        sequence = []
        for start in range(0, len(path), KDA_CHUNK_SIZE):
            chunk = path[start : start + KDA_CHUNK_SIZE]
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
