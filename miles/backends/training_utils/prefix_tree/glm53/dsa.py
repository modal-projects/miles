"""Reuse the pinned kpool selector and SparseMLA with compact tree addresses."""

import torch

from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout


def tree_kpool_select(index_q, index_k, head_weights, gate_score, layout: GlmTreeLayout, *, select):
    """Run the existing detached indexer on each segment's complete logical path.

    Ancestor indexer inputs are gathered, preserving four-token pool phase,
    learned compression APE and unfinished tails at arbitrary forks. Only new
    queries are scored. MLA Q/K/V and all learned projections stay compact.
    Pooling gathers ancestor indexer keys, but scoring and MLA queries stay unique.
    ``select`` receives path-local tensors and returns path-local token addresses.
    """
    selections = []
    with torch.no_grad():
        for segment, path in enumerate(layout.paths):
            length = layout.plan.segment_offsets[segment + 1] - layout.plan.segment_offsets[segment]
            query_rows = path[-length:]
            selected = select(
                index_q.index_select(0, query_rows),
                index_k.index_select(0, path),
                head_weights.index_select(0, query_rows),
                gate_score.index_select(0, path),
                query_start=len(path) - length,
            )
            valid = selected >= 0
            rebased = path[selected.clamp_min(0).long()].to(selected.dtype)
            selections.append(torch.where(valid, rebased, -1))
    return torch.cat(selections, dim=0).contiguous()
