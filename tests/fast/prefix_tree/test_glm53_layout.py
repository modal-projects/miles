"""GLM ancestry maps, including unaligned pools and unequal branch depths."""

import pytest
import torch

from miles.backends.training_utils.prefix_tree.glm53.dsa import tree_kpool_select
from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def test_level_order_restores_dfs_with_nested_forks():
    plan = build_prefix_tree([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6], [9, 10]])
    layout = GlmTreeLayout.from_plan(plan, device="cpu")
    visited = set()
    for level in layout.levels:
        assert all(parent < 0 or parent in visited for parent in level.parents)
        visited.update(level.segments)
    rows = torch.cat([level.rows for level in layout.levels])
    assert torch.equal(rows[layout.restore_rows], torch.arange(len(plan.tokens)))
    for segment, path in enumerate(layout.paths):
        assert [plan.positions[row] for row in path] == list(range(len(path)))
        start, end = plan.segment_offsets[segment : segment + 2]
        assert path[-(end - start) :].tolist() == list(range(start, end))


def test_dsa_pool_phase_tail_and_compact_rebasing():
    sequences = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 8, 9, 10, 11], [1, 2, 12, 13, 14]]
    plan = build_prefix_tree(sequences)
    layout = GlmTreeLayout.from_plan(plan, device="cpu")
    keys = torch.tensor(plan.tokens)[:, None].float()

    def select(q, k, weights, gates, *, query_start):
        assert torch.equal(q, k[query_start:]) and torch.equal(q, weights) and torch.equal(k, gates)
        # Return all four keys of the latest completed pool plus incomplete tail.
        out = torch.full((len(q), 1, 7), -1, dtype=torch.int32)
        for row, position in enumerate(range(query_start, query_start + len(q))):
            completed = ((position + 1) // 4) * 4
            ids = list(range(max(0, completed - 4), position + 1))
            out[row, 0, : len(ids)] = torch.tensor(ids)
        return out

    selected = tree_kpool_select(keys, keys, keys, keys, layout, select=select)
    for sample in plan.sample_nodes:
        for position, row in enumerate(sample):
            actual = selected[row, 0]
            actual = actual[actual >= 0].tolist()
            completed = ((position + 1) // 4) * 4
            assert actual == list(sample[max(0, completed - 4) : position + 1])


def test_multilora_counts_compact_rows_and_refuses_cross_adapter_sharing():
    from miles.backends.training_utils.prefix_tree.glm53.sft import GlmTreeSFTBatch

    sequences = [[1, 2, 3], [1, 2, 4], [1, 2, 5]]
    plan = build_prefix_tree(sequences, sharing_keys=[0, 0, 1])
    batch = GlmTreeSFTBatch.from_plan(plan, sequences, device="cpu", adapter_slots=[0, 0, 1], num_adapter_slots=3)
    assert batch.slot_counts.tolist() == [4, 3, 0]
    with pytest.raises(ValueError, match="shared across adapters"):
        GlmTreeSFTBatch.from_plan(
            build_prefix_tree(sequences), sequences, device="cpu", adapter_slots=[0, 0, 1], num_adapter_slots=2
        )
    with pytest.raises(ValueError, match="Group samples"):
        GlmTreeSFTBatch.from_plan(
            build_prefix_tree(sequences, sharing_keys=[1, 0, 0]),
            sequences,
            device="cpu",
            adapter_slots=[1, 0, 0],
            num_adapter_slots=2,
        )


def test_kda_replays_partial_chunks_and_collapses_long_unary_prefix():
    prefix = list(range(6401))
    sequences = [prefix + [10000 + b] + list(range(11000 + b * 100, 11040 + b * 100)) for b in range(8)]
    plan = build_prefix_tree(sequences)
    layout = GlmTreeLayout.from_plan(plan, device="cpu")
    assert len(layout.recurrent_levels) == 2
    assert layout.recurrent_levels[0].rows.numel() == 6400
    assert sum(level.rows.numel() for level in layout.recurrent_levels) == len(plan.tokens) + 7
    execution = torch.cat([level.rows for level in layout.recurrent_levels])
    assert torch.equal(execution[layout.recurrent_restore_rows], torch.arange(len(plan.tokens)))
