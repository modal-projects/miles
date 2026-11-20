import pytest

from miles.backends.training_utils.prefix_tree.batching import split_prefix_microbatches
from miles.backends.training_utils.prefix_tree.packed import prepare_packed_tree
from miles.utils.seqlen_balancing import expand_bins_by_splitting


@pytest.mark.parametrize("shared,slots", [(True, [0] * 8), (False, [0] * 8), (True, [0, 1] * 4)])
def test_variable_prefixes_stay_within_both_budgets_after_dp_alignment(shared, slots):
    sequences = [[1 if shared else index + 10] * 24 + [index] * (8 + index) for index in range(8)]
    bins = split_prefix_microbatches(
        [list(range(8))], sequences, slots, expanded_limit=160, compact_limit=84, alignment=8, pad_size=8
    )
    if shared and len(set(slots)) == 1:
        assert any(len(batch) > 1 for batch in bins)
    expand_bins_by_splitting(bins, min(8, (len(bins) + 3) // 4 * 4), list(map(len, sequences)))
    assert sorted(index for batch in bins for index in batch) == list(range(8))
    for batch in bins:
        batch.sort(key=lambda index: slots[index])
        rows = sum(len(sequences[index]) for index in batch)
        padded = (rows + 7) // 8 * 8
        assert padded <= 160
        transport = prepare_packed_tree(
            [sequences[index] for index in batch],
            [slots[index] for index in batch],
            padded_rows=padded,
            alignment=8,
            num_adapters=2,
        )
        assert len(transport.plan.tokens) % 8 == 0
        assert len(transport.plan.tokens) <= 84
        for index, path in zip(batch, transport.plan.sample_nodes, strict=False):
            assert [transport.plan.tokens[row] for row in path] == sequences[index]


@pytest.mark.parametrize("expanded_limit,compact_limit", [(24, 100), (100, 24)])
def test_oversized_single_sample_is_rejected_before_training(expanded_limit, compact_limit):
    with pytest.raises(ValueError, match="Sample 0 cannot fit prefix token budgets"):
        split_prefix_microbatches(
            [[0]],
            [[1] * 32],
            [0],
            expanded_limit=expanded_limit,
            compact_limit=compact_limit,
            alignment=2,
            pad_size=8,
        )
