import pytest

from miles.backends.training_utils.prefix_tree.packed import prepare_packed_tree


@pytest.mark.parametrize("share_prefix", [False, True])
@pytest.mark.parametrize("alignment", [1, 2, 8, 128])
@pytest.mark.parametrize("native_padding", [0, 3])
def test_packed_rl_transport_preserves_original_rows_and_adapter_isolation(alignment, native_padding, share_prefix):
    sequences = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 6]]
    slots = [0, 0, 1, 1]
    transport = prepare_packed_tree(
        sequences,
        slots,
        padded_rows=16 + native_padding,
        alignment=alignment,
        num_adapters=3,
        share_prefix=share_prefix,
    )
    expanded = [token for sequence in sequences for token in sequence] + [0] * native_padding
    assert [transport.plan.tokens[row] for row in transport.restore_rows] == expanded
    assert sum(transport.adapter_counts) == len(transport.plan.tokens)
    assert transport.adapter_counts[2] == 0
    assert len(transport.plan.tokens) % alignment == 0
    assert set(transport.plan.sample_nodes[0]).isdisjoint(transport.plan.sample_nodes[2])
    if share_prefix:
        assert transport.plan.sample_nodes[0][:3] == transport.plan.sample_nodes[1][:3]
    else:
        assert set(transport.plan.sample_nodes[0]).isdisjoint(transport.plan.sample_nodes[1])
        assert len(transport.plan.tokens) >= 16 + native_padding
