"""Kpool pools and the eligible-pool window stay inside each packed sequence, as sglang scores per request."""

import pytest

torch = pytest.importorskip("torch")

from miles.kernels.attention.dsa.kpool import pool_boundaries  # noqa: E402

KPOOL = 4
SEQ_LENS = [37, 13, 22]  # none is a multiple of KPOOL


def _cu_seqlens(seq_lens):
    return torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)


def test_pools_are_counted_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)

    counts = (pool_cu_seqlens[1:] - pool_cu_seqlens[:-1]).tolist()
    assert counts == [length // KPOOL for length in SEQ_LENS]

    # A grid laid over the whole buffer would produce more pools than the per-sequence
    # count, and the extra ones are exactly the pools that straddle a boundary.
    assert sum(counts) <= int(cu_seqlens[-1]) // KPOOL


def test_eligible_pool_window_never_leaves_the_query_sequence():
    """Mirrors the index arithmetic in `kpool_select_topk`."""
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)
    total_tokens = int(cu_seqlens[-1])

    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices]
    pool_base = pool_cu_seqlens[seq_indices]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, KPOOL, rounding_mode="floor")

    window_start = pool_base
    window_end = pool_base + eligible_pools
    own_window_end = pool_cu_seqlens[seq_indices + 1]

    assert bool((window_start >= pool_base).all())
    assert bool((window_end <= own_window_end).all()), "a query can see pools belonging to a later sequence"

    # Every query outside the first sequence starts strictly after that sequence's pools,
    # i.e. it can never be handed pool 0 of the packed buffer.
    is_later_seq = seq_indices > 0
    assert bool((window_start[is_later_seq] >= int(pool_cu_seqlens[1])).all())
    assert int(pool_cu_seqlens[1]) > 0


def test_local_positions_restart_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    total_tokens = int(cu_seqlens[-1])
    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    local_positions = token_ids - cu_seqlens[seq_indices]

    expected = torch.cat([torch.arange(length) for length in SEQ_LENS])
    torch.testing.assert_close(local_positions, expected)


@pytest.mark.parametrize("backend", ["canonical", "flashinfer"])
def test_pooled_indexer_reuses_requested_selector_and_causal_bounds(monkeypatch, backend):
    import miles.kernels.attention.dsa as dsa
    from miles.kernels.attention.dsa.kpool import _pool_topk
    from miles_plugins.models.glm5_next.ops import kpool_indexer

    logits = torch.tensor([[9.0, 10.0]]).repeat(6, 1)

    def selector(scores, count, starts, ends):
        columns = torch.arange(scores.shape[1])
        valid = (columns >= starts[:, None]) & (columns < ends[:, None])
        masked = scores.masked_fill(~valid, -torch.inf)
        order = masked.argsort(dim=-1, descending=True, stable=True)[:, :count]
        return order.masked_fill(~valid.gather(1, order), -1).int()

    def requested(name):
        assert name == backend
        return selector

    def expand(scores, seq_base, pool_base, positions, topk, kpool, **kwargs):
        return _pool_topk(scores, topk, kpool, **kwargs)[1]

    monkeypatch.setattr(dsa, "indexer_logits", lambda *args: logits)
    monkeypatch.setattr(kpool_indexer, "get_dsa_topk_fn", requested)
    monkeypatch.setattr(kpool_indexer, "select_expand_tail", expand)
    monkeypatch.setattr(kpool_indexer.indexer_replay_manager, "enabled", False)
    actual = kpool_indexer.kpool_select_topk(
        index_q=torch.zeros(6, 1, 1),
        pooled_k=torch.zeros(2, 1),
        head_weights=torch.ones(6, 1),
        cu_seqlens=torch.tensor([0, 3, 6]),
        pool_cu_seqlens=torch.tensor([0, 1, 2]),
        index_topk=2,
        kpool=2,
        topk_backend=backend,
    )
    assert actual.squeeze(1).tolist() == [[-1], [0], [0], [-1], [1], [1]]
    assert torch.equal(logits, torch.tensor([[9.0, 10.0]]).repeat(6, 1))


def test_default_pool_selection_preserves_torch_values_and_order():
    from miles.kernels.attention.dsa.kpool import _pool_topk

    logits = torch.tensor([[3.0, 3.0, 7.0, -torch.inf], [-torch.inf] * 4])
    scores, indices, count = _pool_topk(logits, 8, 2)
    expected = torch.topk(logits, 4, dim=-1)
    assert count == 4
    assert torch.equal(scores, expected.values)
    assert torch.equal(indices, expected.indices)


def test_empty_pool_rows_keep_the_canonical_selector_alignment(monkeypatch):
    from miles.kernels.attention.dsa.kpool import _pool_topk
    from miles.kernels.attention.dsa.topk import SCORE_ROW_ALIGN
    from miles_plugins.models.glm5_next.ops import kpool_indexer

    def selector(scores, count, starts, ends):
        assert scores.stride(0) % SCORE_ROW_ALIGN == 0
        assert not ends.any()
        return torch.full((len(scores), count), -1, dtype=torch.int32)

    monkeypatch.setattr(kpool_indexer, "get_dsa_topk_fn", lambda backend: selector)
    monkeypatch.setattr(kpool_indexer.indexer_replay_manager, "enabled", False)
    monkeypatch.setattr(
        kpool_indexer,
        "select_expand_tail",
        lambda scores, base, pools, positions, topk, kpool, **kwargs: _pool_topk(scores, topk, kpool, **kwargs)[1],
    )
    actual = kpool_indexer.kpool_select_topk(
        torch.zeros(3, 1, 1),
        torch.zeros(0, 1),
        torch.ones(3, 1),
        torch.tensor([0, 3]),
        torch.tensor([0, 0]),
        8,
        4,
        topk_backend="canonical",
    )
    assert actual.tolist() == [[[-1]], [[-1]], [[-1]]]
