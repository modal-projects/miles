"""Dimuon-lite math without Megatron, covering the block sign, the sqrt(rank) norm, splits and TP sharding."""

import math

import pytest
import torch

from miles.backends.megatron_utils.lora.dimuon import (
    Dimuon,
    FactorSpec,
    factor_direction,
    normalized_sign,
    polar_express,
    rank_scale,
)


def _svd_polar(x):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def _fake_tp(world):
    """gather/slice that emulate a TP group of ``world`` shards held in one process."""
    shards = {}

    def gather(local, dim, group):
        return torch.cat(shards["all"], dim=dim)

    def slice_(full, dim, group):
        return full.chunk(world, dim=dim)[shards["rank"]]

    return shards, gather, slice_


def test_rank_scale_reference():
    assert rank_scale(32) == 1.0
    assert rank_scale(8) == pytest.approx(0.25**0.175)
    assert rank_scale(128) == pytest.approx(4**0.175)


@pytest.mark.parametrize("shape", [(8, 512), (512, 8), (32, 1024)])  # LoRA factors are thin
def test_polar_express_is_near_orthogonal(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    p = polar_express(x)
    sv = torch.linalg.svdvals(p)
    assert sv.min() > 0.6 and sv.max() < 1.25
    cos = (p * _svd_polar(x)).sum() / (p.norm() * _svd_polar(x).norm())
    assert cos > 0.97


def test_normalized_sign_has_frobenius_sqrt_rank():
    torch.manual_seed(0)
    out = normalized_sign(torch.randn(1024, 16), rank=16, row_split=None)
    assert out.norm().item() == pytest.approx(4.0, rel=1e-5)


def test_zero_momentum_gives_zero_direction():
    out = normalized_sign(torch.zeros(16, 256), rank=16, row_split=None)
    assert torch.count_nonzero(out) == 0


def test_padded_rank_stays_zero():
    torch.manual_seed(0)
    rank, max_rank = 4, 16
    a = torch.zeros(max_rank, 256)
    a[:rank] = torch.randn(rank, 256)
    b = torch.zeros(512, max_rank)
    b[:, :rank] = torch.randn(512, rank)
    da = normalized_sign(a, rank=rank, row_split=None)
    db = normalized_sign(b, rank=rank, row_split=None)
    assert torch.count_nonzero(da[rank:]) == 0
    assert torch.count_nonzero(db[:, rank:]) == 0
    assert da.norm().item() == pytest.approx(math.sqrt(rank), rel=1e-5)
    assert db.norm().item() == pytest.approx(math.sqrt(rank), rel=1e-5)


def test_fused_qkv_blocks_are_independent():
    """B of linear_qkv has rows ordered by query group, and each of q, k and v gets its own sign and norm."""
    torch.manual_seed(0)
    groups, q, kv, r = 4, 32, 8, 8
    split = (q, kv, kv)
    full = torch.randn(groups * sum(split), r)
    out = normalized_sign(full, rank=r, row_split=split)
    grouped = full.view(groups, sum(split), r)
    got = out.view(groups, sum(split), r)
    offset = 0
    for size in split:
        block_in = grouped[:, offset : offset + size].reshape(-1, r)
        block_out = got[:, offset : offset + size].reshape(-1, r)
        expected = normalized_sign(block_in, rank=r, row_split=None)
        torch.testing.assert_close(block_out, expected)
        assert block_out.norm().item() == pytest.approx(math.sqrt(r), rel=1e-5)
        offset += size


@pytest.mark.parametrize("partition_dim", [0, 1])
def test_tp_shards_match_unsharded(partition_dim):
    torch.manual_seed(0)
    world, r = 2, 8
    full = torch.randn(256, r) if partition_dim == 0 else torch.randn(r, 256)
    expected = normalized_sign(full, rank=r, row_split=None)
    shards, gather, slice_ = _fake_tp(world)
    shards["all"] = list(full.chunk(world, dim=partition_dim))
    spec = FactorSpec(rank=r, partition_dim=partition_dim)
    for tp_rank in range(world):
        shards["rank"] = tp_rank
        local = shards["all"][tp_rank]
        got = factor_direction(local, spec, gather=gather, slice_=slice_)
        torch.testing.assert_close(got, expected.chunk(world, dim=partition_dim)[tp_rank])


def test_tp_sharded_gate_up_splits_per_shard():
    """Each TP shard of gated linear_fc1 B holds gate rows then up rows, and each is signed separately."""
    torch.manual_seed(0)
    world, ffn_shard, r = 2, 64, 8
    gate = torch.randn(world * ffn_shard, r)
    up = torch.randn(world * ffn_shard, r)
    local = [torch.cat([g, u]) for g, u in zip(gate.chunk(world), up.chunk(world), strict=True)]
    shards, gather, slice_ = _fake_tp(world)
    shards["all"] = local
    spec = FactorSpec(rank=r, row_split=(ffn_shard, ffn_shard), partition_dim=0)
    expected_gate = normalized_sign(gate, rank=r, row_split=None)
    expected_up = normalized_sign(up, rank=r, row_split=None)
    for tp_rank in range(world):
        shards["rank"] = tp_rank
        got = factor_direction(local[tp_rank], spec, gather=gather, slice_=slice_)
        torch.testing.assert_close(got[:ffn_shard], expected_gate.chunk(world)[tp_rank])
        torch.testing.assert_close(got[ffn_shard:], expected_up.chunk(world)[tp_rank])


def test_packed_experts_are_signed_per_expert():
    torch.manual_seed(0)
    experts = torch.randn(3, 128, 8)
    out = factor_direction(experts, FactorSpec(rank=8, batched=True))
    for e in range(3):
        torch.testing.assert_close(out[e], normalized_sign(experts[e], rank=8, row_split=None))


def test_optimizer_step_size_and_momentum():
    torch.manual_seed(0)
    rank, lr, beta = 8, 1e-2, 0.9
    a = torch.nn.Parameter(torch.randn(rank, 64))
    b = torch.nn.Parameter(torch.zeros(128, rank))
    opt = Dimuon([a, b], lr=lr, momentum=beta)
    opt.set_specs({id(a): FactorSpec(rank=rank), id(b): FactorSpec(rank=rank)})

    # Step 1 from the standard init gives A a zero gradient, so A must not move
    g1 = torch.randn(128, rank)
    a.grad, b.grad = torch.zeros_like(a), g1.clone()
    a0 = a.detach().clone()
    opt.step()
    torch.testing.assert_close(a.detach(), a0)
    step1 = -b.detach().clone()
    assert step1.norm().item() == pytest.approx(lr * rank_scale(rank) * math.sqrt(rank), rel=1e-5)

    # Step 2 takes its direction from the momentum EMA over g1 and g2
    g2 = torch.randn(128, rank)
    b.grad = g2.clone()
    before = b.detach().clone()
    opt.step()
    momentum = beta * (1 - beta) * g1 + (1 - beta) * g2
    expected = -lr * rank_scale(rank) * normalized_sign(momentum, rank=rank, row_split=None)
    torch.testing.assert_close(b.detach() - before, expected)


def test_optimizer_requires_specs():
    p = torch.nn.Parameter(torch.randn(4, 8))
    p.grad = torch.randn(4, 8)
    with pytest.raises(KeyError):
        Dimuon([p]).step()
