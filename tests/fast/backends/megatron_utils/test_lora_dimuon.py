"""Dimuon math without Megatron, covering scales, the polar step, splits, TP sharding and state."""

import math
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.dimuon import (
    Dimuon,
    FactorSpec,
    advance_momentum,
    block_direction,
    diagonal_scales,
    dimuon_direction,
    factor_direction,
    fused_row_split,
    polar_express,
    rank_scale,
)


def _svd_polar(x):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def _cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()


def _fake_tp(world):
    """gather/slice that emulate a TP group of ``world`` shards held in one process."""
    shards = {}

    def gather(local, dim, group):
        key = "previous" if local.dtype == torch.bfloat16 else "work"
        return torch.cat(shards[key], dim=dim)

    def slice_(full, dim, group):
        return full.chunk(world, dim=dim)[shards["rank"]]

    return shards, gather, slice_


def _uneven(rows, cols, seed):
    """A momentum whose rows and columns have very different scales."""
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * torch.logspace(-2, 0, rows)[:, None] * torch.logspace(0, -1, cols)[None, :]


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
    assert _cos(p, _svd_polar(x)) > 0.97


def test_polar_express_maps_zero_to_zero():
    assert torch.count_nonzero(polar_express(torch.zeros(8, 64))) == 0


def test_fresh_state_gives_scalar_scales():
    left, right = diagonal_scales(torch.zeros(16, 64, dtype=torch.bfloat16), 1e-4)
    assert torch.all(left == left.flatten()[0]) and torch.all(right == right.flatten()[0])


def test_fresh_state_direction_is_the_normalized_polar_factor():
    torch.manual_seed(0)
    work = torch.randn(16, 256)
    direction = block_direction(work, torch.zeros_like(work), rank=16, damping=1e-4)
    expected = polar_express(work)
    # Scalar scales only change the FP16 rounding inside the polar step
    torch.testing.assert_close(direction, expected * (4.0 / expected.norm()), rtol=0, atol=1e-3)


def test_previous_momentum_rescales_the_direction():
    work = _uneven(16, 256, seed=0)
    fresh = block_direction(work, torch.zeros_like(work), rank=16, damping=1e-4)
    stateful = block_direction(work, work.bfloat16(), rank=16, damping=1e-4)
    assert _cos(fresh, stateful) < 0.99


def test_large_damping_approaches_the_fresh_direction():
    work = _uneven(16, 256, seed=0)
    fresh = block_direction(work, torch.zeros_like(work), rank=16, damping=1e-4)
    damped = block_direction(work, work.bfloat16(), rank=16, damping=1e6)
    assert _cos(fresh, damped) > 0.9999


def test_direction_has_frobenius_sqrt_rank():
    work = _uneven(16, 1024, seed=1)
    direction = block_direction(work, work.bfloat16(), rank=16, damping=1e-4)
    assert direction.norm().item() == pytest.approx(4.0, rel=1e-5)


def test_zero_momentum_gives_zero_direction():
    direction = block_direction(torch.zeros(16, 256), torch.zeros(16, 256), rank=16, damping=1e-4)
    assert torch.count_nonzero(direction) == 0


def test_padded_rank_stays_zero():
    torch.manual_seed(0)
    rank, max_rank = 4, 16
    a = torch.zeros(max_rank, 256)
    a[:rank] = torch.randn(rank, 256)
    b = torch.zeros(512, max_rank)
    b[:, :rank] = torch.randn(512, rank)
    da = dimuon_direction(a, a.bfloat16(), rank=rank, row_split=None)
    db = dimuon_direction(b, b.bfloat16(), rank=rank, row_split=None)
    assert torch.count_nonzero(da[rank:]) == 0
    assert torch.count_nonzero(db[:, rank:]) == 0


def test_fused_qkv_blocks_are_independent():
    """B of linear_qkv has rows ordered by query group, and each of q, k and v gets its own direction."""
    torch.manual_seed(0)
    groups, q, kv, r = 4, 32, 8, 8
    split = (q, kv, kv)
    work = torch.randn(groups * sum(split), r)
    previous = (work * 0.5).bfloat16()
    out = dimuon_direction(work, previous, rank=r, row_split=split).view(groups, sum(split), r)
    work_g, previous_g = work.view(groups, sum(split), r), previous.view(groups, sum(split), r)
    offset = 0
    for size in split:
        rows = slice(offset, offset + size)
        expected = block_direction(work_g[:, rows].reshape(-1, r), previous_g[:, rows].reshape(-1, r), r, 1e-4)
        torch.testing.assert_close(out[:, rows].reshape(-1, r), expected)
        offset += size


@pytest.mark.parametrize("partition_dim", [0, 1])
def test_tp_shards_match_unsharded(partition_dim):
    world, r = 2, 8
    work = _uneven(256, r, seed=2) if partition_dim == 0 else _uneven(r, 256, seed=2)
    previous = (work * 0.7).bfloat16()
    expected = dimuon_direction(work, previous, rank=r, row_split=None)
    shards, gather, slice_ = _fake_tp(world)
    shards["work"] = list(work.chunk(world, dim=partition_dim))
    shards["previous"] = list(previous.chunk(world, dim=partition_dim))
    spec = FactorSpec(rank=r, partition_dim=partition_dim)
    for tp_rank in range(world):
        shards["rank"] = tp_rank
        got = factor_direction(
            shards["work"][tp_rank], shards["previous"][tp_rank], spec, gather=gather, slice_=slice_
        )
        torch.testing.assert_close(got, expected.chunk(world, dim=partition_dim)[tp_rank])


def test_tp_sharded_gate_up_splits_per_shard():
    """Each TP shard of gated linear_fc1 B holds gate rows then up rows, and each is handled separately."""
    torch.manual_seed(0)
    world, ffn_shard, r = 2, 64, 8
    gate, up = torch.randn(world * ffn_shard, r), torch.randn(world * ffn_shard, r)
    local = [torch.cat([g, u]) for g, u in zip(gate.chunk(world), up.chunk(world), strict=True)]
    shards, gather, slice_ = _fake_tp(world)
    shards["work"] = local
    shards["previous"] = [torch.zeros_like(x, dtype=torch.bfloat16) for x in local]
    spec = FactorSpec(rank=r, row_split=(ffn_shard, ffn_shard), partition_dim=0)
    expected_gate = dimuon_direction(gate, torch.zeros_like(gate), rank=r, row_split=None)
    expected_up = dimuon_direction(up, torch.zeros_like(up), rank=r, row_split=None)
    for tp_rank in range(world):
        shards["rank"] = tp_rank
        got = factor_direction(local[tp_rank], shards["previous"][tp_rank], spec, gather=gather, slice_=slice_)
        torch.testing.assert_close(got[:ffn_shard], expected_gate.chunk(world)[tp_rank])
        torch.testing.assert_close(got[ffn_shard:], expected_up.chunk(world)[tp_rank])


def test_packed_experts_are_handled_per_expert():
    torch.manual_seed(0)
    experts = torch.randn(3, 128, 8)
    previous = (experts * 0.5).bfloat16()
    out = factor_direction(experts, previous, FactorSpec(rank=8, batched=True))
    for e in range(3):
        torch.testing.assert_close(out[e], dimuon_direction(experts[e], previous[e], rank=8, row_split=None))


def test_working_momentum_multiplies_then_fused_adds():
    torch.manual_seed(0)
    previous, gradient = torch.randn(8, 32).bfloat16(), torch.randn(8, 32)
    expected = previous.float().mul(0.9)
    expected.add_(gradient, alpha=0.1)
    torch.testing.assert_close(advance_momentum(previous, gradient, 0.9), expected, rtol=0, atol=0)


def _two_factor_optimizer(rank=8, lr=1e-2, beta=0.9):
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(rank, 64))
    b = torch.nn.Parameter(torch.zeros(128, rank))
    opt = Dimuon([a, b], lr=lr, momentum=beta)
    opt.set_specs({id(a): FactorSpec(rank=rank), id(b): FactorSpec(rank=rank)})
    return a, b, opt


def test_optimizer_does_not_move_a_factor_with_zero_gradient():
    a, b, opt = _two_factor_optimizer()
    a.grad, b.grad = torch.zeros_like(a), torch.randn(128, 8)
    a0 = a.detach().clone()
    opt.step()
    torch.testing.assert_close(a.detach(), a0)


def test_optimizer_step_has_size_lr_times_rank_scale_times_sqrt_rank():
    a, b, opt = _two_factor_optimizer()
    a.grad, b.grad = torch.zeros_like(a), torch.randn(128, 8)
    opt.step()
    assert b.detach().norm().item() == pytest.approx(1e-2 * rank_scale(8) * math.sqrt(8), rel=1e-5)


def test_optimizer_stores_bf16_momentum_of_the_fp32_working_ema():
    a, b, opt = _two_factor_optimizer()
    a.grad, b.grad = torch.zeros_like(a), torch.randn(128, 8)
    opt.step()
    stored = opt.state[b]["momentum_buffer"]
    assert stored.dtype == torch.bfloat16
    expected = advance_momentum(torch.zeros(128, 8), b.grad, 0.9).bfloat16()
    torch.testing.assert_close(stored, expected, rtol=0, atol=0)


def test_optimizer_second_step_uses_scales_from_stored_momentum():
    a, b, opt = _two_factor_optimizer()
    a.grad, b.grad = torch.zeros_like(a), _uneven(128, 8, seed=3)
    opt.step()
    previous = opt.state[b]["momentum_buffer"].clone()
    g2 = _uneven(128, 8, seed=4)
    b.grad = g2.clone()
    before = b.detach().clone()
    opt.step()
    work = advance_momentum(previous, g2, 0.9)
    expected = -1e-2 * rank_scale(8) * dimuon_direction(work, previous, rank=8, row_split=None)
    torch.testing.assert_close(b.detach() - before, expected)


def test_optimizer_stays_stateful_at_zero_momentum_coefficient():
    """At beta1 = 0 the update is still rescaled by the previous step's stored momentum."""
    a, b, opt = _two_factor_optimizer(beta=0.0)
    a.grad, b.grad = torch.zeros_like(a), _uneven(128, 8, seed=5)
    opt.step()
    g2 = _uneven(128, 8, seed=6)
    b.grad = g2.clone()
    before = b.detach().clone()
    opt.step()
    stateless = -1e-2 * rank_scale(8) * dimuon_direction(g2, torch.zeros_like(g2), rank=8, row_split=None)
    assert _cos(b.detach() - before, stateless) < 0.999


def test_optimizer_requires_specs():
    p = torch.nn.Parameter(torch.randn(4, 8))
    p.grad = torch.randn(4, 8)
    with pytest.raises(KeyError):
        Dimuon([p]).step()


def _qkv_module(*, attention_output_gate):
    config = SimpleNamespace(
        kv_channels=4, num_attention_heads=4, num_query_groups=2, attention_output_gate=attention_output_gate
    )
    return SimpleNamespace(
        base_linear_name="decoder.layers.0.self_attention.linear_qkv", to_wrap=SimpleNamespace(config=config)
    )


def test_qkv_split_has_query_key_and_value_blocks():
    assert fused_row_split(_qkv_module(attention_output_gate=False)) == (8, 4, 4)


def test_gated_attention_keeps_query_and_gate_in_one_block_like_hf_q_proj():
    assert fused_row_split(_qkv_module(attention_output_gate=True)) == (16, 4, 4)


def _gdn_in_proj_module(tp):
    config = SimpleNamespace(
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4
    )
    # Unsharded rows are q 8, k 8, v 16, z 16, b 4 and a 4
    local_rows = 56 // tp
    adapter = SimpleNamespace(linear_out=SimpleNamespace(weight=torch.zeros(local_rows, 8)))
    return SimpleNamespace(
        base_linear_name="decoder.layers.0.self_attention.in_proj",
        to_wrap=SimpleNamespace(config=config),
        adapters=[adapter],
    )


@pytest.mark.parametrize(("tp", "expected"), [(1, (8, 8, 16, 16, 4, 4)), (2, (4, 4, 8, 8, 2, 2))])
def test_gdn_in_proj_splits_into_q_k_v_z_b_a_blocks_per_tp_shard(tp, expected):
    assert fused_row_split(_gdn_in_proj_module(tp)) == expected


def test_loaded_momentum_stays_bf16():
    a, b, opt = _two_factor_optimizer()
    a.grad, b.grad = torch.zeros_like(a), torch.randn(128, 8)
    opt.step()
    saved = opt.state[b]["momentum_buffer"].clone()
    _, b2, resumed = _two_factor_optimizer()
    resumed.load_state_dict(opt.state_dict())
    torch.testing.assert_close(resumed.state[b2]["momentum_buffer"], saved, rtol=0, atol=0)
