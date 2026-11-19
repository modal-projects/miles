"""Slot isolation on expert-major tokens, including empty slots and padded ranks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.modules import expert_slots
from miles.backends.megatron_utils.lora.native.modules.expert_slots import NativeMultiLoRAExpertAdapter
from miles.backends.megatron_utils.lora.native.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext


def _context():
    return AttachContext(
        lora=LoRAConfig(rank=4, alpha=8, dropout=0, target_modules=None, num_slots=3),
        transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )


@pytest.mark.parametrize("projection", ["gate_up", "up", "down"])
@pytest.mark.parametrize("counts", [(3, 0, 2), (0, 0, 0), (5,), (0,)])
@pytest.mark.parametrize("shared_outer", [True, False])
@pytest.mark.parametrize("kernel_policy", ["native", "rowwise"])
def test_expert_major_forward_and_gradients_match_per_token_reference(
    monkeypatch, projection, counts, shared_outer, kernel_policy
):
    context = _context()
    context = replace(
        context, lora=replace(context.lora, experts_shared_outer_loras=shared_outer, kernel_policy=kernel_policy)
    )
    common = dict(
        hf_prefix="model.layers.0.mlp.experts.",
        reference=torch.empty(()),
        context=context,
        num_local_experts=2,
        moe_intermediate=5,
        width=6,
        is_ep=False,
    )
    build = (
        (lambda: LoRAGroupedFC2(hf_leaf="down_proj", **common))
        if projection == "down"
        else (
            lambda: LoRAGroupedFC1(
                hf_leaves=("up_proj",) if projection == "up" else ("gate_proj", "up_proj"), **common
            )
        )
    )
    adapter = NativeMultiLoRAExpertAdapter([build() for _ in counts], context)
    for slot in range(len(counts)):
        adapter.init_slot(slot, slot + 1, 6)
        with torch.no_grad():
            for name, parameter in adapter.adapters[slot].named_parameters():
                if name.endswith("_B"):
                    parameter.normal_()
                    parameter[..., slot + 1 :].zero_()
    slot_ids = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    # top-2 dispatch duplicates each token once into each expert's segment.
    permutation = torch.arange(sum(counts)).repeat(2)
    dispatched_slots = slot_ids[permutation]
    per_expert = [sum(counts), sum(counts)]
    monkeypatch.setattr(expert_slots, "_co_permute_slot_ids", lambda _d, ids, _n: ids[permutation])
    dispatcher = SimpleNamespace(hidden_shape_before_permute=(sum(counts), 6))
    adapter.routing = expert_slots.build_expert_slot_routing(dispatcher, per_expert, counts, context, "cpu")
    x = torch.randn(len(permutation), 5 if projection == "down" else 6, requires_grad=True)
    actual = adapter(x, nn.Identity())
    expected = []
    for row, slot, expert in zip(x, dispatched_slots.tolist(), [0] * sum(counts) + [1] * sum(counts), strict=True):
        single = adapter.adapters[slot]
        if projection == "down":
            b = single.w2_B if shared_outer else single.w2_B[expert]
            delta = b @ (single.w2_A[expert] @ row)
        else:
            delta = torch.cat([b[expert] @ ((a if shared_outer else a[expert]) @ row) for _, a, b in single.factors()])
        expected.append((6 / (slot + 1)) * delta)
    reference = torch.stack(expected) if expected else actual.new_empty(actual.shape)
    torch.testing.assert_close(actual, reference)
    # A grouped host can return a view whose storage its backward still owns.
    host = F.linear(x, torch.randn(2 * actual.shape[-1], x.shape[-1]))
    out, _ = host.chunk(2, dim=-1)
    before = out.detach().clone()
    expected_sum = out + reference
    summed = adapter.add_delta(out, actual)
    torch.testing.assert_close(summed, expected_sum)
    torch.testing.assert_close(out, before, rtol=0, atol=0)
    assert summed.data_ptr() == actual.data_ptr()
    inputs = (x, *adapter.parameters())
    gradient = torch.randn_like(summed)
    actual_gradients = torch.autograd.grad(summed, inputs, gradient, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_sum, inputs, gradient, allow_unused=True)
    for value, actual_gradient, expected_gradient in zip(inputs, actual_gradients, expected_gradients, strict=True):
        expected_gradient = torch.zeros_like(value) if expected_gradient is None else expected_gradient
        torch.testing.assert_close(actual_gradient, expected_gradient)
    gradients = {id(value): gradient for value, gradient in zip(inputs, actual_gradients, strict=True)}
    for single in adapter.adapters:
        for name, parameter in single.named_parameters():
            grad = gradients[id(parameter)]
            rank = adapter.lora_ranks_by_slot[list(adapter.adapters).index(single)]
            padded = grad[..., rank:, :] if name.endswith("_A") else grad[..., rank:]
            assert torch.count_nonzero(padded) == 0


def test_expert_routing_narrows_the_sequence_parallel_shard(monkeypatch):
    context = _context()
    context = replace(
        context, tp_size=2, tp_rank=1, transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=True)
    )
    seen = []

    def co_permute(_dispatcher, ids, _experts):
        seen.extend(ids.tolist())
        return ids

    monkeypatch.setattr(expert_slots, "_co_permute_slot_ids", co_permute)
    routing = expert_slots.build_expert_slot_routing(
        SimpleNamespace(hidden_shape_before_permute=(3, 6)), [3], (2, 4, 0), context, "cpu"
    )
    assert seen == [1, 1, 1]
    assert routing.counts == ((0,), (3,), (0,))


@pytest.mark.parametrize("alpha", [4, 8])
@pytest.mark.parametrize("per_expert", [(3, 0, 2), (0, 0, 0)])
def test_single_slot_identity_matches_explicit_routing_all_gradients(monkeypatch, alpha, per_expert):
    context = _context()
    single = LoRAGroupedFC2(
        hf_prefix="model.layers.0.mlp.experts.",
        hf_leaf="down_proj",
        reference=torch.empty((), dtype=torch.float64),
        context=context,
        num_local_experts=3,
        moe_intermediate=5,
        width=6,
        is_ep=False,
    )
    adapter = NativeMultiLoRAExpertAdapter([single], context)
    adapter.init_slot(0, 4, alpha)
    with torch.no_grad():
        single.w2_B.normal_()
    rows = sum(per_expert)
    x = torch.randn(rows, 5, dtype=torch.float64, requires_grad=True)
    # A single slot needs neither slot-ID communication nor another row ordering.
    monkeypatch.setattr(
        expert_slots, "_co_permute_slot_ids", lambda *_args: pytest.fail("unnecessary slot communication")
    )
    adapter.routing = expert_slots.build_expert_slot_routing(
        SimpleNamespace(hidden_shape_before_permute=(rows, 5)), per_expert, (rows,), context, "cpu"
    )
    assert adapter.routing.permutation is adapter.routing.inverse is None
    actual = adapter(x, nn.Identity())
    gradient = torch.randn_like(actual)
    inputs = (x, *adapter.parameters())
    actual_gradients = torch.autograd.grad(actual, inputs, gradient)
    identity = torch.arange(rows)
    adapter.routing = expert_slots.ExpertSlotRouting(identity, identity, (per_expert,))
    expected = adapter(x, nn.Identity())
    expected_gradients = torch.autograd.grad(expected, inputs, gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
