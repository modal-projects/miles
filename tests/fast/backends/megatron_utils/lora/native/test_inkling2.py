"""Inkling2 native attention/unembedding slots; no GPU or checkpoint required."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.modules.linear import (
    assert_every_adapter_slotted,
    iter_multi_lora_adapters,
    iter_slot_adapters,
)
from miles.backends.megatron_utils.lora.native.registry import (
    MODEL_SPECS,
    resolve_adapter_targets,
    validate_native_lora_args,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets


class Linear(nn.Module):
    def __init__(self, width: int, rows: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, width), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x @ self.weight.t(), None


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_process = True
        self.layer = nn.Module()
        self.layer.layer_number = 1
        attn = self.layer.self_attention = nn.Module()
        attn.num_heads, attn.head_dim, attn.d_rel = 2, 2, 1
        attn.linear_q = Linear(8, 12)
        attn.linear_k = Linear(8, 4)
        attn.linear_v = Linear(8, 4)
        attn.linear_proj = Linear(4, 8)
        self.output_layer = Linear(8, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.layer.self_attention
        q, _ = attn.linear_q(x)
        k, _ = attn.linear_k(x)
        v, _ = attn.linear_v(x)
        y, _ = attn.linear_proj(q[..., :4] + k + v)
        return self.output_layer(y)[0]


def attach(model: Model, slots: int | None) -> Model:
    spec = MODEL_SPECS["inkling2"]
    context = AttachContext(
        lora=LoRAConfig(rank=4, alpha=8, dropout=0, target_modules=None, num_slots=slots),
        transformer_config=SimpleNamespace(hidden_size=8, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=spec.declared_naming,
    )
    spec.attention.attach(model.layer.self_attention, model.layer, context)
    spec.lm_head.attach(model, SimpleNamespace(), context)
    return model


def test_inkling2_mixed_slots_match_independent_adapters_and_keep_sink_frozen() -> None:
    torch.manual_seed(42)
    base = Model().double()
    multi = attach(deepcopy(base), 2)
    singles = [attach(deepcopy(base), None) for _ in range(2)]
    assert_every_adapter_slotted([multi])
    multis = list(iter_multi_lora_adapters([multi]))
    assert len(multis) == 5  # fused Q/R/gate, K, V, O, and unembedding
    counts = (3, 2)
    for adapter in multis:
        adapter.token_counts_by_slot = counts
        for slot in range(2):
            adapter.init_slot(slot, lora_rank=4, lora_alpha=8)
            with torch.no_grad():
                for p in adapter.adapters[slot].parameters():
                    p.normal_(std=0.05)
    for slot, single in enumerate(singles):
        single_adapters = [
            single.layer.self_attention.lora_q_adapter,
            single.layer.self_attention.lora_k_adapter,
            single.layer.self_attention.lora_v_adapter,
            single.layer.self_attention.lora_o_adapter,
            single.lora_lm_head_adapter,
        ]
        for target, source in zip(single_adapters, iter_slot_adapters([multi], slot), strict=True):
            target.load_state_dict(source.state_dict())
    x = torch.randn(sum(counts), 1, 8, dtype=torch.float64)
    actual = multi(x)
    expected = torch.cat([single(part) for single, part in zip(singles, x.split(counts), strict=True)])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for single, part in zip(singles, x.split(counts), strict=True):
        single(part).square().sum().backward()
    for slot, single in enumerate(singles):
        actual_grads = [p.grad for a in iter_slot_adapters([multi], slot) for p in a.parameters()]
        expected_grads = [p.grad for p in single.parameters() if p.requires_grad]
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)
    q_adapter = multi.layer.self_attention.lora_q_adapter
    delta = q_adapter(x, multi.layer.self_attention.linear_q)
    assert torch.count_nonzero(delta[..., -2:]) == 0
    for slot in range(2):
        exported = dict(export_lora_hf_named(iter_slot_adapters([multi], slot), dtype=torch.float64))
        assert len(exported) == 14
        assert exported["model.llm.unembed.lora_B.weight"].shape == (12, 4)
        assert not any("sink" in name or "experts" in name for name in exported)
    before = multi(x).detach().clone()
    for adapter in multis:
        adapter.clear_slot(0)
    torch.testing.assert_close(multi(x)[counts[0] :], before[counts[0] :])


def test_inkling2_native_targets_reject_experts() -> None:
    config = {"model_type": "inkling2", "text_config": {"n_shared_experts": 0}}
    targets = resolve_hf_lora_targets(config, target_modules=["attn", "unembed"])
    options = dict(hf_modules=[], lora_type="canonical_lora", experts_shared_outer_loras=False)
    served = resolve_adapter_targets(config, targets, **options)
    assert "lm_head" in served
    assert not any("experts" in name for name in targets)
    with pytest.raises(AssertionError, match="shared-outer expert adapters"):
        resolve_adapter_targets(config, [*targets, "model.llm.layers.*.mlp.experts.down_proj"], **options)


def test_inkling2_default_single_lora_retains_expert_targets() -> None:
    config = {"model_type": "inkling2", "text_config": {"n_shared_experts": 0}}
    targets = resolve_hf_lora_targets(config)
    assert any(".experts." in target for target in targets)
    served = resolve_adapter_targets(
        config, targets, hf_modules=[], lora_type="canonical_lora", experts_shared_outer_loras=True
    )
    assert any(".experts." in target for target in served)
    assert MODEL_SPECS["inkling2"].experts is not None


def test_inkling2_multi_lora_rejects_experts_at_startup(tmp_path) -> None:
    config = {"model_type": "inkling2", "text_config": {"n_shared_experts": 0}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(
        overlap_param_gather=False,
        moe_shared_expert_overlap=False,
        overlap_grad_reduce=False,
        hf_checkpoint=str(tmp_path),
        hf_lora_targets=resolve_hf_lora_targets(config),
        multi_lora=True,
    )
    with pytest.raises(AssertionError, match="attention and unembedding only"):
        validate_native_lora_args(args, [])
