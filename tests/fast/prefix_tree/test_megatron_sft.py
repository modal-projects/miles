"""Shared Megatron batch contract, independent of any model or GPU provider."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.prefix_tree.megatron import MegatronTreeSFTBatch
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


class _Layout:
    def __init__(self, plan):
        self.plan = plan

    def prepare_packed(self, model, packed):
        packed.test_layout = self


class _Model:
    config = SimpleNamespace(sequence_parallel=False)
    parallel_output = False

    def __init__(self):
        self.embeddings = torch.randn(17, 17, dtype=torch.float64, requires_grad=True)
        self.packed = None

    def __call__(self, tokens, *, position_ids, attention_mask, packed_seq_params):
        self.packed = packed_seq_params
        return self.embeddings[tokens]


@pytest.fixture
def megatron_contract(monkeypatch):
    # Only transport wrappers are substituted; CE and its gradients are real.
    packed = ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = SimpleNamespace
    utils = ModuleType("megatron.core.utils")
    utils.get_attr_wrapped_model = lambda model, name, **kwargs: getattr(model, name)
    monkeypatch.setitem(sys.modules, packed.__name__, packed)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


@pytest.mark.parametrize("shared", [True, False])
def test_architecture_adapter_and_native_baseline_use_same_loss_and_gradients(megatron_contract, shared):
    sequences = [[1, 2, 3], [1, 2, 4]]
    labels = [[2, 3, 5], [2, 4, 6]]
    plan = build_prefix_tree(sequences, share_prefixes=shared)
    model = _Model()
    batch = MegatronTreeSFTBatch.from_plan(plan, labels, device="cpu", layout=_Layout(plan))
    actual = batch.loss(model, shared=shared)
    expected = torch.nn.functional.cross_entropy(
        model.embeddings[torch.tensor(sequences)].reshape(-1, 17), torch.tensor(labels).reshape(-1)
    )
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, model.embeddings, retain_graph=True)[0]
    b = torch.autograd.grad(expected, model.embeddings)[0]
    torch.testing.assert_close(a, b)
    assert hasattr(model.packed, "test_layout") == shared
    assert model.packed.cu_seqlens_q.tolist() == list(plan.segment_offsets)
    assert batch.position_ids.tolist() == [list(plan.positions)]


def test_native_execution_refuses_shared_rows(megatron_contract):
    sequences = [[1, 2, 3], [1, 2, 4]]
    plan = build_prefix_tree(sequences)
    batch = MegatronTreeSFTBatch.from_plan(plan, sequences, device="cpu", layout=_Layout(plan))
    with pytest.raises(ValueError, match="expanded plan"):
        batch.loss(_Model(), shared=False)


def test_layout_must_match_supervision_plan():
    plan = build_prefix_tree([[1, 2]])
    with pytest.raises(ValueError, match="same plan"):
        MegatronTreeSFTBatch.from_plan(plan, [[2, 3]], device="cpu", layout=_Layout(build_prefix_tree([[1, 2]])))


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_dropout", 0.1),
        ("hidden_dropout", 0.1),
        ("moe_router_force_load_balancing", True),
        ("moe_router_force_biased", 0),
        ("moe_router_jitter_eps", 0.01),
        ("moe_expert_capacity_factor", 1.0),
        ("moe_aux_loss_coeff", [0.0, 0.1]),
    ],
)
def test_common_megatron_sharing_eligibility_rejects_unsupported_semantics(field, value):
    from miles.backends.training_utils.prefix_tree.megatron import validate_megatron_tree_config

    with pytest.raises(ValueError):
        validate_megatron_tree_config(SimpleNamespace(**{field: value}))


def test_common_megatron_router_bias_must_be_fixed():
    from miles.backends.training_utils.prefix_tree.megatron import validate_megatron_tree_config

    validate_megatron_tree_config(
        SimpleNamespace(moe_aux_loss_coeff=[0.0], moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.0)
    )
    with pytest.raises(ValueError, match="fixed router expert bias"):
        validate_megatron_tree_config(
            SimpleNamespace(moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.01)
        )
