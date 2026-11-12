import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/math_sft_data.py"
_SPEC = importlib.util.spec_from_file_location("_math_sft_data", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
math_microbatches = _MODULE.math_microbatches
global_token_scale = _MODULE.global_token_scale


def _rollouts():
    return {
        "prompts": [{"index": i, "prompt_ids": [1] * (3 + i)} for i in range(8)],
        "samples": [
            {"prompt_index": i, "member": j, "completion_ids": [2 + j] * (1 + (i + j) % 4)}
            for i in range(8)
            for j in range(4)
        ],
    }


@pytest.mark.parametrize("multiple", [2, 4])
def test_every_real_token_and_loss_occurrence_survives_alignment_and_sharding(multiple):
    source = _rollouts()
    seen = []
    for rank in range(4):
        batches = math_microbatches(source, dp_rank=rank, dp_size=4, sp_multiple=multiple, max_groups_per_microbatch=1)
        assert len(batches) == 2
        for batch in batches:
            seen.extend(batch.prompt_indices)
            p = source["prompts"][batch.prompt_indices[0]]["prompt_ids"]
            samples = [s for s in source["samples"] if s["prompt_index"] == batch.prompt_indices[0]]
            assert batch.supervised_tokens == sum(len(s["completion_ids"]) for s in samples)
            assert 0 <= batch.padding_tokens < multiple
            assert batch.real_input_tokens == sum(len(p) + len(s["completion_ids"]) - 1 for s in samples)
            for shared in (False, True):
                plan = build_prefix_tree(
                    batch.sequences,
                    sharing_keys=batch.slots,
                    computation_keys=batch.computation_keys,
                    share_prefixes=shared,
                )
                assert len(plan.tokens) % multiple == 0
                layout = SFTLossLayout.from_plan(plan, batch.targets, weights=batch.weights, device="cpu")
                assert len(layout.rows) == batch.supervised_tokens
                logits = torch.randn(len(plan.tokens), 17, dtype=torch.float64)
                expected = []
                for nodes, labels, weights in zip(plan.sample_nodes, batch.targets, batch.weights, strict=True):
                    expected.append(
                        (
                            F.cross_entropy(logits[list(nodes)], torch.tensor(labels), reduction="none")
                            * torch.tensor(weights)
                        ).sum()
                    )
                torch.testing.assert_close(layout.loss(logits), sum(expected) / batch.supervised_tokens)
    assert sorted(seen) == list(range(8))


def test_unequal_group_counts_are_rejected_before_expert_collectives():
    source = _rollouts()
    source["samples"].pop()
    with pytest.raises(ValueError, match="equal numbers"):
        math_microbatches(source, dp_rank=0, dp_size=4)


def test_variable_completion_lengths_and_dp_counts_match_one_global_token_loss():
    source = _rollouts()
    parts = [math_microbatches(source, dp_rank=rank, dp_size=4) for rank in range(4)]
    total = sum(batch.supervised_tokens for batches in parts for batch in batches)
    parameter = torch.randn(17, dtype=torch.float64, requires_grad=True)
    losses, all_labels = [], []
    for batches in parts:
        local = 0
        for batch in batches:
            labels = [
                label
                for targets, weights in zip(batch.targets, batch.weights, strict=True)
                for label, weight in zip(targets, weights, strict=True)
                if weight
            ]
            all_labels.extend(labels)
            mean = F.cross_entropy(parameter.expand(len(labels), -1), torch.tensor(labels))
            local = local + mean * global_token_scale(len(labels), total, 4)
        losses.append(local)
    distributed = torch.stack(losses).mean()
    reference = F.cross_entropy(parameter.expand(len(all_labels), -1), torch.tensor(all_labels))
    torch.testing.assert_close(distributed, reference, atol=1e-14, rtol=1e-14)
    a = torch.autograd.grad(distributed, parameter)[0]
    b = torch.autograd.grad(reference, parameter)[0]
    torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)


def test_packing_is_global_and_preserves_every_trajectory():
    source = _rollouts()
    for budget, expected_steps in ((10000, 1), (1, 2)):
        parts = [math_microbatches(source, dp_rank=rank, dp_size=4, max_microbatch_tokens=budget) for rank in range(4)]
        assert {len(batches) for batches in parts} == {expected_steps}
        assert sorted(i for batches in parts for batch in batches for i in batch.prompt_indices) == list(range(8))
        assert sum(batch.supervised_tokens for batches in parts for batch in batches) == sum(
            len(sample["completion_ids"]) for sample in source["samples"]
        )
        for batches in parts:
            for batch in batches:
                assert list(batch.slots) == sorted(batch.slots)
                assert len(batch.sequences) == 4 * len(batch.prompt_indices)
