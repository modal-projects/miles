"""CPU tests for per-rollout loss denominators (``rollout_mask_sums``).

Covers:
- the rollout-side precomputation in ``convert_samples_to_train_data``;
- ``get_sum_of_sample_mean(denominators=...)`` — legacy per-sample mean when
  ``None``, per-rollout token-weighted mean with precomputed denominators, and
  the strict no-op equivalence between the two when 1 rollout = 1 sample;
- ``aggregate_train_losses(num_rollouts=...)`` per-rollout-mean reduction;
- prompt-mean denominators (``--prompt-mean-loss``).
"""

from __future__ import annotations

import math

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, loss_denominators
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data


@pytest.fixture(autouse=True)
def _parallel_state():
    make_parallel_state()


def _convert(samples, **overrides):
    return convert_samples_to_train_data(
        make_args(rewards_normalization=False, **overrides),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


class TestRolloutMaskSumsPrecompute:
    def test_defaults_to_per_sample_mask_sums(self):
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [2, 3, 4]

    def test_compact_siblings_share_whole_rollout_total(self):
        samples = [make_sample(index=i, response_length=n) for i, n in enumerate((2, 3, 4))]
        # samples 0 and 1 come from the same rollout execution
        samples[0].rollout_id = samples[1].rollout_id = 0
        out = _convert(samples)
        assert out["rollout_mask_sums"] == [5, 5, 4]

    def test_masked_out_tokens_excluded(self):
        s = make_sample(index=0, response_length=4)
        s.loss_mask = [1, 0, 1, 0]
        out = _convert([s])
        assert out["rollout_mask_sums"] == [2]


class TestDenominators:
    def _masks(self, lens):
        return [torch.ones(n, dtype=torch.int) for n in lens]

    def test_none_matches_legacy_per_sample_mean(self):
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens))
        # sample means: (1+3)/2 = 2, (2+2+5)/3 = 3
        assert math.isclose(som(x).item(), 5.0)

    def test_rollout_denoms_give_token_weighted_rollout_mean(self):
        lens = [2, 3]
        x = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
        # both samples belong to one rollout with 5 mask tokens total
        denoms = torch.tensor([5.0, 5.0])
        som = get_sum_of_sample_mean(list(lens), list(lens), self._masks(lens), denominators=denoms)
        # (1+3)/5 + (2+2+5)/5 = 13/5 — one token-weighted mean for the rollout
        assert math.isclose(som(x).item(), 13.0 / 5, rel_tol=1e-6)

    def test_per_sample_denominators_equal_legacy(self):
        """1 rollout = 1 sample: rollout_mask_sums equals each mask's own sum -> strict no-op."""
        lens = [2, 3, 4]
        x = torch.randn(sum(lens))
        masks = self._masks(lens)
        legacy = get_sum_of_sample_mean(list(lens), list(lens), masks)
        denoms = torch.tensor([float(n) for n in lens])
        new = get_sum_of_sample_mean(list(lens), list(lens), masks, denominators=denoms)
        assert torch.allclose(legacy(x), new(x))


class TestAggregateTrainLossesNumRollouts:
    @pytest.fixture(autouse=True)
    def _no_allreduce(self, monkeypatch):
        monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", staticmethod(lambda *a, **k: None))

    def _mb(self, num_samples, values):
        return {"keys": list(values.keys()), "values": torch.tensor([num_samples, *values.values()])}

    def test_num_rollouts_overrides_sample_count(self):
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses, num_rollouts=4)
        assert math.isclose(out["loss"], 2.0)

    def test_legacy_reduction_without_divisor(self):
        losses = [self._mb(2, {"loss": 2.0}), self._mb(2, {"loss": 6.0})]
        out = log_utils.aggregate_train_losses(losses)
        assert math.isclose(out["loss"], 2.0)


class TestPromptMeanDenominators:
    @staticmethod
    def _grouped(lengths_by_group):
        samples = []
        for group_index, lengths in enumerate(lengths_by_group):
            for n in lengths:
                samples.append(make_sample(group_index=group_index, index=len(samples), response_length=n))
        return samples

    @staticmethod
    def _loss(out, per_token_loss):
        """The sample-mean loss: sum of per-sample sums over denominators, over the rollout count."""
        lengths = out["response_lengths"]
        masks = [torch.tensor(mask, dtype=torch.int) for mask in out["loss_masks"]]
        denominators = torch.tensor(loss_denominators(out), dtype=torch.float32)
        reducer = get_sum_of_sample_mean(lengths, lengths, masks, denominators=denominators)
        return reducer(per_token_loss) / len(set(out["rollout_ids"]))

    def test_only_the_flag_adds_prompt_mean_denominators(self):
        samples = self._grouped([(2, 6), (1, 3)])
        assert "loss_denominators" not in _convert(samples)

        out = _convert(samples, prompt_mean_loss=True)
        # T_q * P / N: prompt 0 has 8 tokens, prompt 1 has 4, over 2 prompts and 4 rollouts.
        assert out["loss_denominators"] == [4.0, 4.0, 2.0, 2.0]
        # Per-rollout denominators, which the rollout/* means use, are untouched.
        assert out["rollout_mask_sums"] == [2, 6, 1, 3]

    def test_loss_is_a_token_mean_per_prompt_then_a_mean_over_prompts(self):
        samples = self._grouped([(2, 6), (1, 3, 5), (4,)])
        samples[1].loss_mask = [1, 0, 1, 1, 0, 1]
        out = _convert(samples, prompt_mean_loss=True)
        x = torch.randn(sum(out["response_lengths"]), generator=torch.Generator().manual_seed(0))

        masks = [torch.tensor(mask, dtype=torch.float32) for mask in out["loss_masks"]]
        per_sample = [
            (chunk * mask).sum() for chunk, mask in zip(x.split(out["response_lengths"]), masks, strict=True)
        ]
        per_sample_tokens = [mask.sum() for mask in masks]
        groups = [(0, 1), (2, 3, 4), (5,)]
        expected = sum(
            sum(per_sample[i] for i in group) / sum(per_sample_tokens[i] for i in group) for group in groups
        ) / len(groups)

        torch.testing.assert_close(self._loss(out, x), expected)

    def test_siblings_of_one_rollout_count_as_one_rollout(self):
        samples = self._grouped([(2, 3, 5), (4, 4)])
        samples[0].rollout_id = samples[1].rollout_id = 100
        out = _convert(samples, prompt_mean_loss=True)
        x = torch.randn(sum(out["response_lengths"]), generator=torch.Generator().manual_seed(1))

        # 2 prompts, 4 rollouts: prompt 0 holds 10 tokens, prompt 1 holds 8.
        assert out["loss_denominators"] == [5.0, 5.0, 5.0, 4.0, 4.0]
        head, tail = x[:10], x[10:]
        torch.testing.assert_close(self._loss(out, x), (head.mean() + tail.mean()) / 2)

    def test_prompt_mean_denominators_take_precedence_in_the_loss(self):
        mask_sums, prompt_mean = torch.tensor([2.0]), torch.tensor([4.0])
        assert loss_denominators({"rollout_mask_sums": mask_sums}) is mask_sums
        assert loss_denominators({"rollout_mask_sums": mask_sums, "loss_denominators": prompt_mean}) is prompt_mean
        assert loss_denominators({"rollout_mask_sums": mask_sums, "loss_denominators": None}) is mask_sums
