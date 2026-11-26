"""The score-centering KL coefficient scaled by each response token's age."""

import math

import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _turn

from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import response_token_ages

_SUPPORT_REPLAY = dict(
    rollout_top_p=0.6,
    rollout_top_k=3,
    rollout_sampling_logprobs_mode="support",
    use_sampling_support_replay=True,
)


def _inputs(ages: torch.Tensor | None, kl_direction: str = "forward") -> tuple[torch.Tensor, ScoreCenteringInputs]:
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    log_p = logits.log_softmax(-1)
    log_q = torch.randn(4, 3, generator=generator, dtype=torch.float64).log_softmax(-1)
    return logits, ScoreCenteringInputs(
        train_log_probs=log_p[:, 0],
        train_head_log_probs=log_p,
        rollout_log_probs=log_q[:, 0],
        rollout_head_log_probs=log_q,
        head_mask=torch.ones(4, 3, dtype=torch.bool),
        advantages=torch.tensor([0.5, -0.2, 0.0, 1.1], dtype=torch.float64),
        kl_coef=0.3,
        kl_direction=kl_direction,
        kl_token_ages=ages,
    )


@pytest.mark.parametrize("kl_direction", ["forward", "reverse"])
def test_token_ages_divide_the_kl_coefficient_by_the_square_root_of_age(kl_direction: str) -> None:
    ages = torch.tensor([4, 9, 1, 16])
    _, flat = _inputs(None, kl_direction)
    loss_flat, metrics = score_centering_loss(flat)
    logits, aged = _inputs(ages, kl_direction)
    loss_aged, _ = score_centering_loss(aged)

    kl = metrics.forward_kl if kl_direction == "forward" else metrics.kl
    scale = torch.tensor([1 / 2, 1 / 3, 1.0, 1 / 4], dtype=torch.float64)
    # The KL term is the only part that changes: flat - aged = 0.3 * kl * (1 - scale).
    torch.testing.assert_close(loss_flat - loss_aged, 0.3 * kl * (1 - scale), rtol=1e-12, atol=1e-12)


def test_ages_below_one_keep_the_full_coefficient() -> None:
    _, flat = _inputs(None)
    _, aged = _inputs(torch.tensor([0, -1, 1, 0]))
    torch.testing.assert_close(score_centering_loss(aged)[0], score_centering_loss(flat)[0], rtol=0, atol=0)


def test_token_ages_must_be_per_token() -> None:
    with pytest.raises(ValueError, match="token ages must be"):
        _inputs(torch.tensor([1, 2]))


def _versioned_sample() -> Sample:
    """Prompt [10, 11], a first call generating [20, 21] at version 5, a tool reply [30] with no version, and a
    second call generating [40, 41] at version 7."""
    sample = Sample(tokens=[10, 11, 20, 21, 30, 40, 41], response_length=5, loss_mask=[1, 1, 0, 1, 1])
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="5", abs_start=2, abs_end=4)]),
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="7", abs_start=5, abs_end=7)]),
    ]
    return sample


def test_response_token_ages_follow_the_version_spans() -> None:
    assert response_token_ages(_versioned_sample(), trainer_version=9) == [4, 4, -1, 2, 2]


def test_non_numeric_versions_have_no_age() -> None:
    sample = _versioned_sample()
    sample.weight_versions[1].spans[0] = WeightVersionSpan(version="default", abs_start=5, abs_end=7)
    assert response_token_ages(sample, trainer_version=9) == [4, 4, -1, -1, -1]


def test_age_decay_needs_a_positive_kl_coefficient() -> None:
    with pytest.raises(ValueError, match="age-decay"):
        validate_score_centering_args(_args(score_centering_kl_age_decay="sqrt", **_SUPPORT_REPLAY))
    validate_score_centering_args(
        _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    )


def test_conversion_carries_token_ages_from_the_trainer_version() -> None:
    sample = _turn([5, 6], [3, 2], [0.4, 0.6])
    sample.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=2, abs_end=4)])]
    args = _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    data = convert_samples_to_train_data(args, [sample], {"trainer_weight_version": 8}, None, None)
    assert [list(ages) for ages in data["rollout_token_ages"]] == [[5, 5]]
    with pytest.raises(AssertionError, match="trainer's weight version"):
        convert_samples_to_train_data(args, [sample], {}, None, None)


def test_conversion_without_age_decay_carries_no_ages() -> None:
    sample = _turn([5, 6], [3, 2], [0.4, 0.6])
    data = convert_samples_to_train_data(_args(**_SUPPORT_REPLAY), [sample], {}, None, None)
    assert "rollout_token_ages" not in data
    assert math.isfinite(sum(data["rollout_log_probs"][0]))
