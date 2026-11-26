"""The trainer-to-sampler KL over the sampling support against dense oracles and the estimator it extends."""

import dataclasses
import math

import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils.loss.hub.score_centering import (
    ScoreCenteringInputs,
    score_centering_loss,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.loss.hub.score_centering.importance_sampling import importance_sampling
from miles.backends.training_utils.loss.hub.score_centering.masks import (
    drop_inactive_nan,
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)
from miles.utils.score_centering import validate_score_centering_args


def _estimator_without_kl(inputs: ScoreCenteringInputs) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """The estimator as it was before the KL term, kept verbatim so kl_coef=0 stays pinned to it."""
    sampling = importance_sampling(
        inputs.mode, tis_clip=inputs.tis_clip, mis_low=inputs.mis_low, mis_high=inputs.mis_high
    )
    active = inputs.advantages.detach() != 0
    train_head = sanitize_head_log_probs(inputs.train_head_log_probs, inputs.head_mask, active)
    rollout_head = sanitize_head_log_probs(inputs.rollout_head_log_probs, inputs.head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, inputs.head_mask), head_probs(rollout_head, inputs.head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(inputs.eps) / (1 - p_mass).clamp_min(inputs.eps)
        alpha = sampling.tail_scale(rho, p.dtype)
        residual = sampling.head_mass(p, q, train_head - rollout_head) - alpha.unsqueeze(-1) * p
        weight = sampling.sample_weight(
            drop_inactive_nan(inputs.train_log_probs - inputs.rollout_log_probs, active, 0.0)
        )
    correction = (residual * scored_log_probs(train_head, residual)).sum(-1)
    loss = -inputs.advantages.detach() * (weight * scored_log_probs(inputs.train_log_probs, weight) - correction)
    return loss, (correction.detach(), p_mass, q_mass, rho, weight)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_zero_coefficient_is_bit_identical_to_the_estimator_without_kl(mode: str) -> None:
    generator = torch.Generator().manual_seed(11)
    log_p = torch.randn(6, 7, generator=generator).log_softmax(-1)
    log_q = (torch.randn(6, 7, generator=generator) * 2).log_softmax(-1)
    rollout_head, ids = log_q.topk(4, dim=-1)
    sampled = torch.tensor([[0], [3], [1], [6], [2], [5]])
    train, rollout = log_p.gather(-1, sampled).squeeze(-1), log_q.gather(-1, sampled).squeeze(-1)
    train_head = log_p.gather(-1, ids)
    head_mask = torch.ones(6, 4, dtype=torch.bool)
    head_mask[1, 3] = False
    head_mask[4] = False
    advantages = torch.tensor([1.5, -0.7, 0.0, 0.3, 0.0, -1.1])
    rollout[2] = rollout_head[2, 0] = train_head[4, 1] = torch.nan
    results = []
    for estimator in (score_centering_loss, _estimator_without_kl):
        train_leaf, head_leaf = train.clone().requires_grad_(), train_head.clone().requires_grad_()
        loss, metrics = estimator(
            ScoreCenteringInputs(
                train_log_probs=train_leaf,
                train_head_log_probs=head_leaf,
                rollout_log_probs=rollout,
                rollout_head_log_probs=rollout_head,
                head_mask=head_mask,
                advantages=advantages,
                mode=mode,
                kl_coef=0.0,
            )
        )
        if estimator is score_centering_loss:
            metrics = (
                metrics.correction,
                metrics.train_head_mass,
                metrics.rollout_head_mass,
                metrics.tail_ratio,
                metrics.importance_weight,
            )
        results.append((loss, *torch.autograd.grad(loss.sum(), (train_leaf, head_leaf)), *metrics))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kl_direction", ["forward", "reverse"])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("sampling_support", [False, True])
def test_kl_value_and_logit_gradient_match_the_dense_support_oracle(
    mode: str, sampling_support: bool, kl_direction: str
) -> None:
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(3, 9, generator=generator, dtype=torch.float64)
    support = torch.tensor([[4, 1, 6, -1], [0, 2, 3, 8], [7, 5, -1, -1]])
    valid = support >= 0
    log_q = torch.randn(3, 4, generator=generator, dtype=torch.float64).masked_fill(~valid, -torch.inf).log_softmax(-1)

    def support_log_probs(scores: torch.Tensor) -> torch.Tensor:
        return (scores / 0.7).gather(-1, support.clamp_min(0)).masked_fill(~valid, -torch.inf).log_softmax(-1)

    def run(kl_coef: float):
        scores = logits.clone().requires_grad_()
        if sampling_support:
            ids = torch.cat((support[:, :1], support), dim=-1)
            selected, _ = selected_log_probs_and_entropy(scores, ids, temperature=0.7, sampling_support=True)
        else:
            head = support_log_probs(scores)
            selected = torch.cat((head[:, :1], head), dim=-1)
        loss, metrics = score_centering_loss(
            ScoreCenteringInputs(
                train_log_probs=selected[:, 0],
                train_head_log_probs=selected[:, 1:],
                rollout_log_probs=log_q[:, 0],
                rollout_head_log_probs=log_q,
                head_mask=valid,
                advantages=torch.tensor([1.3, -0.4, 0.8], dtype=torch.float64),
                mode=mode,
                kl_coef=kl_coef,
                kl_direction=kl_direction,
            )
        )
        return loss, metrics, torch.autograd.grad(loss.sum(), scores)[0]

    log_p = support_log_probs(logits)
    log_ratio = torch.where(valid, log_p - log_q, 0.0)
    kl = (log_p.exp() * log_ratio).sum(-1)
    forward_kl = -(log_q.exp() * log_ratio).sum(-1)
    if kl_direction == "forward":
        # d/dz of sum_c q_c (log q_c - log p_c), p the support softmax of z / T: (p - q) / T on the support.
        penalty = forward_kl
        logit_grad = torch.where(valid, log_p.exp() - log_q.exp(), 0.0) / 0.7
    else:
        penalty = kl
        logit_grad = torch.where(valid, log_p.exp() * (log_ratio - kl[:, None]), 0.0) / 0.7
    expected_grad = torch.zeros_like(logits).scatter_add(-1, support.clamp_min(0), logit_grad)

    loss_without_kl, _, grad_without_kl = run(0.0)
    loss, metrics, grad = run(0.3)
    torch.testing.assert_close(metrics.kl, kl, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(metrics.forward_kl, forward_kl, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loss - loss_without_kl, 0.3 * penalty, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad - grad_without_kl, 0.3 * expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kl_direction", ["forward", "reverse"])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_kl_reaches_zero_advantage_rows_and_skips_rows_without_head_slots(mode: str, kl_direction: str) -> None:
    logits = torch.randn(3, 4, generator=torch.Generator().manual_seed(8), dtype=torch.float64, requires_grad=True)
    log_q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64).log().expand(3, 4)
    train_nan = torch.tensor([[False] * 4, [True, False, False, False], [True] * 4])
    rollout_nan = torch.tensor([[False] * 4, [False, True, False, False], [True] * 4])
    train_head = torch.where(train_nan, torch.nan, logits.log_softmax(-1))
    rollout_head = torch.where(rollout_nan, torch.nan, log_q)
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=train_head[:, 0],
            train_head_log_probs=train_head,
            rollout_log_probs=rollout_head[:, 0],
            rollout_head_log_probs=rollout_head,
            head_mask=torch.tensor([[True] * 4, [True] * 4, [False] * 4]),
            advantages=torch.zeros(3, dtype=torch.float64),
            mode=mode,
            kl_coef=0.3,
            kl_direction=kl_direction,
        )
    )
    grad = torch.autograd.grad(loss.sum(), logits)[0]

    log_p = logits.detach()[0].log_softmax(-1)
    if kl_direction == "forward":
        kl = (log_q[0].exp() * (log_q[0] - log_p)).sum()
        expected_grad = log_p.exp() - log_q[0].exp()
    else:
        kl = (log_p.exp() * (log_p - log_q[0])).sum()
        expected_grad = log_p.exp() * (log_p - log_q[0] - kl)
    torch.testing.assert_close(loss[0], 0.3 * kl, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad[0], 0.3 * expected_grad, rtol=1e-12, atol=1e-12)
    assert grad[0].abs().sum() > 0
    assert torch.isfinite(loss[1]) and torch.isfinite(grad[1]).all()
    assert all(torch.isfinite(value).all() for value in metrics.as_log_dict().values())
    assert loss[2] == 0 and (grad[2] == 0).all() and metrics.kl[2] == 0 and metrics.forward_kl[2] == 0


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_matched_policies_have_zero_kl_and_the_sampler_gets_no_gradient(mode: str) -> None:
    logits = torch.randn(2, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64, requires_grad=True)
    log_p = logits.log_softmax(-1)
    rollout = log_p[:, 0].detach().clone().requires_grad_()
    rollout_head = log_p.detach().clone().requires_grad_()
    inputs = ScoreCenteringInputs(
        train_log_probs=log_p[:, 0],
        train_head_log_probs=log_p,
        rollout_log_probs=rollout,
        rollout_head_log_probs=rollout_head,
        head_mask=torch.ones(2, 4, dtype=torch.bool),
        advantages=torch.tensor([0.9, -1.2], dtype=torch.float64),
        mode=mode,
    )
    loss_without_kl, _ = score_centering_loss(inputs)
    grad_without_kl = torch.autograd.grad(loss_without_kl.sum(), logits, retain_graph=True)[0]
    loss, metrics = score_centering_loss(dataclasses.replace(inputs, kl_coef=0.5))
    grad, rollout_grad, rollout_head_grad = torch.autograd.grad(
        loss.sum(), (logits, rollout, rollout_head), allow_unused=True
    )
    assert (metrics.kl == 0).all() and (metrics.forward_kl == 0).all()
    torch.testing.assert_close(loss, loss_without_kl, rtol=0, atol=0)
    torch.testing.assert_close(grad, grad_without_kl, rtol=0, atol=1e-12)
    assert rollout_grad is None and rollout_head_grad is None


def _ratio_outside(log_ratios: list[float], mode: str) -> torch.Tensor:
    tokens = len(log_ratios)
    _, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=torch.tensor(log_ratios, dtype=torch.float64),
            train_head_log_probs=torch.full((tokens, 1), -1.0, dtype=torch.float64),
            rollout_log_probs=torch.zeros(tokens, dtype=torch.float64),
            rollout_head_log_probs=torch.full((tokens, 1), -1.0, dtype=torch.float64),
            head_mask=torch.ones(tokens, 1, dtype=torch.bool),
            advantages=torch.ones(tokens, dtype=torch.float64),
            mode=mode,
            mis_low=0.8,
            mis_high=1.25,
        )
    )
    return metrics.ratio_outside


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_ratio_outside_flags_sampled_ratios_beyond_the_mis_bounds(mode: str) -> None:
    low, high = math.log(0.8), math.log(1.25)
    log_ratios = [
        math.nextafter(low, -math.inf),
        low,
        math.nextafter(low, math.inf),
        0.0,
        math.nextafter(high, -math.inf),
        high,
        math.nextafter(high, math.inf),
        -math.inf,
        math.inf,
    ]
    assert _ratio_outside(log_ratios, mode).tolist() == [1, 0, 0, 0, 0, 0, 1, 1, 1]


def test_nan_sampled_ratio_is_outside() -> None:
    assert _ratio_outside([math.nan], "none").tolist() == [1]


_SUPPORT_REPLAY = dict(
    rollout_top_p=0.6,
    rollout_top_k=3,
    rollout_sampling_logprobs_mode="support",
    use_sampling_support_replay=True,
)


@pytest.mark.parametrize("kl_coef", [math.nan, math.inf, -0.1])
def test_kl_coefficient_must_be_finite_and_non_negative(kl_coef: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_score_centering_args(_args(score_centering_kl_coef=kl_coef, **_SUPPORT_REPLAY))


def test_positive_kl_coefficient_requires_support_log_probs() -> None:
    with pytest.raises(ValueError, match="requires --rollout-sampling-logprobs-mode support"):
        validate_score_centering_args(_args(score_centering_kl_coef=0.1))
    validate_score_centering_args(_args(score_centering_kl_coef=0.0))
    validate_score_centering_args(_args(score_centering_kl_coef=0.1, **_SUPPORT_REPLAY))


def test_kl_direction_must_be_forward_or_reverse() -> None:
    logits = torch.zeros(1, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="forward or reverse"):
        ScoreCenteringInputs(
            train_log_probs=logits[:, 0],
            train_head_log_probs=logits,
            rollout_log_probs=logits[:, 0],
            rollout_head_log_probs=logits,
            head_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=torch.zeros(1, dtype=torch.float64),
            kl_direction="both",
        )
