"""Score-centering KL diagnostics: positions far from the sampler, the KL's pull against the policy gradient's
push, and the forward KL by token age."""

import torch
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss.hub.score_centering_loss import KL_AGE_BUCKETS, score_centering_objective


def _rows(kl_coef: float, ages: torch.Tensor | None = None) -> ScoreCenteringInputs:
    # Row 0 matches the sampler; row 1 is a little off; row 2 is far off.
    log_q = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float64).log()
    log_p = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.95, 0.05]], dtype=torch.float64).log()
    return ScoreCenteringInputs(
        train_log_probs=log_p[:, 0],
        train_head_log_probs=log_p,
        rollout_log_probs=log_q[:, 0],
        rollout_head_log_probs=log_q,
        head_mask=torch.ones(3, 2, dtype=torch.bool),
        advantages=torch.tensor([1.0, -0.5, 0.0], dtype=torch.float64),
        kl_coef=kl_coef,
        kl_token_ages=ages,
    )


def test_far_positions_are_counted_against_both_thresholds() -> None:
    _, metrics = score_centering_loss(_rows(0.1))
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    expected = [
        (q * (q.log() - torch.tensor(p, dtype=torch.float64).log())).sum() for p in ([0.5, 0.5], [0.6, 0.4], [0.95, 0.05])
    ]
    torch.testing.assert_close(metrics.forward_kl, torch.stack(expected), rtol=1e-12, atol=1e-12)
    # KL(q || p): 0, 0.020 and 0.830.
    assert metrics.forward_kl_above_0p1.tolist() == [0.0, 0.0, 1.0]
    assert metrics.forward_kl_above_0p3.tolist() == [0.0, 0.0, 1.0]


def test_pull_is_the_kl_force_and_push_the_policy_gradient_force() -> None:
    ages = torch.tensor([1, 4, 9])
    _, metrics = score_centering_loss(_rows(0.2, ages))
    p = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.95, 0.05]], dtype=torch.float64)
    coefficients = 0.2 / ages.to(torch.float64).sqrt()
    torch.testing.assert_close(metrics.kl_pull, coefficients * (p - 0.5).abs().sum(-1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        metrics.pg_push, 2 * torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64) * (1 - p[:, 0]), rtol=1e-12, atol=1e-12
    )


def test_no_kl_term_means_no_pull() -> None:
    _, metrics = score_centering_loss(_rows(0.0))
    assert (metrics.kl_pull == 0).all()


def test_forward_kl_by_age_reports_each_bucket_sum_and_tokens() -> None:
    log_q = torch.tensor([[0.5, 0.5]] * 4, dtype=torch.float64).log()
    log_p = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]], dtype=torch.float64).log()
    selected = torch.cat((log_p[:, :1], log_p), dim=-1)
    ages = torch.tensor([1, 4, 4, 15])
    args = _args(score_centering_kl_coef=0.0)
    _, log = score_centering_objective(
        args,
        scored={"selected": selected},
        active=torch.tensor([True, True, True, True]),
        rollout_log_probs=log_q[:, 0],
        advantages=torch.tensor([0.3, -0.3, 0.2, -0.2], dtype=torch.float64),
        candidate_ids=torch.tensor([[0, 1]] * 4),
        rollout_head_log_probs=log_q,
        ref_log_probs=None,
        reduce=lambda value: value.sum(),
        token_ages=ages,
    )
    forward = (log_q.exp() * (log_q - log_p)).sum(-1)
    expected = {"0_2": [0], "3_5": [1, 2], "6_8": [], "9_11": [], "12_up": [3]}
    assert {name for name, _, _ in KL_AGE_BUCKETS} == set(expected)
    for name, rows in expected.items():
        torch.testing.assert_close(log[f"sc_forward_kl_age_{name}_sum"], forward[rows].sum(), rtol=1e-12, atol=1e-12)
        assert log[f"sc_forward_kl_age_{name}_tokens"].item() == len(rows)
