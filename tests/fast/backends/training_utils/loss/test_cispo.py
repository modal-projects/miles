"""Correctness tests for Clipped IS-weight Policy Optimization (CISPO)."""

from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.advantages import compute_advantages
from miles.backends.training_utils.loss_hub.math_utils import compute_cispo_loss


def test_cispo_clips_detached_is_weight_without_dropping_token_gradients():
    ratios = torch.tensor([2.0, 0.5, 1.0], requires_grad=True)
    ppo_kl = -ratios.log()
    advantages = torch.tensor([2.0, -1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)

    losses, clipfrac = compute_cispo_loss(
        ppo_kl,
        advantages,
        log_probs,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )

    clipped = torch.tensor([1.2, 0.8, 1.0])
    torch.testing.assert_close(losses, -clipped * advantages * log_probs.detach())
    torch.testing.assert_close(clipfrac, torch.tensor([1.0, 1.0, 0.0]))

    losses.sum().backward()
    assert ratios.grad is None
    torch.testing.assert_close(log_probs.grad, -clipped * advantages)
    assert torch.count_nonzero(log_probs.grad) == log_probs.numel()


def test_cispo_eps_clip_one_disables_the_lower_bound():
    ratios = torch.tensor([10.0, 0.01])
    losses, clipfrac = compute_cispo_loss(
        -ratios.log(),
        torch.ones(2),
        torch.ones(2),
        eps_clip=1.0,
        eps_clip_high=4.0,
    )

    torch.testing.assert_close(losses, torch.tensor([-5.0, -0.01]))
    torch.testing.assert_close(clipfrac, torch.tensor([1.0, 0.0]))


def test_cispo_reuses_group_relative_returns():
    returns, expected = compute_advantages(
        args=Namespace(advantage_estimator="cispo"),
        kl=[torch.zeros(2), torch.zeros(3)],
        rewards=[-1.0, 1.0],
        log_probs=[torch.zeros(2), torch.zeros(3)],
        loss_masks=[torch.ones(2), torch.ones(3)],
        total_lengths=[2, 3],
        response_lengths=[2, 3],
    )

    torch.testing.assert_close(returns[0], torch.full((2,), -1.0))
    torch.testing.assert_close(returns[1], torch.full((3,), 1.0))
    assert all(actual is target for actual, target in zip(returns, expected, strict=True))


def _policy_args() -> Namespace:
    return Namespace(
        advantage_estimator="cispo",
        use_rollout_logprobs=True,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
    )


def test_cispo_policy_loss_uses_actual_rollout_policy(monkeypatch):
    current = torch.tensor([-0.1, -0.4], requires_grad=True)
    rollout = torch.tensor([-0.5, -0.2])
    stale_trainer_recompute = torch.tensor([-4.0, -4.0])
    advantages = torch.tensor([2.0, -1.0])
    batch = {
        "advantages": [advantages],
        "log_probs": [stale_trainer_recompute],
        "rollout_log_probs": [rollout],
        "unconcat_tokens": [torch.tensor([7, 8, 9])],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.ones(2)],
    }

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format, max_seq_lens: loss_masks,
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [current]},
    )

    loss, metrics = loss_utils.policy_loss_function(
        _policy_args(),
        batch,
        logits=torch.zeros((1, 3, 16)),
        sum_of_sample_mean=lambda values: values.mean(),
    )

    ratio = (current.detach() - rollout).exp()
    clipped = ratio.clamp(0.8, 1.2)
    expected = (-clipped * advantages * current.detach()).mean()
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(0.5))

    loss.backward()
    torch.testing.assert_close(current.grad, -clipped * advantages / advantages.numel())


def test_cispo_masks_nonfinite_log_probs_before_reduction(monkeypatch):
    current = torch.tensor([-0.1, float("-inf")], requires_grad=True)
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": [torch.tensor([-0.1, -0.2])],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2])],
        "unconcat_tokens": [torch.tensor([7, 8, 9])],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 0.0])],
    }

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format, max_seq_lens: loss_masks,
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [current]},
    )

    loss, _ = loss_utils.policy_loss_function(
        _policy_args(),
        batch,
        logits=torch.zeros((1, 3, 16)),
        sum_of_sample_mean=lambda values: (values * batch["loss_masks"][0]).sum(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(current.grad).all()
