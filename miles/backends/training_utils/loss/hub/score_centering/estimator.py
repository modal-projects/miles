"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import math

import torch


def _validate_importance_args(mode: str, tis_clip: float, mis_low: float, mis_high: float) -> None:
    if mode == "tis" and (not math.isfinite(tis_clip) or tis_clip <= 0):
        raise ValueError("Score-centering TIS clip must be positive and finite")
    if mode == "mis" and (not math.isfinite(mis_low) or not math.isfinite(mis_high) or not 0 < mis_low <= mis_high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    if mode not in ("none", "tis", "mis"):
        raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> torch.Tensor:
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    _validate_importance_args(mode, tis_clip, mis_low, mis_high)
    if mode == "none":
        return torch.ones_like(log_ratio)
    if torch.isnan(log_ratio).any():
        raise ValueError("Score-centering importance log-ratio contains NaN")
    if mode == "tis":
        return log_ratio.clamp(max=math.log(tis_clip)).exp()
    if mode == "mis":
        inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
        return torch.where(inside, log_ratio.clamp(max=math.log(mis_high)).exp(), 0.0)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def score_centering_loss(
    train_log_probs: torch.Tensor,
    train_head_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    mode: str = "none",
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return unreduced token losses and detached token metrics.

    Inputs have shape [tokens], except head tensors/mask [tokens, candidates].
    Missing head slots have a false mask. Tail masses use the appendix's floor;
    cancellation is exact for a full distribution, approximate for a true tail
    that differs from the modeled, rescaled trainer tail.
    """
    if (
        train_log_probs.ndim != 1
        or rollout_log_probs.shape != train_log_probs.shape
        or advantages.shape != train_log_probs.shape
        or train_head_log_probs.ndim != 2
        or train_head_log_probs.shape[0] != train_log_probs.shape[0]
        or rollout_head_log_probs.shape != train_head_log_probs.shape
        or head_mask.shape != train_head_log_probs.shape
    ):
        raise ValueError("Score-centering sample tensors must be [tokens] and head tensors [tokens, candidates]")
    if not torch.isfinite(advantages).all():
        raise ValueError("Score-centering advantages must be finite")
    _validate_importance_args(mode, tis_clip, mis_low, mis_high)
    active = advantages.detach() != 0
    invalid_head = (
        active[:, None] & head_mask & (torch.isnan(train_head_log_probs) | torch.isnan(rollout_head_log_probs))
    )
    if invalid_head.any():
        raise ValueError("Score centering has a NaN candidate log-probability on an active token")
    # Inactive padding can contain NaN sentinels. Preserve finite values for
    # diagnostics while giving invalid candidates zero mass.
    inactive_nan_train = ~active[:, None] & torch.isnan(train_head_log_probs)
    inactive_nan_rollout = ~active[:, None] & torch.isnan(rollout_head_log_probs)
    safe_train_head = torch.where(inactive_nan_train, -torch.inf, train_head_log_probs)
    safe_rollout_head = torch.where(inactive_nan_rollout, -torch.inf, rollout_head_log_probs)
    head_log_probs = torch.where(head_mask, safe_train_head, 0.0)
    with torch.no_grad():
        p = head_log_probs.exp().masked_fill(~head_mask, 0.0)
        q = safe_rollout_head.exp().masked_fill(~head_mask, 0.0)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        if mode == "none":
            weighted_q, alpha = q, rho
        elif mode == "tis":
            # q * min(p/q, c), including q=0, without 0 * inf.
            weighted_q, alpha = torch.minimum(p, tis_clip * q), (tis_clip * rho).clamp_max(1)
        elif mode == "mis":
            log_ratio = head_log_probs - safe_rollout_head
            inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
            weighted_q = torch.where(inside & head_mask, p, 0.0)
            alpha = ((rho >= 1 / mis_high) & (rho <= 1 / mis_low)).to(p.dtype)
        else:
            raise ValueError(f"Unknown score-centering importance weighting: {mode}")
        residual = weighted_q - alpha.unsqueeze(-1) * p
        sample_log_ratio = train_log_probs - rollout_log_probs
        weight = importance_weights(
            torch.where(~active & torch.isnan(sample_log_ratio), 0.0, sample_log_ratio),
            mode,
            tis_clip=tis_clip,
            mis_low=mis_low,
            mis_high=mis_high,
        )
    finite_head = torch.isfinite(head_log_probs)
    finite_sample = torch.isfinite(train_log_probs)
    invalid = (active[:, None] & (residual != 0) & ~finite_head).any(-1) | (active & (weight != 0) & ~finite_sample)
    if invalid.any():
        raise ValueError("Score centering has a non-finite log-probability with nonzero gradient weight")
    # A zero coefficient contributes no score, even when its log-probability is
    # -inf. Mask before multiplying so 0 * -inf cannot turn the loss into NaN.
    safe_head_log_probs = torch.where((residual != 0) & finite_head, head_log_probs, 0.0)
    safe_train_log_probs = torch.where((weight != 0) & finite_sample, train_log_probs, 0.0)
    correction = (residual * safe_head_log_probs).sum(-1)
    loss = -advantages.detach() * (weight * safe_train_log_probs - correction)
    return loss, {
        "sc_correction": correction.detach(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        "sc_importance_weight": weight,
    }
