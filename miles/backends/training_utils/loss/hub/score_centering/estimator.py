"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import math

import torch


def _token_mask(active: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-token mask [tokens] over values [tokens, ...]."""
    return active.reshape(active.shape + (1,) * (values.ndim - active.ndim))


def drop_inactive_nan(values: torch.Tensor, active: torch.Tensor, fill: float) -> torch.Tensor:
    """Replace NaN on inactive tokens with ``fill``; NaN on active tokens is kept for the caller to reject."""
    return torch.where(~_token_mask(active, values) & torch.isnan(values), fill, values)


def sanitize_head_log_probs(log_probs: torch.Tensor, head_mask: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Head log-probabilities with inactive NaN mapped to ``-inf`` and padding to the placeholder 0."""
    if (_token_mask(active, log_probs) & head_mask & torch.isnan(log_probs)).any():
        raise ValueError("Score centering has a NaN candidate log-probability on an active token")
    return torch.where(head_mask, drop_inactive_nan(log_probs, active, -torch.inf), 0.0)


def head_probs(log_probs: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
    """Probabilities of the head candidates, zero at padding."""
    return log_probs.exp().masked_fill(~head_mask, 0.0)


def scored_log_probs(log_probs: torch.Tensor, coefficient: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Log-probabilities that enter the loss, zeroed wherever the coefficient is zero.

    Masking before the multiply keeps ``0 * -inf`` from turning the loss into NaN.
    """
    finite = torch.isfinite(log_probs)
    scored = coefficient != 0
    if (_token_mask(active, log_probs) & scored & ~finite).any():
        raise ValueError("Score centering has a non-finite log-probability with nonzero gradient weight")
    return torch.where(scored & finite, log_probs, 0.0)


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
    inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
    return torch.where(inside, log_ratio.clamp(max=math.log(mis_high)).exp(), 0.0)


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
    train_head = sanitize_head_log_probs(train_head_log_probs, head_mask, active)
    rollout_head = sanitize_head_log_probs(rollout_head_log_probs, head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, head_mask), head_probs(rollout_head, head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        if mode == "none":
            weighted_q, alpha = q, rho
        elif mode == "tis":
            # q * min(p/q, c), including q=0, without 0 * inf.
            weighted_q, alpha = torch.minimum(p, tis_clip * q), (tis_clip * rho).clamp_max(1)
        else:
            log_ratio = train_head - rollout_head
            inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
            weighted_q = torch.where(inside, p, 0.0)
            alpha = ((rho >= 1 / mis_high) & (rho <= 1 / mis_low)).to(p.dtype)
        residual = weighted_q - alpha.unsqueeze(-1) * p
        weight = importance_weights(
            drop_inactive_nan(train_log_probs - rollout_log_probs, active, 0.0),
            mode,
            tis_clip=tis_clip,
            mis_low=mis_low,
            mis_high=mis_high,
        )
    correction = (residual * scored_log_probs(train_head, residual, active)).sum(-1)
    loss = -advantages.detach() * (weight * scored_log_probs(train_log_probs, weight, active) - correction)
    return loss, {
        "sc_correction": correction.detach(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        "sc_importance_weight": weight,
    }
