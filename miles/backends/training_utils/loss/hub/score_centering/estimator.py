"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import math
from dataclasses import dataclass

import torch

from miles.backends.training_utils.loss_hub.score_centering.masks import (
    drop_inactive_nan,
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)


@dataclass(frozen=True)
class NoWeighting:
    """Vanilla score centering: f(r) = 1."""

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return q

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return rho

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(log_ratio)


@dataclass(frozen=True)
class TruncatedWeighting:
    """Truncated importance sampling (TIS): f(r) = min(r, clip)."""

    clip: float

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # q * min(p/q, c), including q=0, without 0 * inf.
        return torch.minimum(p, self.clip * q)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (self.clip * rho).clamp_max(1)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return log_ratio.clamp(max=math.log(self.clip)).exp()


@dataclass(frozen=True)
class MaskedWeighting:
    """Masked importance sampling (MIS): f(r) = r for low <= r <= high, else 0."""

    low: float
    high: float

    def _inside(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """Whether low <= r <= high, tested on log(r)."""
        return (log_ratio >= math.log(self.low)) & (log_ratio <= math.log(self.high))

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.where(self._inside(log_ratio), p, 0.0)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return ((rho >= 1 / self.high) & (rho <= 1 / self.low)).to(dtype)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return torch.where(self._inside(log_ratio), log_ratio.clamp(max=math.log(self.high)).exp(), 0.0)


def importance_weighting(
    mode: str, *, tis_clip: float, mis_low: float, mis_high: float
) -> NoWeighting | TruncatedWeighting | MaskedWeighting:
    """Validate the parameters of ``mode`` and return its weighting."""
    if mode == "tis" and (not math.isfinite(tis_clip) or tis_clip <= 0):
        raise ValueError("Score-centering TIS clip must be positive and finite")
    if mode == "mis" and (not math.isfinite(mis_low) or not math.isfinite(mis_high) or not 0 < mis_low <= mis_high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    if mode == "none":
        return NoWeighting()
    if mode == "tis":
        return TruncatedWeighting(clip=tis_clip)
    if mode == "mis":
        return MaskedWeighting(low=mis_low, high=mis_high)
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
    weighting = importance_weighting(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    return weighting.sample_weight(log_ratio)


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
    weighting = importance_weighting(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    active = advantages.detach() != 0
    train_head = sanitize_head_log_probs(train_head_log_probs, head_mask, active)
    rollout_head = sanitize_head_log_probs(rollout_head_log_probs, head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, head_mask), head_probs(rollout_head, head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = weighting.tail_scale(rho, p.dtype)
        residual = weighting.head_mass(p, q, train_head - rollout_head) - alpha.unsqueeze(-1) * p
        weight = weighting.sample_weight(drop_inactive_nan(train_log_probs - rollout_log_probs, active, 0.0))
    correction = (residual * scored_log_probs(train_head, residual, active)).sum(-1)
    loss = -advantages.detach() * (weight * scored_log_probs(train_log_probs, weight, active) - correction)
    return loss, {
        "sc_correction": correction.detach(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        "sc_importance_weight": weight,
    }
