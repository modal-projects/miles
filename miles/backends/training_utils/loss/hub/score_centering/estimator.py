"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import torch

from miles.backends.training_utils.loss_hub.score_centering.importance_sampling import importance_sampling
from miles.backends.training_utils.loss_hub.score_centering.masks import (
    drop_inactive_nan,
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)


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
    sampling = importance_sampling(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    active = advantages.detach() != 0
    train_head = sanitize_head_log_probs(train_head_log_probs, head_mask, active)
    rollout_head = sanitize_head_log_probs(rollout_head_log_probs, head_mask, active)
    with torch.no_grad():
        p, q = head_probs(train_head, head_mask), head_probs(rollout_head, head_mask)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = sampling.tail_scale(rho, p.dtype)
        residual = sampling.head_mass(p, q, train_head - rollout_head) - alpha.unsqueeze(-1) * p
        weight = sampling.sample_weight(drop_inactive_nan(train_log_probs - rollout_log_probs, active, 0.0))
    correction = (residual * scored_log_probs(train_head, residual, active)).sum(-1)
    loss = -advantages.detach() * (weight * scored_log_probs(train_log_probs, weight, active) - correction)
    return loss, {
        "sc_correction": correction.detach(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        "sc_importance_weight": weight,
    }
