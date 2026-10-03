"""Compare gradients on saved verifier logits with an independent dense oracle.

The trainer is deliberately perturbed from the saved target logits. This tests
off-policy arithmetic without claiming to reproduce a trained checkpoint.
The dense surrogate can differ by a term whose gradient is zero; compare its
gradient, not its scalar loss, to the efficient top-K estimator.
"""

import numpy as np
import torch

from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.utils.types import Sample


def _weight(ratio: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones_like(ratio)
    if mode == "tis":
        return ratio.clamp_max(2.0)
    return torch.where((ratio >= 0.5) & (ratio <= 5.0), ratio, 0.0)


def compare_gradient(logits: np.ndarray, sample: Sample, position: int, *, temperature: float, mode: str) -> dict:
    ids = torch.as_tensor(sample.rollout_topk_token_ids[position].astype(np.int64))
    valid = ids >= 0
    ids = ids[valid]
    q_head = torch.as_tensor(sample.rollout_topk_log_probs[position][valid.numpy()], dtype=torch.float64).exp()
    q_selected = torch.tensor(sample.rollout_log_probs[position], dtype=torch.float64).exp()
    selected = sample.tokens[len(sample.tokens) - sample.response_length + position]
    raw = torch.as_tensor(logits, dtype=torch.float64) / temperature
    # Deterministic off-policy perturbation; gradients must not pass through q.
    raw = raw + 0.4 * torch.cos(torch.arange(raw.numel(), dtype=torch.float64) * 0.13)
    if sample.rollout_sampling_mask is not None:
        raw = raw[ids]
        selected = int(torch.nonzero(ids == selected).item())
        ids = torch.arange(len(ids))
    train_logits = raw.clone().requires_grad_()
    logp = train_logits.log_softmax(-1)
    advantage = torch.tensor([1.3 if position % 2 else -0.7], dtype=torch.float64)
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp[selected].reshape(1),
            train_head_log_probs=logp[ids].reshape(1, -1),
            rollout_log_probs=q_selected.log().reshape(1),
            rollout_head_log_probs=q_head.log().reshape(1, -1),
            head_mask=torch.ones(1, len(ids), dtype=torch.bool),
            advantages=advantage,
            mode=mode,
        )
    )
    actual = torch.autograd.grad(loss.sum(), train_logits)[0]
    reference_logits = raw.clone().requires_grad_()
    reference_logp = reference_logits.log_softmax(-1)
    p = reference_logp.detach().exp()
    rho = (1 - q_head.sum()).clamp_min(1e-6) / (1 - p[ids].sum()).clamp_min(1e-6)
    reconstructed_q = p * rho
    reconstructed_q[ids] = q_head
    coefficients = reconstructed_q * _weight(p / reconstructed_q, mode)
    sampled_weight = _weight(p[selected] / q_selected, mode)
    reference = -advantage * (sampled_weight * reference_logp[selected] - (coefficients * reference_logp).sum())
    expected = torch.autograd.grad(reference.sum(), reference_logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    return {
        "mode": mode,
        "loss": float(loss.detach().sum()),
        "max_gradient_error": float((actual - expected).abs().max()),
        "relative_gradient_l2_error": float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
        "correction": float(metrics.correction.item()),
    }
