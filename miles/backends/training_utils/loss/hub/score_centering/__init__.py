"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

from miles.backends.training_utils.loss_hub.score_centering.estimator import importance_weights, score_centering_loss
from miles.backends.training_utils.loss_hub.score_centering.selected_log_probs import (
    selected_log_probs,
    selected_log_probs_and_entropy,
)

__all__ = ["importance_weights", "score_centering_loss", "selected_log_probs", "selected_log_probs_and_entropy"]
