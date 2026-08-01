from argparse import Namespace

import torch

from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights


def test_mismatch_metrics_do_not_require_tis_config_when_tis_is_disabled():
    train_log_probs = [torch.tensor([-0.1, -0.2])]
    rollout_log_probs = [torch.tensor([-0.2, -0.4])]
    loss_masks = [torch.ones(2)]

    weights, modified_masks, metrics = compute_mis_weights(
        Namespace(use_tis=False),
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    assert weights is None
    assert modified_masks is loss_masks
    assert "log_ppl_abs_diff" in metrics
