"""Train diagnostics split by the rollout weight view (precision) that produced each sample.

A heterogeneous fleet serves one policy as several serialized views
(``--update-weight-views``), and each sample records the ``<pool>:<view>`` source
that generated it. Each per-view diagnostic is a numerator/denominator pair:
``aggregate_train_losses`` divides both by the same step normalizer and reports
their ratio, so a view's value is its own mean however the batch splits across views.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch

NUMERATOR_SUFFIX = "/__numerator"
DENOMINATOR_SUFFIX = "/__denominator"
UNKNOWN_VIEW = -1


def rollout_view_names(args: Namespace) -> tuple[str, ...]:
    return tuple(getattr(args, "update_weight_views", None) or ())


def rollout_view_index(view_names: Sequence[str], rollout_source: str | None) -> int:
    """Index of a ``<pool>:<view>`` source's view in ``view_names``; UNKNOWN_VIEW if it has none."""
    if not rollout_source or ":" not in rollout_source:
        return UNKNOWN_VIEW
    view = rollout_source.rsplit(":", 1)[1]
    return view_names.index(view) if view in view_names else UNKNOWN_VIEW


def add_rollout_view_metrics(
    args: Namespace,
    batch: Mapping,
    log: dict[str, torch.Tensor],
    local_loss_masks: Sequence[torch.Tensor],
    token_metrics: Mapping[str, torch.Tensor],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Add each token metric's per-view numerator and denominator to ``log``.

    ``local_loss_masks`` are the per-sample local response masks in batch order, whose
    concatenation is the layout of every tensor in ``token_metrics``. Every view is
    reported in every micro-batch, zero when absent, so all ranks reduce the same keys.
    """
    view_ids = batch.get("rollout_view_ids")
    names = rollout_view_names(args)
    if view_ids is None or not names:
        return
    device = next(iter(token_metrics.values())).device
    for index, name in enumerate(names):
        weight = torch.cat(
            [
                mask.to(device=device, dtype=torch.float32) * float(view == index)
                for mask, view in zip(local_loss_masks, view_ids, strict=True)
            ]
        )
        denominator = sum_of_sample_mean(weight).detach()
        for key, value in token_metrics.items():
            log[f"{key}/{name}{NUMERATOR_SUFFIX}"] = sum_of_sample_mean(value.detach() * weight).detach()
            log[f"{key}/{name}{DENOMINATOR_SUFFIX}"] = denominator


def resolve_ratio_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Replace each numerator/denominator pair with its ratio, dropping pairs with no weight."""
    resolved = {}
    for key, value in metrics.items():
        if key.endswith(DENOMINATOR_SUFFIX):
            continue
        if key.endswith(NUMERATOR_SUFFIX):
            base = key.removesuffix(NUMERATOR_SUFFIX)
            denominator = metrics[base + DENOMINATOR_SUFFIX]
            if denominator > 0:
                resolved[base] = value / denominator
            continue
        resolved[key] = value
    return resolved
