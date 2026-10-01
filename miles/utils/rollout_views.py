"""Train-vs-rollout mismatch split by the rollout that produced each sample.

A heterogeneous fleet serves one policy as several serialized views
(``--update-weight-views``), and each sample records the ``<pool>:<view>`` source
that generated it and the weight versions it was generated under. The mismatch is
reported for all tokens and per view, per rollout source (``--rollout-sources``) and
per staleness bucket. Every value is a mean over loss tokens, whatever the loss
aggregation, so runs that aggregate their loss differently stay comparable.

Each value is a numerator/denominator pair: ``aggregate_train_losses`` divides both by
the same step normalizer and reports their ratio, so a group's value is its own mean
however the batch splits and under any context-parallel layout. All values are
detached; they never enter the loss.
"""

import math
from argparse import Namespace
from collections.abc import Mapping, Sequence

import torch

NUMERATOR_SUFFIX = "/__numerator"
DENOMINATOR_SUFFIX = "/__denominator"
UNKNOWN_VIEW = -1
UNKNOWN_LAG = -1
# Version lag = the version a batch trains on minus a sample's oldest generation version.
LAG_BUCKETS = ("lag_0_1", "lag_2_3", "lag_4_5", "lag_6_plus")
# Tokens whose trainer/sampler ratio leaves [1/5, 5]: those a [0.2, 5] masked
# importance sampler (IcePop, MIS) drops.
TAIL_LOG_RATIO = math.log(5.0)
TOKEN_METRICS = ("train_rollout_kl", "train_rollout_ratio_tail_frac")


def rollout_view_names(args: Namespace) -> tuple[str, ...]:
    return tuple(getattr(args, "update_weight_views", None) or ())


def rollout_view_index(view_names: Sequence[str], rollout_source: str | None) -> int:
    """Index of a ``<pool>:<view>`` source's view in ``view_names``; UNKNOWN_VIEW if it has none."""
    if not rollout_source or ":" not in rollout_source:
        return UNKNOWN_VIEW
    view = rollout_source.rsplit(":", 1)[1]
    return view_names.index(view) if view in view_names else UNKNOWN_VIEW


def rollout_source_names(args: Namespace) -> tuple[str, ...]:
    return tuple(getattr(args, "rollout_sources", None) or ())


def rollout_source_index(source_names: Sequence[str], rollout_source: str | None) -> int:
    """Index of a sample's rollout source in ``source_names``; UNKNOWN_VIEW if it is not one."""
    return source_names.index(rollout_source) if rollout_source in source_names else UNKNOWN_VIEW


def rollout_lag_bucket(lag: int | None) -> int:
    """Index into LAG_BUCKETS for a sample's version lag; UNKNOWN_LAG without a version."""
    if lag is None:
        return UNKNOWN_LAG
    return min(max(lag, 0) // 2, len(LAG_BUCKETS) - 1)


def _groups(args: Namespace, batch: Mapping) -> dict[str, list[bool]]:
    """Per-sample membership of every view, source and staleness group the batch is tagged with."""
    groups = {}
    for key, names, prefix in (
        ("rollout_view_ids", rollout_view_names(args), "/"),
        ("rollout_source_ids", rollout_source_names(args), "/by_source/"),
        ("rollout_lag_buckets", LAG_BUCKETS, "/"),
    ):
        ids = batch.get(key)
        if ids is not None and names:
            groups.update({f"{prefix}{name}": [i == index for i in ids] for index, name in enumerate(names)})
    return groups


def _to_device(values: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Copy host values to ``device`` without waiting for the queued GPU work."""
    host = torch.tensor(values, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else host


def add_train_rollout_diagnostics(
    args: Namespace,
    batch: Mapping,
    log: dict[str, torch.Tensor],
    *,
    local_loss_masks: Sequence[torch.Tensor],
    abs_diff: torch.Tensor,
    kl: torch.Tensor,
) -> None:
    """Add the mismatch for all tokens and every tagged group to ``log``.

    ``abs_diff`` is |trainer - rollout| log-probability and ``kl`` the losses' KL
    estimate on each token, both zero outside the loss mask. ``local_loss_masks`` are
    the per-sample local response masks whose concatenation is their layout. Nothing is
    added unless the batch carries view, source or staleness tags; then every group is
    reported in every micro-batch, zero when absent, so all ranks reduce the same keys.
    """
    groups = _groups(args, batch)
    if not groups:
        return
    with torch.no_grad():
        device = kl.device
        num_samples = len(local_loss_masks)
        mask = torch.cat(local_loss_masks).to(device=device, dtype=torch.float32)
        lengths = _to_device([m.numel() for m in local_loss_masks], torch.long, device)
        sample = torch.repeat_interleave(torch.arange(num_samples, device=device), lengths, output_size=mask.numel())
        # Rows: KL, tail indicator, loss tokens; summed per sample, then per group.
        per_token = torch.stack([kl.detach().float(), (abs_diff.detach() > TAIL_LOG_RATIO).float(), mask]) * mask
        per_sample = per_token.new_zeros(3, num_samples).index_add_(1, sample, per_token)
        membership = _to_device([[True] * num_samples, *groups.values()], torch.float32, device)
        totals = membership @ per_sample.T
        for suffix, (kl_sum, tail_sum, tokens) in zip(("/all", *groups), totals, strict=True):
            for key, value in zip(TOKEN_METRICS, (kl_sum, tail_sum), strict=True):
                log[f"{key}{suffix}{NUMERATOR_SUFFIX}"] = value
                log[f"{key}{suffix}{DENOMINATOR_SUFFIX}"] = tokens


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
