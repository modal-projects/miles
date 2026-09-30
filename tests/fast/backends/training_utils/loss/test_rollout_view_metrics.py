"""Per-rollout-view mismatch diagnostics must partition the global ones without touching the loss."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.rollout_views import (
    DENOMINATOR_SUFFIX,
    NUMERATOR_SUFFIX,
    UNKNOWN_VIEW,
    resolve_ratio_metrics,
    rollout_view_index,
)

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

VIEWS = {"bf16": "/checkpoints/bf16", "fp8": "/checkpoints/fp8", "nvfp4": "/checkpoints/nvfp4"}
METRICS = ("train_rollout_logprob_abs_diff", "train_rollout_kl")


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("rollout-view-metrics") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _run(args, inputs, view_ids):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    batch = make_batch(inputs, "policy_loss")
    if view_ids is not None:
        batch["rollout_view_ids"] = view_ids
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _inputs(args):
    inputs = make_inputs(
        seed=7,
        batch_size=4,
        prompt_lens=[12, 30, 8, 20],
        response_lens=[10, 24, 6, 16],
        vocab_size=64,
        args=args,
    )
    # A partial loss mask keeps per-sample means and token sums distinct.
    inputs["loss_masks"][1][::3] = 0
    return inputs


@pytest.mark.parametrize("per_token", [False, True])
def test_views_partition_the_global_mismatch_and_leave_training_unchanged(process_group, per_token):
    args = make_args(
        calculate_per_token_loss=per_token,
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        update_weight_views=VIEWS,
    )
    inputs = _inputs(args)

    plain_loss, plain_metrics, plain_grad = _run(args, inputs, view_ids=None)
    loss, metrics, grad = _run(args, inputs, view_ids=[0, 1, 1, 2])

    # The tag is diagnostic only: loss, gradient and every global metric are unchanged.
    assert torch.equal(loss, plain_loss)
    assert torch.equal(grad, plain_grad)
    for key, value in plain_metrics.items():
        assert torch.equal(metrics[key], value), key

    masks = inputs["loss_masks"]
    weights = [mask.sum() if per_token else torch.tensor(1.0) for mask in masks]
    for metric in METRICS:
        numerators = [metrics[f"{metric}/{view}{NUMERATOR_SUFFIX}"] for view in VIEWS]
        denominators = [metrics[f"{metric}/{view}{DENOMINATOR_SUFFIX}"] for view in VIEWS]
        torch.testing.assert_close(sum(numerators), metrics[metric])
        torch.testing.assert_close(sum(denominators), sum(weights))
        # fp8 holds samples 1 and 2.
        torch.testing.assert_close(denominators[1], weights[1] + weights[2])


def test_one_view_batch_reports_the_global_mean(process_group):
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        update_weight_views=VIEWS,
    )
    inputs = _inputs(args)

    _, metrics, _ = _run(args, inputs, view_ids=[1, 1, 1, 1])
    resolved = resolve_ratio_metrics({key: value.item() for key, value in metrics.items()})

    for metric in METRICS:
        assert resolved[f"{metric}/fp8"] == pytest.approx(metrics[metric].item() / 4)
        # Absent views carry no weight and report nothing rather than a zero mean.
        assert f"{metric}/bf16" not in resolved
        assert f"{metric}/nvfp4" not in resolved
    assert not any(key.endswith((NUMERATOR_SUFFIX, DENOMINATOR_SUFFIX)) for key in resolved)


def test_every_view_is_reported_in_every_micro_batch(process_group):
    """Ranks all-reduce one fixed key list, so a micro-batch without a view still carries its keys."""
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        update_weight_views=VIEWS,
    )
    inputs = _inputs(args)

    _, first, _ = _run(args, inputs, view_ids=[0, 0, 0, 0])
    _, second, _ = _run(args, inputs, view_ids=[2, UNKNOWN_VIEW, 2, 2])

    assert list(first) == list(second)
    assert first[f"train_rollout_kl/nvfp4{DENOMINATOR_SUFFIX}"].item() == 0.0


def test_runs_without_views_add_no_keys(process_group):
    args = make_args(entropy_coef=0.0, kl_coef=0.0, observe_training_entropy=False, true_on_policy_mode=False)
    inputs = _inputs(args)

    _, metrics, _ = _run(args, inputs, view_ids=[0, 1, 1, 2])

    assert not any(key.endswith((NUMERATOR_SUFFIX, DENOMINATOR_SUFFIX)) for key in metrics)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ServerH100FP8:fp8", 1),
        ("ServerB300NVFP4W4A16:nvfp4", 2),
        ("Server:with:colons:bf16", 0),
        ("ServerH100FP8", UNKNOWN_VIEW),
        ("ServerX:int4", UNKNOWN_VIEW),
        (None, UNKNOWN_VIEW),
        ("", UNKNOWN_VIEW),
    ],
)
def test_rollout_source_maps_to_its_view(source, expected):
    assert rollout_view_index(tuple(VIEWS), source) == expected


def test_ratio_resolution_keeps_plain_metrics_and_drops_empty_views():
    resolved = resolve_ratio_metrics(
        {
            "loss": 0.5,
            f"kl/fp8{NUMERATOR_SUFFIX}": 0.3,
            f"kl/fp8{DENOMINATOR_SUFFIX}": 0.6,
            f"kl/bf16{NUMERATOR_SUFFIX}": 0.0,
            f"kl/bf16{DENOMINATOR_SUFFIX}": 0.0,
        }
    )

    assert resolved == {"loss": 0.5, "kl/fp8": pytest.approx(0.5)}
