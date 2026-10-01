"""Train-vs-rollout mismatch split by view, rollout source and staleness.

The split must partition the all-token mismatch, weight loss tokens whatever the loss
aggregation, keep one key list across micro-batches, and leave the loss untouched.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.rollout_views import (
    DENOMINATOR_SUFFIX,
    LAG_BUCKETS,
    NUMERATOR_SUFFIX,
    UNKNOWN_LAG,
    UNKNOWN_VIEW,
    add_train_rollout_diagnostics,
    resolve_ratio_metrics,
    rollout_lag_bucket,
    rollout_source_index,
    rollout_view_index,
)

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

VIEWS = {"bf16": "/checkpoints/bf16", "fp8": "/checkpoints/fp8", "nvfp4": "/checkpoints/nvfp4"}
# Two pools serve fp8, so a view mixes hardware that a source keeps apart.
SOURCES = ("ServerA100BF16:bf16", "ServerH100FP8:fp8", "ServerH200FP8:fp8", "ServerB300NVFP4:nvfp4")
METRICS = ("train_rollout_kl", "train_rollout_ratio_tail_frac")
TAGS = {"rollout_view_ids": [0, 1, 1, 2], "rollout_source_ids": [0, 1, 2, 3], "rollout_lag_buckets": [0, 1, 3, 3]}


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("rollout-views") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _args(**overrides):
    return make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        update_weight_views=VIEWS,
        rollout_sources=SOURCES,
        **overrides,
    )


def _inputs(args):
    inputs = make_inputs(
        seed=7, batch_size=4, prompt_lens=[12, 30, 8, 20], response_lens=[10, 24, 6, 16], vocab_size=64, args=args
    )
    # A partial loss mask keeps token and sample weightings apart.
    inputs["loss_masks"][1][::3] = 0
    return inputs


def _run(args, inputs, tags):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    batch = make_batch(inputs, "policy_loss") | tags
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
        denominators=batch.get("loss_denominators"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _pairs(metrics):
    return {key: value for key, value in metrics.items() if key.endswith((NUMERATOR_SUFFIX, DENOMINATOR_SUFFIX))}


def test_split_matches_hand_computation():
    """Sample A (bf16, lag 1) has |log r| = [0, log 3, log 6]; sample B (fp8, lag 7) has [log 1.5, masked]."""
    make_parallel_state()
    log: dict[str, torch.Tensor] = {}
    add_train_rollout_diagnostics(
        make_args(update_weight_views=VIEWS),
        {"rollout_view_ids": [0, 1], "rollout_lag_buckets": [rollout_lag_bucket(1), rollout_lag_bucket(7)]},
        log,
        local_loss_masks=[torch.ones(3), torch.tensor([1.0, 0.0])],
        abs_diff=torch.tensor([0.0, math.log(3), math.log(6), math.log(1.5), math.log(7)]),
        kl=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    got = resolve_ratio_metrics({key: value.item() for key, value in log.items()})

    # Means over loss tokens; among them only log 6 leaves [log 1/5, log 5].
    assert got["train_rollout_kl/all"] == pytest.approx(6 / 4)
    assert got["train_rollout_ratio_tail_frac/all"] == pytest.approx(1 / 4)
    assert got["train_rollout_kl/bf16"] == pytest.approx(3 / 3)
    assert got["train_rollout_ratio_tail_frac/bf16"] == pytest.approx(1 / 3)
    assert got["train_rollout_kl/fp8"] == pytest.approx(3.0)
    assert got["train_rollout_ratio_tail_frac/fp8"] == pytest.approx(0.0)
    assert got["train_rollout_kl/lag_0_1"] == got["train_rollout_kl/bf16"]
    assert got["train_rollout_kl/lag_6_plus"] == got["train_rollout_kl/fp8"]
    # Empty groups report nothing rather than a zero mean.
    assert not any(key.endswith(("/nvfp4", "/lag_2_3", "/lag_4_5")) for key in got)


@pytest.mark.parametrize("per_token", [False, True])
def test_split_partitions_the_mismatch_and_leaves_training_unchanged(process_group, per_token):
    args = _args(calculate_per_token_loss=per_token)
    inputs = _inputs(args)

    plain_loss, plain_metrics, plain_grad = _run(args, inputs, {})
    loss, metrics, grad = _run(args, inputs, TAGS)

    # The tags are diagnostic only: loss, gradient and every existing metric are unchanged.
    assert torch.equal(loss, plain_loss)
    assert torch.equal(grad, plain_grad)
    for key, value in plain_metrics.items():
        assert torch.equal(metrics[key], value), key

    tokens = sum(mask.sum() for mask in inputs["loss_masks"]).float()
    for metric in METRICS:
        total = metrics[f"{metric}/all{NUMERATOR_SUFFIX}"]
        torch.testing.assert_close(metrics[f"{metric}/all{DENOMINATOR_SUFFIX}"], tokens)
        for names in (VIEWS, [f"by_source/{source}" for source in SOURCES], LAG_BUCKETS):
            torch.testing.assert_close(sum(metrics[f"{metric}/{name}{NUMERATOR_SUFFIX}"] for name in names), total)
        # The two fp8 pools make up the fp8 view.
        torch.testing.assert_close(
            metrics[f"{metric}/by_source/ServerH100FP8:fp8{NUMERATOR_SUFFIX}"]
            + metrics[f"{metric}/by_source/ServerH200FP8:fp8{NUMERATOR_SUFFIX}"],
            metrics[f"{metric}/fp8{NUMERATOR_SUFFIX}"],
        )
    if per_token:
        # A per-token loss reports the same KL token sum as its plain global metric.
        torch.testing.assert_close(metrics[f"train_rollout_kl/all{NUMERATOR_SUFFIX}"], metrics["train_rollout_kl"])


def test_split_does_not_depend_on_the_loss_aggregation(process_group):
    """Token-, sample- and prompt-mean losses report the same split, so their runs compare."""
    inputs = _inputs(_args())
    prompt_mean = {"loss_denominators": torch.tensor([20.0, 20.0, 12.0, 12.0])}
    runs = [
        _run(_args(calculate_per_token_loss=per_token), inputs, TAGS | extra)
        for per_token, extra in ((True, {}), (False, {}), (False, prompt_mean))
    ]

    losses = [loss for loss, _, _ in runs]
    assert not torch.equal(losses[0], losses[1]) and not torch.equal(losses[1], losses[2])
    token, *others = (_pairs(metrics) for _, metrics, _ in runs)
    for other in others:
        assert other.keys() == token.keys()
        for key, value in token.items():
            torch.testing.assert_close(other[key], value, msg=key)


def test_every_group_is_reported_in_every_micro_batch(process_group):
    """Ranks all-reduce one fixed key list, so a micro-batch missing a group still carries its keys."""
    args = _args()
    inputs = _inputs(args)

    _, first, _ = _run(args, inputs, {key: [0] * 4 for key in TAGS})
    _, second, _ = _run(
        args,
        inputs,
        {
            "rollout_view_ids": [2, UNKNOWN_VIEW, 2, 2],
            "rollout_source_ids": [3, UNKNOWN_VIEW, 3, 3],
            "rollout_lag_buckets": [3, UNKNOWN_LAG, 2, 2],
        },
    )

    assert list(_pairs(first)) == list(_pairs(second))
    assert first[f"train_rollout_kl/nvfp4{DENOMINATOR_SUFFIX}"].item() == 0.0
    assert first[f"train_rollout_kl/by_source/ServerH200FP8:fp8{DENOMINATOR_SUFFIX}"].item() == 0.0


def test_untagged_batches_add_no_split(process_group):
    args = _args()
    _, metrics, _ = _run(args, _inputs(args), {})

    assert not _pairs(metrics)


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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ServerH100FP8:fp8", 1),
        ("ServerH200FP8:fp8", 2),
        ("ServerB200NVFP4:nvfp4", UNKNOWN_VIEW),
        (None, UNKNOWN_VIEW),
    ],
)
def test_rollout_source_maps_to_its_own_index(source, expected):
    assert rollout_source_index(SOURCES, source) == expected


@pytest.mark.parametrize(
    ("lag", "bucket"),
    [(None, UNKNOWN_LAG), (-1, 0), (0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (50, 3)],
)
def test_version_lag_buckets(lag, bucket):
    assert rollout_lag_bucket(lag) == bucket


def test_ratio_resolution_keeps_plain_metrics_and_drops_empty_groups():
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
