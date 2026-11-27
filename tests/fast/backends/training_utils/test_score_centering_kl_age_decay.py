"""The score-centering KL coefficient scaled by each response token's age."""

import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _turn

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.metrics import log_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import response_token_ages

_SUPPORT_REPLAY = dict(
    rollout_top_p=0.6,
    rollout_top_k=3,
    rollout_sampling_logprobs_mode="support",
    use_sampling_support_replay=True,
)


def _inputs(ages: torch.Tensor | None, kl_direction: str = "forward") -> tuple[torch.Tensor, ScoreCenteringInputs]:
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    log_p = logits.log_softmax(-1)
    log_q = torch.randn(4, 3, generator=generator, dtype=torch.float64).log_softmax(-1)
    return logits, ScoreCenteringInputs(
        train_log_probs=log_p[:, 0],
        train_head_log_probs=log_p,
        rollout_log_probs=log_q[:, 0],
        rollout_head_log_probs=log_q,
        head_mask=torch.ones(4, 3, dtype=torch.bool),
        advantages=torch.tensor([0.5, -0.2, 0.0, 1.1], dtype=torch.float64),
        kl_coef=0.3,
        kl_direction=kl_direction,
        kl_token_ages=ages,
    )


@pytest.mark.parametrize("kl_direction", ["forward", "reverse"])
def test_token_ages_divide_the_kl_coefficient_by_the_square_root_of_age(kl_direction: str) -> None:
    ages = torch.tensor([4, 9, 1, 16])
    _, flat = _inputs(None, kl_direction)
    loss_flat, metrics = score_centering_loss(flat)
    logits, aged = _inputs(ages, kl_direction)
    loss_aged, _ = score_centering_loss(aged)

    kl = metrics.forward_kl if kl_direction == "forward" else metrics.kl
    scale = torch.tensor([1 / 2, 1 / 3, 1.0, 1 / 4], dtype=torch.float64)
    # The KL term is the only part that changes: flat - aged = 0.3 * kl * (1 - scale).
    torch.testing.assert_close(loss_flat - loss_aged, 0.3 * kl * (1 - scale), rtol=1e-12, atol=1e-12)


def test_ages_below_one_keep_the_full_coefficient() -> None:
    _, flat = _inputs(None)
    _, aged = _inputs(torch.tensor([0, -1, 1, 0]))
    torch.testing.assert_close(score_centering_loss(aged)[0], score_centering_loss(flat)[0], rtol=0, atol=0)


def test_token_ages_must_be_per_token() -> None:
    with pytest.raises(ValueError, match="token ages must be"):
        _inputs(torch.tensor([1, 2]))


def _versioned_sample() -> Sample:
    """Prompt [10, 11], a first call generating [20, 21] at version 5, a tool reply [30] with no version, and a
    second call generating [40, 41] at version 7."""
    sample = Sample(tokens=[10, 11, 20, 21, 30, 40, 41], response_length=5, loss_mask=[1, 1, 0, 1, 1])
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="5", abs_start=2, abs_end=4)]),
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="7", abs_start=5, abs_end=7)]),
    ]
    return sample


def test_response_token_ages_follow_the_version_spans() -> None:
    assert response_token_ages(_versioned_sample(), trainer_version=9) == [4, 4, -1, 2, 2]


def test_non_numeric_versions_have_no_age() -> None:
    sample = _versioned_sample()
    sample.weight_versions[1].spans[0] = WeightVersionSpan(version="default", abs_start=5, abs_end=7)
    assert response_token_ages(sample, trainer_version=9) == [4, 4, -1, -1, -1]


def test_age_decay_needs_a_positive_kl_coefficient() -> None:
    with pytest.raises(ValueError, match="age-decay"):
        validate_score_centering_args(_args(score_centering_kl_age_decay="sqrt", **_SUPPORT_REPLAY))
    validate_score_centering_args(
        _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    )


def test_conversion_carries_token_ages_from_the_trainer_version() -> None:
    sample = _turn([5, 6], [3, 2], [0.4, 0.6])
    sample.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=2, abs_end=4)])]
    args = _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    data = convert_samples_to_train_data(args, [sample], {"trainer_weight_version": 8}, None, None)
    assert [list(ages) for ages in data["rollout_token_ages"]] == [[5, 5]]
    with pytest.raises(AssertionError, match="trainer's weight version"):
        convert_samples_to_train_data(args, [sample], {}, None, None)


def test_conversion_sends_ages_only_with_age_decay() -> None:
    sample = _turn([5, 6], [3, 2], [0.4, 0.6])
    sample.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=2, abs_end=4)])]
    data = convert_samples_to_train_data(_args(**_SUPPORT_REPLAY), [sample], {"trainer_weight_version": 4}, None, None)
    assert "rollout_token_ages" not in data
    assert math.isfinite(sum(data["rollout_log_probs"][0]))


def test_age_decay_refuses_a_trained_token_without_a_version() -> None:
    sample = _turn([5, 6], [3, 2], [0.4, 0.6])
    sample.weight_versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=2, abs_end=3)])]
    args = _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    with pytest.raises(AssertionError, match="1 trained tokens carry no numeric weight version"):
        convert_samples_to_train_data(args, [sample], {"trainer_weight_version": 8}, None, None)
    sample.loss_mask = [1, 0]
    data = convert_samples_to_train_data(args, [sample], {"trainer_weight_version": 8}, None, None)
    assert [list(ages) for ages in data["rollout_token_ages"]] == [[5, -1]]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_steps_per_rollout=2),
        dict(update_weights_interval=2),
        dict(global_batch_size=2),
    ],
)
def test_age_decay_refuses_batches_trained_after_another_step(overrides: dict) -> None:
    args = _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY, **overrides)
    with pytest.raises(ValueError, match="one optimizer step per rollout"):
        validate_score_centering_args(args)


def test_age_decay_takes_a_dynamic_global_batch_as_one_step() -> None:
    args = _args(
        score_centering_kl_age_decay="sqrt",
        score_centering_kl_coef=0.1,
        use_dynamic_global_batch_size=True,
        global_batch_size=2,
        **_SUPPORT_REPLAY,
    )
    validate_score_centering_args(args)


def test_age_decay_runs_only_on_megatron() -> None:
    args = _args(score_centering_kl_age_decay="sqrt", score_centering_kl_coef=0.1, **_SUPPORT_REPLAY)
    args.train_backend = "fsdp"
    with pytest.raises(ValueError, match="Megatron"):
        validate_score_centering_args(args)


def test_the_kl_direction_defaults_to_reverse() -> None:
    from miles.utils.args.configs.algo import AlgoConfig

    assert AlgoConfig.model_fields["score_centering_kl_direction"].default == "reverse"


@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "cp_rank", "expected_indices"),
    [
        ("thd", 1, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("thd", 2, 0, [6]),
        ("thd", 2, 1, [0, 1, 2, 3, 4, 5]),
        ("bshd", 2, 0, [0]),
        ("bshd", 2, 1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_the_trainer_slices_token_ages_with_the_rollout_log_probs(
    monkeypatch: pytest.MonkeyPatch, qkv_format: str, cp_size: int, cp_rank: int, expected_indices: list[int]
) -> None:
    log_probs = torch.tensor([-0.2, -1.3, -0.7, -2.1, -0.4, -3.2, -1.8])
    ages = [3, 3, -1, 2, 2, 0, 0]
    rollout_data = {
        "tokens": [list(range(11))],
        "loss_masks": [[1] * 7],
        "total_lengths": [11],
        "response_lengths": [7],
        "rollout_log_probs": [log_probs.tolist()],
        "rollout_token_ages": [ages],
    }
    group = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=cp_rank, size=cp_size, group=None)
    parallel_state = ParallelState(
        intra_dp=group, intra_dp_cp=cp, cp=cp, tp=group, pp=group, ep=group, etp=group, indep_dp=group
    )
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    args = Namespace(
        enable_witness=False,
        multimodal_token_space_processing=False,
        qkv_format=qkv_format,
        data_pad_size_multiplier=16,
        true_on_policy_mode=False,
        backend=Namespace(compress_ratios=[], bf16=False, fp16=False),
    )

    loaded, _ = data_utils.get_rollout_data(
        args=args,
        rollout_data_ref=object(),
        train_parallel_config=parallel_state.train_parallel_config(supports_precomputed_schedule=True),
    )

    [sliced] = loaded["rollout_token_ages"]
    assert sliced.dtype == torch.int32
    assert sliced.tolist() == [ages[i] for i in expected_indices]
    assert len(sliced) == len(loaded["rollout_log_probs"][0])


def test_rollout_metrics_leave_the_token_ages_out(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        effective_dp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils, "gather_log_data", lambda name, args, rollout_id, log_dict: captured.update(log_dict)
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-1.0, -3.0])],
        "rollout_token_ages": [torch.tensor([2, 2], dtype=torch.int32)],
    }
    args = Namespace(
        ci_test=False, qkv_format="thd", log_multi_turn=False, log_passrate=False, log_correct_samples=False
    )

    log_utils.log_rollout_data(0, args, rollout_data)

    assert "rollout_token_ages" not in captured
    assert captured["log_probs"][0] == pytest.approx(-2.0)
