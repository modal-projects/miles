"""Score centering with DFlash requires exact sampled verification of draft tokens."""

from argparse import Namespace

import pytest
import torch

from miles.utils.score_centering import validate_score_centering_args, validate_score_centering_speculative_config


def _args(**overrides):
    return Namespace(
        **{
            "loss_type": "score_centering",
            "rollout_top_logprobs_num": 128,
            "score_centering_tis_clip": 2.0,
            "score_centering_mis_low": 0.5,
            "score_centering_mis_high": 5.0,
            "rollout_top_k": -1,
            "rollout_temperature": 0.7,
            "advantage_estimator": "grpo",
            "use_sampling_support_replay": False,
            "sglang_speculative_algorithm": "DFLASH",
            **overrides,
        }
    )


def _check(environ=None, filtered_sampling=False, **server_args):
    validate_score_centering_speculative_config(
        {"speculative_algorithm": "DFLASH", **server_args},
        environ=environ or {},
        filtered_sampling=filtered_sampling,
    )


@pytest.mark.parametrize("capture_k", [32, 128])
def test_dflash_is_accepted_with_unfiltered_sampling(capture_k):
    validate_score_centering_args(_args(rollout_top_logprobs_num=capture_k))


def test_dflash_with_filtered_sampling_is_rejected():
    with pytest.raises(ValueError, match="only unfiltered sampling"):
        validate_score_centering_args(_args(rollout_top_k=64, use_sampling_support_replay=True))


@pytest.mark.parametrize("deferred", [{"sglang_config": "groups.yaml"}, {"rollout_external": True}])
def test_global_flags_do_not_describe_server_groups_or_external_servers(deferred):
    validate_score_centering_args(_args(sglang_speculative_algorithm="EAGLE", **deferred))


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3", "NGRAM", "DSPARK"])
def test_other_speculative_algorithms_are_rejected(algorithm):
    with pytest.raises(ValueError, match="only with --sglang-speculative-algorithm DFLASH"):
        _check(speculative_algorithm=algorithm)


def test_non_speculative_servers_are_not_checked():
    validate_score_centering_speculative_config(
        {"speculative_algorithm": None, "device": "npu"}, environ={}, filtered_sampling=True
    )


@pytest.mark.parametrize("field", ["speculative_accept_threshold_single", "speculative_accept_threshold_acc"])
@pytest.mark.parametrize("value", [0.9, 1.1, float("nan")])
def test_relaxed_acceptance_thresholds_are_rejected(field, value):
    with pytest.raises(ValueError, match="exact DFlash sampling"):
        _check(**{field: value})


@pytest.mark.parametrize("value", ["1", "0.5", "nan", "bad-value"])
def test_simulated_acceptance_is_rejected(value):
    with pytest.raises(ValueError, match="SGLANG_SIMULATE_ACC_LEN"):
        _check(environ={"SGLANG_SIMULATE_ACC_LEN": value})


@pytest.mark.parametrize("device", ["npu", "musa", "cpu"])
def test_non_cuda_devices_are_rejected(device):
    with pytest.raises(ValueError, match="requires CUDA"):
        _check(device=device)


def test_rocm_builds_are_rejected(monkeypatch):
    # SGLang's DFlash worker on ROCm verifies sampled requests greedily.
    monkeypatch.setattr(torch.version, "hip", "6.4")
    with pytest.raises(ValueError, match="requires CUDA"):
        _check(device="cuda")
