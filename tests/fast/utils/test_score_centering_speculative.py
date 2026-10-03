"""Speculative score centering requires the worker's actual sampling policy."""

from argparse import Namespace
from copy import deepcopy

import pytest

from miles.utils.score_centering import (
    validate_score_centering_args,
    validate_score_centering_server_info,
    validate_score_centering_speculative_config,
)


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
            "sglang_speculative_algorithm": "DFLASH",
            **overrides,
        }
    )


def _server_info(**state_overrides):
    return {
        "speculative_algorithm": "DFLASH",
        "speculative_accept_threshold_single": 1.0,
        "speculative_accept_threshold_acc": 1.0,
        "internal_states": [
            {
                "speculative_algorithm": "DFLASH",
                "speculative_accept_threshold_single": 1.0,
                "speculative_accept_threshold_acc": 1.0,
                "dflash_sampling_verify_available": True,
                "env_vars": {},
                **state_overrides,
            }
        ],
    }


@pytest.mark.parametrize("sampling_top_k,capture_k", [(-1, 32), (-1, 128), (32, 32), (64, 128)])
def test_dflash_accepts_unfiltered_heads_and_bounded_filtered_support(sampling_top_k, capture_k):
    validate_score_centering_args(_args(rollout_top_k=sampling_top_k, rollout_top_logprobs_num=capture_k))


@pytest.mark.parametrize("defer", [{"sglang_config": "groups.yaml"}, {"rollout_external": True}])
def test_global_speculation_does_not_override_effective_group_or_external_configuration(defer):
    validate_score_centering_args(_args(sglang_speculative_algorithm="EAGLE", **defer))


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3", "NGRAM", "DSPARK"])
def test_other_speculative_algorithms_remain_unsupported(algorithm):
    with pytest.raises(ValueError, match="only.*DFLASH"):
        validate_score_centering_speculative_config({"speculative_algorithm": algorithm}, environ={})


@pytest.mark.parametrize("field", ["speculative_accept_threshold_single", "speculative_accept_threshold_acc"])
@pytest.mark.parametrize("value", [0.9, 1.1, float("nan"), float("inf")])
def test_relaxed_or_invalid_acceptance_thresholds_are_rejected(field, value):
    with pytest.raises(ValueError, match="exact DFlash sampling"):
        validate_score_centering_speculative_config({"speculative_algorithm": "DFLASH", field: value}, environ={})


@pytest.mark.parametrize("value", ["1", "0.5", "nan", "inf", "bad-value"])
def test_simulated_acceptance_is_rejected(value):
    with pytest.raises(ValueError, match="SGLANG_SIMULATE_ACC_LEN"):
        validate_score_centering_speculative_config(
            {"speculative_algorithm": "DFLASH"}, environ={"SGLANG_SIMULATE_ACC_LEN": value}
        )


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "YES", "Y"])
def test_original_logprobs_reject_every_sglang_true_spelling(value, monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", value)
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        validate_score_centering_args(_args())


def test_runtime_checks_each_dp_worker_and_uses_effective_thresholds():
    info = _server_info()
    validate_score_centering_server_info(info)
    second = deepcopy(info["internal_states"][0])
    second["speculative_accept_threshold_acc"] = 0.9
    info["internal_states"].append(second)
    with pytest.raises(ValueError, match="exact DFlash sampling"):
        validate_score_centering_server_info(info)


@pytest.mark.parametrize(
    "missing,match",
    [
        ("dflash_sampling_verify_available", "dflash_sampling_verify_available"),
        ("speculative_accept_threshold_single", "speculative_accept_threshold_single"),
        ("speculative_accept_threshold_acc", "speculative_accept_threshold_acc"),
        ("env_vars", "SGLANG_EXPOSE_OWN_ENV_VARS"),
    ],
)
def test_runtime_does_not_infer_worker_guarantees_from_startup_values(missing, match):
    info = _server_info()
    del info["internal_states"][0][missing]
    with pytest.raises(ValueError, match=match):
        validate_score_centering_server_info(info)


@pytest.mark.parametrize("capability", [False, None, "true", 1])
def test_runtime_rejects_greedy_fallback_and_unrecognized_capability_values(capability):
    with pytest.raises(ValueError, match="dflash_sampling_verify_available"):
        validate_score_centering_server_info(_server_info(dflash_sampling_verify_available=capability))


@pytest.mark.parametrize(
    "env,match",
    [
        ({"SGLANG_SIMULATE_ACC_LEN": "4"}, "SGLANG_SIMULATE_ACC_LEN"),
        ({"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"}, "SGLANG_RETURN_ORIGINAL_LOGPROB"),
        ({"SGLANG_RETURN_ORIGINAL_LOGPROB": "true"}, "SGLANG_RETURN_ORIGINAL_LOGPROB"),
        ({"SGLANG_RETURN_ORIGINAL_LOGPROB": "yes"}, "SGLANG_RETURN_ORIGINAL_LOGPROB"),
        ({"SGLANG_RETURN_ORIGINAL_LOGPROB": "Y"}, "SGLANG_RETURN_ORIGINAL_LOGPROB"),
    ],
)
def test_runtime_checks_remote_environment(env, match):
    with pytest.raises(ValueError, match=match):
        validate_score_centering_server_info(_server_info(env_vars=env))


def test_speculative_server_must_report_worker_states():
    with pytest.raises(ValueError, match="internal_states"):
        validate_score_centering_server_info({"speculative_algorithm": "DFLASH"})


def test_non_speculative_workers_do_not_need_speculative_capabilities():
    validate_score_centering_server_info({"speculative_algorithm": None, "internal_states": [{}]})
    validate_score_centering_speculative_config({"speculative_algorithm": None}, environ={})


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "YES", "Y"])
def test_non_speculative_workers_reject_reported_original_logprobs(value):
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        validate_score_centering_server_info(
            {
                "speculative_algorithm": None,
                "internal_states": [{"env_vars": {"SGLANG_RETURN_ORIGINAL_LOGPROB": value}}],
            }
        )


def test_non_speculative_workers_accept_reported_sampling_logprobs():
    validate_score_centering_server_info(
        {
            "speculative_algorithm": None,
            "internal_states": [{"env_vars": {"SGLANG_RETURN_ORIGINAL_LOGPROB": "0"}}],
        }
    )


@pytest.mark.parametrize("info", [{}, {"internal_states": [{}]}])
def test_missing_algorithm_is_not_assumed_to_mean_non_speculative(info):
    with pytest.raises(ValueError, match="report speculative_algorithm"):
        validate_score_centering_server_info(info)


def test_worker_algorithm_is_used_when_startup_algorithm_is_absent():
    info = _server_info()
    del info["speculative_algorithm"]
    validate_score_centering_server_info(info)
