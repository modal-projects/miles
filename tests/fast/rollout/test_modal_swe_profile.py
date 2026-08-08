from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import sys
from argparse import Namespace
from pathlib import Path

import pytest

MODAL_SWE_DIR = Path(__file__).parents[3] / "examples" / "experimental" / "modal-swe"
sys.path.insert(0, str(MODAL_SWE_DIR))

from modal_swe_profile import (  # noqa: E402
    FLEET_COUNTERS,
    ProfileRequest,
    _event_summary,
    _fleet_counter_delta,
    _observed_phase,
    _parse_prometheus_counters,
    _step_phases,
    _step_targets,
    _validate_request,
)
from modal_swe_profile_hook import profile_rollout_request_hook  # noqa: E402


def _request(**overrides) -> ProfileRequest:
    values = {
        "mode": "scale",
        "target_app": "stitch-glm5-2-nvfp4-lbtrain1",
        "target_router_class": "Router",
        "target_server_class": "Server",
        "target_environment": "stitch-dev",
        "groups_per_step": 32,
        "samples_per_prompt": 8,
        "warmup_steps": 2,
        "measure_steps": 1,
        "max_groups": 0,
        "all_dataset": False,
        "concurrency": 384,
        "session_servers": 64,
        "controller_processes": 48,
        "controller_threads": 16,
        "max_agent_steps": 256,
        "episode_timeout": 7200,
        "overall_timeout": 43_200,
        "preflight_only": False,
    }
    values.update(overrides)
    return ProfileRequest(**values)


def test_profile_is_locked_to_router_gateway_and_server_replicas():
    _validate_request(_request())

    with pytest.raises(ValueError, match="router=Router, server=Server"):
        _validate_request(_request(target_router_class="Server"))
    with pytest.raises(ValueError, match="router=Router, server=Server"):
        _validate_request(_request(target_server_class="Router"))


def test_all_dataset_uses_unique_groups_and_a_partial_tail():
    request = _request(all_dataset=True)
    targets = _step_targets(request, dataset_groups=731)

    assert len(targets) == 23
    assert targets[:-1] == [32] * 22
    assert targets[-1] == 27
    assert sum(targets) == 731
    assert _step_phases(request, targets) == [
        *(["warmup"] * 2),
        *(["measure"] * 19),
        "cooldown",
        "cooldown",
    ]


def test_default_scale_reserves_a_full_concurrency_cooldown():
    request = _request()
    targets = _step_targets(request, dataset_groups=731)

    assert targets == [32, 32, 32, 32, 16]
    assert _step_phases(request, targets) == [
        "warmup",
        "warmup",
        "measure",
        "cooldown",
        "cooldown",
    ]


def test_observed_phase_requires_near_full_occupancy_at_both_boundaries():
    assert (
        _observed_phase(
            "measure",
            require_steady_occupancy=True,
            active_start=384,
            active_end=346,
            active_limit=384,
        )
        == "measure"
    )
    assert (
        _observed_phase(
            "measure",
            require_steady_occupancy=True,
            active_start=384,
            active_end=345,
            active_limit=384,
        )
        == "cooldown"
    )
    assert (
        _observed_phase(
            "warmup",
            require_steady_occupancy=True,
            active_start=0,
            active_end=384,
            active_limit=384,
        )
        == "warmup"
    )


def test_step_targets_refuse_dataset_wraparound():
    with pytest.raises(ValueError, match="never wraps or pads"):
        _step_targets(_request(max_groups=732), dataset_groups=731)


def test_request_hook_sets_unconstrained_weight_and_sticky_session():
    args = Namespace(
        rollout_session_affinity_header="Modal-Session-ID",
        rollout_request_retry_attempts=1200,
        rollout_request_retry_sleep=1.0,
    )
    context = Namespace(session_id="session-7")
    request = {"payload": {}, "headers": {"x-existing": "yes"}}

    asyncio.run(profile_rollout_request_hook(args, context, request))

    assert request == {
        "payload": {"weight_version": {"min_version": None, "exact_version": 0}},
        "headers": {
            "x-existing": "yes",
            "Modal-Session-ID": "session-7",
        },
        "max_retries": 1200,
        "retry_sleep": 1.0,
    }


def test_event_summary_uses_common_fleet_wall_window():
    events = [
        {
            "proxy_started_at_unix": 10.0,
            "proxy_finished_at_unix": 14.0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "status_code": 200,
        },
        {
            "proxy_started_at_unix": 12.0,
            "proxy_finished_at_unix": 16.0,
            "prompt_tokens": 80,
            "completion_tokens": 10,
            "status_code": 200,
        },
    ]

    summary = _event_summary(events)

    assert summary["request_event_wall_window_seconds"] == 6.0
    assert summary["request_event_interval_seconds_sum"] == 8.0
    assert summary["request_event_effective_concurrency"] == pytest.approx(8 / 6)
    assert summary["completion_tokens_per_request_event_wall_second"] == 5.0


def test_fleet_counter_parser_and_delta_sum_direct_replicas():
    before = {
        "replica-a": _parse_prometheus_counters('sglang:generation_tokens_total{model_name="m"} 10\nsglang:prompt_tokens_total{model_name="m"} 20\nsglang:num_requests_total{model_name="m"} 2\n'),
        "replica-b": _parse_prometheus_counters("sglang:generation_tokens_total 30\nsglang:prompt_tokens_total 40\nsglang:num_requests_total 4\n"),
    }
    after = {
        replica: {key: value + increment for key, value in counters.items()}
        for replica, counters, increment in (
            ("replica-a", before["replica-a"], 5),
            ("replica-b", before["replica-b"], 7),
        )
    }

    metrics = _fleet_counter_delta(before, after, boundary_seconds=2.0)

    assert metrics["generation_tokens_delta"] == 12
    assert metrics["prompt_tokens_delta"] == 12
    assert metrics["requests_delta"] == 12
    assert metrics["generation_tokens_per_boundary_second"] == 6


def test_fleet_counter_delta_tolerates_replica_churn():
    before = {
        "stable": {key: 10.0 for key in FLEET_COUNTERS},
        "lost": {key: 20.0 for key in FLEET_COUNTERS},
    }
    after = {
        "stable": {key: 15.0 for key in FLEET_COUNTERS},
        "added": {key: 3.0 for key in FLEET_COUNTERS},
    }

    metrics = _fleet_counter_delta(before, after, boundary_seconds=1.0)

    assert metrics["generation_tokens_delta"] == 5
    assert metrics["replicas_before"] == 2
    assert metrics["replicas_after"] == 2
    assert metrics["replicas_common"] == 1
    assert metrics["replicas_lost"] == 1
    assert metrics["replicas_added"] == 1
