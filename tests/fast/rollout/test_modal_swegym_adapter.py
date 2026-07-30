import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest
from jinja2 import Template

SWEGYM_DIR = Path(__file__).parents[3] / "examples" / "experimental" / "modal-swegym"
sys.path.insert(0, str(SWEGYM_DIR))

import modal_swegym_agent_function as agent_function_module  # noqa: E402
from modal_swegym_agent_function import (  # noqa: E402
    _AgentWorker,
    _OBSERVATION_TEMPLATE,
    _RayAgentWorkerPool,
    _attach_client_model_timings,
    _failure,
    _instrument_model_requests,
    _is_context_limit_error,
    _is_sandbox_not_found_error,
    _parse_reward,
)
from modal_swegym_sandbox import (  # noqa: E402
    _BOUNDED_COMMAND_RUNNER,
    ModalSWEGymEnvironment,
    SandboxExecResult,
    _parse_bounded_command_response,
    sandbox_settings,
)
from modal_swegym_metrics import log_rollout_data  # noqa: E402
from miles.utils.types import Sample


def _run_bounded(
    command: str,
    *,
    timeout: float = 30,
    output_hard_limit: int = 16 * 1024 * 1024,
):
    started = time.perf_counter()
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _BOUNDED_COMMAND_RUNNER,
            "5000",
            "5000",
            str(timeout),
            str(output_hard_limit),
        ],
        input=command.encode(),
        capture_output=True,
        check=True,
    )
    return _parse_bounded_command_response(
        stdout=process.stdout,
        stderr=process.stderr,
        fallback_return_code=process.returncode,
        client_seconds=time.perf_counter() - started,
    )


def test_sandbox_settings_rejects_invalid_modal_app_name(monkeypatch):
    invalid_name = "x" * 65
    monkeypatch.setenv("MODAL_SWEGYM_SANDBOX_APP", invalid_name)

    with pytest.raises(ValueError, match=r"1-64 characters.*65 chars"):
        sandbox_settings()


def test_sandbox_settings_accepts_64_character_modal_app_name(monkeypatch):
    valid_name = "x" * 64
    monkeypatch.setenv("MODAL_SWEGYM_SANDBOX_APP", valid_name)

    assert sandbox_settings()["app_name"] == valid_name


def test_invalid_command_wrapper_response_is_infrastructure_failure():
    with pytest.raises(RuntimeError, match="invalid response"):
        _parse_bounded_command_response(
            stdout=b"not-json",
            stderr=b"wrapper traceback",
            fallback_return_code=1,
            client_seconds=0.1,
        )


def test_policy_failure_keeps_trajectory_trainable_with_zero_reward():
    result = _failure("command_timeout", infrastructure=False)

    assert "_miles_abort" not in result
    assert result["reward"] == 0.0
    assert result["agent_metrics"]["policy_failure"] == 1
    assert result["agent_metrics"]["infra_error"] == 0


def test_infrastructure_failure_requests_scheduler_abort():
    result = _failure("sandbox_infra_error")

    assert result["_miles_abort"] is True
    assert result["agent_metrics"]["infra_error"] == 1


def test_client_model_request_instrumentation_records_exact_attempts():
    class Model:
        def _query(self, value):
            return value * 2

    model = Model()
    durations = []
    phases = []
    _instrument_model_requests(model, durations, phases.append)

    assert model._query(3) == 6
    assert len(durations) == 1
    assert durations[0] >= 0
    assert phases == ["model_generation", "interaction"]

    metrics = _attach_client_model_timings({}, durations)
    assert metrics["client_model_request_durations_seconds"] == durations
    assert metrics["model_request_count"] == 1
    assert metrics["model_request_time"] == pytest.approx(sum(durations))


def test_client_model_timings_split_generation_from_interaction():
    metrics = _attach_client_model_timings(
        {"total_time": 10.0, "total_tool_time": 3.0},
        [2.0, 3.0],
    )

    assert metrics["model_request_count"] == 2
    assert metrics["model_request_time"] == 5.0
    assert metrics["interaction_time"] == 5.0
    assert metrics["interaction_sandbox_time"] == 3.0
    assert metrics["interaction_unattributed_time"] == 2.0
    assert metrics["generation_time_ratio"] == 0.5
    assert metrics["interaction_time_ratio"] == 0.5
    assert metrics["generation_bound"] == 0


def test_client_model_timings_classify_pre_generation_failure():
    metrics = _attach_client_model_timings({"total_time": 10.0}, [])

    assert metrics["model_request_count"] == 0
    assert metrics["model_request_time"] == 0
    assert metrics["interaction_time"] == 10.0
    assert metrics["interaction_sandbox_time"] == 0.0
    assert metrics["interaction_unattributed_time"] == 10.0
    assert metrics["generation_time_ratio"] == 0.0
    assert metrics["interaction_time_ratio"] == 1.0


def test_rollout_metrics_aggregate_adapter_owned_timings():
    samples = [
        Sample(
            metadata={
                "exit_status": "Submitted",
                "agent_metrics": {
                    "total_time": 10.0,
                    "total_tool_time": 4.0,
                    "agent_tool_output_hard_limit_count": 1,
                    "client_model_request_durations_seconds": [2.0, 3.0],
                },
                "session_collect/total_seconds": 0.5,
                "model_request/durations_seconds": [1.5, 2.5],
                "model_request/completion_tokens": 40,
                "model_request/non_200_count": 0,
            }
        ),
        Sample(
            metadata={
                "exit_status": "LimitsExceeded",
                "agent_metrics": {
                    "total_time": 20.0,
                    "total_tool_time": 8.0,
                    "agent_tool_output_hard_limit_count": 0,
                    "context_limit_exceeded": 1,
                    "client_model_request_durations_seconds": [4.0],
                },
                "session_collect/total_seconds": 1.0,
                "model_request/durations_seconds": [3.5],
                "model_request/completion_tokens": 20,
                "model_request/non_200_count": 1,
            }
        ),
    ]
    metrics = {}

    assert log_rollout_data(0, None, samples, metrics, 0.0) is False

    assert metrics["rollout_agent/total_time_mean"] == 15
    assert metrics["rollout_session/total_seconds_mean"] == 0.75
    assert metrics["rollout_model/request_count"] == 3
    assert metrics["rollout_model/completion_tokens"] == 60
    assert metrics["rollout_model/client_minus_backend_request_count"] == 0
    assert metrics["rollout_model/client_minus_backend_seconds_signed"] == pytest.approx(1.5)
    assert metrics["rollout_agent/agent_tool_output_hard_limit_count_mean"] == 0.5
    assert metrics["rollout_agent/context_limit_exit_ratio"] == 0.5
    assert "model_request/durations_seconds" not in samples[0].metadata
    assert (
        "client_model_request_durations_seconds"
        not in samples[0].metadata["agent_metrics"]
    )


def test_reward_parser_rejects_ambiguous_mapping():
    output = "\n".join(
        [
            "__MILES_SWEGYM_REWARD_START__",
            '{"passed": 1, "failed": 0}',
            "__MILES_SWEGYM_REWARD_END__",
        ]
    )

    assert _parse_reward(output) is None


def test_bounded_runner_preserves_small_stdout_stderr_and_return_code():
    result = _run_bounded("printf stdout; printf stderr >&2; exit 7")

    assert result.return_code == 7
    assert result.output == "stdoutstderr"
    assert result.output_total_bytes == 12
    assert not result.output_truncated
    assert result.remote_seconds > 0


def test_bounded_runner_caps_output_before_transport():
    result = _run_bounded("python -c \"import sys; sys.stdout.write('a' * 600000); sys.stderr.write('b' * 400000)\"")

    assert result.return_code == 0
    assert result.output == ""
    assert result.output_total_bytes == 1_000_000
    assert result.output_truncated
    assert result.output_head == "a" * 5000
    assert result.output_tail == "b" * 5000
    assert result.transferred_bytes < 20_000
    assert not result.output_limited


def test_bounded_runner_stops_commands_that_produce_runaway_output():
    result = _run_bounded(
        "python -c \"import sys; chunk=b'x'*65536; [sys.stdout.buffer.write(chunk) for _ in range(1000)]\"",
        output_hard_limit=200_000,
    )

    assert result.return_code == 125
    assert result.output_truncated
    assert result.output_limited
    assert result.output_total_bytes < 1_000_000
    assert result.transferred_bytes < 20_000
    assert result.remote_seconds < 5


def test_output_limit_is_explicit_agent_feedback(monkeypatch):
    env = ModalSWEGymEnvironment.__new__(ModalSWEGymEnvironment)
    env.cwd = "/testbed"
    result = SandboxExecResult(
        return_code=125,
        output="",
        output_head="first lines\n",
        output_tail="\nlast lines",
        output_total_bytes=200_000,
        output_truncated=True,
        remote_seconds=0.1,
        client_seconds=0.2,
        transferred_bytes=10_000,
        output_limited=True,
    )
    monkeypatch.setattr(env, "exec_detailed", lambda *_args, **_kwargs: result)

    observation = env.execute({"command": "yes"})

    assert observation["returncode"] == 125
    assert observation["output_limited"]
    rendered = Template(_OBSERVATION_TEMPLATE).render(output=observation)
    assert "terminated after producing too much output" in rendered
    assert "first lines" in rendered
    assert "last lines" in rendered


def test_bounded_runner_enforces_deadline_and_keeps_partial_output():
    result = _run_bounded(
        "printf started; sleep 30",
        timeout=0.2,
    )

    assert result.return_code == 124
    assert result.timed_out
    assert result.output == "started"
    assert result.remote_seconds < 5


def test_bounded_runner_streams_commands_larger_than_modal_cmd_limit():
    command = "#" + ("x" * 200_000) + "\nprintf streamed"
    result = _run_bounded(command)

    assert result.return_code == 0
    assert result.output == "streamed"


class _FakeWorker:
    pass


def test_worker_pool_round_robins_ties_and_prefers_least_loaded():
    workers = [_FakeWorker() for _ in range(3)]
    pool = _RayAgentWorkerPool(workers)

    acquired = [pool._acquire()[0] for _ in range(6)]
    assert acquired == [0, 1, 2, 0, 1, 2]
    assert pool.in_flight == [2, 2, 2]

    pool._release(1)
    index, worker = pool._acquire()
    assert index == 1
    assert worker is workers[1]


def test_empty_worker_pool_is_rejected():
    pool = _RayAgentWorkerPool([])
    with pytest.raises(ValueError):
        pool._acquire()


@pytest.mark.asyncio
async def test_cancelled_dispatch_retains_capacity_until_remote_episode_finishes():
    finish = asyncio.Event()

    class RemoteMethod:
        def remote(self, _payload):
            async def run():
                await finish.wait()
                return {"reward": 0}

            return run()

    worker = _FakeWorker()
    worker.run_episode = RemoteMethod()
    pool = _RayAgentWorkerPool([worker])

    dispatch = asyncio.create_task(pool.run_episode({}))
    await asyncio.sleep(0)
    assert pool.in_flight == [1]

    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert pool.in_flight == [1]

    finish.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pool.in_flight == [0]


@pytest.mark.asyncio
async def test_agent_worker_records_live_phase_accounting(monkeypatch):
    def fake_episode(*, phase_callback, **_kwargs):
        phase_callback("sandbox_boot")
        phase_callback("model_generation")
        phase_callback("interaction")
        return {"agent_metrics": {}}

    monkeypatch.setattr(
        agent_function_module,
        "_run_episode_sync",
        fake_episode,
    )
    worker = _AgentWorker(worker_index=3, threads=1)
    result = await worker.run_episode(
        {
            "submitted_at_unix": time.time(),
            "base_url": "http://example",
            "prompt": "prompt",
            "request_kwargs": {},
            "metadata": {},
        }
    )

    metrics = result["agent_metrics"]
    assert metrics["agent_worker_index"] == 3
    assert metrics["phase_accounted_seconds"] >= 0
    assert "phase_executor_queue_seconds" in metrics
    assert "phase_sandbox_boot_seconds" in metrics
    assert "phase_model_generation_seconds" in metrics
    assert "phase_interaction_seconds" in metrics
    assert (await worker.stats())["active"] == 0


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("TITO context limit reached: prompt has 65536 tokens, configured max_seq_len is 65536"),
        RuntimeError("Requested token count exceeds the model's maximum context length"),
        RuntimeError("Input length (65530 tokens) exceeds the maximum allowed length (65530 tokens)"),
    ],
)
def test_context_limit_errors_are_recognized(error):
    assert _is_context_limit_error(error)


def test_unrelated_bad_request_is_not_a_context_limit():
    assert not _is_context_limit_error(RuntimeError("appended message has role='assistant'"))


def test_modal_sandbox_not_found_is_a_distinct_infra_error():
    class SandboxNotFoundError(RuntimeError):
        __module__ = "modal.exception"

    assert _is_sandbox_not_found_error(SandboxNotFoundError("Sandbox not found"))
    assert not _is_sandbox_not_found_error(RuntimeError("file not found"))
