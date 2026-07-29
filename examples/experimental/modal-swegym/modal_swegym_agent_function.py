"""mini-swe-agent rollouts for SWE-Gym with command execution in Modal."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import threading
import time
import tomllib
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, Callable

from modal_swegym_sandbox import ModalSWEGymEnvironment, SandboxCommandTimeoutError, sandbox_settings

logger = logging.getLogger(__name__)


# The agent keeps complete command output in its in-memory trajectory. Emitting
# every tool observation (often full test tracebacks) and every successful HTTP
# request to the cluster console adds substantial log I/O without improving
# operations. Keep warnings/errors; rollout metrics and verifier evidence are
# logged separately by this adapter.
def _configure_dependency_logging() -> None:
    for dependency_logger in ("agent", "minisweagent", "litellm", "LiteLLM", "httpx"):
        logging.getLogger(dependency_logger).setLevel(logging.WARNING)


_configure_dependency_logging()

_REWARD_START = "__MILES_SWEGYM_REWARD_START__"
_REWARD_END = "__MILES_SWEGYM_REWARD_END__"
_VERIFIER_LOG_TAIL_CHARS = 4000
_PYTEST_REPORTER = """\
def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            status = "PASSED"
        elif report.failed:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"\\n{status} {report.nodeid}", flush=True)
    elif report.failed:
        print(f"\\nERROR {report.nodeid}", flush=True)
"""

# Preserve mini-swe-agent's existing observation semantics while allowing the
# environment to report that it already bounded output inside the Sandbox.
# Without this branch, mini-swe-agent truncates only after Modal has transported
# the complete output (371 MiB in one observed rollout).
_OBSERVATION_TEMPLATE = """\
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output_truncated | default(false) -%}
{% if output.output_limited | default(false) -%}
<warning>
Your last command was terminated after producing too much output.
Narrow the command or redirect verbose output to a file, then inspect that file selectively with head, tail, sed, or grep.
</warning>
{% else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{% endif -%}
<output_head>
{{ output.output_head }}
</output_head>
<elided_bytes>
{{ output.output_elided_bytes }} bytes elided before sandbox transport
</elided_bytes>
<output_tail>
{{ output.output_tail }}
</output_tail>
{% elif output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
</warning>
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ output.output | length - 10000 }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""


def _task_dir(metadata: dict[str, Any]) -> Path:
    explicit = metadata.get("task_dir")
    instance_id = metadata.get("instance_id")
    if explicit:
        path = Path(explicit)
    elif instance_id:
        path = Path(os.getenv("MODAL_SWEGYM_TASKS_DIR", "/data/harbor-tasks/swegym")) / str(instance_id).lower()
    else:
        raise ValueError("SWE-Gym metadata must contain task_dir or instance_id")
    if not path.is_dir():
        raise FileNotFoundError(f"SWE-Gym task directory does not exist: {path}")
    return path


def _parse_reward(output: str) -> float | None:
    lines = output.splitlines()
    try:
        start = lines.index(_REWARD_START) + 1
        end = lines.index(_REWARD_END, start)
    except ValueError:
        return None

    raw = "\n".join(lines[start:end]).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, dict):
        if "reward" in decoded:
            decoded = decoded["reward"]
        elif len(decoded) == 1:
            decoded = next(iter(decoded.values()))
        else:
            return None
    try:
        return float(decoded)
    except (TypeError, ValueError):
        return None


def _is_context_limit_error(error: Exception) -> bool:
    """Recognize both Miles and OpenAI/SGLang context-limit error shapes."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "tito context limit reached",
                "maximum context length",
                "context window exceeded",
                "context_length_exceeded",
                "exceeds the maximum allowed length",
            )
        ):
            return True
        if "input length" in message and "maximum allowed length" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sandbox_not_found_error(error: Exception) -> bool:
    """Recognize Modal sandbox disappearance without importing Modal internals."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_type = type(current)
        qualified_name = f"{error_type.__module__}.{error_type.__name__}".lower()
        message = str(current).lower()
        if (
            ("modal" in qualified_name and "notfound" in qualified_name)
            or "sandbox not found" in message
            or ("sandbox" in message and "not found" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _verifier_timeout(task_dir: Path, configured_timeout: int) -> int:
    """Return the smaller of the operator cap and Harbor's task timeout."""
    if configured_timeout <= 0:
        raise ValueError(f"Verifier timeout must be positive, got {configured_timeout}")
    task_config = task_dir / "task.toml"
    if not task_config.is_file():
        return configured_timeout
    try:
        config = tomllib.loads(task_config.read_text())
        task_timeout = int(config["verifier"]["timeout_sec"])
        if task_timeout <= 0:
            raise ValueError
        return min(configured_timeout, task_timeout)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        logger.warning("Invalid verifier timeout in %s; using %ss", task_config, configured_timeout)
        return configured_timeout


def run_verifier(
    env: ModalSWEGymEnvironment,
    task_dir: Path,
    *,
    configured_timeout: int,
) -> dict[str, Any]:
    """Run Harbor's verifier and retain enough evidence to audit its verdict."""
    timeout = _verifier_timeout(task_dir, configured_timeout)
    # Verifier tests must not exist in the policy sandbox while the agent is
    # acting. Upload them only after the episode terminates.
    env.upload_tree(task_dir / "tests", "/tests")
    verify_command = (
        "mkdir -p /logs/verifier; "
        "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json; "
        # SWE-Gym's generated parser matches FAIL_TO_PASS against per-test
        # status lines. Its test scripts and pytest-pretty omit those records,
        # so even a passing oracle otherwise scores 0. This hook emits only the
        # records that the existing parser expects; it does not alter outcomes.
        f"printf %s {shlex.quote(_PYTEST_REPORTER)} > /tmp/miles_pytest_reporter.py; "
        'export PYTHONPATH="/tmp:${PYTHONPATH:-}"; '
        'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p miles_pytest_reporter"; '
        # The sandbox runner bounds output before transport and preserves
        # partial diagnostics when the verifier reaches its deadline.
        "bash /tests/test.sh 2>&1; "
        "status=$?; "
        f"echo {_REWARD_START}; "
        "if [ -f /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
        "elif [ -f /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi; "
        f"echo; echo {_REWARD_END}; "
        "exit $status"
    )
    return_code, output = env.exec(verify_command, cwd="/testbed", timeout=timeout)
    return {
        "reward": _parse_reward(output),
        "return_code": return_code,
        "timeout_sec": timeout,
        "output_tail": output[-_VERIFIER_LOG_TAIL_CHARS:],
    }


def _failure(
    reason: str,
    *,
    infrastructure: bool = True,
    total_time: float | None = None,
    agent_queue_time: float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    metrics: dict[str, float | int] = {
        "infra_error": int(infrastructure),
        "policy_failure": int(not infrastructure),
    }
    if total_time is not None:
        metrics["total_time"] = total_time
    if agent_queue_time is not None:
        metrics["agent_queue_time"] = agent_queue_time
    metadata = dict(metadata)
    if isinstance(metadata.get("agent_metrics"), dict):
        metrics.update(metadata.pop("agent_metrics"))
    result = {
        "exit_status": reason,
        "eval_report": {},
        "agent_metrics": metrics,
        **metadata,
    }
    if infrastructure:
        result["_miles_abort"] = True
    else:
        # Preserve recorded policy tokens and train the terminal outcome with
        # zero reward. The normal reward function reads this field.
        result["reward"] = 0.0
    return result


@dataclass(frozen=True)
class _EnvironmentSnapshot:
    command_count: int
    exec_time: float
    exec_remote_time: float
    upload_time: float
    upload_bytes: int
    command_timeout_count: int
    output_bytes: int
    transferred_bytes: int
    output_truncated_count: int
    output_hard_limit_count: int

    @classmethod
    def capture(cls, env: ModalSWEGymEnvironment) -> _EnvironmentSnapshot:
        return cls(
            command_count=env.command_count,
            exec_time=env.exec_time,
            exec_remote_time=env.exec_remote_time,
            upload_time=env.upload_time,
            upload_bytes=env.upload_bytes,
            command_timeout_count=env.command_timeout_count,
            output_bytes=env.command_output_bytes,
            transferred_bytes=env.command_transferred_bytes,
            output_truncated_count=env.command_output_truncated_count,
            output_hard_limit_count=env.command_output_hard_limit_count,
        )


def _environment_metrics(
    env: ModalSWEGymEnvironment,
    *,
    agent_queue_time: float,
    agent_dispatch_queue_time: float = 0.0,
    total_time: float,
    agent_snapshot: _EnvironmentSnapshot | None = None,
) -> dict[str, float | int]:
    """Return a non-overlapping sandbox/tool timing breakdown."""
    agent_snapshot = agent_snapshot or _EnvironmentSnapshot.capture(env)

    agent_durations = env.exec_durations[: agent_snapshot.command_count]
    agent_remote_durations = env.exec_remote_durations[: agent_snapshot.command_count]
    agent_transport_durations = env.exec_transport_durations[: agent_snapshot.command_count]
    agent_input_sizes = env.command_input_sizes[: agent_snapshot.command_count]
    verifier_exec_time = max(0.0, env.exec_time - agent_snapshot.exec_time)
    verifier_exec_remote_time = max(
        0.0,
        env.exec_remote_time - agent_snapshot.exec_remote_time,
    )
    verifier_upload_time = max(0.0, env.upload_time - agent_snapshot.upload_time)
    sandbox_non_generation_time = env.boot_time + env.exec_time + env.upload_time
    return {
        "agent_queue_time": agent_queue_time,
        "agent_dispatch_queue_time": agent_dispatch_queue_time,
        "sandbox_boot_time": env.boot_time,
        "tool_calls": agent_snapshot.command_count,
        "agent_tool_exec_time": agent_snapshot.exec_time,
        "agent_tool_remote_exec_time": agent_snapshot.exec_remote_time,
        "agent_tool_transport_time": max(
            0.0,
            agent_snapshot.exec_time - agent_snapshot.exec_remote_time,
        ),
        "agent_tool_exec_mean": (sum(agent_durations) / len(agent_durations) if agent_durations else 0.0),
        "agent_tool_remote_exec_mean": (sum(agent_remote_durations) / len(agent_remote_durations) if agent_remote_durations else 0.0),
        "agent_tool_transport_mean": (sum(agent_transport_durations) / len(agent_transport_durations) if agent_transport_durations else 0.0),
        "agent_tool_exec_p90": (sorted(agent_durations)[round((len(agent_durations) - 1) * 0.90)] if agent_durations else 0.0),
        "agent_tool_exec_max": max(agent_durations, default=0.0),
        "agent_tool_input_mib": sum(agent_input_sizes) / (1024 * 1024),
        "agent_tool_input_mean_bytes": (sum(agent_input_sizes) / len(agent_input_sizes) if agent_input_sizes else 0.0),
        "agent_tool_input_max_bytes": max(agent_input_sizes, default=0),
        "agent_tool_input_over_64k_count": sum(size > 65536 for size in agent_input_sizes),
        "agent_tool_input_over_64k_ratio": (sum(size > 65536 for size in agent_input_sizes) / len(agent_input_sizes) if agent_input_sizes else 0.0),
        "tool_timeout_count": agent_snapshot.command_timeout_count,
        "agent_tool_output_mib": agent_snapshot.output_bytes / (1024 * 1024),
        "agent_tool_transferred_mib": agent_snapshot.transferred_bytes / (1024 * 1024),
        "agent_tool_output_truncated_count": agent_snapshot.output_truncated_count,
        "agent_tool_output_truncated_ratio": (
            agent_snapshot.output_truncated_count / agent_snapshot.command_count
            if agent_snapshot.command_count
            else 0.0
        ),
        "agent_tool_output_hard_limit_count": agent_snapshot.output_hard_limit_count,
        "agent_tool_output_hard_limit_ratio": (
            agent_snapshot.output_hard_limit_count / agent_snapshot.command_count
            if agent_snapshot.command_count
            else 0.0
        ),
        "verifier_upload_time": verifier_upload_time,
        "verifier_upload_mib": max(0, env.upload_bytes - agent_snapshot.upload_bytes)
        / (1024 * 1024),
        "verifier_exec_time": verifier_exec_time,
        "verifier_remote_exec_time": verifier_exec_remote_time,
        "verifier_transport_time": max(
            0.0,
            verifier_exec_time - verifier_exec_remote_time,
        ),
        "verifier_output_mib": max(
            0,
            env.command_output_bytes - agent_snapshot.output_bytes,
        )
        / (1024 * 1024),
        "verifier_transferred_mib": max(
            0,
            env.command_transferred_bytes - agent_snapshot.transferred_bytes,
        )
        / (1024 * 1024),
        "total_tool_time": sandbox_non_generation_time,
        "total_time": total_time,
    }


def _attach_client_model_timings(
    metrics: dict[str, Any],
    durations: list[float],
) -> dict[str, Any]:
    """Attach exact timings; the rollout log hook computes batch summaries."""
    metrics["client_model_request_durations_seconds"] = list(durations)
    model_request_time = sum(durations)
    total_time = float(metrics.get("total_time", 0.0))
    metrics["model_request_count"] = len(durations)
    metrics["model_request_time"] = model_request_time
    interaction_time = max(0.0, total_time - model_request_time)
    sandbox_time = min(
        interaction_time,
        max(0.0, float(metrics.get("total_tool_time", 0.0))),
    )
    metrics["interaction_time"] = interaction_time
    metrics["interaction_sandbox_time"] = sandbox_time
    # Agent setup, prompt rendering, Python scheduling, and any other time not
    # spent in a model HTTP request or a measured Sandbox operation. Keeping
    # this residual explicit makes instrumentation gaps visible.
    metrics["interaction_unattributed_time"] = max(
        0.0,
        interaction_time - sandbox_time,
    )
    if total_time > 0:
        metrics["generation_time_ratio"] = min(1.0, model_request_time / total_time)
        metrics["interaction_time_ratio"] = max(0.0, 1.0 - metrics["generation_time_ratio"])
        metrics["generation_bound"] = int(metrics["generation_time_ratio"] > 0.5)
    return metrics


def _instrument_model_requests(
    model: Any,
    durations: list[float],
    phase_callback: Callable[[str], None] | None = None,
) -> None:
    """Measure each real LiteLLM HTTP attempt as perceived by the agent."""
    query = getattr(model, "_query", None)
    if not callable(query):
        logger.warning(
            "Model %s has no callable _query; client request timing unavailable",
            type(model).__name__,
        )
        return

    def timed_query(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        if phase_callback is not None:
            phase_callback("model_generation")
        try:
            return query(*args, **kwargs)
        finally:
            durations.append(time.perf_counter() - started)
            if phase_callback is not None:
                phase_callback("interaction")

    try:
        model._query = timed_query
    except (AttributeError, TypeError):
        logger.warning(
            "Model %s does not allow _query instrumentation; client request timing unavailable",
            type(model).__name__,
        )


def _run_episode_sync(
    *,
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    queued_at: float,
    dispatch_queue_time: float = 0.0,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from minisweagent.agents import get_agent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model

    # mini-swe-agent configures its logger during import, so enforce the
    # production console levels once more after those imports complete.
    _configure_dependency_logging()

    agent_queue_time = time.perf_counter() - queued_at
    settings = sandbox_settings()
    task_dir = _task_dir(metadata)
    verifier_timeout = _verifier_timeout(task_dir, int(settings["verify_timeout"]))
    started = time.perf_counter()
    if phase_callback is not None:
        phase_callback("sandbox_boot")
    env = ModalSWEGymEnvironment(
        task_dir,
        lifetime=int(settings["episode_timeout"]) + verifier_timeout + 300,
        exec_timeout=int(settings["exec_timeout"]),
        app_name=str(settings["app_name"]),
    )

    try:
        if phase_callback is not None:
            phase_callback("agent_setup")
        config = get_config_from_spec("swebench")
        client_model_request_durations: list[float] = []
        model_kwargs = {
            **request_kwargs,
            "api_base": f"{base_url.rstrip('/')}/v1",
            "api_key": "EMPTY",
            "drop_params": True,
            # LiteLLM's shorter implicit default can expire a valid long-context
            # decode while the episode itself still has ample wall-clock budget.
            "timeout": float(os.getenv("MODAL_SWEGYM_MODEL_REQUEST_TIMEOUT", "1800")),
            # A hidden client retry can overlap the original request after a
            # transport disconnect. The Miles session server also serializes
            # same-session requests, but disabling LiteLLM's inner retry avoids
            # duplicate model work. Production configs also set mini-SWE-Agent's
            # outer attempt limit to one.
            "num_retries": 0,
        }
        model_config = {
            **config.get("model", {}),
            "model_name": f"openai/{os.getenv('AGENT_MODEL_NAME', 'model')}",
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
            "observation_template": _OBSERVATION_TEMPLATE,
        }
        agent_config = {
            **config.get("agent", {}),
            "step_limit": int(os.getenv("MODAL_SWEGYM_MAX_STEPS", "100")),
            "wall_time_limit_seconds": int(settings["episode_timeout"]),
            "cost_limit": 0.0,
            "output_path": None,
        }
        model = get_model(config=model_config)
        _instrument_model_requests(
            model,
            client_model_request_durations,
            phase_callback,
        )
        # BadRequest errors are deterministic for a fixed request. Retrying a
        # TITO validation or context-limit 400 ten times only burns rollout
        # slots. Transient transport/server errors remain retryable.
        try:
            import litellm

            # LiteLLM prints an issue URL and debugging hint directly to stderr
            # for every mapped client error, bypassing Python logger levels.
            # Context exhaustion is an expected terminal condition for these
            # long trajectories, so retain it in aggregate outcome metrics
            # without flooding the distributed job log.
            litellm.suppress_debug_info = True
            if hasattr(model, "abort_exceptions"):
                model.abort_exceptions = list(
                    dict.fromkeys(
                        [
                            *model.abort_exceptions,
                            litellm.exceptions.BadRequestError,
                        ]
                    )
                )
        except (AttributeError, ImportError):
            logger.warning("Unable to mark LiteLLM BadRequestError as non-retryable")
        agent = get_agent(model, env, agent_config, default_type="default")
        # DefaultAgent logs every message (including complete command output)
        # at DEBUG. Miles configures the process root logger independently, so
        # logger levels alone can be reset by initialization order; disabling
        # this trajectory logger is deterministic. The messages remain in
        # ``agent.messages`` and therefore in the returned training sample.
        agent.logger.disabled = True
        context_limit_exceeded = False
        try:
            if phase_callback is not None:
                phase_callback("interaction")
            result = agent.run(str(prompt))
        except SandboxCommandTimeoutError:
            logger.warning("SWE-Gym command timed out for %s", task_dir.name)
            elapsed = time.perf_counter() - started
            agent_metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
            )
            _attach_client_model_timings(
                agent_metrics,
                client_model_request_durations,
            )
            return _failure(
                "command_timeout",
                infrastructure=False,
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                agent_metrics=agent_metrics,
            )
        except Exception as error:
            if _is_context_limit_error(error):
                # Reaching the configured context budget is a normal policy
                # limit, not an infrastructure failure. Grade the current
                # sandbox state just like mini-SWE-Agent's step limit.
                context_limit_exceeded = True
                result = {
                    "exit_status": "LimitsExceeded",
                    "submission": "",
                }
                logger.debug(
                    "SWE-Gym context limit reached for %s; running verifier",
                    task_dir.name,
                )
            else:
                logger.warning(
                    "SWE-Gym agent failed for %s: %s: %s",
                    task_dir.name,
                    type(error).__name__,
                    str(error)[:500],
                )
                elapsed = time.perf_counter() - started
                reason = "sandbox_not_found" if _is_sandbox_not_found_error(error) else "agent_error"
                agent_metrics = _environment_metrics(
                    env,
                    agent_queue_time=agent_queue_time,
                    agent_dispatch_queue_time=dispatch_queue_time,
                    total_time=elapsed,
                )
                _attach_client_model_timings(
                    agent_metrics,
                    client_model_request_durations,
                )
                return _failure(
                    reason,
                    total_time=elapsed,
                    agent_queue_time=agent_queue_time,
                    agent_error=f"{type(error).__name__}: {error}"[:1000],
                    agent_metrics=agent_metrics,
                )

        agent_snapshot = _EnvironmentSnapshot.capture(env)
        verify_started = time.perf_counter()
        if phase_callback is not None:
            phase_callback("verification")
        try:
            verifier = run_verifier(
                env,
                task_dir,
                configured_timeout=int(settings["verify_timeout"]),
            )
        except SandboxCommandTimeoutError as error:
            diagnostic = ""
            if error.result is not None:
                diagnostic = (error.result.output_tail if error.result.output_truncated else error.result.output)[-_VERIFIER_LOG_TAIL_CHARS:]
            logger.warning(
                "SWE-Gym verifier timed out for %s after %ss",
                task_dir.name,
                verifier_timeout,
            )
            elapsed = time.perf_counter() - started
            metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_snapshot=agent_snapshot,
            )
            metrics["verifier_timeout"] = 1
            metrics["context_limit_exceeded"] = int(context_limit_exceeded)
            _attach_client_model_timings(
                metrics,
                client_model_request_durations,
            )
            return _failure(
                "verifier_timeout",
                # The verifier executes the repository state produced by the
                # policy. Making tests exceed the benchmark deadline is a
                # policy outcome, not missing infrastructure; preserve the
                # trajectory and train it with zero reward.
                infrastructure=False,
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                verifier_timeout_sec=_verifier_timeout(task_dir, int(settings["verify_timeout"])),
                verifier_output_tail=diagnostic,
                agent_metrics=metrics,
            )
        verify_time = time.perf_counter() - verify_started
        reward = verifier["reward"]
        if reward is None:
            logger.error(
                "SWE-Gym verifier produced no reward for %s (rc=%s, timeout=%ss)",
                task_dir.name,
                verifier["return_code"],
                verifier["timeout_sec"],
            )
            elapsed = time.perf_counter() - started
            metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_snapshot=agent_snapshot,
            )
            metrics["verifier_return_code"] = verifier["return_code"]
            metrics["verifier_reward_missing"] = 1
            metrics["context_limit_exceeded"] = int(context_limit_exceeded)
            _attach_client_model_timings(
                metrics,
                client_model_request_durations,
            )
            return _failure(
                "verifier_infra_error",
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                verifier_return_code=verifier["return_code"],
                verifier_timeout_sec=verifier["timeout_sec"],
                verifier_output_tail=verifier["output_tail"],
                agent_metrics=metrics,
            )

        elapsed = time.perf_counter() - started
        agent_metrics = _environment_metrics(
            env,
            agent_queue_time=agent_queue_time,
            agent_dispatch_queue_time=dispatch_queue_time,
            total_time=elapsed,
            agent_snapshot=agent_snapshot,
        )
        agent_metrics["turns"] = agent.n_calls
        agent_metrics["eval_time"] = verify_time
        agent_metrics["context_limit_exceeded"] = int(context_limit_exceeded)
        agent_metrics["verifier_return_code"] = verifier["return_code"]
        agent_metrics["verifier_reward_missing"] = 0
        agent_metrics["verifier_timeout"] = 0
        _attach_client_model_timings(
            agent_metrics,
            client_model_request_durations,
        )
        return {
            "reward": reward,
            "exit_status": result.get("exit_status", "completed"),
            "eval_report": {
                "reward": reward,
                "verifier_return_code": verifier["return_code"],
            },
            "verifier_return_code": verifier["return_code"],
            "verifier_timeout_sec": verifier["timeout_sec"],
            "agent_metrics": agent_metrics,
        }
    finally:
        if phase_callback is not None:
            phase_callback("cleanup")
        env.stop()


def _threads_per_agent_process() -> int:
    return int(
        os.getenv(
            "MODAL_SWEGYM_AGENT_THREADS_PER_PROCESS",
            os.getenv("MODAL_SWEGYM_AGENT_THREADS", "32"),
        )
    )


def _agent_process_count() -> int:
    return int(os.getenv("MODAL_SWEGYM_AGENT_PROCESSES", "1"))


@cache
def _local_agent_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_threads_per_agent_process(),
        thread_name_prefix="modal-swegym",
    )


class _AgentWorker:
    """One Ray process with its own Modal SDK event loop and bounded thread fan-out."""

    def __init__(self, worker_index: int, threads: int) -> None:
        self.worker_index = worker_index
        self.threads = threads
        self.executor = ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix=f"modal-swegym-{worker_index}",
        )
        self._phase_lock = threading.Lock()
        self._episode_phases: dict[str, tuple[str, float, float]] = {}

    async def ping(self) -> dict[str, Any]:
        """Prove the controller actor imported and started before rollout."""
        return {
            "worker_index": self.worker_index,
            "threads": self.threads,
            "pid": os.getpid(),
        }

    async def stats(self) -> dict[str, Any]:
        """Return a cheap live phase snapshot while executor threads are busy."""
        now = time.monotonic()
        with self._phase_lock:
            entries = list(self._episode_phases.values())
        phase_counts = Counter(phase for phase, _, _ in entries)
        oldest_phase_seconds: dict[str, float] = {}
        for phase, phase_started, _ in entries:
            oldest_phase_seconds[phase] = max(
                oldest_phase_seconds.get(phase, 0.0),
                now - phase_started,
            )
        return {
            "worker_index": self.worker_index,
            "active": len(entries),
            "phase_counts": dict(phase_counts),
            "oldest_phase_seconds": oldest_phase_seconds,
            "oldest_episode_seconds": max(
                (now - episode_started for _, _, episode_started in entries),
                default=0.0,
            ),
        }

    async def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        dispatch_queue_time = max(
            0.0,
            time.time() - float(payload.pop("submitted_at_unix")),
        )
        queued_at = time.perf_counter()
        episode_id = uuid.uuid4().hex
        episode_started = time.monotonic()
        current_phase = "executor_queue"
        phase_started = episode_started
        phase_durations: Counter[str] = Counter()

        def set_phase(phase: str) -> None:
            nonlocal current_phase, phase_started
            now = time.monotonic()
            phase_durations[current_phase] += max(0.0, now - phase_started)
            current_phase = phase
            phase_started = now
            with self._phase_lock:
                self._episode_phases[episode_id] = (
                    phase,
                    phase_started,
                    episode_started,
                )

        with self._phase_lock:
            self._episode_phases[episode_id] = (
                current_phase,
                phase_started,
                episode_started,
            )
        result = None
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                partial(
                    _run_episode_sync,
                    **payload,
                    queued_at=queued_at,
                    dispatch_queue_time=dispatch_queue_time,
                    phase_callback=set_phase,
                ),
            )
        finally:
            now = time.monotonic()
            phase_durations[current_phase] += max(0.0, now - phase_started)
            with self._phase_lock:
                self._episode_phases.pop(episode_id, None)
        if isinstance(result, dict):
            metrics = result.setdefault("agent_metrics", {})
            metrics["agent_worker_index"] = self.worker_index
            for phase, duration in phase_durations.items():
                metrics[f"phase_{phase}_seconds"] = duration
            metrics["phase_accounted_seconds"] = sum(phase_durations.values())
        return result


class _RayAgentWorkerPool:
    """Load-balanced handles for independent Modal-controller processes."""

    def __init__(self, workers: list[Any]) -> None:
        self.workers = workers
        self.in_flight = [0] * len(workers)
        self.next_tie_break = 0
        self.progress_reporter: asyncio.Task | None = None

    def _acquire(self) -> tuple[int, Any]:
        minimum = min(self.in_flight)
        for offset in range(len(self.workers)):
            index = (self.next_tie_break + offset) % len(self.workers)
            if self.in_flight[index] == minimum:
                self.in_flight[index] += 1
                self.next_tie_break = (index + 1) % len(self.workers)
                return index, self.workers[index]
        raise AssertionError("Agent worker pool is unexpectedly empty")

    def _release(self, index: int) -> None:
        self.in_flight[index] -= 1
        assert self.in_flight[index] >= 0

    async def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        index, worker = self._acquire()
        future = asyncio.ensure_future(worker.run_episode.remote(payload))

        def release_when_finished(completed: asyncio.Future) -> None:
            # Retrieve failures even when the caller was cancelled so asyncio
            # does not report an unobserved task exception.
            if not completed.cancelled():
                completed.exception()
            self._release(index)

        future.add_done_callback(release_when_finished)
        try:
            # Cancelling an asyncio waiter cannot stop the executor thread in
            # the Ray actor. Keep the remote call alive and retain its capacity
            # accounting until the episode actually exits.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            raise

    def ensure_progress_reporter(self) -> None:
        if self.progress_reporter is None or self.progress_reporter.done():
            self.progress_reporter = asyncio.create_task(self._report_progress())

    async def _report_progress(self) -> None:
        """Log one aggregate phase heartbeat for the complete controller pool."""
        while True:
            await asyncio.sleep(30)
            stats_tasks = [asyncio.ensure_future(worker.stats.remote()) for worker in self.workers]
            done, pending = await asyncio.wait(stats_tasks, timeout=5)
            for task in pending:
                task.cancel()
            snapshots = []
            for task in done:
                try:
                    snapshots.append(task.result())
                except Exception as error:
                    logger.warning(
                        "One Modal agent-pool stats shard failed: %s",
                        error,
                    )
            if not snapshots:
                logger.warning(
                    "Modal agent-pool phase metrics unavailable from all %d workers",
                    len(self.workers),
                )
                continue

            phases: Counter[str] = Counter()
            oldest_by_phase: dict[str, float] = {}
            for snapshot in snapshots:
                phases.update(snapshot["phase_counts"])
                for phase, seconds in snapshot["oldest_phase_seconds"].items():
                    oldest_by_phase[phase] = max(
                        oldest_by_phase.get(phase, 0.0),
                        float(seconds),
                    )
            dispatched = sum(self.in_flight)
            logger.info(
                "Modal agent-pool progress: responsive_workers=%d/%d dispatched=%d "
                "active=%d capacity=%d phases=%s oldest_phase_seconds=%s "
                "oldest_episode=%.1fs",
                len(snapshots),
                len(self.workers),
                dispatched,
                sum(phases.values()),
                len(self.workers) * _threads_per_agent_process(),
                dict(sorted(phases.items())),
                {phase: round(seconds, 1) for phase, seconds in sorted(oldest_by_phase.items())},
                max(
                    (float(snapshot["oldest_episode_seconds"]) for snapshot in snapshots),
                    default=0.0,
                ),
            )


@cache
def _ray_agent_pool() -> _RayAgentWorkerPool:
    import ray

    process_count = _agent_process_count()
    threads = _threads_per_agent_process()
    if process_count <= 1:
        raise ValueError("Ray agent pool requires more than one process")
    remote_worker = ray.remote(_AgentWorker)
    workers = [
        remote_worker.options(
            num_cpus=1,
            # Reserve actor concurrency for ping/stats while all executor
            # slots are occupied by episodes. This does not increase
            # episode concurrency; the thread pool remains the hard cap.
            max_concurrency=threads + 2,
            max_pending_calls=(threads + 2) * 2,
            scheduling_strategy="SPREAD",
        ).remote(index, threads)
        for index in range(process_count)
    ]
    # Actor construction is asynchronous. Resolve one cheap call on every
    # worker so an import/scheduling failure is surfaced once at pool startup
    # rather than converting early trajectories into infrastructure failures.
    ready = ray.get([worker.ping.remote() for worker in workers])
    pool = _RayAgentWorkerPool(workers)
    logger.info(
        "Started %s Modal agent-controller processes with %s threads each (pids=%s)",
        process_count,
        threads,
        [item["pid"] for item in ready],
    )
    return pool


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run one SWE-Gym episode without blocking the rollout event loop."""
    del kwargs
    queued_at = time.perf_counter()
    payload = {
        "base_url": base_url,
        "prompt": prompt,
        "request_kwargs": request_kwargs or {},
        "metadata": metadata or {},
        "submitted_at_unix": time.time(),
    }
    try:
        if _agent_process_count() > 1:
            agent_pool = _ray_agent_pool()
            agent_pool.ensure_progress_reporter()
            episode = agent_pool.run_episode(payload)
        else:
            payload.pop("submitted_at_unix")
            episode = asyncio.get_running_loop().run_in_executor(
                _local_agent_executor(),
                partial(
                    _run_episode_sync,
                    **payload,
                    queued_at=queued_at,
                ),
            )
        # The mini-swe-agent wall limit, per-command/verifier deadlines, and
        # Modal Sandbox lifetime already bound every blocking phase. A second
        # asyncio timeout cannot stop the worker thread and would release pool
        # accounting while its sandbox kept running.
        return await episode
    except Exception as error:
        logger.warning(
            "Modal SWE-Gym episode failed: %s: %s",
            type(error).__name__,
            str(error)[:500],
        )
        return _failure(
            "sandbox_infra_error",
            total_time=time.perf_counter() - queued_at,
            sandbox_error=f"{type(error).__name__}: {error}"[:1000],
        )
