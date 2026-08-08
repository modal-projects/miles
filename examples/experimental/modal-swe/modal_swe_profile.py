"""Standalone throughput profiler for the Stitch-backed Modal SWE rollout path.

This app intentionally runs rollout generation only.  It sends rollout traffic
through the pool's ``Router`` while discovering and scraping the underlying
``Server`` replicas directly.  It uses the same Miles dataset, TITO session
servers, mini-swe-agent adapter, reward hook, and fully asynchronous producer as
the GLM-5.2 Stitch experiment, without starting a trainer or publishing weights.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from argparse import Namespace
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import modal

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "miles.modal_swe_rollout_profile.v1"
TARGET_ENVIRONMENT = "stitch-dev"
TARGET_APP = "stitch-glm5-2-nvfp4-lbtrain1"
TARGET_ROUTER_CLASS = "Router"
TARGET_SERVER_CLASS = "Server"
TARGET_RUN_ID = "lbtrain1"
# Match the fixed B200+ rollout fleet's Flash target, leaving four request slots
# per replica between normal admission and SGLang's max-running limit of 16.
EXPECTED_REPLICAS = 32
TARGET_REQUESTS_PER_REPLICA = 12
EXPERIMENT_VOLUME = "stitch-miles-glm5-2-nvfp4"
CHECKPOINT = "/checkpoints/glm5-2-nvfp4"
PROMPT_DATA = "/data/swebench-pro/test.jsonl"
TASKS_DIR = "/data/swebench-pro/tasks"
BASE_POINTER = "lbtrain1/weight_v000000"
PROFILE_APP_NAME = "miles-swe-rollout-profile"
PROFILE_IMAGE_TAG = "radixark/miles:dev-202607290235"
PROFILE_TIMEOUT_SECONDS = 24 * 60 * 60
SESSION_PORT_START = 40_000
STEADY_ACTIVE_FRACTION = 0.90
FLEET_COUNTERS = {
    "generation_tokens": "sglang:generation_tokens_total",
    "prompt_tokens": "sglang:prompt_tokens_total",
    "requests": "sglang:num_requests_total",
}


def _source_root() -> Path:
    """Find the local checkout while remaining importable in Modal containers."""
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / "miles").is_dir() and (parent / "examples").is_dir():
            return parent
    # Modal imports this module as /root/modal_swe_profile.py. Image definitions
    # are already hydrated there, so this fallback only needs to be well-formed.
    return Path("/root/miles")


_SOURCE_ROOT = _source_root()
_EXACT_AGENT_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.1",
)
# The GLM config's three FlashInfer install commands support CUDA training-side
# conversion. This controller is CPU-only and never imports that stack; the
# already-deployed Server image owns its inference-time FlashInfer build.


def _profile_image() -> modal.Image:
    """Build the CPU controller from the experiment's dated trainer base."""
    return (
        modal.Image.from_registry(PROFILE_IMAGE_TAG)
        .entrypoint([])
        .pip_install(*_EXACT_AGENT_PACKAGES)
        .env(
            {
                "PYTHONPATH": ("/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/modal-swe"),
                "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
                "AGENT_MODEL_NAME": "model",
                "MSWEA_SILENT_STARTUP": "1",
                "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
                "LITELLM_LOG": "ERROR",
                "MODAL_SWE_TASKS_DIR": TASKS_DIR,
                "MODAL_SWE_SANDBOX_APP": "glm5-2-nvfp4-swebench-pro-sandbox",
                "MODAL_SWE_EXEC_TIMEOUT": "120",
                "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES": str(16 * 1024 * 1024),
                "MODAL_SWE_SETUP_TIMEOUT": "600",
                "MODAL_SWE_VERIFY_TIMEOUT": "3600",
                "MODAL_SWE_INJECT_PYTEST_REPORTER": "0",
                "MODAL_SWE_CPUS": "2",
                "MODAL_SWE_MEMORY_MIB": "16384",
            }
        )
        .add_local_dir(
            _SOURCE_ROOT,
            remote_path="/root/miles",
            copy=True,
            ignore=[".git", ".venv", "**/__pycache__", "**/*.pyc"],
        )
    )


app = modal.App(PROFILE_APP_NAME)
profile_image = _profile_image()
control_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "httpx==0.28.1",
    "modal==1.5.1",
)
data_volume = modal.Volume.from_name("miles-data", environment_name=TARGET_ENVIRONMENT, version=2)
checkpoint_volume = modal.Volume.from_name("miles-checkpoints", environment_name=TARGET_ENVIRONMENT, version=2)
experiment_volume = modal.Volume.from_name(EXPERIMENT_VOLUME, environment_name=TARGET_ENVIRONMENT, version=2)
profile_volumes = {
    "/data": data_volume.read_only(),
    "/checkpoints": checkpoint_volume.read_only(),
}


@dataclass(frozen=True)
class ProfileRequest:
    mode: str
    target_app: str
    target_router_class: str
    target_server_class: str
    target_environment: str
    groups_per_step: int
    samples_per_prompt: int
    warmup_steps: int
    measure_steps: int
    max_groups: int
    all_dataset: bool
    concurrency: int
    session_servers: int
    controller_processes: int
    controller_threads: int
    max_agent_steps: int
    episode_timeout: int
    overall_timeout: int
    preflight_only: bool


class AdmissionBudgetExhausted(RuntimeError):
    """The finite profiler producer cannot supply another accepted group."""


def _finite_rollout_class():
    """Create the bounded subclass only inside the dependency-complete image."""
    from miles.rollout.base_types import RolloutFnTrainInput
    from miles.rollout.failures import is_non_retryable_failure
    from miles.rollout.fully_async_rollout import (
        FullyAsyncRolloutFn,
        _iter_samples,
        _mask_non_retryable_failures,
    )

    class FiniteAdmissionFullyAsyncRolloutFn(FullyAsyncRolloutFn):
        def __init__(self, input: Any, *, admission_budget: int):
            super().__init__(input)
            if admission_budget < 1:
                raise ValueError("admission_budget must be positive")
            self._admission_budget = admission_budget
            self._max_additional_submissions = max(32, math.ceil(admission_budget * 0.10))
            self._fresh_groups_submitted = 0
            self._retry_groups_submitted = 0
            self._replacement_slots = 0
            self._rejected_candidates: list[dict[str, Any]] = []
            self._attempted_instances: set[str] = set()
            self._admission_changed = asyncio.Event()
            self._drain_in_progress = False
            self._predicted_accepted = 0
            self._quiesced = asyncio.Event()
            self._hold_worker_open = asyncio.Event()

        def _buffered_groups(self) -> int:
            return int(self.data_source.get_buffer_length() or 0)

        def _fresh_groups_remaining(self) -> int:
            return self._admission_budget + self._replacement_slots - self._fresh_groups_submitted

        def _has_admission(self) -> bool:
            return self._buffered_groups() > 0 or self._fresh_groups_remaining() > 0

        def _submit_bounded_group(self) -> asyncio.Task:
            retry = self._buffered_groups() > 0
            additional = self._retry_groups_submitted + max(
                0,
                self._fresh_groups_submitted - self._admission_budget,
            )
            if (retry or self._fresh_groups_submitted >= self._admission_budget) and (additional >= self._max_additional_submissions):
                raise AdmissionBudgetExhausted(f"finite profiler exceeded its retry/replacement allowance ({additional}/{self._max_additional_submissions})")

            samples = self.data_source.get_samples(1)
            self._scheduler.on_submit(samples)
            [prompt_group] = samples
            self._groups_submitted += 1
            if retry:
                self._retry_groups_submitted += 1
            else:
                self._fresh_groups_submitted += 1
            instance_id = str(prompt_group[0].metadata.get("instance_id") or "")
            if instance_id:
                self._attempted_instances.add(instance_id)
            return asyncio.create_task(self._generate_group(prompt_group))

        async def _worker_loop(self) -> None:
            active: set[asyncio.Task] = set()
            try:
                while True:
                    self._scheduler.arm()
                    capacity = self._scheduler.available_group_slots(
                        pending_groups=len(active),
                        group_budget=self._max_in_flight_groups(),
                    )
                    for _ in range(capacity):
                        if not self._has_admission():
                            break
                        active.add(self._submit_bounded_group())

                    if active:
                        done, active = await self._scheduler.wait_for_progress(active)
                        for task in done:
                            await self._output.put(task.result())
                        continue

                    if self._has_admission():
                        raise RuntimeError("finite rollout scheduler reported no capacity without active groups")
                    if self._groups_dequeued < self._groups_finished:
                        self._admission_changed.clear()
                        if self._has_admission() or self._groups_dequeued >= self._groups_finished:
                            continue
                        await self._admission_changed.wait()
                        continue
                    if self._drain_in_progress:
                        self._admission_changed.clear()
                        if self._has_admission() or not self._drain_in_progress:
                            continue
                        await self._admission_changed.wait()
                        continue
                    self._quiesced.set()
                    await self._hold_worker_open.wait()
                    raise AssertionError("finite rollout worker was released unexpectedly")
            finally:
                for task in active:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)

        def _candidate_disposition(self, completed: Any) -> str:
            aborted = [sample for sample in _iter_samples(completed.group) if sample.status == sample.Status.ABORTED]
            if not aborted:
                return "accept"
            if any(not is_non_retryable_failure(sample) for sample in aborted):
                return "retry"
            masked, _, _ = _mask_non_retryable_failures(completed.group)
            return "replace" if masked is None else "accept"

        def _record_rejected_candidate(self, completed: Any) -> None:
            self._replacement_slots += 1
            if self._replacement_slots > self._max_additional_submissions:
                raise AdmissionBudgetExhausted(f"finite profiler exceeded its terminal-group replacement allowance ({self._replacement_slots}/{self._max_additional_submissions})")
            samples = list(_iter_samples(completed.group))
            self._rejected_candidates.append(
                {
                    "instance_id": next(
                        (sample.metadata.get("instance_id") for sample in samples if sample.metadata),
                        None,
                    ),
                    "sample_indices": [sample.index for sample in samples],
                    "group_index": samples[0].group_index if samples else None,
                    "reason": "non_retryable_group_dropped",
                }
            )

        async def _next_group(self):
            queue_get = asyncio.create_task(self._output.get())
            quiesced = asyncio.create_task(self._quiesced.wait())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {queue_get, quiesced, self._worker},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=30.0,
                    )
                    if self._worker in done:
                        self._worker.result()
                        raise RuntimeError("finite rollout worker exited unexpectedly")
                    if queue_get in done:
                        self._groups_dequeued += 1
                        completed = queue_get.result()
                        disposition = self._candidate_disposition(completed)
                        if disposition == "replace":
                            self._record_rejected_candidate(completed)
                        elif disposition == "accept":
                            self._predicted_accepted += 1
                        self._admission_changed.set()
                        return completed
                    if quiesced in done:
                        await asyncio.sleep(0)
                        if queue_get.done():
                            continue
                        if self._output.empty():
                            raise AdmissionBudgetExhausted("finite admission budget was exhausted before the requested accepted groups; inspect dropped failures")
                    if not done:
                        logger.warning(
                            "No finite-profile group completed for 30s (submitted=%d finished=%d dequeued=%d)",
                            self._groups_submitted,
                            self._groups_finished,
                            self._groups_dequeued,
                        )
            finally:
                for task in (queue_get, quiesced):
                    if not task.done():
                        task.cancel()

        def _recycle(self, prompt_group: list[Any]) -> None:
            super()._recycle(prompt_group)
            self._admission_changed.set()

        async def drain_step(self, *, rollout_id: int, target_groups: int):
            """Drain one full or final-partial learner batch."""
            if target_groups < 1 or target_groups > self.args.rollout_batch_size:
                raise ValueError(f"target_groups must be in [1, {self.args.rollout_batch_size}], got {target_groups}")
            original = self.args.rollout_batch_size
            rejected_before = len(self._rejected_candidates)
            self.args.rollout_batch_size = target_groups
            self._drain_in_progress = True
            self._predicted_accepted = 0
            try:
                output = await self(RolloutFnTrainInput(rollout_id=rollout_id))
                if self._predicted_accepted != target_groups:
                    raise RuntimeError(f"finite profiler candidate accounting diverged from drain output ({self._predicted_accepted}/{target_groups})")
                output.metrics.update(
                    {
                        "rollout_profile/fresh_groups_submitted": self._fresh_groups_submitted,
                        "rollout_profile/retry_groups_submitted": self._retry_groups_submitted,
                        "rollout_profile/replacement_groups_submitted": max(
                            0,
                            self._fresh_groups_submitted - self._admission_budget,
                        ),
                        "rollout_profile/rejected_candidate_groups": len(self._rejected_candidates),
                        "rollout_profile/unique_instances_attempted": len(self._attempted_instances),
                    }
                )
                output.profile_rejected_candidates = self._rejected_candidates[rejected_before:]
                return output
            finally:
                self._drain_in_progress = False
                self._admission_changed.set()
                self.args.rollout_batch_size = original

        def active_trajectories(self) -> int:
            return int(self._scheduler.samples_in_flight)

        def admission_stats(self) -> dict[str, Any]:
            settled = self._groups_submitted == self._groups_finished == self._groups_dequeued and self._buffered_groups() == 0 and self.active_trajectories() == 0
            return {
                "dataset_groups": self._admission_budget,
                "groups_submitted": self._groups_submitted,
                "groups_finished": self._groups_finished,
                "groups_dequeued": self._groups_dequeued,
                "fresh_groups_submitted": self._fresh_groups_submitted,
                "retry_groups_submitted": self._retry_groups_submitted,
                "replacement_groups_submitted": max(
                    0,
                    self._fresh_groups_submitted - self._admission_budget,
                ),
                "rejected_candidate_groups": len(self._rejected_candidates),
                "unique_instances_attempted": len(self._attempted_instances),
                "buffered_groups": self._buffered_groups(),
                "active_trajectories": self.active_trajectories(),
                "settled": settled,
                "terminal_replacement_policy": "next_epoch_dataset_draw",
            }

        async def wait_until_quiesced(self) -> None:
            quiesced = asyncio.create_task(self._quiesced.wait())
            try:
                done, _ = await asyncio.wait(
                    {quiesced, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("finite rollout worker exited before quiescing")
                await quiesced
            finally:
                if not quiesced.done():
                    quiesced.cancel()

    return FiniteAdmissionFullyAsyncRolloutFn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _at_base(info: dict[str, Any]) -> bool:
    applied = info.get("applied")
    return applied in (BASE_POINTER, 0, "0", "weight_v000000")


async def _resolve_gateway(
    target_app: str,
    target_router_class: str,
    target_environment: str,
) -> str:
    server = modal.Server.from_name(
        target_app,
        target_router_class,
        environment_name=target_environment,
    )
    url = await server.get_url.aio()
    if not url:
        raise RuntimeError(f"{target_app}.{target_router_class} has no deployed gateway URL")
    return str(url).rstrip("/")


def _discover_server_containers(
    target_app: str,
    target_server_class: str,
    target_environment: str,
) -> dict[str, Any]:
    """Isolate Flash discovery from the active ``modal run`` client."""
    helper = Path(__file__).with_name("modal_swe_replicas.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--app",
            target_app,
            "--server",
            target_server_class,
            "--environment",
            target_environment,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or not isinstance(result.get("containers"), list):
        raise TypeError("direct-replica discovery returned an invalid payload")
    return result


def _container_host(container: Any) -> str | None:
    host = container.get("host") if isinstance(container, dict) else getattr(container, "host", None)
    if not host:
        return None
    host = str(host).rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"


async def _fetch_server_info(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    response = await client.get(f"{url}/server_info")
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"server_info from {url} was not an object")
    return data


async def _wake_replicas(
    client: httpx.AsyncClient,
    replicas: list[str],
) -> None:
    async def wake(url: str) -> None:
        response = await client.post(f"{url}/wake")
        response.raise_for_status()

    results = await asyncio.gather(*(wake(url) for url in replicas), return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"failed to wake {len(failures)}/{len(replicas)} Stitch replicas: {type(failures[0]).__name__}: {failures[0]}")


async def _fetch_replica_infos(
    client: httpx.AsyncClient,
    replicas: list[str],
    *,
    tolerate_failures: bool = False,
) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *(_fetch_server_info(client, url) for url in replicas),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures and not tolerate_failures:
        raise RuntimeError(f"failed to read {len(failures)}/{len(replicas)} replicas: {type(failures[0]).__name__}: {failures[0]}")
    if failures:
        print(
            json.dumps(
                {
                    "event": "replica_preflight_partial",
                    "replicas_requested": len(replicas),
                    "replicas_read": len(replicas) - len(failures),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return [result for result in results if isinstance(result, dict)]


async def _wait_for_base(
    *,
    client: httpx.AsyncClient,
    gateway: str,
    replicas: list[str],
    timeout_seconds: int,
    min_ready_replicas: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    min_ready_replicas = len(replicas) if min_ready_replicas is None else min_ready_replicas
    deadline = time.monotonic() + timeout_seconds
    last_log = 0.0
    while True:
        replica_infos = await _fetch_replica_infos(
            client,
            replicas,
            tolerate_failures=min_ready_replicas < len(replicas),
        )
        gateway_info = await _fetch_server_info(client, gateway)
        ready = len(replica_infos) >= min_ready_replicas and all(_at_base(info) and info.get("sync_state") != "ERROR" and int(info.get("active_requests") or 0) == 0 for info in replica_infos) and _at_base(gateway_info)
        if ready:
            return gateway_info, replica_infos

        now = time.monotonic()
        if now >= deadline:
            at_base = sum(_at_base(info) for info in replica_infos)
            busy = sum(int(info.get("active_requests") or 0) > 0 for info in replica_infos)
            raise TimeoutError(f"Stitch pool did not converge to the idle base revision: replicas_at_base={at_base}/{len(replicas)}, busy={busy}, gateway_applied={gateway_info.get('applied')!r}")
        if now - last_log >= 10.0:
            print(
                json.dumps(
                    {
                        "event": "claim_progress",
                        "replicas_at_base": sum(_at_base(info) for info in replica_infos),
                        "replicas_total": len(replicas),
                        "gateway_applied": gateway_info.get("applied"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_log = now
        await asyncio.sleep(2.0)


@app.function(
    image=control_image,
    cpu=1,
    memory=1024,
    region="us",
    volumes={"/stitch": experiment_volume},
    timeout=15 * 60,
)
async def claim_base() -> dict[str, Any]:
    """Idempotently commit the fixed profiler base pointer."""
    started = _utc_now()
    await experiment_volume.reload.aio()
    pointer = Path("/stitch") / TARGET_RUN_ID / "latest"
    previous = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None
    if previous not in (None, BASE_POINTER):
        raise RuntimeError(f"refusing to overwrite non-base Stitch pointer {previous!r}; expected absent or {BASE_POINTER!r}")

    committed = previous is None
    if committed:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=pointer.parent, prefix=".profile-claim-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(BASE_POINTER)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, pointer)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        await experiment_volume.commit.aio()
    return {
        "started_at": started,
        "finished_at": _utc_now(),
        "action": "claimed" if committed else "validated_existing_base",
        "pointer_before": previous,
        "pointer_after": BASE_POINTER,
        "committed": committed,
    }


@app.function(
    image=control_image,
    cpu=1,
    memory=1024,
    region="us",
    volumes={"/stitch": experiment_volume},
    timeout=5 * 60,
)
async def release_base() -> dict[str, Any]:
    """Remove only the exact profiler base pointer; never overwrite training state."""
    await experiment_volume.reload.aio()
    pointer = Path("/stitch") / TARGET_RUN_ID / "latest"
    previous = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None
    if previous is None:
        return {"action": "already_absent", "pointer_before": None, "released": False}
    if previous != BASE_POINTER:
        raise RuntimeError(f"refusing to remove non-base Stitch pointer {previous!r}; expected {BASE_POINTER!r}")
    pointer.unlink()
    await experiment_volume.commit.aio()
    return {
        "action": "released_base",
        "pointer_before": previous,
        "released": True,
        "finished_at": _utc_now(),
    }


def _validate_request(request: ProfileRequest) -> None:
    if (
        request.target_app,
        request.target_router_class,
        request.target_server_class,
        request.target_environment,
    ) != (
        TARGET_APP,
        TARGET_ROUTER_CLASS,
        TARGET_SERVER_CLASS,
        TARGET_ENVIRONMENT,
    ):
        raise ValueError(f"this profiler is hard-locked to {TARGET_ENVIRONMENT}:{TARGET_APP} (router={TARGET_ROUTER_CLASS}, server={TARGET_SERVER_CLASS})")
    positive = {
        "groups_per_step": request.groups_per_step,
        "samples_per_prompt": request.samples_per_prompt,
        "concurrency": request.concurrency,
        "session_servers": request.session_servers,
        "controller_processes": request.controller_processes,
        "controller_threads": request.controller_threads,
        "max_agent_steps": request.max_agent_steps,
        "episode_timeout": request.episode_timeout,
        "overall_timeout": request.overall_timeout,
        "measure_steps": request.measure_steps,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"profile values must be positive: {invalid}")
    if request.warmup_steps < 0 or request.max_groups < 0:
        raise ValueError("warmup_steps and max_groups must be non-negative")
    if request.all_dataset and request.max_groups:
        raise ValueError("--all-dataset and --max-groups are mutually exclusive")
    if request.concurrency < request.samples_per_prompt:
        raise ValueError("concurrency must admit at least one complete prompt group")
    if request.concurrency % request.samples_per_prompt:
        raise ValueError("concurrency must be divisible by samples_per_prompt")
    if request.mode not in {"canary", "scale"}:
        raise ValueError("mode must be 'canary' or 'scale'")


def _step_targets(
    request: ProfileRequest,
    *,
    dataset_groups: int,
) -> list[int]:
    if request.all_dataset:
        total = dataset_groups
    elif request.max_groups:
        total = request.max_groups
    else:
        total = request.groups_per_step * (request.warmup_steps + request.measure_steps)
        if request.mode == "scale":
            total += request.concurrency // request.samples_per_prompt
    if total > dataset_groups:
        raise ValueError(f"requested {total} unique groups, but dataset contains {dataset_groups}; the profiler never wraps or pads the dataset")
    targets = [request.groups_per_step] * (total // request.groups_per_step)
    if tail := total % request.groups_per_step:
        targets.append(tail)
    if request.warmup_steps >= len(targets):
        raise ValueError(f"warmup_steps={request.warmup_steps} leaves no measured step for {len(targets)} planned steps")
    return targets


def _step_phases(
    request: ProfileRequest,
    targets: list[int],
) -> list[str]:
    """Separate steady-state measurements from warmup and fleet drain-down."""
    phases = ["measure"] * len(targets)
    for step in range(request.warmup_steps):
        phases[step] = "warmup"

    if request.mode != "scale":
        return phases
    if not (request.all_dataset or request.max_groups):
        measure_end = request.warmup_steps + request.measure_steps
        for step in range(measure_end, len(phases)):
            phases[step] = "cooldown"
        return phases

    reserve = request.concurrency // request.samples_per_prompt
    cooldown_start = len(targets)
    reserved = 0
    while cooldown_start > request.warmup_steps + 1 and reserved < reserve:
        cooldown_start -= 1
        reserved += targets[cooldown_start]
    if reserved < reserve:
        raise ValueError("fixed group budget is too small to retain one measured step and a full-concurrency cooldown")
    for step in range(cooldown_start, len(phases)):
        phases[step] = "cooldown"
    return phases


def _observed_phase(
    planned_phase: str,
    *,
    require_steady_occupancy: bool,
    active_start: int,
    active_end: int,
    active_limit: int,
) -> str:
    """Exclude underfilled drain boundaries from steady-state aggregates."""
    if planned_phase != "measure" or not require_steady_occupancy:
        return planned_phase
    threshold = math.ceil(active_limit * STEADY_ACTIVE_FRACTION)
    return "measure" if min(active_start, active_end) >= threshold else "cooldown"


def _configure_agent_environment(request: ProfileRequest) -> None:
    settings = {
        "MODAL_SWE_MAX_STEPS": request.max_agent_steps,
        "MODAL_SWE_EPISODE_TIMEOUT": request.episode_timeout,
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": 1800,
        "MODAL_SWE_AGENT_PROCESSES": request.controller_processes,
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": request.controller_threads,
    }
    os.environ.update({key: str(value) for key, value in settings.items()})


def _session_port_range(server_count: int) -> list[int]:
    from miles.utils.http_utils import is_port_available

    for start in range(SESSION_PORT_START, 60_001 - server_count):
        if all(is_port_available(port) for port in range(start, start + server_count)):
            return [start] if server_count == 1 else [start, start + server_count]
    raise RuntimeError(f"could not find {server_count} contiguous ports for session servers")


def _build_rollout_args(request: ProfileRequest, *, gateway: str) -> Namespace:
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    chat_template_path, template_kwargs = resolve_fixed_chat_template("glm47")
    session_ports = _session_port_range(request.session_servers)
    return Namespace(
        apply_chat_template=False,
        apply_chat_template_kwargs=template_kwargs,
        async_max_concurrent_samples=request.concurrency,
        buffer_filter_path=None,
        chat_template_path=chat_template_path,
        custom_agent_function_path="modal_swe_agent_function.run",
        custom_generate_function_path="miles.rollout.generate_hub.agentic_tool_call.generate",
        custom_rm_path="modal_swe_agent_function.reward_func",
        custom_rollout_request_hook_path="modal_swe_profile_hook.profile_rollout_request_hook",
        dump_details=None,
        dynamic_sampling_filter_path=None,
        group_rm=False,
        hf_checkpoint=CHECKPOINT,
        input_key="prompt",
        label_key=None,
        load=None,
        lora_adapter_path=None,
        lora_rank=0,
        mask_offpolicy_in_partial_rollout=False,
        max_seq_len=65_536,
        max_weight_staleness=None,
        metadata_key="metadata",
        miles_router_timeout=600.0,
        moe_router_topk=8,
        multi_lora=False,
        multimodal_keys=None,
        n_samples_per_prompt=request.samples_per_prompt,
        num_layers=78,
        partial_rollout=False,
        prompt_data=PROMPT_DATA,
        reward_key=None,
        rollout_batch_size=request.groups_per_step,
        rollout_endpoint_url=gateway,
        rollout_global_dataset=True,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        rollout_max_response_len=8192,
        rollout_num_gpus=0,
        rollout_num_gpus_per_engine=4,
        rollout_request_retry_attempts=1200,
        rollout_request_retry_sleep=1.0,
        rollout_request_timeout_secs=300.0,
        rollout_sample_completion_backfill=True,
        rollout_sample_filter_path=None,
        rollout_seed=42,
        rollout_session_affinity_header="Modal-Session-ID",
        rollout_shuffle=True,
        rollout_skip_special_tokens=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_temperature=1.0,
        rollout_top_k=-1,
        rollout_top_p=1.0,
        save="/tmp/miles-swe-profile",
        save_debug_trajectory_data=None,
        session_server_ip=None,
        session_server_port=session_ports,
        # Sixty-four spawn children import transformers while 48 Ray actors
        # provide 768 episode slots. This gate is outside measured rollout time.
        session_server_startup_timeout_seconds=600,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip=None,
        sglang_router_policy=None,
        sglang_router_port=None,
        sglang_server_concurrency=request.concurrency,
        sglang_speculative_algorithm=None,
        tito_model="glm47",
        tito_session_mismatch_sample_rate=0.0625,
        tool_key=None,
        use_rollout_indexer_replay=False,
        use_rollout_routing_replay=True,
        use_distributed_post=False,
        use_session_server=True,
    )


def _validate_dataset() -> dict[str, Any]:
    checkpoint = Path(CHECKPOINT)
    prompt_path = Path(PROMPT_DATA)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint mount is missing: {checkpoint}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt dataset is missing: {prompt_path}")

    instance_ids: list[str] = []
    missing: list[str] = []
    with prompt_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            instance_id = str(metadata.get("instance_id") or "")
            task_dir = Path(str(metadata.get("task_dir") or ""))
            if not instance_id or not task_dir.is_dir():
                missing.append(f"line {line_number}: {instance_id or '<no instance_id>'}")
                continue
            required = (
                task_dir / "environment" / "Dockerfile",
                task_dir / "tests" / "test.sh",
            )
            if not all(path.is_file() for path in required):
                missing.append(f"line {line_number}: {instance_id}")
            instance_ids.append(instance_id)
    if missing:
        raise FileNotFoundError(f"{len(missing)} SWE-bench Pro tasks are incomplete; first={missing[:5]}")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("SWE-bench Pro prompt data contains duplicate instance_id values")
    return {
        "checkpoint": str(checkpoint),
        "prompt_data": str(prompt_path),
        "dataset_groups": len(instance_ids),
        "unique_instances": len(set(instance_ids)),
        "task_artifacts_validated": len(instance_ids),
    }


def _flatten(groups: Iterable[list[Any]]) -> list[Any]:
    return [sample for group in groups for sample in group]


def _request_events(samples: Iterable[Any]) -> list[dict[str, Any]]:
    events: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in samples:
        raw_events = sample.metadata.get("model_request/events", [])
        for event in raw_events if isinstance(raw_events, list) else []:
            if not isinstance(event, dict):
                continue
            key = (str(event.get("session_id", "")), int(event.get("request_index", -1)))
            events[key] = event
    return list(events.values())


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [
        (float(event["proxy_started_at_unix"]), float(event["proxy_finished_at_unix"]))
        for event in events
        if isinstance(event.get("proxy_started_at_unix"), (int, float)) and isinstance(event.get("proxy_finished_at_unix"), (int, float)) and float(event["proxy_finished_at_unix"]) >= float(event["proxy_started_at_unix"])
    ]
    window = max((end for _, end in intervals), default=0.0) - min((start for start, _ in intervals), default=0.0)
    interval_sum = sum(end - start for start, end in intervals)
    prompt_tokens = sum(int(event.get("prompt_tokens") or 0) for event in events)
    completion_tokens = sum(int(event.get("completion_tokens") or 0) for event in events)
    statuses = Counter(str(event.get("status_code")) for event in events)
    denominator = window if window > 0 else None
    return {
        "request_count": len(events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "request_event_wall_window_seconds": max(0.0, window),
        "request_event_interval_seconds_sum": interval_sum,
        "request_event_effective_concurrency": interval_sum / denominator if denominator else 0.0,
        "requests_per_request_event_wall_second": len(events) / denominator if denominator else 0.0,
        "prompt_tokens_per_request_event_wall_second": prompt_tokens / denominator if denominator else 0.0,
        "completion_tokens_per_request_event_wall_second": completion_tokens / denominator if denominator else 0.0,
        "total_tokens_per_request_event_wall_second": ((prompt_tokens + completion_tokens) / denominator if denominator else 0.0),
        "status_counts": dict(sorted(statuses.items())),
    }


def _parse_prometheus_counters(text: str) -> dict[str, float]:
    values = {key: 0.0 for key in FLEET_COUNTERS}
    found = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, _, raw_value = line.rpartition(" ")
        for key, name in FLEET_COUNTERS.items():
            if metric == name or metric.startswith(f"{name}{{"):
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite fleet counter {name}={raw_value}")
                values[key] += value
                found.add(key)
    missing = set(FLEET_COUNTERS) - found
    if missing:
        raise KeyError(f"replica /metrics is missing counters: {sorted(missing)}")
    return values


async def _scrape_fleet_counters(
    client: httpx.AsyncClient,
    replicas: list[str],
) -> dict[str, dict[str, float]]:
    async def scrape(url: str) -> tuple[str, dict[str, float]]:
        response = await client.get(f"{url}/metrics")
        response.raise_for_status()
        return url, _parse_prometheus_counters(response.text)

    results = await asyncio.gather(
        *(scrape(url) for url in replicas),
        return_exceptions=True,
    )
    counters: dict[str, dict[str, float]] = {}
    failures = []
    for url, result in zip(replicas, results, strict=True):
        if isinstance(result, BaseException):
            failures.append({"url": url, "error": f"{type(result).__name__}: {result}"})
            continue
        result_url, values = result
        counters[result_url] = values
    if failures:
        print(
            json.dumps(
                {
                    "event": "fleet_counter_scrape_partial",
                    "replicas_requested": len(replicas),
                    "replicas_read": len(counters),
                    "failures": failures,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return counters


def _fleet_counter_delta(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    *,
    boundary_seconds: float,
) -> dict[str, Any]:
    before_replicas = set(before)
    after_replicas = set(after)
    common_replicas = before_replicas & after_replicas
    deltas = {key: 0.0 for key in FLEET_COUNTERS}
    for replica in common_replicas:
        for key in FLEET_COUNTERS:
            delta = after[replica][key] - before[replica][key]
            if delta < 0:
                raise RuntimeError(f"fleet counter reset on {replica}: {key} delta={delta}")
            deltas[key] += delta
    divisor = boundary_seconds if boundary_seconds > 0 else float("inf")
    return {
        "boundary_wall_seconds": boundary_seconds,
        "replicas_before": len(before_replicas),
        "replicas_after": len(after_replicas),
        "replicas_common": len(common_replicas),
        "replicas_lost": len(before_replicas - after_replicas),
        "replicas_added": len(after_replicas - before_replicas),
        "generation_tokens_delta": deltas["generation_tokens"],
        "prompt_tokens_delta": deltas["prompt_tokens"],
        "requests_delta": deltas["requests"],
        "generation_tokens_per_boundary_second": deltas["generation_tokens"] / divisor,
        "prompt_tokens_per_boundary_second": deltas["prompt_tokens"] / divisor,
        "requests_per_boundary_second": deltas["requests"] / divisor,
    }


def _sample_failures(samples: list[Any], *, step: int) -> list[dict[str, Any]]:
    from miles.rollout.failures import is_infrastructure_failure
    from miles.utils.types import Sample

    failures = []
    for sample in samples:
        if sample.status != Sample.Status.ABORTED and not is_infrastructure_failure(sample):
            continue
        failures.append(
            {
                "step": step,
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "instance_id": sample.metadata.get("instance_id"),
                "exit_status": sample.metadata.get("exit_status"),
                "infrastructure_failure": is_infrastructure_failure(sample),
                "non_retryable_failure": bool(sample.metadata.get("_miles_non_retryable_failure")),
            }
        )
    return failures


def _step_report(
    *,
    step: int,
    phase: str,
    planned_phase: str,
    active_start: int,
    active_end: int,
    active_limit: int,
    target_groups: int,
    started_at_unix: float,
    finished_at_unix: float,
    drain_seconds: float,
    fleet_counters: dict[str, Any],
    output: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from modal_swe_metrics import add_metrics

    samples = _flatten(output.samples)
    metrics = dict(output.metrics or {})
    add_metrics(
        samples,
        metrics,
        rollout_time=drain_seconds,
    )
    events = _request_events(samples)
    instance_ids = {str(sample.metadata["instance_id"]) for sample in samples if sample.metadata.get("instance_id") is not None}
    report = {
        "step": step,
        "phase": phase,
        "planned_phase": planned_phase,
        "active_trajectories_start": active_start,
        "active_trajectories_end": active_end,
        "active_trajectory_limit": active_limit,
        "steady_state_active_threshold": math.ceil(active_limit * STEADY_ACTIVE_FRACTION),
        "target_groups": target_groups,
        "accepted_groups": len(output.samples),
        "trajectories": len(samples),
        "unique_instances": len(instance_ids),
        "started_at_unix": started_at_unix,
        "finished_at_unix": finished_at_unix,
        "learner_batch_drain_seconds": drain_seconds,
        "metrics": metrics,
        "request_events": _event_summary(events),
        "fleet_counters": fleet_counters,
    }
    rejected = [{"step": step, **failure} for failure in getattr(output, "profile_rejected_candidates", [])]
    return report, events, [*_sample_failures(samples, step=step), *rejected]


def _summary(
    steps: list[dict[str, Any]],
    measured_events: list[dict[str, Any]],
) -> dict[str, Any]:
    measured = [step for step in steps if step["phase"] == "measure"]
    drain_seconds = sum(float(step["learner_batch_drain_seconds"]) for step in measured)
    groups = sum(int(step["accepted_groups"]) for step in measured)
    trajectories = sum(int(step["trajectories"]) for step in measured)
    fleet_boundary_seconds = sum(float(step["fleet_counters"]["boundary_wall_seconds"]) for step in measured)
    fleet_generation_tokens = sum(float(step["fleet_counters"]["generation_tokens_delta"]) for step in measured)
    fleet_prompt_tokens = sum(float(step["fleet_counters"]["prompt_tokens_delta"]) for step in measured)
    fleet_requests = sum(float(step["fleet_counters"]["requests_delta"]) for step in measured)
    event_summary = _event_summary(measured_events)
    denominator = drain_seconds if drain_seconds > 0 else None
    fleet_divisor = fleet_boundary_seconds if fleet_boundary_seconds > 0 else float("inf")
    return {
        "measured_steps": len(measured),
        "measured_groups": groups,
        "measured_trajectories": trajectories,
        "learner_batch_drain_seconds_sum": drain_seconds,
        "groups_per_learner_batch_drain_second": groups / denominator if denominator else 0.0,
        "trajectories_per_learner_batch_drain_second": trajectories / denominator if denominator else 0.0,
        "step_attributed_requests_per_learner_batch_drain_second": (event_summary["request_count"] / denominator if denominator else 0.0),
        "step_attributed_prompt_tokens_per_learner_batch_drain_second": (event_summary["prompt_tokens"] / denominator if denominator else 0.0),
        "step_attributed_completion_tokens_per_learner_batch_drain_second": (event_summary["completion_tokens"] / denominator if denominator else 0.0),
        "step_attributed_total_tokens_per_learner_batch_drain_second": (event_summary["total_tokens"] / denominator if denominator else 0.0),
        "fleet_boundary_wall_seconds_sum": fleet_boundary_seconds,
        "fleet_generation_tokens_delta": fleet_generation_tokens,
        "fleet_prompt_tokens_delta": fleet_prompt_tokens,
        "fleet_requests_delta": fleet_requests,
        "fleet_generation_tokens_per_boundary_second": fleet_generation_tokens / fleet_divisor,
        "fleet_prompt_tokens_per_boundary_second": fleet_prompt_tokens / fleet_divisor,
        "fleet_requests_per_boundary_second": fleet_requests / fleet_divisor,
        **event_summary,
    }


async def _run_rollouts(
    *,
    request: ProfileRequest,
    gateway: str,
    replicas: list[str],
    targets: list[int],
    phases: list[str],
    steps: list[dict[str, Any]],
    measured_events: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    from miles.rollout.base_types import RolloutFnConstructorInput
    from miles.ray.rollout.router_manager import start_session_server
    from miles.rollout.data_source import RolloutDataSourceWithBuffer
    from miles.utils.http_utils import init_http_client

    _configure_agent_environment(request)
    if request.controller_processes > 1:
        import ray

        ray.init(
            num_cpus=request.controller_processes,
            include_dashboard=False,
            log_to_driver=True,
        )

    args = _build_rollout_args(request, gateway=gateway)
    data_source = RolloutDataSourceWithBuffer(args)
    init_http_client(args)
    await asyncio.to_thread(start_session_server, args)
    finite_rollout = _finite_rollout_class()
    rollout = finite_rollout(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        admission_budget=sum(targets),
    )
    accepted_groups = 0
    accepted_instances: set[str] = set()

    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as metrics_client:
            fleet_before = await _scrape_fleet_counters(metrics_client, replicas)
            boundary_started = time.monotonic()
            for step, target in enumerate(targets):
                planned_phase = phases[step]
                active_start = rollout.active_trajectories()
                started = time.time()
                started_monotonic = time.monotonic()
                output = await rollout.drain_step(rollout_id=step, target_groups=target)
                accepted_groups += len(output.samples)
                accepted_instances.update(str(sample.metadata["instance_id"]) for sample in _flatten(output.samples) if sample.metadata.get("instance_id") is not None)
                drain_seconds = time.monotonic() - started_monotonic
                finished = time.time()
                fleet_after = await _scrape_fleet_counters(
                    metrics_client,
                    replicas,
                )
                boundary_finished = time.monotonic()
                active_end = rollout.active_trajectories()
                phase = _observed_phase(
                    planned_phase,
                    require_steady_occupancy=request.mode == "scale",
                    active_start=active_start,
                    active_end=active_end,
                    active_limit=request.concurrency,
                )
                fleet_metrics = _fleet_counter_delta(
                    fleet_before,
                    fleet_after,
                    boundary_seconds=boundary_finished - boundary_started,
                )
                fleet_before = fleet_after
                boundary_started = boundary_finished
                report, events, step_failures = _step_report(
                    step=step,
                    phase=phase,
                    planned_phase=planned_phase,
                    active_start=active_start,
                    active_end=active_end,
                    active_limit=request.concurrency,
                    target_groups=target,
                    started_at_unix=started,
                    finished_at_unix=finished,
                    drain_seconds=drain_seconds,
                    fleet_counters=fleet_metrics,
                    output=output,
                )
                steps.append(report)
                failures.extend(step_failures)
                if phase == "measure":
                    measured_events.extend(events)
                print(
                    json.dumps(
                        {
                            "event": "profile_step",
                            "step": step,
                            "phase": phase,
                            "planned_phase": planned_phase,
                            "active_trajectories_start": active_start,
                            "active_trajectories_end": active_end,
                            "groups": len(output.samples),
                            "seconds": drain_seconds,
                            "fleet_generation_tokens_per_second": fleet_metrics["generation_tokens_per_boundary_second"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            await rollout.wait_until_quiesced()
            if request.mode == "scale" and not any(step["phase"] == "measure" for step in steps):
                raise RuntimeError("no learner drain began and ended at steady-state rollout occupancy")
    except BaseException:
        try:
            await asyncio.wait_for(rollout.close(), timeout=120.0)
        except Exception as cleanup_error:
            logger.warning("Timed out stopping rollout admission: %s", cleanup_error)
        try:
            from modal_swe_agent_function import abort

            await abort(args)
        except Exception as cleanup_error:
            logger.warning("Modal SWE cooperative abort failed: %s", cleanup_error)
        raise
    else:
        await rollout.close()
    admission = rollout.admission_stats()
    admission.update(
        {
            "accepted_groups": accepted_groups,
            "accepted_unique_instances": len(accepted_instances),
            "duplicate_accepted_groups": accepted_groups - len(accepted_instances),
            "all_dataset_instances_attempted": (admission["unique_instances_attempted"] == sum(targets)),
        }
    )
    if accepted_groups != sum(targets):
        raise RuntimeError(f"accepted group count diverged from the dataset target ({accepted_groups}/{sum(targets)})")
    if not admission["all_dataset_instances_attempted"]:
        raise RuntimeError(f"finite profiler did not attempt every unique dataset instance ({admission['unique_instances_attempted']}/{sum(targets)})")
    if not admission["settled"]:
        raise RuntimeError(f"finite profiler did not settle all submissions: {admission}")
    return admission


async def _run_profile(
    payload: dict[str, Any],
    claim_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    request = ProfileRequest(**payload)
    _validate_request(request)
    profile_id = uuid.uuid4().hex
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "profile_id": profile_id,
            "started_at": _utc_now(),
            "request": asdict(request),
        },
        "preflight": {"claim": claim_provenance},
        "admission": {},
        "steps": [],
        "summary": {},
        "failures": [],
    }
    steps: list[dict[str, Any]] = report["steps"]
    measured_events: list[dict[str, Any]] = []
    try:
        dataset = _validate_dataset()
        gateway = await _resolve_gateway(
            request.target_app,
            request.target_router_class,
            request.target_environment,
        )
        replicas = list(claim_provenance.get("replicas") or []) if isinstance(claim_provenance, dict) else []
        if not replicas:
            raise ValueError("local preflight did not provide direct replica URLs")
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            gateway_info, replica_infos = await _wait_for_base(
                client=client,
                gateway=gateway,
                replicas=replicas,
                timeout_seconds=120,
                min_ready_replicas=max(1, len(replicas) - 4),
            )
        if not _at_base(gateway_info):
            raise RuntimeError(f"Stitch pool is not serving {BASE_POINTER}; run with --claim-base first (applied={gateway_info.get('applied')!r})")
        invalid_replicas = [index for index, info in enumerate(replica_infos) if not _at_base(info) or info.get("sync_state") == "ERROR" or int(info.get("active_requests") or 0) != 0]
        if invalid_replicas:
            raise RuntimeError(f"direct replica preflight failed for indices {invalid_replicas}: every replica must be idle at {BASE_POINTER}")
        targets = _step_targets(request, dataset_groups=int(dataset["dataset_groups"]))
        phases = _step_phases(request, targets)
        report["preflight"].update(
            {
                **dataset,
                "gateway": gateway,
                "router_class": request.target_router_class,
                "server_class": request.target_server_class,
                "gateway_server_info": gateway_info,
                "replica_server_infos": replica_infos,
                "replicas_total": len(replica_infos),
                "step_targets": targets,
                "step_phases": phases,
                "admission_budget_groups": sum(targets),
                "admission_budget_trajectories": sum(targets) * request.samples_per_prompt,
                "isolated_at_start": True,
            }
        )
        if not request.preflight_only:
            report["admission"] = await asyncio.wait_for(
                _run_rollouts(
                    request=request,
                    gateway=gateway,
                    replicas=replicas,
                    targets=targets,
                    phases=phases,
                    steps=steps,
                    measured_events=measured_events,
                    failures=report["failures"],
                ),
                timeout=request.overall_timeout,
            )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        report["failures"].append(
            {
                "stage": "orchestrator",
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
        )
    if steps:
        report["summary"] = _summary(steps, measured_events)
    report["run"]["finished_at"] = _utc_now()
    report["run"]["ok"] = not report["failures"]
    return _jsonable(report)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


@app.function(
    image=profile_image,
    cpu=16,
    memory=65_536,
    region="us",
    volumes=profile_volumes,
    timeout=PROFILE_TIMEOUT_SECONDS,
    single_use_containers=True,
)
async def run_canary(
    payload: dict[str, Any],
    claim_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await _run_profile(payload, claim_provenance)


@app.function(
    image=profile_image,
    cpu=64,
    memory=262_144,
    region="us",
    volumes=profile_volumes,
    timeout=PROFILE_TIMEOUT_SECONDS,
    single_use_containers=True,
)
async def run_scale(
    payload: dict[str, Any],
    claim_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await _run_profile(payload, claim_provenance)


def _mode_default(mode: str, canary: int, scale: int, supplied: int) -> int:
    return supplied if supplied > 0 else (canary if mode == "canary" else scale)


@app.local_entrypoint()
async def main(
    mode: str = "canary",
    target_app: str = TARGET_APP,
    target_router_class: str = TARGET_ROUTER_CLASS,
    target_server_class: str = TARGET_SERVER_CLASS,
    target_environment: str = TARGET_ENVIRONMENT,
    groups_per_step: int = 0,
    samples_per_prompt: int = 8,
    warmup_steps: int = -1,
    measure_steps: int = 0,
    max_groups: int = 0,
    all_dataset: bool = False,
    concurrency: int = 0,
    session_servers: int = 0,
    controller_processes: int = 0,
    controller_threads: int = 0,
    max_agent_steps: int = 256,
    episode_timeout: int = 7200,
    overall_timeout: int = 43_200,
    claim_pool_base: bool = True,
    claim_timeout: int = 600,
    release_pool_base: str = "auto",
    preflight_only: bool = False,
    output: str = "",
) -> None:
    """Run a small canary or the production-shaped CPU rollout controller."""
    mode = mode.lower().strip()
    if mode not in {"canary", "scale"}:
        raise ValueError("--mode must be canary or scale")
    request = ProfileRequest(
        mode=mode,
        target_app=target_app,
        target_router_class=target_router_class,
        target_server_class=target_server_class,
        target_environment=target_environment,
        groups_per_step=_mode_default(mode, 1, 32, groups_per_step),
        samples_per_prompt=samples_per_prompt,
        warmup_steps=(0 if mode == "canary" else 2) if warmup_steps < 0 else warmup_steps,
        measure_steps=_mode_default(mode, 1, 3, measure_steps),
        max_groups=max_groups,
        all_dataset=all_dataset,
        concurrency=_mode_default(
            mode,
            samples_per_prompt,
            EXPECTED_REPLICAS * TARGET_REQUESTS_PER_REPLICA,
            concurrency,
        ),
        session_servers=_mode_default(mode, 1, 64, session_servers),
        controller_processes=_mode_default(mode, 1, 48, controller_processes),
        controller_threads=_mode_default(mode, samples_per_prompt, 16, controller_threads),
        max_agent_steps=max_agent_steps,
        episode_timeout=episode_timeout,
        overall_timeout=overall_timeout,
        preflight_only=preflight_only,
    )
    _validate_request(request)
    release_pool_base = release_pool_base.lower().strip()
    if release_pool_base not in {"auto", "always", "never"}:
        raise ValueError("--release-pool-base must be auto, always, or never")

    provenance: dict[str, Any] = {
        "action": "claim_skipped",
        "committed": False,
    }
    result = None
    try:
        if claim_pool_base:
            provenance = await claim_base.remote.aio()
        gateway = await _resolve_gateway(
            target_app,
            target_router_class,
            target_environment,
        )
        discovery = await asyncio.to_thread(
            _discover_server_containers,
            target_app,
            target_server_class,
            target_environment,
        )
        containers = discovery["containers"]
        replicas = [host for item in containers if (host := _container_host(item))]
        if not replicas:
            raise RuntimeError(f"no live replicas discovered for {target_app}.{target_server_class}")
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            await _wake_replicas(client, replicas)
            gateway_info, replica_infos = await _wait_for_base(
                client=client,
                gateway=gateway,
                replicas=replicas,
                timeout_seconds=claim_timeout,
            )
        provenance.update(
            {
                "gateway": gateway,
                "router_class": target_router_class,
                "server_class": target_server_class,
                "server_function_id": discovery.get("function_id"),
                "gateway_server_info": gateway_info,
                "replicas": replicas,
                "replica_server_infos": replica_infos,
                "replicas_total": len(replicas),
                "replicas_at_base": sum(_at_base(info) for info in replica_infos),
            }
        )

        function = run_canary if mode == "canary" else run_scale
        result = await function.remote.aio(asdict(request), provenance)
    finally:
        should_release = release_pool_base == "always" or (release_pool_base == "auto" and bool(provenance.get("committed")))
        if should_release:
            try:
                released = await release_base.remote.aio()
                if result is not None:
                    result["run"]["release"] = released
            except Exception as error:
                if result is None:
                    raise
                result["failures"].append(
                    {
                        "stage": "release_base",
                        "type": type(error).__name__,
                        "message": str(error)[:2000],
                    }
                )
                result["run"]["ok"] = False

    assert result is not None
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if output:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    print(encoded)
    if not result["run"]["ok"]:
        raise RuntimeError("Modal SWE rollout profile failed; see emitted JSON report")
