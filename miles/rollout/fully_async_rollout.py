"""Continuous rollout generation for asynchronous training.

The rollout function owns a persistent producer on Miles' shared rollout event
loop. Training calls only drain completed prompt groups; unfinished groups keep
running across learner steps.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from copy import deepcopy

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnInput, RolloutFnTrainOutput
from miles.rollout.failures import clear_infrastructure_failure, is_infrastructure_failure
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_PROGRESS_REPORT_SECONDS = 30.0

Trajectory = Sample | list[Sample]
Group = list[Trajectory]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for trajectory in group:
        if isinstance(trajectory, list):
            yield from trajectory
        else:
            yield trajectory


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _numeric_versions(group: Group) -> list[int]:
    return [int(version) for sample in _iter_samples(group) for version in sample.weight_versions if str(version).isdigit()]


def group_oldest_weight_version(group: Group) -> int | None:
    versions = [sample.oldest_weight_version for sample in _iter_samples(group)]
    versions = [version for version in versions if version is not None]
    return min(versions) if versions else None


def group_newest_weight_version(group: Group) -> int | None:
    versions = _numeric_versions(group)
    return max(versions) if versions else None


def _reset_for_retry(sample: Sample) -> None:
    """Reset generated state without carrying an old failure classification."""
    sample.reset_for_retry()
    clear_infrastructure_failure(sample)


def mask_infrastructure_failures(group: Group) -> tuple[Group | None, int]:
    """Replace infra-aborted trajectories with zero-loss, shape-safe rows.

    GRPO still needs the successfully completed siblings from the prompt
    group.  An aborted sample often has no tokens, so it cannot simply pass
    through conversion with ``remove_sample=True``.  Clone the shortest valid
    sibling's tensor-shaped fields, preserve the failed sample's identity and
    diagnostics, and zero its loss later via ``remove_sample``.

    At least two real siblings are required: one remaining reward has no
    relative advantage and is not useful for a group-relative update.
    """
    samples = list(_iter_samples(group))
    aborted = [
        sample
        for sample in samples
        if sample.status == Sample.Status.ABORTED
        and is_infrastructure_failure(sample)
    ]
    valid = [
        sample
        for sample in samples
        if sample.status != Sample.Status.ABORTED
        and sample.reward is not None
        and sample.response_length > 0
        and len(sample.tokens) >= sample.response_length
    ]
    if not aborted or len(valid) < 2:
        return None, 0

    template = min(valid, key=lambda sample: (sample.response_length, len(sample.tokens)))
    replacements: dict[int, Sample] = {}
    for failed in aborted:
        masked = deepcopy(template)
        masked.index = failed.index
        masked.group_index = failed.group_index
        masked.prompt = failed.prompt
        masked.label = failed.label
        masked.metadata = {
            **failed.metadata,
            "_fully_async_infra_masked": True,
            "_fully_async_mask_template_index": template.index,
        }
        masked.remove_sample = True
        masked.status = Sample.Status.COMPLETED
        masked.reward = 0.0
        masked.routing_key = failed.routing_key
        replacements[id(failed)] = masked

    masked_group: Group = []
    for trajectory in group:
        if isinstance(trajectory, list):
            masked_group.append([replacements.get(id(sample), sample) for sample in trajectory])
        else:
            masked_group.append(replacements.get(id(trajectory), trajectory))
    return masked_group, len(aborted)


class FullyAsyncRolloutFn:
    """Persistent, completion-ordered rollout producer.

    ``--async-max-concurrent-samples`` controls live trajectories. The number
    of admitted prompt groups (active plus completed-queued) can be bounded
    independently; otherwise it is derived with 50% headroom so a slow sibling
    does not leave trajectory slots idle.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)

        trajectory_limit = input.args.async_max_concurrent_samples or input.args.rollout_batch_size * input.args.n_samples_per_prompt
        minimum_groups = math.ceil(trajectory_limit / input.args.n_samples_per_prompt)
        self.trajectory_limit = trajectory_limit
        self.pool_group_limit = getattr(
            input.args,
            "async_max_active_groups",
            None,
        ) or max(
            input.args.rollout_batch_size,
            math.ceil(minimum_groups * 1.5),
        )
        self.trajectory_slots = asyncio.Semaphore(trajectory_limit)
        self.output: asyncio.Queue[tuple[int, Group]] = asyncio.Queue(maxsize=self.pool_group_limit)
        self.pool_space_available = asyncio.Event()

        self.worker: asyncio.Task | None = None
        self.progress_reporter: asyncio.Task | None = None
        self.refill = True
        # Anchor the trainer's publication schedule to the first numeric
        # version returned by SGLang.  Sample metadata is the authoritative
        # record of which weights generated each token; polling every engine
        # through the router here puts control-plane I/O on the learner's
        # critical drain path and can stall an otherwise full output queue.
        self.weight_version_base: int | None = None
        self.stats: Counter[str] = Counter()
        self.active_groups = 0
        self.active_trajectories = 0
        self.waiting_trajectories = 0
        self.max_active_groups = 0
        self.max_active_trajectories = 0
        self.active_group_started: dict[asyncio.Task, float] = {}
        self.drain_rollout_id: int | None = None
        self.drain_accepted_groups = 0
        self.drain_completed_groups = 0
        self.drain_aborted_groups = 0
        self.drain_dropped_groups = 0
        self.last_report_stats: Counter[str] = Counter()
        self.last_report_time = time.monotonic()
        self.last_progress_stats: Counter[str] = Counter()
        self.last_progress_time = time.monotonic()

    def _target_weight_version(self, rollout_id: int, observed_newest: int | None) -> int | None:
        """Return the version expected when ``rollout_id`` reaches training.

        ``train_async`` publishes once before rollout 0 and then according to
        ``update_weights_interval``.  The first observed numeric SGLang version
        anchors that schedule, which also makes resumed runs independent of a
        hard-coded initial version.
        """
        interval = max(int(self.args.update_weights_interval), 1)
        scheduled_updates = rollout_id // interval
        if self.weight_version_base is None and observed_newest is not None:
            self.weight_version_base = observed_newest - scheduled_updates
        if self.weight_version_base is None:
            return None
        return self.weight_version_base + scheduled_updates

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise ValueError("FullyAsyncRolloutFn does not serve evaluation; configure a separate eval function")
        self._ensure_worker()
        return await self._drain(input.rollout_id)

    def _ensure_worker(self) -> None:
        if self.worker is None:
            # Construction happens before rollout engines finish loading and
            # initial weights are published.  Start throughput windows here so
            # the first report measures rollout work rather than cold start.
            now = time.monotonic()
            self.last_report_time = now
            self.last_progress_time = now
            self.last_report_stats = Counter(self.stats)
            self.last_progress_stats = Counter(self.stats)
            self.worker = asyncio.create_task(self._producer())
            self.progress_reporter = asyncio.create_task(self._report_progress())
            logger.info(
                "Started fully-async rollout producer: groups=%d trajectories=%d",
                self.pool_group_limit,
                self.trajectory_limit,
            )
        elif self.worker.done():
            self.worker.result()
            raise RuntimeError("fully-async rollout producer exited unexpectedly")

    async def _generate_one(self, sample: Sample, sampling_params: dict) -> Trajectory:
        started = time.monotonic()
        if sample.status in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
            sample.metadata["_fully_async_slot_wait_seconds"] = 0.0
            result = await generate_and_rm(self.state, sample, sampling_params, evaluation=False)
            for item in result if isinstance(result, list) else [result]:
                item.metadata["_fully_async_trajectory_wall_seconds"] = time.monotonic() - started
            return result

        waiting_started = time.monotonic()
        self.waiting_trajectories += 1
        try:
            await self.trajectory_slots.acquire()
        finally:
            self.waiting_trajectories -= 1
        try:
            sample.metadata["_fully_async_slot_wait_seconds"] = time.monotonic() - waiting_started
            self.active_trajectories += 1
            self.max_active_trajectories = max(self.max_active_trajectories, self.active_trajectories)
            self.stats["trajectories_started"] += 1
            result: Trajectory | None = None
            try:
                result = await generate_and_rm(self.state, sample, sampling_params, evaluation=False)
                for item in result if isinstance(result, list) else [result]:
                    item.metadata["_fully_async_trajectory_wall_seconds"] = time.monotonic() - started
                return result
            except asyncio.CancelledError:
                self.stats["trajectories_cancelled"] += 1
                raise
            finally:
                self.active_trajectories -= 1
                self.stats["trajectories_finished"] += 1
                if result is not None:
                    result_samples = result if isinstance(result, list) else [result]
                    self.stats["generated_tokens"] += sum(
                        item.effective_response_length
                        for item in result_samples
                    )
                    for item in result if isinstance(result, list) else [result]:
                        self.stats[f"trajectory_status/{item.status.value}"] += 1
        finally:
            self.trajectory_slots.release()

    async def _generate_group(self, group: Group) -> Group:
        started = time.monotonic()
        if policy_uses_routing_key(self.args):
            for sample in _iter_samples(group):
                if sample.routing_key is None:
                    sample.routing_key = str(uuid.uuid4())

        tasks = []
        for index, trajectory in enumerate(group):
            if isinstance(trajectory, list):
                raise TypeError("nested input trajectories are not supported")
            sampling_params = self.state.sampling_params.copy()
            if getattr(self.args, "sglang_enable_deterministic_inference", False):
                sampling_params["sampling_seed"] = self.args.rollout_seed + index
            tasks.append(asyncio.create_task(self._generate_one(trajectory, sampling_params)))

        try:
            generated = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if self.args.group_rm:
            flat = list(_iter_samples(generated))
            await batched_async_rm(self.args, flat, inplace_set_reward_field=True)

        wall_seconds = time.monotonic() - started
        completion_offsets = [
            float(sample.metadata["_fully_async_trajectory_wall_seconds"])
            for sample in _iter_samples(generated)
            if isinstance(
                sample.metadata.get("_fully_async_trajectory_wall_seconds"),
                (int, float),
            )
        ]
        spread = max(completion_offsets) - min(completion_offsets) if completion_offsets else 0.0
        for sample in _iter_samples(generated):
            sample.metadata["_fully_async_group_wall_seconds"] = wall_seconds
            sample.metadata["_fully_async_group_completion_spread_seconds"] = spread
        return generated

    async def _producer(self) -> None:
        active: dict[asyncio.Task, int] = {}
        next_group_id = 0
        try:
            while self.refill or active:
                # Bound every admitted group, not just currently generating
                # groups. Completed groups retain their slot until the learner
                # drains them, preventing a fast producer from building a much
                # larger stale queue behind this nominal pool limit.
                while self.refill and len(active) + self.output.qsize() < self.pool_group_limit:
                    [group] = self.data_source.get_samples(1)
                    task = asyncio.create_task(self._generate_group(group))
                    active[task] = next_group_id
                    self.active_group_started[task] = time.monotonic()
                    next_group_id += 1
                    self.active_groups += 1
                    self.max_active_groups = max(self.max_active_groups, self.active_groups)
                    self.stats["groups_started"] += 1

                if not active:
                    if not self.refill:
                        return
                    # Every admitted group may finish before the learner drains
                    # the bounded queue. That is backpressure, not completion
                    # of the persistent producer. Wait for `_next_group` to
                    # release one admitted-group slot, then refill it.
                    self.pool_space_available.clear()
                    if self.output.qsize() >= self.pool_group_limit:
                        await self.pool_space_available.wait()
                    continue

                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    group_id = active.pop(task)
                    self.active_group_started.pop(task, None)
                    try:
                        group = task.result()
                        if any(sample.status == Sample.Status.ABORTED for sample in _iter_samples(group)):
                            self.stats["groups_with_abort"] += 1
                        await self.output.put((group_id, group))
                        self.stats["groups_queued"] += 1
                    finally:
                        self.active_groups -= 1
                        self.stats["groups_finished"] += 1
        finally:
            # RolloutManager.dispose() closes the persistent producer after the
            # final learner update. Explicitly cancel child groups so they do
            # not keep Ray/Modal work alive during process teardown.
            pending = list(active)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                for task in pending:
                    self.active_group_started.pop(task, None)
                self.active_groups -= len(pending)
                self.stats["groups_cancelled"] += len(pending)
                self.stats["groups_finished"] += len(pending)

    async def close(self) -> None:
        """Stop persistent rollout work after the final learner update."""
        self.refill = False
        tasks = [task for task in (self.worker, self.progress_reporter) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.worker = None
        self.progress_reporter = None
        logger.info(
            "Closed fully-async rollout producer: cancelled_groups=%d cancelled_trajectories=%d",
            self.stats["groups_cancelled"],
            self.stats["trajectories_cancelled"],
        )

    async def _report_progress(self) -> None:
        """Emit pool state independently of learner-batch completion.

        W&B only receives rollout metrics after ``_drain`` returns. This
        reporter keeps a zero-step run diagnosable and also exposes whether
        capacity is decoding, waiting for trajectory slots, or accumulating
        completed groups.
        """
        while self.worker is not None and not self.worker.done():
            await asyncio.sleep(_PROGRESS_REPORT_SECONDS)
            now = time.monotonic()
            elapsed = max(now - self.last_progress_time, 1e-9)
            deltas = {key: value - self.last_progress_stats.get(key, 0) for key, value in self.stats.items()}
            oldest_group_seconds = max(
                (now - started for started in self.active_group_started.values()),
                default=0.0,
            )
            logger.info(
                "Fully-async progress: rollout=%s accepted=%d/%d drained=%d "
                "aborted=%d dropped=%d queued=%d active_groups=%d "
                "active_trajectories=%d waiting_trajectories=%d "
                "finished_groups_delta=%d finished_trajectories_delta=%d "
                "group_rate=%.3f/s trajectory_rate=%.3f/s oldest_group=%.1fs",
                self.drain_rollout_id,
                self.drain_accepted_groups,
                self.args.rollout_batch_size,
                self.drain_completed_groups,
                self.drain_aborted_groups,
                self.drain_dropped_groups,
                self.output.qsize(),
                self.active_groups,
                self.active_trajectories,
                self.waiting_trajectories,
                deltas.get("groups_finished", 0),
                deltas.get("trajectories_finished", 0),
                deltas.get("groups_finished", 0) / elapsed,
                deltas.get("trajectories_finished", 0) / elapsed,
                oldest_group_seconds,
            )
            self.last_progress_time = now
            self.last_progress_stats = Counter(self.stats)

    async def _next_group(self) -> tuple[int, Group]:
        get_task = asyncio.create_task(self.output.get())
        try:
            done, _ = await asyncio.wait(
                {get_task, self.worker},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                result = get_task.result()
                self.pool_space_available.set()
                return result
            self.worker.result()
            raise RuntimeError("fully-async rollout producer exited before filling the learner batch")
        finally:
            if not get_task.done():
                get_task.cancel()
                await asyncio.gather(get_task, return_exceptions=True)

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        started = time.monotonic()
        groups: list[Group] = []
        observed_groups: list[Group] = []
        completed_groups = 0
        aborted_groups = 0
        dropped_groups = 0
        unusable_groups = 0
        infra_masked_groups = 0
        infra_masked_trajectories = 0
        stale_groups = 0
        staleness: list[int] = []
        newest_staleness: list[int] = []
        version_spans: list[int] = []
        accepted_staleness: list[int] = []
        accepted_newest_staleness: list[int] = []
        accepted_version_spans: list[int] = []
        recycled_staleness: list[int] = []
        recycled_newest_staleness: list[int] = []
        recycled_version_spans: list[int] = []
        target_weight_version: int | None = None
        self.drain_rollout_id = rollout_id
        self.drain_accepted_groups = 0
        self.drain_completed_groups = 0
        self.drain_aborted_groups = 0
        self.drain_dropped_groups = 0

        while len(groups) < self.args.rollout_batch_size:
            _, group = await self._next_group()
            observed_groups.append(group)
            completed_groups += 1
            self.drain_completed_groups = completed_groups

            aborted_samples = [
                sample
                for sample in _iter_samples(group)
                if sample.status == Sample.Status.ABORTED
            ]
            if aborted_samples:
                aborted_groups += 1
                self.drain_aborted_groups = aborted_groups
                if any(
                    not is_infrastructure_failure(sample)
                    for sample in aborted_samples
                ):
                    # A non-infrastructure abort has no shape-safe training
                    # trajectory. Policy-terminal outcomes should normally
                    # retain their recorded tokens and zero reward before this
                    # layer; an abort here is unusable data.
                    unusable_groups += 1
                    dropped_groups += 1
                    self.drain_dropped_groups = dropped_groups
                    continue
                masked_group, masked_count = mask_infrastructure_failures(group)
                if masked_group is not None:
                    group = masked_group
                    infra_masked_groups += 1
                    infra_masked_trajectories += masked_count
                else:
                    dropped_groups += 1
                    self.drain_dropped_groups = dropped_groups
                if masked_group is None:
                    continue

            oldest = group_oldest_weight_version(group)
            newest = group_newest_weight_version(group)
            if oldest is not None:
                current = self._target_weight_version(rollout_id, newest)
                if current is not None:
                    if newest is not None:
                        # A sample can complete just after a publication.  Its
                        # embedded version is stronger evidence than the
                        # schedule anchor and prevents negative lag.
                        current = max(current, newest)
                    target_weight_version = max(target_weight_version or current, current)
                    lag = current - oldest
                    staleness.append(lag)
                    if newest is not None:
                        newest_staleness.append(current - newest)
                        version_spans.append(newest - oldest)
                    if self.args.max_weight_staleness is not None and lag > self.args.max_weight_staleness:
                        recycled_staleness.append(lag)
                        if newest is not None:
                            recycled_newest_staleness.append(current - newest)
                            recycled_version_spans.append(newest - oldest)
                        for sample in _iter_samples(group):
                            _reset_for_retry(sample)
                        self.data_source.add_samples([group])
                        stale_groups += 1
                        continue
                    accepted_staleness.append(lag)
                    if newest is not None:
                        accepted_newest_staleness.append(current - newest)
                        accepted_version_spans.append(newest - oldest)

            groups.append(group)
            self.drain_accepted_groups = len(groups)

        groups.sort(key=lambda group: next(_iter_samples(group)).index)
        metrics = self._metrics(
            started=started,
            completed_groups=completed_groups,
            aborted_groups=aborted_groups,
            dropped_groups=dropped_groups,
            unusable_groups=unusable_groups,
            infra_masked_groups=infra_masked_groups,
            infra_masked_trajectories=infra_masked_trajectories,
            stale_groups=stale_groups,
            staleness=staleness,
            newest_staleness=newest_staleness,
            version_spans=version_spans,
            accepted_staleness=accepted_staleness,
            accepted_newest_staleness=accepted_newest_staleness,
            accepted_version_spans=accepted_version_spans,
            recycled_staleness=recycled_staleness,
            recycled_newest_staleness=recycled_newest_staleness,
            recycled_version_spans=recycled_version_spans,
            target_weight_version=target_weight_version,
            accepted_groups=groups,
            observed_groups=observed_groups,
        )

        if rollout_id + 1 >= self.args.num_rollout:
            self.refill = False
            metrics["rollout_async/refill_stopped_after_final_batch"] = 1.0

        return RolloutFnTrainOutput(samples=groups, metrics=metrics)

    def _metrics(
        self,
        *,
        started: float,
        completed_groups: int,
        aborted_groups: int,
        dropped_groups: int,
        unusable_groups: int,
        infra_masked_groups: int,
        infra_masked_trajectories: int,
        stale_groups: int,
        staleness: list[int],
        newest_staleness: list[int],
        version_spans: list[int],
        accepted_staleness: list[int],
        accepted_newest_staleness: list[int],
        accepted_version_spans: list[int],
        recycled_staleness: list[int],
        recycled_newest_staleness: list[int],
        recycled_version_spans: list[int],
        target_weight_version: int | None,
        accepted_groups: list[Group],
        observed_groups: list[Group],
    ) -> dict[str, float]:
        now = time.monotonic()
        report_seconds = now - self.last_report_time
        deltas = {key: value - self.last_report_stats.get(key, 0) for key, value in self.stats.items()}
        self.last_report_time = now
        self.last_report_stats = Counter(self.stats)

        metrics: dict[str, float] = {
            "rollout_async/batch_wait_seconds": now - started,
            "rollout_async/queue_groups": self.output.qsize(),
            "rollout_async/active_groups": self.active_groups,
            "rollout_async/admitted_groups": self.active_groups + self.output.qsize(),
            "rollout_async/active_trajectories": self.active_trajectories,
            "rollout_async/waiting_trajectories": self.waiting_trajectories,
            "rollout_async/max_active_groups": self.max_active_groups,
            "rollout_async/max_active_trajectories": self.max_active_trajectories,
            "rollout_async/pool_group_limit": self.pool_group_limit,
            "rollout_failure/completed_groups": completed_groups,
            "rollout_failure/aborted_groups": aborted_groups,
            "rollout_failure/dropped_groups": dropped_groups,
            "rollout_failure/unusable_groups": unusable_groups,
            "rollout_failure/infra_masked_groups": infra_masked_groups,
            "rollout_failure/infra_masked_trajectories": infra_masked_trajectories,
            "rollout_staleness/recycled_groups": stale_groups,
            "rollout_staleness/filter_enabled": float(self.args.max_weight_staleness is not None),
            "rollout_async/report_window_seconds": report_seconds,
        }
        if target_weight_version is not None:
            metrics["rollout_staleness/target_weight_version"] = target_weight_version
        for key, value in self.stats.items():
            metrics[f"rollout_async/lifetime/{key}"] = value
        for key, value in deltas.items():
            metrics[f"rollout_async/batch_delta/{key}"] = value
        if report_seconds > 0:
            metrics["rollout_async/trajectory_completions_per_sec"] = deltas.get("trajectories_finished", 0) / report_seconds
            metrics["rollout_async/group_completions_per_sec"] = deltas.get("groups_finished", 0) / report_seconds
            metrics["rollout_async/generated_tokens_per_sec"] = deltas.get("generated_tokens", 0) / report_seconds

        for population, population_metrics in (
            (
                "candidate",
                (
                    ("oldest_lag", staleness),
                    ("newest_lag", newest_staleness),
                    ("within_group_version_span", version_spans),
                ),
            ),
            (
                "accepted",
                (
                    ("oldest_lag", accepted_staleness),
                    ("newest_lag", accepted_newest_staleness),
                    ("within_group_version_span", accepted_version_spans),
                ),
            ),
            (
                "recycled",
                (
                    ("oldest_lag", recycled_staleness),
                    ("newest_lag", recycled_newest_staleness),
                    ("within_group_version_span", recycled_version_spans),
                ),
            ),
        ):
            metrics[f"rollout_staleness/{population}_groups"] = len(population_metrics[0][1])
            for name, values in population_metrics:
                if values:
                    metrics[f"rollout_staleness/{population}/{name}_mean"] = sum(values) / len(values)
                    metrics[f"rollout_staleness/{population}/{name}_p50"] = _percentile(values, 0.50)
                    metrics[f"rollout_staleness/{population}/{name}_p90"] = _percentile(values, 0.90)
                    metrics[f"rollout_staleness/{population}/{name}_max"] = max(values)

        observed_versions = [version for group in observed_groups for version in _numeric_versions(group)]
        if observed_versions:
            metrics["rollout_staleness/observed_sample_version_min"] = min(observed_versions)
            metrics["rollout_staleness/observed_sample_version_max"] = max(observed_versions)
            metrics["rollout_staleness/observed_sample_version_spread"] = max(observed_versions) - min(observed_versions)

        for name in (
            "_fully_async_slot_wait_seconds",
            "_fully_async_trajectory_wall_seconds",
            "_fully_async_group_wall_seconds",
            "_fully_async_group_completion_spread_seconds",
        ):
            values = [
                float(sample.metadata[name])
                for group in accepted_groups
                for sample in _iter_samples(group)
                if isinstance(sample.metadata.get(name), (int, float))
            ]
            if values:
                key = name.removeprefix("_fully_async_")
                metrics[f"rollout_async/{key}_mean"] = sum(values) / len(values)
                metrics[f"rollout_async/{key}_p90"] = _percentile(values, 0.90)
                metrics[f"rollout_async/{key}_max"] = max(values)

        observed_samples = [sample for group in observed_groups for sample in _iter_samples(group)]
        accepted_samples = [sample for group in accepted_groups for sample in _iter_samples(group)]
        metrics["rollout_async/observed_trajectories"] = len(observed_samples)
        metrics["rollout_async/accepted_trajectories"] = len(accepted_samples)
        metrics["rollout_async/trainable_trajectories"] = sum(not sample.remove_sample for sample in accepted_samples)
        metrics["rollout_async/masked_trajectory_ratio"] = (
            sum(sample.remove_sample for sample in accepted_samples)
            / len(accepted_samples)
            if accepted_samples
            else 0.0
        )
        return metrics
