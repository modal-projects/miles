"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from copy import deepcopy

import httpx

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.rollout.failures import (
    clear_failure_classification,
    is_infrastructure_failure,
    is_loss_masked_failure,
    is_non_retryable_failure,
    mark_loss_masked_failure,
)
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    SubmissionScheduler,
    generate_and_rm_group,
)
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [version for sample in _iter_samples(group) if not sample.remove_sample and (version := sample.oldest_weight_version) is not None]
    return min(versions) if versions else None


def _mask_non_retryable_failures(group: Group) -> tuple[Group | None, int, int]:
    """Make terminal failures shape-safe without adding them to the loss.

    A zero-loss placeholder retains the fixed prompt-group shape while reward
    normalization uses only real siblings. Two real siblings are required for
    a group-relative update; otherwise the entire group is dropped.
    """
    if any(isinstance(item, list) for item in group):
        return None, 0, 0

    samples = list(_iter_samples(group))
    failures = [sample for sample in samples if sample.status == Sample.Status.ABORTED and is_non_retryable_failure(sample)]
    valid = [sample for sample in samples if sample.status != Sample.Status.ABORTED and sample.reward is not None and sample.response_length > 0 and len(sample.tokens) >= sample.response_length]
    if not failures or len(valid) < 2:
        return None, 0, 0

    template = min(valid, key=lambda sample: (sample.response_length, len(sample.tokens)))
    replacements: dict[int, Sample] = {}
    infrastructure_count = 0
    for failed in failures:
        masked = deepcopy(template)
        masked.index = failed.index
        masked.group_index = failed.group_index
        masked.prompt = failed.prompt
        masked.label = failed.label
        masked.metadata = {
            **failed.metadata,
            "_fully_async_mask_template_index": template.index,
        }
        mark_loss_masked_failure(masked)
        masked.remove_sample = True
        masked.status = Sample.Status.COMPLETED
        masked.reward = 0.0
        masked.routing_key = failed.routing_key
        replacements[id(failed)] = masked
        infrastructure_count += int(is_infrastructure_failure(failed))

    return (
        [replacements.get(id(sample), sample) for sample in samples],
        len(failures),
        infrastructure_count,
    )


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router's /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = float("-inf")

    async def get(self, args) -> int | None:
        # Throttles failures too: the drain queries once per group, and an unreachable
        # router would otherwise cost every one of them the full timeout.
        if (time.monotonic() - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            data = await asyncio.wait_for(get(url), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
            self._value = int(data["weight_version"])
        except (
            httpx.HTTPError,
            asyncio.TimeoutError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            # Transient router unavailability; the staleness filter is best-effort.
            logger.debug(f"Failed to query engine weight version: {e}")
        finally:
            # Stamped on completion, so a router slower than the TTL still gets throttled.
            self._last_query = time.monotonic()
        return self._value


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._scheduler = SubmissionScheduler(input.args)
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: asyncio.Queue[tuple[list[Sample], Group]] | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            self._output = asyncio.Queue(maxsize=OUTPUT_QUEUE_MAX_GROUPS)
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input.rollout_id)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(self, prompt_group: list[Sample]) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return prompt_group, result

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        try:
            while True:
                await self._producer_resumed.wait()
                self._scheduler.arm()
                while self._scheduler.has_capacity(
                    pending_groups=len(active),
                    group_budget=self._max_in_flight_groups(),
                ):
                    active.add(self._submit_one_group())
                done, active = await self._scheduler.wait_for_progress(active)
                for task in done:
                    # Blocks when the queue is full: training lagging behind rollout
                    # production pauses submission instead of growing the queue unboundedly.
                    await self._output.put(task.result())
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)

    async def close(self) -> None:
        """Stop the persistent producer and every in-flight prompt group."""
        worker = self._worker
        if worker is None:
            return
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        self._worker = None

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> tuple[list[Sample], Group]:
        queue_get = asyncio.create_task(self._output.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        if not args.rollout_global_dataset:
            raise ValueError("FullyAsyncRolloutFn requires --rollout-global-dataset")

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        aborted_groups_recycled = 0
        non_retryable_groups_dropped = 0
        non_retryable_trajectories_masked = 0
        infrastructure_trajectories_masked = 0
        stale_groups_recycled = 0
        staleness_values: list[int] = []
        metric_gatherer = MetricGatherer()
        do_print = True

        while len(data) < target_data_size:
            prompt_group, group = await self._next_group()
            assert len(group) == args.n_samples_per_prompt

            aborted = [sample for sample in _iter_samples(group) if sample.status == Sample.Status.ABORTED]
            if aborted:
                if any(not is_non_retryable_failure(sample) for sample in aborted):
                    # SGLang marks in-flight requests ABORTED while publishing
                    # weights. Repeating those attempts is expected and safe.
                    self._recycle(prompt_group)
                    aborted_groups_recycled += 1
                    continue

                group, masked_count, infrastructure_count = _mask_non_retryable_failures(group)
                if group is None:
                    non_retryable_groups_dropped += 1
                    continue
                non_retryable_trajectories_masked += masked_count
                infrastructure_trajectories_masked += infrastructure_count

            if args.max_weight_staleness is not None:
                oldest = group_oldest_weight_version(group)
                current = await self._weight_version.get(args)
                if oldest is not None and current is not None:
                    staleness = current - oldest
                    staleness_values.append(staleness)
                    if staleness > args.max_weight_staleness:
                        self._recycle(prompt_group)
                        stale_groups_recycled += 1
                        logger.info(f"Recycled stale group (oldest_version={oldest}, current={current}, staleness={staleness} > max={args.max_weight_staleness})")
                        continue

            # A placeholder exists only to preserve the fixed tensor shape. It
            # must not make an otherwise uniform valid group appear dynamic.
            filter_group = [
                sample for sample in _iter_samples(group) if not is_loss_masked_failure(sample)
            ]
            filter_output = call_dynamic_filter(self._dynamic_filter, args, filter_group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            if do_print:
                sample = _first_sample(group)
                logger.info(f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}")
                do_print = False

            data.append(group)

        sample = _first_sample(data[-1])
        logger.info(f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}")

        data.sort(key=lambda group: _first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        metrics = {
            "rollout/fully_async/queue_size": self._output.qsize(),
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/non_retryable_groups_dropped": non_retryable_groups_dropped,
            "rollout/fully_async/non_retryable_trajectories_masked": non_retryable_trajectories_masked,
            "rollout/fully_async/infrastructure_trajectories_masked": infrastructure_trajectories_masked,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
            **metric_gatherer.collect(),
        }
        if staleness_values:
            metrics["rollout/fully_async/avg_staleness"] = sum(staleness_values) / len(staleness_values)
            metrics["rollout/fully_async/max_staleness"] = max(staleness_values)

        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            clear_failure_classification(sample)
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])
