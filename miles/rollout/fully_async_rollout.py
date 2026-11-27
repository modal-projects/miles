"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import copy
import logging
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from miles.ray.rollout.external_rollout import ExternalRollout
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    compute_kv_cache_namespace,
    stamp_kv_cache_namespace,
)
from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    UnusedReason,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
from miles.rollout.session.session_load import SESSION_LOAD_KEY
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.function_registry import load_function
from miles.utils.http_utils import get
from miles.utils.output_store.dropped import record_dropped_samples
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        assert self.args.buffer_filter_path is None, "fully-async rollout does not support buffer_filter_path"
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else self._drop_unused
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._worker_error_cancel_targets: set[asyncio.Task] = set()
        self._eval_prompt_dataset_cache: dict = {}
        self._curr_kv_cache_namespace: str | None = None
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._admission_changed = asyncio.Event()
        default_buffer_cls = DefaultMultiDataBuffer if self.args.raw_megatron.is_multi_policy else DefaultDataBuffer
        buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
        assert (
            not self.args.async_submit_within_staleness or buffer_cls.admits is not DataBuffer.admits
        ), f"--async-submit-within-staleness requires {buffer_cls.__name__} to implement admits()"
        self._output: DataBuffer = buffer_cls(
            DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
        )
        self._retry_buffer: deque[list[Sample]] = deque()
        self._running_tasks: list[_RunningTask] = []
        # An external rollout service sizes its engines from each training step.
        control_url = self.args.rollout_external_control_url
        self._external_rollout = ExternalRollout(control_url) if control_url else None
        self._step_reports: set[asyncio.Task] = set()
        self._refresher: asyncio.Task | None = None
        self._session_load: Counter[str] = Counter()
        self._recent_loads: deque[Counter[str]] = deque(maxlen=10)
        # Each minute, the requests the engines run and the samples in flight.
        self._recent_use: deque[tuple[int, int]] = deque(maxlen=10)
        self._admitted_groups = 0
        self._request_capacity: int | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        self._curr_kv_cache_namespace = compute_kv_cache_namespace(self.args, input)
        if self._worker is None:
            assert (
                input.weight_version is not None or self.args.debug_rollout_only or self.args.debug_skip_weight_update
            ), (
                "the orchestration publishes the restored weight version before the producer starts, or every "
                "group a checkpoint restored is filtered out as stale"
            )
            # Admission starts from the inference side's capacity, not the args' maximum.
            await self._refresh_capacity()
            self._refresher = asyncio.create_task(self._refresh_capacity_every_minute())
            self._worker = asyncio.create_task(self._worker_loop())
            self._worker.add_done_callback(self._on_worker_error)
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        if (refresher := self._refresher) is not None:
            await _end_worker(refresher)
        if (worker := self._worker) is None:
            return
        await _end_worker(worker)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(
                input.generate_state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(
                self.state,
                self._eval_prompt_dataset_cache,
                kv_cache_namespace=compute_kv_cache_namespace(self.args, input),
            )
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_samples(self) -> int:
        n = self.args.n_samples_per_prompt
        if (x := self.args.async_max_concurrent_samples) is not None:
            samples = x
        else:
            samples = self.args.rollout_batch_size * n
        if self._request_capacity is not None:
            running = sum(requests for requests, _ in self._recent_use)
            in_flight = sum(samples for _, samples in self._recent_use)
            # A sample holds a request for only part of its life: the requests running per sample in flight.
            generating_share = running / in_flight if running and in_flight else 1.0
            samples = min(samples, int(self._request_capacity / generating_share))
        # Whole groups are submitted, so the budget floors to whole groups of the default size.
        return max(1, samples // n) * n

    def _next_group_size(self) -> int:
        # A recycled group's size is known; the data source's next one is not drawn yet, so the default size bounds it.
        return len(self._retry_buffer[0]) if self._retry_buffer else self.args.n_samples_per_prompt

    def _submit_one_group(self) -> asyncio.Task:
        samples = [self._retry_buffer.popleft()] if self._retry_buffer else self.data_source.get_samples(1)
        stamp_kv_cache_namespace(samples, namespace=self._curr_kv_cache_namespace)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        _check_group_size(self.args, prompt_group)
        for sample in prompt_group:
            sample.request_priority = self._admitted_groups
        self._admitted_groups += 1
        task = asyncio.create_task(self._generate_group(prompt_group))
        self._running_tasks.append(_RunningTask(prompt_group=prompt_group, task=task))
        return task

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            self._admission_changed.clear()
            while (admits := self._output.admits(in_flight=len(active))) and self._scheduler.has_capacity(
                pending_samples=self._pending_samples(),
                group_size=self._next_group_size(),
                sample_budget=self._max_in_flight_samples(),
            ):
                active.add(self._submit_one_group())
            if admits:
                done, active = await self._scheduler.wait_for_progress(active)
            else:
                done, active = await self._wait_for_admission(active)
            for task in done:
                prompt_group = next(x.prompt_group for x in self._running_tasks if x.task is task)
                entry = self._collect_group_result(task, prompt_group)
                self._add_session_load(entry.group)
                await self._output.put(entry)
                self._running_tasks = [x for x in self._running_tasks if x.task is not task]

    async def _wait_for_admission(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        """Wait for a drain to start, an unused group to leave, or a group to finish (its put may reject it)."""
        waiter = asyncio.create_task(self._admission_changed.wait())
        try:
            done, pending = await asyncio.wait(pendings | {waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        return done - {waiter}, pending - {waiter}

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    def _add_session_load(self, group: Group) -> None:
        # A v2 session gives several samples, each carrying the whole session's load; a single-turn sample is its own.
        loads = {
            sample.metadata.get("session_id", id(sample)): sample.metadata[SESSION_LOAD_KEY]
            for sample in iter_samples(group)
            if SESSION_LOAD_KEY in sample.metadata
        }
        for load in loads.values():
            self._session_load.update(load)

    async def _refresh_capacity(self) -> None:
        load, self._session_load = self._session_load, Counter()
        self._recent_loads.append(load)
        if self._external_rollout is None:
            engines = await _own_engines_load(self.args)
        else:
            engines = await self._external_rollout.report_load(self._output.output() or {})
        if engines["running"] is not None:
            in_flight = self._scheduler.in_flight(pending_samples=self._pending_samples())
            self._recent_use.append((engines["running"], in_flight))
        if engines["capacity"] is not None:
            self._request_capacity = engines["capacity"]
        self._admission_changed.set()
        recent, kept = sum(self._recent_loads, Counter()), ""
        if prefix := recent["prefix_tokens"]:
            kept = f"; later turns found {1 - recent['uncached_prefix_tokens'] / prefix:.0%} of earlier context cached"
        logger.info(
            "Admission: at most %d samples in flight for %s requests at once, %s running%s",
            self._max_in_flight_samples(),
            self._request_capacity,
            engines["running"],
            kept,
        )

    def _pending_samples(self) -> int:
        return sum(len(x.prompt_group) for x in self._running_tasks)

    async def _refresh_capacity_every_minute(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._refresh_capacity()

    # -------------------------- consumer --------------------------

    def _on_worker_error(self, worker: asyncio.Task) -> None:
        # The worker loop never returns normally, so any completion fails the waiting step.
        logger.error("fully-async rollout worker ended", exc_info=None if worker.cancelled() else worker.exception())
        for target in list(self._worker_error_cancel_targets):
            target.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        assert not self._worker.done(), "the fully-async rollout worker has ended"
        self._worker_error_cancel_targets.add(asyncio.current_task())
        self._admission_changed.set()
        entries = await self._output.get(
            current_version=input.weight_version,
            num_groups=args.rollout_batch_size,
            trainer_model_id=input.trainer_model_id,
        )
        self._worker_error_cancel_targets.discard(asyncio.current_task())
        if self._external_rollout is not None and (step := self._output.take_step(input.trainer_model_id)):
            # Off the step: a slow or failed report never holds the trainer.
            report = asyncio.create_task(self._external_rollout.report_step(step))
            self._step_reports.add(report)
            report.add_done_callback(self._step_reports.discard)

        data: list[Group] = []
        do_print = True

        for entry in entries:
            if args.use_partial_groups_on_abort:
                assert 0 < len(entry.group) <= len(entry.prompt_group)
            else:
                assert len(entry.group) == len(entry.prompt_group)

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        before_filter = SampleOwnershipRecorder.flatten_samples(data)
        if self._sample_filter is not None:
            self._sample_filter(args, data)
        kept = {id(sample) for sample in SampleOwnershipRecorder.flatten_samples(data)}
        record_dropped_samples(sample for sample in before_filter if id(sample) not in kept)
        SampleOwnershipRecorder.log_dropped_groups(
            args=args, before=before_filter, after=data, reason="rollout_sample_filter"
        )

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics(input.trainer_model_id))

    def _recycle(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self._retry_buffer.append(prompt_group)
        self._admission_changed.set()

    def _drop_unused(self, prompt_group: list[Sample], reason: UnusedReason) -> None:
        SampleOwnershipRecorder.log_dropped_samples(args=self.args, samples=prompt_group, reason=reason.value)
        self._admission_changed.set()

    def save(self, directory: Path) -> None:
        state = _FullyAsyncRolloutState(
            retry_buffer=list(self._retry_buffer),
            running=[_copy_and_reset_for_retry(x.prompt_group) for x in self._running_tasks],
            output=self._output.state_dict(),
        )
        save_simple_checkpoint(directory=directory, data=state)

    def load(self, directory: Path) -> None:
        assert self._worker is None, "restore the fully async rollout state before producer startup"
        state: _FullyAsyncRolloutState = load_simple_checkpoint(directory=directory)
        assert not self._retry_buffer, "restore the fully async rollout state before any group was recycled"
        self._retry_buffer = deque([*state.retry_buffer, *state.running])
        self._output.load_state_dict(state.output)

    def drop_saved_in_flight(self, directory: Path) -> None:
        state: _FullyAsyncRolloutState = load_simple_checkpoint(directory=directory)
        prompt_groups = [*state.retry_buffer, *state.running]
        record_dropped_samples(prompt_groups)
        samples = [sample for group in prompt_groups for sample in group]
        SampleOwnershipRecorder.log_dropped_samples(args=self.args, samples=samples, reason="data_reset_on_load")
        # the finished groups are in the buffer's own layout, which a custom buffer class may change
        self._output.drop_saved_state(state.output, reason="data_reset_on_load")
        logger.info(f"Dropped {len(prompt_groups)} saved queued or running prompt groups drawn from the reset data")


@dataclass(frozen=True)
class _RunningTask:
    prompt_group: list[Sample]
    task: asyncio.Task


@dataclass(frozen=True)
class _FullyAsyncRolloutState:
    retry_buffer: list[list[Sample]]
    running: list[list[Sample]]
    output: Any


def _check_group_size(args, prompt_group: list[Sample]) -> None:
    size, n = len(prompt_group), args.n_samples_per_prompt
    assert 0 < size <= n, (
        f"a prompt group holds 1 to --n-samples-per-prompt ({n}) samples, the most the in-flight budget reserves "
        f"for a group not yet drawn, but this one holds {size}"
    )
    if size == n:
        return
    assert args.use_dynamic_global_batch_size, (
        f"a prompt group of {size} samples, not --n-samples-per-prompt ({n}), varies the sample count, so it requires "
        f"--use-dynamic-global-batch-size"
    )
    assert not args.log_passrate, (
        f"--log-passrate estimates pass@k over groups of --n-samples-per-prompt ({n}) samples, so it cannot take a "
        f"prompt group of {size}"
    )


def _copy_and_reset_for_retry(prompt_group: list[Sample]) -> list[Sample]:
    copied = copy.deepcopy(prompt_group)
    for sample in copied:
        sample.reset_for_retry()
    return copied


async def _own_engines_load(args) -> dict[str, int | None]:
    try:
        async with asyncio.timeout(30):
            urls = await get_worker_urls(args)
            loads = await asyncio.gather(*(get(f"{url}/v1/loads?include=core") for url in urls))
    except Exception as error:
        logger.warning(f"could not read the engines' capacity: {error!r}")
        return {"capacity": None, "running": None}
    ranks = [rank for load in loads for rank in load["loads"]]
    return {
        "capacity": sum(rank["max_running_requests"] for rank in ranks) or None,
        "running": sum(rank["num_running_reqs"] for rank in ranks),
    }


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
