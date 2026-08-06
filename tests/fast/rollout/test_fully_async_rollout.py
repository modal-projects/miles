from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
import time
from argparse import Namespace
from collections import deque
from dataclasses import replace

import httpx
import pytest

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.failures import (
    is_infrastructure_failure,
    is_loss_masked_failure,
    mark_infrastructure_failure,
    mark_non_retryable_failure,
)
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

N_SAMPLES_PER_PROMPT = 2


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        self.recycled.extend(groups)


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            tokens=[100 + i],
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=status,
            weight_versions=list(weight_versions or []),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        async_max_concurrent_samples=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_sample_completion_backfill=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    fn = fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))

    class NoWeightVersion:
        async def get(self, args):
            return None

    fn._finish_weight_version = NoWeightVersion()
    fn._weight_version = NoWeightVersion()
    return fn


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_eval_without_fleet_pauses_producer(monkeypatch):
    """Shared-engine eval: producer submissions pause during eval and resume after."""
    release = asyncio.Event()

    async def blocking_generate(
        state,
        group,
        sampling_params,
        evaluation=False,
        sample_done_callback=None,
    ):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch, make_args(rollout_batch_size=2, eval_num_gpus=0), data_source, generate=blocking_generate
    )

    eval_started = asyncio.Event()
    eval_release = asyncio.Event()
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}

    async def fake_run_eval_datasets(state, cache):
        assert state is fn.state  # shared-engine eval uses the train state
        eval_started.set()
        await eval_release.wait()
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    # Start the producer via a train call, then run eval concurrently.
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == submitted_before_eval

    eval_release.set()
    output = await eval_task
    assert output.data == eval_results

    # Producer resumes and the train drain completes.
    assert (await drain).samples


async def test_eval_runs_on_dedicated_fleet(monkeypatch):
    """RolloutManager (not the fn) decides fleet-vs-shared and builds the fleet's
    GenerateState; it hands it in via RolloutFnEvalInput.generate_state. The fn must
    use that state as-is (not self.state) and must not touch the producer/data_source.
    Building/caching the fleet state itself is EvalFleetSession's job, covered in
    tests/fast/rollout/test_checkpoint_eval.py.
    """
    args = make_args(eval_num_gpus=1, eval_num_gpus_per_engine=1)
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, args, data_source)

    fleet_state = FakeGenerateState(args)
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}
    seen_states = []

    async def fake_run_eval_datasets(state, cache):
        seen_states.append(state)
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    output = await fn(RolloutFnEvalInput(rollout_id=0, generate_state=fleet_state, weight_version="0"))

    assert output.data == eval_results
    assert seen_states == [fleet_state]  # used the fleet's state, not fn.state
    # Eval must not start the producer or consume training prompts.
    assert fn._worker is None
    assert data_source.num_get_calls == 0


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 1


async def test_non_retryable_infrastructure_failure_is_masked(monkeypatch):
    group = make_group(1)
    for sample in group:
        sample.rollout_sampling_mask_ids = [sample.tokens[-1], 999]
        sample.rollout_sampling_mask_offsets = [0, 2]
    group.append(replace(group[-1], index=12))
    failed = group[-1]
    failed.status = Sample.Status.ABORTED
    mark_infrastructure_failure(failed)
    data_source = FakeDataSource(scripted=[group])
    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, n_samples_per_prompt=3),
        data_source,
    )

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    masked = output.samples[0][-1]
    assert masked.remove_sample
    assert is_loss_masked_failure(masked)
    assert masked.status == Sample.Status.COMPLETED
    assert masked.index == failed.index
    assert is_infrastructure_failure(masked)
    assert masked.rollout_sampling_mask_ids == group[0].rollout_sampling_mask_ids
    assert masked.rollout_sampling_mask_offsets == group[0].rollout_sampling_mask_offsets
    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/non_retryable_trajectories_masked"] == 1
    assert output.metrics["rollout/fully_async/infrastructure_trajectories_masked"] == 1


async def test_failure_placeholder_does_not_affect_dynamic_filter(monkeypatch):
    group = make_group(1)
    group.append(replace(group[-1], index=12))
    failed = group[-1]
    failed.status = Sample.Status.ABORTED
    mark_infrastructure_failure(failed)
    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, n_samples_per_prompt=3),
        FakeDataSource(scripted=[group]),
    )
    filtered_groups = []

    def keep_valid_siblings(_args, filter_group):
        filtered_groups.append(filter_group)
        return DynamicFilterOutput(keep=True, reason=None)

    fn._dynamic_filter = keep_valid_siblings

    await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(filtered_groups) == 1
    assert [sample.index for sample in filtered_groups[0]] == [10, 11]


async def test_non_retryable_group_without_two_valid_siblings_is_dropped(monkeypatch):
    group = make_group(1)
    group[0].status = Sample.Status.ABORTED
    mark_non_retryable_failure(group[0])
    data_source = FakeDataSource(scripted=[group])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert output.samples[0][0].group_index != 1
    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/non_retryable_groups_dropped"] == 1


async def test_stale_group_recycled(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = list(data_source_fresh_versions)
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    class FakeWeightVersion:
        async def get(self, args):
            return 10

    fn._weight_version = FakeWeightVersion()

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 5


async def test_metrics_split_generation_and_post_finish_staleness(monkeypatch):
    first = make_group(1, weight_versions=["5", "7"])
    never = asyncio.Event()

    async def generate_first_only(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        if group[0].group_index != 1:
            await never.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=6),
        FakeDataSource(scripted=[first]),
        generate=generate_first_only,
    )

    class FixedWeightVersion:
        def __init__(self, value):
            self.value = value

        async def get(self, args):
            return self.value

    fn._finish_weight_version = FixedWeightVersion(8)
    fn._weight_version = FixedWeightVersion(10)

    output = await fn(RolloutFnTrainInput(rollout_id=0))
    await fn.close()

    metrics = output.metrics
    assert metrics["rollout_staleness/accepted/generation_lag_versions_mean"] == 3
    assert metrics["rollout_staleness/accepted/post_finish_lag_versions_mean"] == 2
    assert metrics["rollout_staleness/accepted/within_group_version_span_mean"] == 2
    assert metrics["rollout_staleness/current_version_available_ratio"] == 1
    assert metrics["rollout_staleness/filter_current_version_fallback_ratio"] == 0
    assert metrics["rollout/fully_async/avg_staleness"] == 5
    assert metrics["rollout_async/accepted_queue_wait_seconds_mean"] >= 0
    assert metrics["rollout_async/group_completions_per_sec"] > 0
    assert metrics["rollout_async/trainer_consumption_groups_per_sec"] > 0
    assert metrics["rollout_async/active_trajectories"] == 2


async def test_unavailable_current_version_is_not_reported_as_zero_post_finish_lag(monkeypatch):
    group = make_group(group_index=0, weight_versions=[5])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=6),
        FakeDataSource(scripted=[group]),
        generate=generate,
    )

    class UnavailableWeightVersion:
        available = False
        success_age_seconds = None

        async def get(self, args):
            return None

    fn._finish_weight_version = UnavailableWeightVersion()
    fn._weight_version = UnavailableWeightVersion()

    output = await fn(RolloutFnTrainInput(rollout_id=0))
    await fn.close()

    metrics = output.metrics
    assert "rollout_staleness/accepted/post_finish_lag_versions_mean" not in metrics
    assert metrics["rollout_staleness/current_version_available_ratio"] == 0
    assert metrics["rollout_staleness/filter_current_version_fallback_ratio"] == 1


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_close_cancels_in_flight_groups(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_generate(
        state,
        group,
        sampling_params,
        evaluation=False,
        sample_done_callback=None,
    ):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1),
        FakeDataSource(),
        generate=blocking_generate,
    )
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await started.wait()

    await fn.close()
    await cancelled.wait()
    await asyncio.gather(drain, return_exceptions=True)

    assert fn._worker is None


async def test_worker_bounds_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_worker_failure_beats_queued_groups(monkeypatch):
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        raise RuntimeError("generation exploded")

    fn._output = asyncio.Queue(maxsize=fully_async.OUTPUT_QUEUE_MAX_GROUPS)
    group = make_group(1)
    await fn._output.put(
        fully_async._CompletedGroup(
            prompt_group=group,
            group=group,
            finished_at=time.monotonic(),
            finish_weight_version=None,
        )
    )
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(
        state,
        group,
        sampling_params,
        evaluation=False,
        sample_done_callback=None,
    ):
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
    assert len(submitted) > 1
    assert len(output.samples) == 1


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    def reject_group_1(args, group, **kwargs):
        keep = group[0].group_index != 1
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_group_1

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Unlike a recycle, a filtered group is not returned to the data source for re-sampling.
    assert data_source.recycled == []
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_weight_version_throttles_failed_queries(monkeypatch):
    """A drain queries once per group, so an unreachable router must not cost one timeout each."""
    calls = []

    async def unreachable_router(url):
        calls.append(url)
        raise httpx.ConnectError("router down")

    monkeypatch.setattr(fully_async, "get", unreachable_router)
    args = make_args()

    throttled = fully_async._CachedWeightVersion(ttl=60.0)
    assert await throttled.get(args) is None
    assert await throttled.get(args) is None
    assert len(calls) == 1

    calls.clear()
    expired = fully_async._CachedWeightVersion(ttl=0.0)
    assert await expired.get(args) is None
    assert await expired.get(args) is None
    assert len(calls) == 2


async def test_weight_version_coalesces_concurrent_queries(monkeypatch):
    calls = 0

    async def router_version(url):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"weight_version": 9}

    monkeypatch.setattr(fully_async, "get", router_version)
    version = fully_async._CachedWeightVersion(ttl=60.0)

    values = await asyncio.gather(*(version.get(make_args()) for _ in range(10)))

    assert values == [9] * 10
    assert calls == 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"weight_version": 9}, 9),
        ({"weight_version": "9"}, 9),
        ({"weight_version": {"min_version": 9}}, 9),
        ({"weight_version": {"min_version": 8, "exact_version": 9}}, 9),
    ],
)
async def test_weight_version_accepts_stitch_constraint_shape(monkeypatch, response, expected):
    async def router_version(_url):
        return response

    monkeypatch.setattr(fully_async, "get", router_version)
    version = fully_async._CachedWeightVersion(ttl=60.0)

    assert await version.get(make_args()) == expected
    assert version.available


async def test_backfill_submits_replacement_before_the_group_returns(monkeypatch):
    """With --rollout-sample-completion-backfill, finished samples free slots immediately."""
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1, rollout_sample_completion_backfill=True)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    # Report every sample of the still-pending group as finished.
    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await asyncio.sleep(0.01)

    # A replacement group went out even though the first group has not returned.
    assert data_source.num_get_calls == 2

    release.set()
    output = await drain
    assert len(output.samples) == 1
