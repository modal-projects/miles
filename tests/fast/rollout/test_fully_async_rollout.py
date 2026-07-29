import asyncio
import math
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.rollout import fully_async_rollout
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.failures import mark_infrastructure_failure
from miles.utils.types import Sample


def _args(**overrides):
    values = {
        "async_max_concurrent_samples": 1024,
        "async_max_active_groups": None,
        "rollout_batch_size": 16,
        "n_samples_per_prompt": 8,
        "sglang_router_policy": None,
        "sglang_enable_deterministic_inference": False,
        "rollout_seed": 42,
        "group_rm": False,
        "max_weight_staleness": 6,
        "update_weights_interval": 1,
        "num_rollout": 200,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _rollout(args=None):
    rollout = fully_async_rollout.FullyAsyncRolloutFn.__new__(fully_async_rollout.FullyAsyncRolloutFn)
    rollout.args = args or _args()
    rollout.state = SimpleNamespace(aborted=False, sampling_params={})
    rollout.trajectory_limit = rollout.args.async_max_concurrent_samples
    rollout.trajectory_slots = asyncio.Semaphore(rollout.trajectory_limit)
    rollout.output = asyncio.Queue()
    rollout.pool_space_available = asyncio.Event()
    rollout.worker = None
    rollout.progress_reporter = None
    rollout.refill = True
    rollout.stats = Counter()
    rollout.active_groups = 0
    rollout.active_trajectories = 0
    rollout.waiting_trajectories = 0
    rollout.max_active_groups = 0
    rollout.max_active_trajectories = 0
    minimum_groups = math.ceil(rollout.trajectory_limit / rollout.args.n_samples_per_prompt)
    rollout.pool_group_limit = rollout.args.async_max_active_groups or max(rollout.args.rollout_batch_size, math.ceil(minimum_groups * 1.5))
    rollout.active_group_started = {}
    rollout.drain_rollout_id = None
    rollout.drain_accepted_groups = 0
    rollout.drain_completed_groups = 0
    rollout.drain_aborted_groups = 0
    rollout.drain_dropped_groups = 0
    rollout.last_report_stats = Counter()
    rollout.last_report_time = time.monotonic()
    rollout.last_progress_stats = Counter()
    rollout.last_progress_time = time.monotonic()
    rollout.weight_version_base = None
    return rollout


def test_pool_depth_is_derived_from_trajectory_limit(monkeypatch):
    monkeypatch.setattr(
        fully_async_rollout,
        "GenerateState",
        lambda args: SimpleNamespace(args=args),
    )

    rollout = fully_async_rollout.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=SimpleNamespace()))

    assert rollout.trajectory_limit == 1024
    assert rollout.pool_group_limit == 192


def test_pool_depth_can_be_bounded_independently(monkeypatch):
    monkeypatch.setattr(
        fully_async_rollout,
        "GenerateState",
        lambda args: SimpleNamespace(args=args),
    )

    rollout = fully_async_rollout.FullyAsyncRolloutFn(
        RolloutFnConstructorInput(
            args=_args(async_max_concurrent_samples=544, async_max_active_groups=80),
            data_source=SimpleNamespace(),
        )
    )

    assert rollout.trajectory_limit == 544
    assert rollout.pool_group_limit == 80


@pytest.mark.asyncio
async def test_later_group_starts_when_one_trajectory_slot_opens(monkeypatch):
    rollout = _rollout(_args(async_max_concurrent_samples=2))
    rollout.trajectory_slots = asyncio.Semaphore(2)
    started: asyncio.Queue[int] = asyncio.Queue()
    releases = {index: asyncio.Event() for index in range(6)}

    async def fake_generate(state, sample, sampling_params, evaluation=False):
        del state, sampling_params, evaluation
        await started.put(sample.index)
        await releases[sample.index].wait()
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0
        return sample

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm", fake_generate)
    groups = [[Sample(index=group * 2), Sample(index=group * 2 + 1)] for group in range(3)]
    tasks = [asyncio.create_task(rollout._generate_group(group)) for group in groups]

    assert {await started.get(), await started.get()} == {0, 1}
    releases[0].set()
    assert await asyncio.wait_for(started.get(), timeout=1) == 2
    assert not tasks[0].done()

    for event in releases.values():
        event.set()
    await asyncio.gather(*tasks)

    assert rollout.active_trajectories == 0
    assert rollout.max_active_trajectories == 2
    assert rollout.stats["trajectories_finished"] == 6


@pytest.mark.asyncio
async def test_producer_returns_first_completed_groups_across_prompt_pool():
    rollout = _rollout(_args(rollout_batch_size=2, n_samples_per_prompt=1))
    rollout.pool_group_limit = 6
    releases = {}
    submitted = []

    class DataSource:
        def get_samples(self, count):
            assert count == 1
            index = len(submitted)
            submitted.append(index)
            releases[index] = asyncio.Event()
            return [[Sample(index=index)]]

    async def fake_generate_group(group):
        await releases[group[0].index].wait()
        return group

    rollout.data_source = DataSource()
    rollout._generate_group = fake_generate_group
    rollout.worker = asyncio.create_task(rollout._producer())

    while len(submitted) < rollout.pool_group_limit:
        await asyncio.sleep(0)

    releases[4].set()
    releases[1].set()
    completed = {group_id for group_id, _ in [await rollout._next_group(), await rollout._next_group()]}

    assert completed == {1, 4}
    # The persistent producer may already refill the two freed coordinator
    # slots before this coroutine resumes.
    assert submitted[:6] == list(range(6))

    rollout.refill = False
    for release in releases.values():
        release.set()
    await rollout.worker


@pytest.mark.asyncio
async def test_completed_queue_counts_toward_group_pool_limit():
    rollout = _rollout(_args(rollout_batch_size=1, n_samples_per_prompt=1))
    rollout.pool_group_limit = 2
    rollout.output = asyncio.Queue(maxsize=rollout.pool_group_limit)
    releases: dict[int, asyncio.Event] = {}
    submitted: list[int] = []

    class DataSource:
        def get_samples(self, count):
            assert count == 1
            index = len(submitted)
            submitted.append(index)
            releases[index] = asyncio.Event()
            return [[Sample(index=index)]]

    async def fake_generate_group(group):
        await releases[group[0].index].wait()
        return group

    rollout.data_source = DataSource()
    rollout._generate_group = fake_generate_group
    rollout.worker = asyncio.create_task(rollout._producer())

    while len(submitted) < 2:
        await asyncio.sleep(0)
    releases[0].set()
    releases[1].set()
    while rollout.output.qsize() < 2:
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert submitted == [0, 1]

    await rollout._next_group()
    while len(submitted) < 3:
        await asyncio.sleep(0)
    assert submitted == [0, 1, 2]

    rollout.refill = False
    for release in releases.values():
        release.set()
    await rollout.worker


@pytest.mark.asyncio
async def test_close_cancels_persistent_groups_and_background_tasks():
    rollout = _rollout(_args(rollout_batch_size=1, n_samples_per_prompt=1))
    rollout.pool_group_limit = 2
    started = asyncio.Event()

    class DataSource:
        def get_samples(self, count):
            assert count == 1
            return [[Sample(index=rollout.stats["groups_started"])]]

    async def blocked_group(group):
        started.set()
        await asyncio.Event().wait()
        return group

    rollout.data_source = DataSource()
    rollout._generate_group = blocked_group
    rollout._ensure_worker()
    await asyncio.wait_for(started.wait(), timeout=1)

    await rollout.close()

    assert rollout.worker is None
    assert rollout.progress_reporter is None
    assert rollout.active_groups == 0
    assert rollout.stats["groups_cancelled"] == 2


@pytest.mark.asyncio
async def test_completed_retry_sibling_does_not_take_a_slot(monkeypatch):
    rollout = _rollout(_args(async_max_concurrent_samples=1))
    rollout.trajectory_slots = asyncio.Semaphore(1)
    active_during_call = {}

    async def fake_generate(state, sample, sampling_params, evaluation=False):
        del state, sampling_params, evaluation
        active_during_call[sample.index] = rollout.active_trajectories
        if sample.status == Sample.Status.ABORTED:
            sample.status = Sample.Status.COMPLETED
            sample.reward = 0
        return sample

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm", fake_generate)
    completed = Sample(index=0, status=Sample.Status.COMPLETED, reward=0)
    retry = Sample(index=1, status=Sample.Status.ABORTED)

    result = await rollout._generate_group([completed, retry])

    assert [sample.status for sample in result] == [
        Sample.Status.COMPLETED,
        Sample.Status.COMPLETED,
    ]
    assert active_during_call == {0: 0, 1: 1}


def test_partial_infrastructure_failure_is_masked_without_changing_shape():
    valid = [
        Sample(
            index=index,
            group_index=7,
            tokens=list(range(10 + index)),
            response_length=2 + index,
            reward=float(index % 2),
            rollout_log_probs=[-0.1] * (2 + index),
            status=Sample.Status.COMPLETED,
            weight_versions=["4"],
        )
        for index in range(2)
    ]
    failed = Sample(
        index=2,
        group_index=7,
        status=Sample.Status.ABORTED,
        metadata={"exit_status": "session_record_timeout"},
    )
    mark_infrastructure_failure(failed)

    masked_group, masked_count = fully_async_rollout.mask_infrastructure_failures([*valid, failed])

    assert masked_count == 1
    assert masked_group is not None
    assert len(masked_group) == 3
    placeholder = masked_group[2]
    assert isinstance(placeholder, Sample)
    assert placeholder.index == failed.index
    assert placeholder.group_index == failed.group_index
    assert placeholder.remove_sample
    assert placeholder.reward == 0.0
    assert placeholder.status == Sample.Status.COMPLETED
    assert placeholder.response_length == valid[0].response_length
    assert placeholder.rollout_log_probs == valid[0].rollout_log_probs
    assert placeholder.metadata["exit_status"] == "session_record_timeout"
    assert placeholder.metadata["_fully_async_infra_masked"] is True
    assert failed.status == Sample.Status.ABORTED


def test_partial_infrastructure_failure_needs_two_valid_siblings():
    valid = Sample(
        index=0,
        tokens=[1, 2],
        response_length=1,
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    failed = Sample(index=1, status=Sample.Status.ABORTED)
    mark_infrastructure_failure(failed)

    assert fully_async_rollout.mask_infrastructure_failures([valid, failed]) == (None, 0)


def test_non_infrastructure_abort_is_not_masked():
    valid = [
        Sample(
            index=index,
            tokens=[1, 2],
            response_length=1,
            reward=float(index),
            status=Sample.Status.COMPLETED,
        )
        for index in range(2)
    ]
    policy_abort = Sample(
        index=2,
        status=Sample.Status.ABORTED,
        metadata={"exit_status": "prompt_exceeds_max_seq_len"},
    )

    assert fully_async_rollout.mask_infrastructure_failures(
        [*valid, policy_abort]
    ) == (None, 0)


@pytest.mark.asyncio
async def test_producer_failure_is_propagated():
    rollout = _rollout()

    async def fail():
        raise RuntimeError("producer failed")

    rollout.worker = asyncio.create_task(fail())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="producer failed"):
        rollout._ensure_worker()


def test_weight_version_helpers_cover_multiturn_group():
    group = [
        Sample(weight_versions=["3", "4"]),
        Sample(weight_versions=["default", "5"]),
    ]

    assert fully_async_rollout.group_oldest_weight_version(group) == 3
    assert fully_async_rollout.group_newest_weight_version(group) == 5


def test_target_weight_version_uses_publication_schedule():
    rollout = _rollout(_args(update_weights_interval=1))

    assert rollout._target_weight_version(0, observed_newest=7) == 7
    assert rollout._target_weight_version(1, observed_newest=7) == 8
    assert rollout._target_weight_version(5, observed_newest=10) == 12


def test_target_weight_version_honors_update_interval():
    rollout = _rollout(_args(update_weights_interval=2))

    assert rollout._target_weight_version(0, observed_newest=7) == 7
    assert rollout._target_weight_version(1, observed_newest=7) == 7
    assert rollout._target_weight_version(2, observed_newest=7) == 8


@pytest.mark.asyncio
async def test_drain_uses_sample_versions_without_control_plane_io():
    rollout = _rollout(_args(rollout_batch_size=2, n_samples_per_prompt=1))
    rollout.worker = asyncio.create_task(asyncio.Event().wait())

    try:
        for index in range(2):
            await rollout.output.put(
                (
                    index,
                    [
                        Sample(
                            index=index,
                            status=Sample.Status.COMPLETED,
                            reward=0,
                            weight_versions=["1"],
                        )
                    ],
                )
            )
        first = await asyncio.wait_for(rollout._drain(0), timeout=0.1)
        assert len(first.samples) == 2
        assert first.metrics["rollout_staleness/target_weight_version"] == 1
        assert first.metrics["rollout_staleness/candidate/oldest_lag_max"] == 0
        assert first.metrics["rollout_staleness/accepted/oldest_lag_max"] == 0
        assert first.metrics["rollout_staleness/recycled_groups"] == 0

        for index in range(2, 4):
            await rollout.output.put(
                (
                    index,
                    [
                        Sample(
                            index=index,
                            status=Sample.Status.COMPLETED,
                            reward=0,
                            weight_versions=["1"],
                        )
                    ],
                )
            )
        second = await asyncio.wait_for(rollout._drain(1), timeout=0.1)
        assert len(second.samples) == 2
        assert second.metrics["rollout_staleness/target_weight_version"] == 2
        assert second.metrics["rollout_staleness/candidate/oldest_lag_max"] == 1
        assert second.metrics["rollout_staleness/accepted/oldest_lag_max"] == 1
    finally:
        rollout.worker.cancel()
        await asyncio.gather(rollout.worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_drain_separates_accepted_and_recycled_staleness():
    added = []
    rollout = _rollout(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            max_weight_staleness=0,
        )
    )
    rollout.data_source = SimpleNamespace(add_samples=added.extend)
    rollout.weight_version_base = 1
    rollout.worker = asyncio.create_task(asyncio.Event().wait())
    stale = Sample(
        index=0,
        status=Sample.Status.COMPLETED,
        reward=0,
        weight_versions=["1"],
    )
    fresh = Sample(
        index=1,
        status=Sample.Status.COMPLETED,
        reward=0,
        weight_versions=["3"],
    )
    await rollout.output.put((0, [stale]))
    await rollout.output.put((1, [fresh]))

    try:
        result = await asyncio.wait_for(rollout._drain(2), timeout=0.1)
    finally:
        rollout.worker.cancel()
        await asyncio.gather(rollout.worker, return_exceptions=True)

    assert result.samples == [[fresh]]
    assert added == [[stale]]
    assert result.metrics["rollout_staleness/candidate/oldest_lag_max"] == 2
    assert result.metrics["rollout_staleness/accepted/oldest_lag_max"] == 0
    assert result.metrics["rollout_staleness/recycled/oldest_lag_max"] == 2
    assert result.metrics["rollout_staleness/accepted_groups"] == 1
    assert result.metrics["rollout_staleness/recycled_groups"] == 1


@pytest.mark.asyncio
async def test_drain_keeps_valid_siblings_from_partial_infrastructure_failure():
    rollout = _rollout(
        _args(
            rollout_batch_size=1,
            n_samples_per_prompt=3,
        )
    )
    rollout.worker = asyncio.create_task(asyncio.Event().wait())
    group = [
        Sample(
            index=index,
            group_index=0,
            tokens=[1, 2, 3],
            response_length=1,
            rollout_log_probs=[-0.1],
            reward=float(index),
            status=Sample.Status.COMPLETED,
            weight_versions=["1"],
        )
        for index in range(2)
    ]
    group.append(
        Sample(
            index=2,
            group_index=0,
            status=Sample.Status.ABORTED,
            metadata={"exit_status": "verifier_transport_error"},
        )
    )
    mark_infrastructure_failure(group[-1])
    await rollout.output.put((0, group))

    try:
        result = await asyncio.wait_for(rollout._drain(0), timeout=0.1)
    finally:
        rollout.worker.cancel()
        await asyncio.gather(rollout.worker, return_exceptions=True)

    accepted = result.samples[0]
    assert len(accepted) == 3
    assert sum(sample.remove_sample for sample in accepted) == 1
    assert result.metrics["rollout_failure/infra_masked_groups"] == 1
    assert result.metrics["rollout_failure/infra_masked_trajectories"] == 1
    assert result.metrics["rollout_failure/dropped_groups"] == 0
    assert result.metrics["rollout_async/trainable_trajectories"] == 2


def test_metrics_include_generic_lifetime_progress():
    rollout = _rollout()
    rollout.stats.update(
        {
            "groups_finished": 2,
            "trajectories_finished": 16,
            "trajectory_status/completed": 8,
            "trajectory_status/aborted": 8,
        }
    )
    accepted = Sample(
        status=Sample.Status.COMPLETED,
        metadata={
            "session_collect/get_seconds": 10.0,
            "agent_metrics": {
                "total_time": 10.0,
                "model_request_time": 8.0,
                "interaction_time": 2.0,
                "generation_time_ratio": 0.8,
            },
        },
    )
    rejected = Sample(
        status=Sample.Status.ABORTED,
        metadata={
            "exit_status": "verifier_timeout",
            "session_collect/get_seconds": 20.0,
            "agent_metrics": {
                "total_time": 20.0,
                "model_request_time": 5.0,
                "interaction_time": 15.0,
                "generation_time_ratio": 0.25,
            },
        },
    )

    metrics = rollout._metrics(
        started=time.monotonic() - 1,
        completed_groups=2,
        aborted_groups=1,
        dropped_groups=1,
        unusable_groups=0,
        infra_masked_groups=0,
        infra_masked_trajectories=0,
        stale_groups=0,
        staleness=[],
        newest_staleness=[],
        version_spans=[],
        accepted_staleness=[],
        accepted_newest_staleness=[],
        accepted_version_spans=[],
        recycled_staleness=[],
        recycled_newest_staleness=[],
        recycled_version_spans=[],
        target_weight_version=4,
        accepted_groups=[[accepted]],
        observed_groups=[[accepted], [rejected]],
    )

    assert metrics["rollout_async/lifetime/groups_finished"] == 2
    assert metrics["rollout_async/batch_delta/trajectory_status/aborted"] == 8
    assert metrics["rollout_staleness/target_weight_version"] == 4
