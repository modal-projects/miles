from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import logging
from argparse import Namespace

import pytest

from miles.rollout.submission_scheduler import (
    GroupLevelSubmission,
    SampleBackfillSubmission,
    make_submission_scheduler,
)
from miles.utils.types import Sample

GROUP_SIZE = 2


def _make_group(group_index: int) -> list[Sample]:
    return [
        Sample(group_index=group_index, index=group_index * 10 + i, prompt="prompt", response="ok")
        for i in range(GROUP_SIZE)
    ]


def _parked_task() -> asyncio.Task:
    return asyncio.create_task(asyncio.Event().wait())


async def _settle(times: int = 20) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class TestWaitForProgress:
    async def test_the_groups_that_are_still_running_stay_pending(self) -> None:
        """The driver resubmits from what it gets back, so a still-running group must not be reported done."""
        scheduler = GroupLevelSubmission()
        released = asyncio.Event()
        finishing = asyncio.create_task(released.wait())
        parked = _parked_task()

        waiting = asyncio.create_task(scheduler.wait_for_progress({finishing, parked}))
        await _settle()
        assert not waiting.done()

        released.set()
        done, pending = await waiting

        assert done == {finishing}
        assert pending == {parked}
        parked.cancel()


class TestSampleBackfillLeak:
    def test_a_leaked_in_flight_count_is_reported_before_it_is_reset(self, caplog) -> None:
        """Silently resetting the count would hide a generate function that never reports its samples."""
        scheduler = SampleBackfillSubmission()
        scheduler.on_submit([_make_group(1)])

        with caplog.at_level(logging.WARNING, logger="miles.rollout.submission_scheduler"):
            assert scheduler.has_capacity(pending_samples=0, group_size=GROUP_SIZE, sample_budget=2 * GROUP_SIZE)

        assert scheduler.samples_in_flight == 0
        assert "resetting to 0" in caplog.text


class TestGroupsOfTheirOwnSize:
    def test_the_next_group_fits_by_its_own_size_once_enough_samples_finish(self) -> None:
        """A smaller group may go out as soon as its own samples' worth of budget frees, not a default group's."""
        scheduler = SampleBackfillSubmission()
        scheduler.on_submit([[Sample(index=index) for index in range(6)]])
        scheduler.sample_done_callback()

        assert scheduler.has_capacity(pending_samples=6, group_size=3, sample_budget=8)
        assert not scheduler.has_capacity(pending_samples=6, group_size=4, sample_budget=8)

    def test_a_pending_group_holds_every_sample_it_was_drawn_with(self) -> None:
        """At group granularity a group gives its budget back only when it returns, however small it is."""
        scheduler = GroupLevelSubmission()

        assert scheduler.has_capacity(pending_samples=3, group_size=5, sample_budget=8)
        assert not scheduler.has_capacity(pending_samples=4, group_size=5, sample_budget=8)


class TestMakeSubmissionScheduler:
    def test_an_unknown_granularity_is_refused(self) -> None:
        """Falling back silently would pace the run in a way the user never asked for."""
        args = Namespace(rollout_submission_granularity="batch", n_samples_per_prompt=GROUP_SIZE)

        with pytest.raises(AssertionError, match="unknown submission granularity"):
            make_submission_scheduler(args, default="sample")
