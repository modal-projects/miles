"""Paces prompt-group submission for the rollout drivers (--rollout-submission-granularity).

The driver owns the pending group tasks and passes in the samples they hold as ``pending_samples``,
with the size of the group it would submit next; the scheduler only accounts samples in flight.
"""

import asyncio
import logging
from argparse import Namespace

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class GroupLevelSubmission:
    """A submission slot frees only when the whole group task returns."""

    sample_done_callback = None

    def has_capacity(self, *, pending_samples: int, group_size: int, sample_budget: int) -> bool:
        return pending_samples + group_size <= sample_budget

    def on_submit(self, groups: list[list[Sample]]) -> None:
        pass

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)


class SampleBackfillSubmission:
    """Each finished sample frees its own slot; the next group fits once enough samples finish for its size."""

    def __init__(self):
        self.samples_in_flight = 0
        self._sample_done = asyncio.Event()

    def sample_done_callback(self) -> None:
        self.samples_in_flight -= 1
        self._sample_done.set()

    def has_capacity(self, *, pending_samples: int, group_size: int, sample_budget: int) -> bool:
        """A False return arms the sample wakeup: only later completions wake ``wait_for_progress``."""
        if pending_samples == 0 and self.samples_in_flight:
            # a group returned without spawning its sample tasks, or a stub skipped the callback
            logger.warning(f"samples_in_flight={self.samples_in_flight} with no pending groups; resetting to 0")
            self.samples_in_flight = 0
        if self.samples_in_flight + group_size <= sample_budget:
            return True
        self._sample_done.clear()
        return False

    def on_submit(self, groups: list[list[Sample]]) -> None:
        self.samples_in_flight += sum(len(group) for group in groups)

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        waiter = asyncio.create_task(self._sample_done.wait())
        try:
            done, pending = await asyncio.wait(pendings | {waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        return done - {waiter}, pending - {waiter}


def make_submission_scheduler(args: Namespace, *, default: str) -> GroupLevelSubmission | SampleBackfillSubmission:
    granularity = args.rollout_submission_granularity or default
    if granularity == "group":
        return GroupLevelSubmission()
    assert granularity == "sample", f"unknown submission granularity: {granularity}"
    return SampleBackfillSubmission()
