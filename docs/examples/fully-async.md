---
title: Fully Async Rollout
description: Continuously generate completion-ordered prompt groups while the learner trains.
---

Fully async rollout separates rollout production from learner batches. A persistent
producer keeps prompt groups in flight, and each learner call drains whichever complete
first. Slow groups remain active and may be consumed by a later learner step.

## Quick start

Start from a normal async-training recipe and make two changes:

```diff
- python3 train.py ...
+ python3 train_async.py ...
+   --rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn
```

The Qwen example is runnable directly:

```bash
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```

The rollout function uses Miles' class-based rollout API and shared async event loop.
It does not create a separate thread or example-local global worker.

## Scheduling

`--async-max-concurrent-samples` is the trajectory-level concurrency limit. The
producer derives its prompt-group pool from that value and adds headroom so one slow
sibling does not leave trajectory capacity idle. `--rollout-batch-size` is only the
number of completed prompt groups returned to one learner step.

`--async-max-active-groups` optionally sets the prompt-group pool explicitly. Its
limit covers active generation plus completed groups waiting in the queue; completed
work cannot accumulate behind the configured bound.

For example, with eight trajectories per prompt and 1,024 concurrent trajectories,
the producer maintains up to 192 prompt groups. A learner batch of 32 groups can
therefore contain groups submitted at very different times, based only on completion.

The completed queue is bounded. If the learner falls behind, backpressure stops the
producer from pulling an unbounded number of prompts.

## Failures and staleness

Policy outcomes, including command and episode limits, are completed samples and retain
their assigned reward. Environment adapters mark infrastructure failures explicitly.
When a group has at least two valid siblings, only those marked failures become
zero-loss placeholders and the valid trajectories are still trained. Otherwise the
group is dropped and replaced by the next fresh prompt to finish.

When `--max-weight-staleness` is set, groups whose oldest generated token is more
than that many learner publications behind are recycled. When it is unset, staleness
is observed and logged but not filtered.

## Metrics

The primitive reports generic scheduler metrics under:

- `rollout_async/`: pool occupancy, queue depth, throughput, slot wait, and group wall time.
- `rollout_staleness/`: oldest/newest lag, within-group version span, and observed sample-version spread.
- `rollout_failure/`: infrastructure abort, masking, and drop counts.

Environment-specific metrics belong in `--custom-rollout-log-function-path`, not in
the scheduler.

## Limitations

- Evaluation requires a separate rollout function.
- Selection is completion ordered; returned groups are sorted by sample index only for
  deterministic downstream presentation.
