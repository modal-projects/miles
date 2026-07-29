# Fully Asynchronous Rollout Example

This example enables Miles' persistent, completion-ordered rollout producer. It
keeps generating while the learner trains and returns whichever prompt groups
finish first.

## Files
* `miles/rollout/fully_async_rollout.py`: the core class-based persistent rollout pool.
* `run-qwen3-4b-fully_async.sh`: example launch script with Qwen3‑4B.

## Prerequisite
First set up model & environment following the Qwen3-4B example.

## Quick Start
```bash
cd miles
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```
You should see `Started fully-async rollout producer` in the rollout manager log.

## How It Works (Very Short)
* The class runs on Miles' shared rollout event loop.
* `--async-max-concurrent-samples` limits live trajectories.
* The prompt-group pool is derived from that limit with headroom for slow siblings.
  `--async-max-active-groups` can set an explicit active-plus-completed group bound.
* Each learner call drains `--rollout-batch-size` completed groups; unfinished groups continue.
* The completed queue is bounded and applies backpressure when the learner falls behind.

## Limitations
* No evaluation mode.
* Selection is completion ordered (then sorted by sample index for presentation).
* Partial infrastructure failures preserve valid siblings; unusable groups follow
  the next fresh prompt group to finish.

## Config Differences (2 Key Points)
To enable the continuous producer there are two required changes compared to a
normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Set the rollout function path:
	```bash
	--rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn
	```

For immediate weight publication while rollouts remain in flight, also set:

```bash
--pause-generation-mode in_place \
--update-weights-with-inflight-rollouts
```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the class owns a persistent producer on Miles' shared rollout event loop. Each call from `train_async.py` drains completed samples while the same producer continuously refills the pool.
