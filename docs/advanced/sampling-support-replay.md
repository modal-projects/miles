---
title: Sampling-Support Replay
description: Train against the same bounded sampling distribution used during rollout.
---

Top-p and top-k change the distribution that generates a rollout token. They
remove tokens from the vocabulary and renormalize the remaining probability
mass. Recomputing a token's log probability over the full vocabulary therefore
does not reproduce its behavior-policy probability, even when the rollout and
trainer weights are identical.

Sampling-support replay preserves that distribution. For every generated token,
SGLang returns the realized set of token IDs that remained after sampling
filters. Miles transports this ragged support with the sample and applies it to
the actor logits before the softmax:

$$
q_\theta(a_t \mid S_t) =
\frac{\exp(z_\theta(a_t) / T)}
{\sum_{j \in S_t} \exp(z_\theta(j) / T)}.
$$

The support `S_t` is fixed rollout data; `z_θ` comes from the actor being
trained. This gives PPO/GRPO a denominator and numerator in the same sampled
probability space while retaining gradients through the current actor logits.

## Enable replay

```bash
ROLLOUT_ARGS+=(
  --rollout-temperature 1.0
  --rollout-top-p 0.95
  --rollout-top-k 64
  --use-miles-router
)
```

Replay is enabled whenever `--rollout-top-p` is below `1` or
`--rollout-top-k` is positive. A finite top-p run must also set a positive
top-k so that the captured support is bounded. Miles asks SGLang to capture its
native sampling support with `return_sampling_mask` and uses the returned
support-normalized log probability as the rollout log probability.

`--rollout-top-k` is the default for rollout requests, not a global upper bound
on request-specific top-k values. SGLang owns support capacity through
`--sglang-sampling-mask-max-tokens` (or the matching server-group override) and
rejects requests whose realized support cannot be represented.

## Correctness requirements

Miles rejects configurations that it cannot replay faithfully:

- Every training request must state its top-p, top-k, and temperature.
- Request temperature must match `--rollout-temperature`.
- Frequency, presence, and repetition penalties and `logit_bias` are not
  supported because the trainer does not replay those logit transformations.
- `--recompute-logprobs-via-prefill` is incompatible because that path does not
  preserve the per-token support.
- The Miles router is currently required as a transport compatibility measure.
  The SGLang v0.5.20 gateway deserializes chat requests through a typed schema
  that does not retain `return_sampling_mask`, so the engine never receives the
  capture request. The gateway's non-streaming response path does preserve raw
  response bytes; response stripping is not the issue.

SGLang owns compatibility with custom logit processors and the physical support
limit. Speculative decoding is not restricted by Miles; it works when the
SGLang backend returns the same native support and normalized log probability
for every accepted output token.

Tool and environment tokens are recorded with singleton support. Multi-turn
sessions concatenate supports across assistant turns and singleton supports
across observations, so every response token remains aligned with exactly one
support. Evaluation requests do not capture this training-only metadata.

## Current objective limitation

The actor produces full-vocabulary logits, but the current loss interface
returns one actor score per token. With replay enabled, that score is normalized
over the captured support for the policy ratio. Reference KL and on-policy
distillation also need the actor score normalized over the full vocabulary, so
Miles currently rejects `--use-kl-loss`, nonzero `--kl-coef`, and `--use-opd`
with replay.

This is an interface limitation, not a mathematical conflict. Both scores can
be derived from the same actor forward pass once the loss path carries them as
separate values.

## Monitor replay

Track `train/train_rollout_logprob_abs_diff`. On the first update from identical
rollout and actor weights, the value should be close to the numerical tolerance
of the two inference paths. Later, it also includes legitimate policy staleness
and weight updates, so interpret it together with version-lag and clipping
metrics rather than as a standalone correctness test.
