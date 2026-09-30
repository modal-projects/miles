---
title: "Score Centering"
description: "Center off-policy policy gradients using the sampler's top-k probabilities, with optional TIS or MIS weights."
# Generated from examples/infra_features/score_centering/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Correct off-policy score drift using the sampler's top-k probabilities, with optional truncated or masked importance weights.

This implements [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807), including the efficient top-k approximation in Appendix A.

## Enable it

Add these arguments to an existing text-only GRPO training recipe:

```bash
--loss-type score_centering \
--advantage-estimator grpo \
--rollout-top-logprobs-num 128 \
--score-centering-is none \
--rollout-temperature 1.0 \
--rollout-top-p 1.0 \
--rollout-top-k -1 \
--use-rollout-logprobs \
--disable-grpo-std-normalization \
--calculate-per-token-loss
```

For session-server rollouts using more than 20 candidates, also set `--use-miles-router`.

Keep reward mean subtraction enabled. Disabling standard-deviation normalization gives the paper's group-centered rewards. This is a separate REINFORCE-style loss: PPO clipping parameters do not apply. Existing batch size and update scheduling still control how many updates consume a rollout batch; choose them explicitly when reproducing an experiment.

Choose `--score-centering-is tis` for weights clipped at `--score-centering-tis-clip` (default 2). Choose `mis` to retain ratios in `[--score-centering-mis-low, --score-centering-mis-high]` (defaults 0.5 and 5), setting other weights to zero. These weights are centered together with the score. Use these options instead of `--use-tis` or a custom TIS function.

The existing entropy and reference-KL loss options remain available. They are separate regularizers; the score-centering identity applies to the policy-gradient term.
With filtered sampling, sampling-support replay rules prohibit reference KL; entropy is computed over the recorded support.

Training logs include `train/train_rollout_logprob_abs_diff` and `train/train_rollout_kl`. The latter uses the same masked, sampled-token k3 estimator of KL(rollout || train) as the policy loss. It is a detached diagnostic and is emitted even when reference-KL regularization is disabled.

## How it works

When the rollout distribution differs from the current trainer, even a constant reward can produce an unwanted average policy update. Score centering subtracts the expected weighted score under the rollout distribution.

Let `p` be the current trainer distribution, `q` the distribution that actually sampled the token, `H` the stored candidates, `A` the detached advantage, and `f` the selected importance-weight function. The implementation computes:

```text
rho   = max(1 - sum(q[H]), 1e-6) / max(1 - sum(p[H]), 1e-6)
alpha = rho * f(1 / rho)
loss  = -A * (stop_gradient(f(p[token] / q[token])) * log(p[token])
             - sum(stop_gradient(q[H] * f(p[H] / q[H]) - alpha * p[H]) * log(p[H])))
```

Outside `H`, the approximation models `q` as `rho * p`. The sampled token always uses its recorded `q[token]`, including when it is outside `H`. All weights and correction coefficients are detached. Full-distribution centering cancels the expected constant-reward gradient; the top-k version approximates the true tail and does not guarantee exact cancellation for an arbitrary tail.

`--rollout-top-logprobs-num K` plays a different role in the two sampling modes:

- **Unfiltered sampling** (`--rollout-top-p 1.0 --rollout-top-k -1`): Miles automatically uses `selected` mode. The sampler draws from the full vocabulary, so Miles records only its top `K` tokens as `H` and the tail model covers the rest. `K` is a real truncation and must be set; a larger `K` leaves less to the tail model.
- **Filtered sampling** (top-p/top-k): Miles automatically uses `support` mode when candidate recording is enabled. The sampler draws only from the realized support, and Miles records that whole support with its post-filter probabilities. `H` is then the full sampling distribution and nothing is approximated. `K` only sizes the arrays: it must be at least `--rollout-top-k` and hold the realized support, which cutoff ties can make larger; a larger support fails validation. The trainer renormalizes its log probabilities over the same support.

SGLang reports three kinds of log probability:

| SGLang field | Distribution | Filtered | Sums to 1 on the support |
| --- | --- | --- | --- |
| `output_token_sampling_logprobs` (selected or support mode) | `q'`: renormalized after top-k/top-p filtering | yes | yes |
| `output_token_logprobs`, `output_top_logprobs` | `p_T = softmax(logits / T)` over the full vocabulary | no | no, less than 1 under filtering |
| `output_token_logprobs`, `output_top_logprobs` with `SGLANG_RETURN_ORIGINAL_LOGPROB=1` | `log_softmax(logits)`, without the temperature | no | — |

The first row is the sampler under filtering and the second the sampler without filtering. Miles-managed rollout workers set `SGLANG_RETURN_ORIGINAL_LOGPROB=0`, so the third row does not occur. Each configuration reads:

| Configuration | Sampling-support replay (`append_sampling_metadata`) | Candidates (`append_rollout_topk_logprobs`) | SGLang source |
| --- | --- | --- | --- |
| Sampling-support replay only (`--rollout-top-logprobs-num 0`) | stores the support; returns the sampled-token log probability | nothing (`K = 0`) | `output_token_sampling_mask`, `output_token_sampling_logprobs` |
| Score centering, unfiltered | not called (no `return_sampling_mask`) | stores the top `K` candidates | `output_top_logprobs` |
| Score centering, filtered | stores the support; picks the sampled-token log probability from its row | stores `[n, K]`: the same support IDs with their row log probabilities | both read the same `output_token_sampling_mask`, `output_token_sampling_logprobs` |

The trainer computes only the requested log probabilities from each vocabulary shard, reducing normalization scalars and selected logits across tensor-parallel ranks. It excludes padded vocabulary entries and computes probabilities in float32 for BF16/FP16 models. `--log-probs-chunk-size` controls the temporary computation size; `--recompute-loss-function` can trade computation for saved activations. No full vocabulary is gathered across ranks.

## Rollout and data contract

Native SGLang generation, the legacy rollout path, and both session-server versions request candidate probabilities at generation time when `--rollout-top-logprobs-num` is positive. Miles derives the logprob mode at startup from `--rollout-top-p` and `--rollout-top-k`. On these training requests, the configured count and derived mode override any client-supplied `top_logprobs`, `top_logprobs_num`, or `sampling_logprobs_mode`. A count of zero disables candidate recording without requesting support-wide probabilities. Each `Sample` carries:

- `rollout_topk_token_ids`: int32 array shaped `[response_length, k]`.
- `rollout_topk_log_probs`: float32 array of the same shape.
- `rollout_log_probs`: the actual sampled-token log probabilities.

Evaluation requests skip this collection and may use independent sampling settings, including greedy decoding. The built-in agentic producer marks evaluation sessions when creating them; custom session clients should create them with `POST /sessions` with JSON body `{"evaluation": true}`.

For unfiltered session rollouts, more than 20 candidates require `--use-miles-router`. The SGLang Rust router caps OpenAI `top_logprobs` at 20, while MilesRouter forwards a larger request to SGLang unchanged. To use the SGLang router for unfiltered session rollouts, set `--rollout-top-logprobs-num 20` or less. Filtered rollouts request support probabilities instead of `top_logprobs` and do not use that cap. Native `/generate` rollouts support either router.

Unused candidate slots and non-trained observation rows contain token ID `-1` and log probability `-inf`. Tool-observation masks, multi-turn merging, retries, trailing-token trimming, and truncation preserve row alignment. Session serialization and data-parallel sharding retain both arrays. Candidates stay on CPU until the trainer selects its context-parallel rows.

Custom rollout producers must supply these fields with probabilities from the actual generation call and no repeated non-negative token ID in a row; Miles does not check uniqueness. Missing candidates or disagreeing sampled/candidate probabilities fail validation. Rescoring old rollouts with newer weights is not a substitute. With the feature disabled, requests and the session wire format remain unchanged.

## Supported configurations and limits

- The shared loss is wired into Megatron and FSDP. Candidate selection supports tensor parallelism, packed (`thd`) and padded (`bshd`) zigzag context parallelism, and packed all-gather context parallelism.
- Sampling requires a fixed positive temperature and `min_p=0` on every call. Filtered sampling, including top-p filtering, automatically uses support mode and requires a positive `top_k`, for example `--rollout-top-p 0.9 --rollout-top-k 64 --rollout-top-logprobs-num 128`. Global filtered rollout settings automatically enable sampling-support replay; per-request overrides are checked when each request is built. See the [sampling-support replay guide](/advanced/sampling-support-replay) for its request and server requirements.
- Filtered sampling requires SGLang with support log probabilities (SGLang PR [#40932](https://github.com/sgl-project/sglang/pull/40932), included in the `sglang-miles` branch by [#41047](https://github.com/sgl-project/sglang/pull/41047)). External servers must set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` like Miles-managed workers. OpenAI session responses must expose the SGLang fields above in `choices[0].meta_info`; a generic OpenAI-compatible server without that metadata is insufficient.
- Constrained/custom sampling, speculative decoding, true-on-policy mode, OPD, multi-LoRA/Tinker losses, sequence masking, custom policy-loss reducers, custom train-data converters, and logprob recomputation via prefill are rejected. Multimodal token expansion is not supported. The initial advantage estimator is GRPO.
- Retaining `k=128` uses about 1 KiB per response position for the two arrays, before transport overhead. Larger `k` improves the tail approximation at additional storage and compute cost.

## Metrics and verification

Training logs include `sc_correction`, `sc_train_head_mass`, `sc_rollout_head_mass`, `sc_tail_ratio`, `sc_importance_weight`, `train_rollout_kl`, and `train_rollout_logprob_abs_diff`, under the usual `train/` namespace. Small head mass means more of the distribution is approximated by the tail model. Large tail ratios indicate a substantial mismatch in remaining mass.
For filtered sampling, the head covers the full support, so both head masses and the tail ratio should be approximately one.

The numerical tests compare gradients against an independent dense-distribution oracle for all three weighting modes, including sampled tokens outside the head, constant rewards, and tiny tails. Pipeline tests exercise real native/session producers, serialization, data-parallel splitting, masks, advantage computation, regularization, and checkpointed loss scaling. Run the focused tests in the repository's test environment:

```bash
python -m pytest tests/fast/backends/training_utils/test_score_centering.py \
    tests/fast/backends/training_utils/test_score_centering_pipeline.py \
    tests/fast/backends/training_utils/test_score_centering_filtered.py
python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
MILES_TEST_CUDA_DISTRIBUTED=1 python -m pytest \
    tests/fast/backends/training_utils/test_score_centering_distributed.py
```

The distributed tests use four CPU/Gloo processes or four CUDA/NCCL processes (TP=2, CP=2), all three context layouts and weighting modes, plus BF16 selected-probability gradients. The independent dense-gradient oracle also runs on CUDA when available. These are correctness tests, not a reproduction of the paper's GPU training results.

For a real SGLang server, also run the opt-in protocol probe:

```bash
MILES_LIVE_SCORE_CENTERING_ENDPOINT=http://127.0.0.1:30000 \
MILES_LIVE_SCORE_CENTERING_MODEL=/path/to/model \
MILES_LIVE_SCORE_CENTERING_SERVED_MODEL=your-served-model \
python -m pytest --confcutdir=tests/manual tests/manual/test_score_centering_live.py
```

Set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` on the server before starting it. The probe checks native and OpenAI response metadata using the production candidate collector and validator, and checks temperature scaling at 0.7, 1.0 and 1.3. It warms the shared prompt first so that cached and uncached prefills do not confound the temperature comparison. Set `MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR` to keep the raw responses.
