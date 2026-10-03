# DFlash score-centering validation

These probes separate sampling correctness, captured metadata, loss arithmetic,
model-forward numerical differences, and actual training integration. Matching
generated strings at the same random seed is not a correctness criterion.

## Revisions and scope

- Miles stack base: `1145f1e8a451435b469095e38caf64161b4a9b09`
  ([upstream #3732](https://github.com/radixark/miles/pull/3732)).
- SGLang base: `sglang-miles` at
  `2b7313dc21a57251f568553247d36d78e585a614`, with the individually listed
  dependencies in [modal-projects/sglang#69](https://github.com/modal-projects/sglang/pull/69).
  `stitch-sglang-v0.5.20` is a patch reference, not an integration dependency.
- Target: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Draft: `incoai/Qwen3.8-27B-DFlash2`, revision
  `015e795645c74b1a0eeef3b570031fb62e769bc5`.

Record the final Miles/SGLang commits, dirty-file hashes, model revisions,
image digest, server arguments, package versions and GPU model with each run.
Image-installed package Git metadata can be stale when using source overlays;
preserve the imported source paths and an archive hash too.

The experiment image is
`radixark/miles@sha256:2b6fa5afa2b52f53fe870b842b62d7fcb2e258cac420018fb34d7a6ab41f1f0b`.
The source overlay reused its `rust_tree_core/mem_cache` extension, SHA256
`254d907ca70c4c0190bbccae85420f9e63ff63e55b83fd10c21e1affca951a1e`, after
checking that the Python adapters and Rust sources match the image baseline.

The tested target uses BF16, FA3 full attention, float32 Mamba state and the
`extra_buffer` Mamba cache strategy. DFlash2 uses block size 8. Both arms use
explicit `top_k_first` filtering. The upstream filtering-order patch changes a
default; that change is not a mathematical requirement of score centering.

This is initially DFLASH support. DFlash1 and DFlash2 share the algorithm name,
but have different verification paths. Synthetic CUDA tests cover both paths;
the real-model experiment exercises DFlash2. Neither demonstrates all-model or
all-draft compatibility, and three training updates do not establish long-run
stability.

The categorical verification kernels consume probabilities and token IDs, so
their distribution-law tests do not depend on the model architecture. The draft
method still chooses the verification path and constructs and aligns its inputs;
the live checks establish that those tensors and returned metadata correspond
to the right target prefixes for this model and DFlash2 implementation.

## Inference and estimator checks

Use an environment with the Miles and dependency PR sources installed and the
pinned checkpoints downloaded as `/models/Qwen3.8-27B` and
`/models/Qwen3.8-27B-DFlash2`. Source overlays must retain the image's compatible
compiled SGLang extensions. Install the optional observer from the Miles root:

```bash
pip install --no-deps tests/manual/score_centering_spec/plugin
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
export SGLANG_EXPOSE_OWN_ENV_VARS=1
export MILES_SCORE_CENTERING_TRACE_DIR=/artifacts/traces
python -m sglang.launch_server \
  --model-path /models/Qwen3.8-27B --host 127.0.0.1 --port 30000 \
  --tp 1 --dtype bfloat16 --attention-backend fa3 \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
  --kv-cache-dtype bfloat16 --mem-fraction-static 0.75 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --sampling-mask-max-tokens 128 --sampling-filter-order top_k_first \
  --random-seed 20261003 --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8 \
  --speculative-accept-threshold-single 1.0 \
  --speculative-accept-threshold-acc 1.0
```

Run from another shell with the same source environment:

```bash
python -m tests.manual.score_centering_spec.probe \
  --endpoint http://127.0.0.1:30000 --model /models/Qwen3.8-27B \
  --output /artifacts/spec-matrix --concurrency 8
python -m tests.manual.score_centering_spec.check_trace \
  --responses /artifacts/spec-matrix/responses.jsonl \
  --trace-dir /artifacts/traces --output /artifacts/spec-matrix/trace-check.json
python -m tests.manual.score_centering_spec.endpoint_checks \
  --endpoint http://127.0.0.1:30000 --model /models/Qwen3.8-27B \
  --output /artifacts/spec-endpoints
```

The matrix contains 12 sampling/capture settings and output limits of
1, 7, 8, 9, 17, 32 and 64 tokens. It tests unfiltered capture K=32/128,
sampling top-k=32, and top-k=64/top-p=0.9, at temperatures 0.7/1/1.3.
Filtered requests reserve capture width 128 because ties can make realized
top-k support larger than the requested sampling top-k. An overflow must fail;
truncating support or renormalizing only the captured subset is invalid.

The observer checks the exact target probabilities passed to stochastic
verification for each committed row and saves full logits for a bounded subset.
It covers accepted drafts,
rejection bonuses and all-accepted bonuses, then aligns them with emitted
metadata. The first response token comes from ordinary prefill and is outside
the DFlash trace. Extra committed rows after termination are counted separately.
The probe checks complete retained-row coverage and rejects aborted responses.

An independent NumPy float64 reference checks the actual verifier distribution
at L1 tolerance `2e-5`; emitted log probabilities must agree within `2e-5`.
The top-k/top-p reference preserves cutoff ties. Full-logit snapshots also check
the production none/TIS/MIS score-centering gradients against an independent
dense surrogate at absolute tolerance `1e-7` and relative tolerance `1e-5`.
This gradient check tests estimator arithmetic; it does not substitute for
training or prove independent model-forward equivalence.

`endpoint_checks` checks natural EOS and string stops through both native
generation and non-streaming OpenAI Chat Completions, including the production
Miles session-record assembler. Repeat it against the actual router URL to
check its request/response field preservation; direct-engine success alone does
not establish router compatibility.

## Non-speculative controls and performance

Restart the server without the three speculative model/algorithm/block flags
and the two acceptance-threshold flags, keeping the target and other settings
fixed. Replay identical recorded prefixes:

```bash
python -m tests.manual.score_centering_spec.replay \
  --responses /artifacts/spec-matrix/responses.jsonl \
  --endpoint http://127.0.0.1:30000 --output /artifacts/regular-replay
```

Each prefix is evaluated twice, reporting both cross-arm and reference-to-itself
differences. Prefill, decode and speculative verification can use different
floating-point kernels; these comparisons diagnose model-forward differences
separately from the exact-logit oracle. Also run a small regular control using
the unmodified pinned `sglang-miles` source. For filtered baseline comparisons,
use its existing order and explicitly select that same order in the patched
control, so dependency changes are not confused with speculative decoding.

For timings, **restart with `MILES_SCORE_CENTERING_TRACE_DIR` unset**. The observer
copies GPU tensors to the CPU and synchronizes. Measure regular/speculative and
capture on/off, with identical sampling settings, prompts, output limits and
concurrency. For example, repeat the following with separate output directories
and then with `--no-capture`:

```bash
python -m tests.manual.score_centering_spec.probe \
  --endpoint http://127.0.0.1:30000 --model /models/Qwen3.8-27B \
  --output /artifacts/timing-regular-capture-1 --concurrency 8 \
  --cases unfiltered128_t1,topk64_topp90_t1 --prompts 32 --lengths 256
```

Warmup exercises multiple verification/decode blocks and the selected capture
mode outside timing. Report medians and spread over repeated runs. The reported
rate includes HTTP transport and result serialization; expensive diagnostic
sample validation happens after the timer. Compare capture overhead within
each arm and spec/non-spec speed at the same capture setting.

## Short training integration

`train_probe` uses the existing Qwen dense recipe on one eight-GPU node, trainer
TP4/DP2, two TP4 rollout replicas, CPU optimizer offload, and three optimizer
steps. The initial synchronization and two post-update synchronizations exercise
two rollouts after target updates. The standard training loop skips the final
handoff because no rollout follows it. Both arms start from the same
converted checkpoint and use the same dataset, objective and training settings.
The target checkpoint also contains a vision encoder that this text-only recipe
does not train. Both arms use `--check-weight-update-selector target` and the
narrow `--check-weight-update-skip-list visual.` rule: startup preserves the
frozen vision tensors and checks every language-model tensor after transfer.
Without that rule the startup checker randomizes the vision encoder, which a
language-only update cannot restore.

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 python -m tests.manual.score_centering_spec.train_probe \
  --stage prepare --model-dir /models --checkpoint-dir /results --data-dir /datasets
python -m tests.manual.score_centering_spec.train_probe \
  --stage run --model-dir /models --checkpoint-dir /results --data-dir /datasets \
  --output-dir /results/training --arm regular --sampling unfiltered
python -m tests.manual.score_centering_spec.train_probe \
  --stage run --model-dir /models --checkpoint-dir /results --data-dir /datasets \
  --output-dir /results/training --arm dflash --sampling unfiltered
```

Repeat both arms with `--sampling filtered` for top-k=64/top-p=0.9. `--stage print`
prints the exact argument manifest without launching. Preserve rollouts,
training tensors, events and TensorBoard metrics. Require finite probabilities,
losses and gradients, nonzero group advantages, actual target updates, successful
weight checks, and valid metadata on rollouts after those updates. The fixed
draft is excluded from the target weight-reset/check protocol. If all groups
have zero advantages, parameter movement from weight decay is inconclusive.

## Local regression checks

```bash
python -m pytest -q tests/manual/score_centering_spec/test_contract.py \
  tests/fast/launch_scripts/test_score_centering_train_probe.py \
  tests/fast/utils/test_score_centering_speculative.py
```

The SGLang dependency PR contains separate synthetic distribution-law, CDF,
cutoff-tie, support-capacity and filtering regression tests. Retain failed
experiment records when a defect is found and rerun affected checks after the
fix; do not relax tolerances to turn a discrepancy into a pass.
