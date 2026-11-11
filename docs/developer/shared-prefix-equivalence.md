# Shared-prefix SFT: equivalence and determinism investigation

This extends the initial Qwen SFT experiment. It distinguishes mathematical
equivalence, repeatability of one packed computation, and numerical agreement
between expanded and compact low-precision computations.

## Findings

The first adapter did not propagate `torch.use_deterministic_algorithms(True)`
to the custom FlashAttention callable. FA4 has its own `deterministic` argument,
defaulting to false. Its deterministic backward orders the dQ reductions and,
for grouped-query attention, dK/dV reductions using semaphores. Torch's global
setting alone cannot select that custom-kernel implementation.

The adapter now forwards `deterministic=True` whenever Torch deterministic
algorithms are enabled. This does not change the default fast path or mutate
global Torch settings. A backend that does not support the requested flag
fails rather than silently ignoring the determinism requirement.

An optional `deterministic_reduction=True` batch setting additionally replaces
shared K/V and loss-row gather backward with fixed-order segmented FP32 sums
(FP64 for FP64 inputs). CPU preparation uses stable counting sort. The GPU
reduction assigns each parent row its own segment and uses no shared-output
scatter atomics. It rounds back to the original tensor dtype once at the end.

For a complete repeatability experiment, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before CUDA initialization, disable TF32, enable Torch deterministic algorithms,
and keep model weights, dropout, packing and collective configuration fixed:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
batch = TreeSFTBatch.from_plan(
    plan, targets, device="cuda", deterministic_reduction=True,
)
loss = batch.loss(model, varlen_func=flash_attn_varlen_func)
loss.backward()
```

Do not infer that this makes two *different* numerical graphs bitwise equal.
Canonicalize samples and their supervision together before packing if arrival
order should not change the packed graph. The diagnostic sorts by sequence and
target values; a production work-unit planner should use stable datum, adapter,
weight-version and replay identities.

## Determinism does not remove BF16 rounding boundaries

A one-weight linear layer is a counterexample, with no attention, routing,
parallelism or nondeterministic operations. Let its input be `1.5`, and its two
output adjoints be `1` and `1/256`:

```text
Expanded BF16 backward: round_bf16(1.5 * 1 + 1.5 * (1/256)) = 1.5078125
Shared BF16 backward:   round_bf16(1.5 * round_bf16(1 + 1/256)) = 1.5
FP32, either layout:   1.505859375
```

The shared node receives a merged adjoint. Even if that merge sums in FP32,
ordinary autograd returns it to the BF16 output tensor before the parent's
backward runs. That extra rounding point changes the result. FP32 master
parameters and final `.grad` buffers do not remove intermediate BF16 rounding.
`tests/fast/prefix_tree/test_rounding.py` checks these exact values with Torch
deterministic algorithms enabled.

Matching the expanded BF16 backward bit-for-bit therefore requires more than
choosing a summation order. FP32 adjoints through shared-operation backward
interfaces are an accuracy improvement to investigate, not a guarantee of
expanded-BF16 equality. Literal equality requires preserving the expanded
backward's rounding boundaries and reduction structure, potentially including
per-child contributions through shared operations. That can surrender much of
the backward sharing benefit. Neither approach is implemented by this adapter.

## Measurements

All runs keep weights fixed; there is no optimizer update between repeats.
They compare every parameter-gradient shard, mapped attention/MLP forward
outputs, and MoE selected expert IDs. Relative L2 sums squared errors and
reference norms over all rank-local shards; replicated parameters are counted
on each rank. It is not a gradient sample or a checksum-only comparison.

The dense ablation uses pretrained Qwen3-0.6B, 513 shared prompt tokens,
approximately 65 suffix tokens and four branches. With a deterministic expanded
reference:

| Tree execution | Repeat gradient relative L2 | Relative L2 versus expanded |
| --- | ---: | ---: |
| Native backward | 0.003716 | 0.00435–0.00440 |
| Torch deterministic, FA deterministic | **0, bitwise** | 0.004312 |
| Plus fixed-order FP32 K/V and loss-row sums | **0, bitwise** | 0.004072 |

Reversing branches after enabling deterministic controls changes gradients by
`1.52e-5` relative L2. Canonical packing of the same samples arriving in reverse
order gives bitwise-identical repeat gradients. All checked BF16 attention/MLP
forward outputs match expanded execution bitwise. This isolates the observed
BF16 gap to backward for this experiment.

The initial all-precision ablation also compared FP32 math SDPA, BF16 math
SDPA, and an FP32 final LM head. Math attention still exhibited a BF16 gap;
promoting only the LM head did not remove it. Thus the remaining discrepancy
is not exclusively FA4, a sampling mask, or final-logit precision.

### Larger models and parallelism

The larger tests use full pretrained Qwen3-4B and Qwen3-30B-A3B with **native
Transformers TP=2**, in both BF16 autocast and FP32 math attention. Parameters
and stored gradients are FP32. These are standalone HF TP tests, not a claim
that the distributed Miles Megatron actor integration is complete. MoE experts
use the eager expert implementation; EP and the Miles grouped-LoRA kernels are
not exercised.

Inputs have 129 shared prompt tokens, approximately 17 suffix tokens and four
branches. Full gradient references are kept on CPU so only one model and one
gradient buffer reside on each GPU. Each TP group uses single-node NCCL
Ring/Simple collectives with the Socket network plugin. RDMA provisioning is enabled
and reserved capacity is excluded.

The full ablation found:

- Native BF16 expanded repeats themselves differ: 0.300% for 4B and 0.227%
  for 30B MoE. These references must not be treated as deterministic ground truth.
- Deterministic tree repeats are bitwise identical on both models and both
  precisions. Replicated Q/K norm gradients agree across ranks.
- MoE selected expert IDs match at all 28,176 token-layer positions (587 expanded
  tokens × 48 layers), including reversed and canonical tree layouts.
- All checked BF16 attention/MLP forward outputs match expanded execution
  bitwise. FP32 forward differences start at the first attention output at about
  `3e-7`–`6e-7` maximum absolute error, consistent with regrouped floating-point
  arithmetic; these results are not labeled bitwise-equivalent to expanded.
- With deterministic controls, reversing branches changes gradients by
  `2.00e-5` relative L2 for 4B BF16 and `1.60e-5` for 30B MoE BF16. These are
  much smaller than the original approximately `4e-3` equivalence gap.

The final focused calibration enables deterministic mode for **both** expanded
and tree backward. Both expanded and compact repeats compare bitwise equal on
all rank-local gradients:

| Model | TP | BF16 tree vs expanded relative L2 | FP32 tree vs expanded relative L2 |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 2 | 0.00587088 (0.587%) | 0.00000694364 (0.000694%) |
| Qwen3-30B-A3B | 2 | 0.00411185 (0.411%) | 0.00000181285 (0.000181%) |

Expert sets and ordered expert selections still match at all 28,176 token-layer
positions. Q/K norm gradient replicas match across ranks. These results validate
repeatability and a much smaller FP32 numerical discrepancy, not expanded-BF16
bitwise equivalence. No statistical training-convergence claim is made.

[`shared-prefix-equivalence-results.csv`](shared-prefix-equivalence-results.csv)
contains the per-run metrics. `*_native_reference` protocols intentionally use
the original nondeterministic reference; `*_deterministic_reference` protocols
use the corrected reference. Error fields are raw fractions, not percentages;
blank trace/repeat fields mean that measurement was not collected on that run.
The GPU validation apps stopped with zero remaining containers.

## Reproduction and scope

Run `tests/manual/prefix_tree/diagnose_equivalence.py` on Blackwell with Torch
2.10.0, Transformers 5.12.1, Accelerate, kernels 0.14.0, CuTe DSL 4.8.1, and
FA4 version 1 (inspected source revision
`a7822b61ba2665652a742b5b3d8c1ef01aa4b588`). `--focus` compares only deterministic
expanded/tree repeats; `--short` includes native and reordered-tree controls.
The full diagnostic deliberately bypasses automatic flag propagation in its
attention wrapper to isolate the Torch-only and FA-deterministic ablations.

```bash
PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone \
  --nproc-per-node=2 tests/manual/prefix_tree/diagnose_equivalence.py \
  --model Qwen/Qwen3-30B-A3B --prefix 129 --suffix 17 --branches 4 \
  --focus --output /tmp/moe-tp2-equivalence.json
```

The CPU suite now has 33 passing tests, including FP64 gradient oracles,
checkpointing, native MoE route equivalence, fixed-order reduction gradcheck,
the BF16 rounding counterexample and deterministic-flag forwarding. No upstream
Bridge/attention files, shared image pins, Spindle code or default trainer path
are changed. GLM DSA/KDA, multi-LoRA work-unit integration and cross-microbatch
gradient relay remain separate work.
