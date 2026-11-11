# Shared-prefix SFT: initial implementation

This is an opt-in Qwen3 SFT implementation in Miles. It establishes
packing, attention, loss and gradient correctness before integration with the
distributed trainer. It does not enable a launch flag, change the default
trainer, modify Spindle, or patch/vend Megatron-Bridge or FlashAttention.
Standalone dense/MoE TP diagnostics and the deterministic-backward fix are
described in [the equivalence investigation](shared-prefix-equivalence.md).

## Computation and loss identity

`build_prefix_tree` compacts identical input prefixes, collapses unary chains
into contiguous segments, and records logical positions, ancestor KV paths,
and every original sample's token mapping. Its construction cost is linear in
expanded input tokens plus materialized segment-path rows. There is no dense
tree attention mask.

`sharing_keys` must identify adapter and weight version; `computation_keys`
must identify replay/stochastic side inputs at each token. A mismatch forks
the rest of the trajectory: matching later tokens never remerge. These keys
are an explicit caller contract, not automatic validation of router replay.

Targets and weights are separate original loss occurrences. Two samples can
share a hidden state while assigning different targets or weights to its
prediction. Ignored targets are `-100`. SFT normalization divides by the sum
of original supervised weights, never the number of unique tree tokens.
The first implementation gathers logits for loss occurrences; a specialized
mapped vocabulary kernel is not required for correctness.

## Execution

The Qwen adapter uses public Transformers attention registration. Parameters,
projections, norms and decoder layers are unchanged. Explicit logical
`position_ids` apply RoPE before attention. The batch supplies an explicit mask
mapping to avoid constructing a physical-order quadratic causal mask.

Q is computed once per unique token. Each segment gathers KV from its complete
ancestor path and uses asymmetric varlen attention with bottom-right causal
alignment. Autograd sums repeated KV contributions. The reference backend uses
PyTorch SDPA per segment; the performance probe supplies stock FA4. There is
no custom CUDA attention kernel or mutable global current-tree context.

```python
from miles.backends.training_utils.prefix_tree import build_prefix_tree
from miles.backends.training_utils.prefix_tree.qwen import TreeSFTBatch, enable_qwen_prefix_tree

enable_qwen_prefix_tree(model)  # dense Qwen3, zero dropout, default RoPE
plan = build_prefix_tree(sequences)
batch = TreeSFTBatch.from_plan(plan, targets, device=model.device)
loss = batch.loss(model, varlen_func=flash_attn_varlen_func)
loss.backward()
```

Prepare the batch once outside repeated forward/backward measurements. Without
`varlen_func`, the reference implementation is intended for correctness only.
Do not reuse a batch across model/adapter versions with cached activations;
this implementation stores metadata, not a persistent activation cache.

## Correctness checks

```bash
PYTHONPATH=. python -m pytest tests/fast/prefix_tree
```

The Qwen oracle runs original trajectories independently through ordinary
causal attention. It compares loss, every parameter gradient and an SGD step.
An eight-branch, 129-token shared-prefix case also checks an AdamW step at
FP64 tolerance `1e-12`.
Cases include nested/uneven forks, duplicate trajectories, an endpoint inside
another trajectory, unrelated roots, differing labels and weights, ignored
targets, and checkpoint recomputation. Planner tests check adapter/version
isolation and replay-key differences. Attention has an independent FP64 Q/K/V
gradient oracle at `atol=rtol=1e-12`.

The full-Qwen FP64 oracle substitutes native FP64 RMSNorm and uses ordinary
SDPA so all operations retain FP64. This test-only substitution is necessary:
native Qwen RMSNorm and eager softmax explicitly downcast to FP32. Native Qwen
is also tested separately in FP64/FP32 with tolerances accounting for those
operations. The production adapter does not replace norms or softmax.

Mathematical gradient equivalence is the invariant. Bitwise equality is not
promised: sharing changes floating-point reduction order. GPU reports record
bitwise equality, global/per-parameter relative L2 and maximum absolute error
rather than describing approximate agreement as exact.

## GPU performance probe

`tests/manual/prefix_tree/qwen_sft.py` uses public pretrained Qwen3-0.6B,
FP32 parameters/gradients, BF16 autocast, and stock FA4 on Blackwell. Both the
expanded and compact measurements use the same varlen backend. Data packing
is measured separately. Two warmup iterations precede five timed
forward/backward iterations; CUDA is synchronized at measurement boundaries.
The same updated model weights are used for both performance measurements.
Timing excludes loading, compilation, optimizer steps and planning.

The probe also runs expanded trajectories as separate microbatches, with the
same global loss normalization. This measures BF16 numerical changes caused
by regrouping alone. It reports a separate strict pointwise SGD-weight check
(`atol=5e-5`, `rtol=1e-4`). A failure is retained in the JSON and the CLI exits
nonzero **after** writing the diagnostic throughput report. Performance numbers
do not imply that the strict numerical gate passed.

Report expanded input tokens/sec, supervised tokens/sec, peak allocated
memory, raw iteration times, token compaction, loss and gradient errors.
Expanded input tokens/sec counts the same original work for both layouts;
it is not the number of physical tokens executed by the compact decoder.

Optional probe dependencies are separate from Miles' core requirements:
Torch 2.10.0, Transformers 5.12.1 (the existing Miles pin), kernels 0.14.0,
einops, apache-tvm-ffi, nvidia-cutlass-dsl 4.8.1 and compatible CUDA Python bindings.
The probe loads `kernels-community/flash-attn4` version 1. No default image or
shared dependency pin is changed by this feature.

## Initial results, 2026-11-11

Seventeen isolated CPU checks pass with Torch 2.10 and Transformers 5.12.1.
The normal Miles test runner additionally needs Ray and the other Miles test
dependencies; the lightweight local run used `pytest --noconftest`.

Pretrained Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
was measured on B200, using FP32 weights/gradients and BF16 autocast with TF32
disabled. Eight independent single-GPU cases ran concurrently on one 8-GPU
node, in separate processes with four CPU threads each. These are preliminary
single-device SFT measurements, not distributed Miles or RL throughput.
The Modal node used RDMA and excluded reserved capacity; it stopped with zero
remaining containers after the run.

| Shared prefix | Approx. suffix | Branches | Expanded input tok/s | Tree input tok/s | Speedup | Tree grad relative L2 | Expanded-only regrouping relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | 33 | 2 | 4,348 | 4,243 | 0.98x | 0.580% | 0.502% |
| 257 | 65 | 4 | 17,933 | 18,421 | 1.03x | 0.447% | 0.359% |
| 513 | 129 | 8 | 56,256 | 69,174 | 1.23x | 0.389% | 0.242% |
| 1,025 | 257 | 8 | 61,532 | 125,011 | 2.03x | 0.381% | 0.201% |
| 2,049 | 257 | 8 | 63,723 | 185,217 | 2.91x | 0.388% | 0.201% |
| 4,097 | 513 | 8 | 63,670 | 203,929 | 3.20x | 0.389% | 0.203% |
| 8,193 | 513 | 4 | 59,740 | 159,511 | 2.67x | 0.348% | 0.226% |
| 2,049 | 1,025 | 8 | 60,798 | 119,399 | 1.96x | 0.358% | 0.194% |

Suffix length varies by up to two tokens between branches. The 4,097-token
case reduced peak allocated GPU memory from 109.8 GB to 38.6 GB. Loss scalars
matched in all eight cases. All passed the probe's relative-gradient checks
(global 2%, per-parameter 5%) but **all failed the stricter pointwise optimizer
check**. Maximum post-SGD weight differences were 0.000156 to 0.000352.
The FP64 tests establish the mathematical gradient equivalence; these BF16
results do not establish bitwise equivalence or justify a production enablement.
The expanded-only control demonstrates that regrouping itself introduces
rounding differences, but does not explain away the tree's additional error.

The machine-readable summary is in
[`shared-prefix-sft-results.csv`](shared-prefix-sft-results.csv). Before GLM
or distributed integration, investigate FP32 accumulation for repeated KV/loss
rows and agree a numerical criterion appropriate for low-precision training.

## Follow-up boundaries

1. Integrate this explicit plan into Miles' common work-unit preparation,
   Ray schema, DP grouping, compact batch loader and distributed scheduler.
2. Add model-specific GLM DSA pooling/address mapping and differentiable KDA/
   convolution state branching. Those are not validated by the Qwen tests.
3. Add cross-microbatch activation/state sharing and gradient relay. Current
   sharing is limited to a single forward/backward tree; budget gathered KV
   and activation memory as well as unique tokens.
4. Validate multi-LoRA, TP/SP/EP, replayed routing, and optimizer finalization
   before enabling this through Spindle or running RL comparisons.

Scaled/dynamic RoPE, sliding-window attention, dropout, other model families,
and distributed execution are not currently supported by this entrypoint.
