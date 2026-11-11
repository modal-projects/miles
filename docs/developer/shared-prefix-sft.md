# Shared-prefix SFT: initial implementation

This is an opt-in, single-device Qwen3 implementation in Miles. It establishes
packing, attention, loss and gradient correctness before integration with the
distributed trainer. It does not enable a launch flag, change the default
trainer, modify Spindle, or patch/vend Megatron-Bridge or FlashAttention.

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

Report expanded input tokens/sec, supervised tokens/sec, peak allocated
memory, raw iteration times, token compaction, loss and gradient errors.
Expanded input tokens/sec counts the same original work for both layouts;
it is not the number of physical tokens executed by the compact decoder.

Optional probe dependencies are separate from Miles' core requirements:
Torch 2.10.0, Transformers 5.12.1 (the existing Miles pin), kernels 0.18.2,
einops, apache-tvm-ffi, nvidia-cutlass-dsl and compatible CUDA Python bindings.
The probe loads `kernels-community/flash-attn4` version 1. No default image or
shared dependency pin is changed by this feature.

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
