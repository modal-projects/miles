---
title: Inkling-2-Small
description: Native Megatron training and attention/unembedding multi-LoRA for Inkling-2-Small.
---

Inkling-2-Small uses the native Megatron implementation and the matching SGLang kernels in `radixark/miles:inkling-2-small`. See [the launcher](https://github.com/modal-projects/miles/blob/spindle-miles-v0.2.0/scripts/run_inkling2.py) for checkpoint preparation, conversion, and full-model LoRA launch arguments.

Single-adapter LoRA keeps the complete attention, routed-expert, and unembedding layout. Multi-LoRA supports the attention and unembedding projections with the routed experts frozen:

```text
--megatron-to-hf-mode raw
--target-modules attn,unembed
--multi-lora-n-adapters 2
```

Use TP=CP=PP=1 and shard the experts with EP. Multi-LoRA requires separate trainer and sampler GPUs and per-slot Adam optimizers; do not use the launcher's colocated/offloaded single-adapter settings. Expert targets are rejected at startup for multi-LoRA.
