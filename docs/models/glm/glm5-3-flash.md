---
title: GLM-5.3-Flash
description: RL recipe for GLM-5.3-Flash, a KDA + DSA hybrid MoE with mHC hyper-connections and NoPE MLA.
---

Implementation: [`radixark/miles#2786`](https://github.com/radixark/miles/pull/2786). It goes
with the SGLang
[`sglang-miles-glm53next`](https://github.com/sgl-project/sglang/tree/sglang-miles-glm53next)
branch and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89); the
image in section 3 pins all three.

## 1. Model Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`model_type: glm5_next`) is a
**45-layer KDA + DSA hybrid MoE**. It is a different architecture from the 744 B GLM5 and
GLM5.2 flagships, not a smaller cut of them.

- **45 layers, hybrid**: 34 KDA linear-attention layers + 11 DSA sparse-attention layers.
- **288-expert MoE**, sigmoid routing at top-8; the first 3 layers are dense.
- **mHC hyper-connections** at every block.
- **NoPE MLA** — multi-latent attention with the positional half of the QK head empty.
- **kpool-compressed lightning indexer** picks which keys the DSA layers attend.
- Hidden 4096, FFN 12288, 64 attention heads, vocab 154880, rotary base 800000.
- MTP is dropped for training.

## 2. Supported Variants

| Variant | `--model-name` | Layers |
|---|---|---|
| Full | `GLM-5.3-Flash` | 45 |
| 4-layer slice | `GLM-5.3-Flash-4layer` | 4 (launcher default) |

## 3. Environment Setup

### Experimental Bridge multi-LoRA image

For the Modal fork's Bridge multi-LoRA consumer, build the separate trainer image
from the repository root:

```bash
docker build -f docker/Dockerfile.glm53 -t miles:glm53 .
```

This overlays the digest-pinned Miles v0.1.2 release with this fork's Miles code,
Megatron-Bridge `6527b18e8bb0db994a267e6dfd4db7dafc669df9`, Transformers 5.12.1
and TileLang 0.1.12. The provider and kernels are imported directly from Bridge;
Miles carries no copy. The GLM provider registers a fallback config on this
Transformers version. Transformers 5.16 collides with this release's SGLang
`qwen3_asr` registration, so do not upgrade it independently.

The Bridge pin is the unmerged
[`radixark/Megatron-Bridge#35`](https://github.com/radixark/Megatron-Bridge/pull/35)
branch, based on Bridge 0.5.0. It replaces the release's newer Bridge 0.7.0 only
in this GLM image. In particular, it lacks the newer MiMo provider; the shared
Dockerfile retains its release Bridge pin. When the GLM provider lands upstream,
update this dependency pin and validate the replacement before removing the
overlay.

Use Bridge mode and context parallelism 1. Native multi-LoRA in
[`radixark/miles#4662`](https://github.com/radixark/miles/pull/4662) still rejects
grouped expert adapters, so it does not cover this workload's routed/shared MoE
targets. This trainer overlay does not update the sampler: the Spindle consumer
uses the GLM-capable SGLang v0.5.23 fork separately, and applies its existing
generic grouped-weight import cache memory bound.

### Upstream full-finetuning recipe

Use `docker.io/radixark/miles:glm53next` — the rolling `radixark/miles:dev` image with the
three moving parts checked out at the versions this recipe was built against, multi-arch so
the same tag serves GB300 and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2786`](https://github.com/radixark/miles/pull/2786) `1cd14c00` |
| SGLang | `sglang-miles-glm53next` `9a26e749` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

```bash
hf download zai-org/GLM-5.3-Flash --local-dir /root/models/GLM-5.3-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

The reference checkpoint has to be converted first — `--ref-load` resolves to
`<--ckpt-dir>/<megatron_model_type>_torch_dist`:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py glm5.3-flash)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM \
  torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/models/GLM-5.3-Flash \
    --save          /root/ckpt/glm5.3-flash_torch_dist
```

## 4. Launch

Bring up a ray cluster across the nodes, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then on the
head node:

```bash
cd /root/miles
python scripts/run_glm5_3_flash.py train \
  --model-name GLM-5.3-Flash \
  --num-nodes 16 --num-gpus-per-node 4 \
  --num-rollout 20 --rollout-max-response-len 4096
```

Smoke slice on one node:

```bash
python scripts/run_glm5_3_flash.py train --num-nodes 1 --num-gpus-per-node 8
```

| Shape | TP | PP | EP | Rollout engine |
|---|---|---|---|---|
| 16 × 4 (full) | 8 | 4 | 16 | 8 GPUs, SGLang TP 8 / EP 8 |
| 8 × 4 (full) | 8 | 4 | 8 | 8 GPUs, SGLang TP 8 / EP 8 |
| 2 × 4 or 1 × 8 (4-layer slice) | 2 | 2 | 2 | 4 GPUs, SGLang TP 4 / EP 4 |

The PP-4 shapes run 11 / 11 / 11 / 12 layers per stage, since 45 does not divide by 4.
GRPO on DAPO-Math-17k, Adam at `lr 1e-6`, `max_tokens_per_gpu 8192`, full uniform recompute.
Rollout is colocated, with the trainer offloaded to disk; both DSA paths run on tilelang and
the KV cache is BF16. Routing replay turns on with `--enable-r3`.

## 5. What a Healthy Run Looks Like

From the validation run in [#2786](https://github.com/radixark/miles/pull/2786) — 16 nodes ×
4 GB300, DAPO on DAPO-Math-17k:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.0068 – 0.0106 across the first 11 rollouts |
| `rollout/raw_reward` | 0.5 → 0.94 within 10 rollouts |
| `train/ppo_kl` | ~2.6e-4 |
| `train/grad_norm` | 0.31 – 0.49 |

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up: it
covers the KDA, DSA and hyper-connection paths at once.
