"""Beam 501B BF16 GRPO LoRA on DAPO-Math-17K with AIME25 eval, on 4 x 8 H200 GPUs.

The math recipe of scripts/run_beam.py (data, reward, GRPO + TIS, batch and
sequence sizes, async training, eval) with a LoRA adapter in place of
full-parameter training. Only what LoRA needs changes:
  - rank-16 / alpha-32 LoRA on every Beam linear layer: attention q/k/v/o and
    the output gate, the dense MLPs and the routed experts (per-expert
    factors); the router, embeddings and LM head stay frozen;
  - lr peaks at 1e-4 after a 10-step warmup and decays (cosine) to 1e-5 by
    step 400, instead of a constant 1e-6. At a constant 1e-5 the zero-initialized
    LoRA B barely moves (no learning in 75 steps); a constant 2e-4 learns fast
    but turns unstable after ~100 steps, as the adapter's effective step grows
    with |B|;
  - rollout routing replay is on by default;
  - the adapter's optimizer state (~4.7B params) fits on the GPUs, so the
    NVMe-streamed optimizer is dropped (it does not support LoRA);
  - expert-TP 1: expert LoRA under expert-TP > 1 is wrong in Megatron-Bridge
    (forward, backward and export), so experts shard by EP only.

32 GPUs = 16 trainers (TP8/EP16/expert-TP1, DP2) + one TP16 SGLang engine.
The frozen BF16 base is ~62 GB per trainer GPU. Weight updates broadcast only
the adapter (~8.7 GB) into the engine; requires Reflection SGLang with streamed
LoRA updates (register_lora_adapter + weight-update sessions).

AIME25 eval prompts carry the training prompts' answer-format instruction
(Answer: \\boxed{...}, see `prepare`): the grader requires \\boxed{}, and on bare
AIME problems the policy drops it as its final message gets shorter.

Requires an existing 4-node Ray cluster built from docker/Dockerfile.beam,
and a Beam HF checkpoint and the datasets on storage shared by every node
(downloaded by `prepare`).

Args:
  --resume: Continue from the latest adapter checkpoint in output-dir (adapter,
      optimizer and LR scheduler). Rollout ids and the dataset position restart
      at 0: Miles does not restore them when the base loads from HF.
  --num-rollout: Total rollout count.
  --model-id: Hugging Face repository to download during prepare.
  --model-dir: BeamForCausalLM HF checkpoint directory, including tokenizer.
  --data-dir: Parent of dapo-math-17k/ and aime-2025/ (see `prepare`).
  --output-dir: Shared checkpoint root (adapters under iter_*/adapter); dumps go
      to output-dir/dumps.
  --megatron-path: Installed compatible Megatron-LM source directory.
  --lora-rank / --lora-alpha: Adapter rank and alpha (scaling alpha / rank).
  --global-batch-size: Samples per optimizer step; a divisor of the rollout's
      responses, each used once (default: all of them, one step per rollout).
  --enable-eval / --no-enable-eval: AIME25 every five rollouts (default on).
  --async-training / --no-async-training: train_async.py overlap, or synchronous train.py.
  --use-rollout-routing-replay / --no-use-rollout-routing-replay: Rollout routing
      replay and the Miles router (default on).
  --extra-args: Extra Miles/Megatron/SGLang flags, appended last so they override.

Examples:
  python scripts/run_beam_lora.py prepare --data-dir /data/datasets

  MASTER_ADDR=10.0.0.1 RAY_ADDRESS=http://10.0.0.1:8265 \
    python scripts/run_beam_lora.py execute --num-rollout 100 \
      --model-dir /root/models/Beam --data-dir /data/datasets \
      --output-dir /root/shared_data/beam-lora-run
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

NUM_NODES = 4
NUM_GPUS_PER_NODE = 8
TRAIN_NODES = 2
ROLLOUT_GPUS_PER_ENGINE = 16
PROMPT_LENGTH = 1024
RESPONSE_LENGTH = 4096
CONTEXT_LENGTH = PROMPT_LENGTH + RESPONSE_LENGTH
# DAPO-Math-17K wraps every problem in this answer-format instruction; AIME25 ships bare problems
# while the grader requires \boxed{}, so eval prompts get the same wrapping (see `prepare`).
ANSWER_FORMAT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the "
    "form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
)
ANSWER_FORMAT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
EVAL_FILE = "aime-2025/aime-2025-answer-format.jsonl"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    resume: bool = False
    num_rollout: int = 100
    model_id: str = "reflection/Beam-1.0-501B-A23B"
    model_dir: str = "/root/models/Beam"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    lora_rank: int = 16
    lora_alpha: int = 32
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    save_interval: int = 20
    enable_eval: bool = True
    async_training: bool = True
    use_rollout_routing_replay: bool = True
    extra_args: str = ""
    # Ray: by default this submits to a cluster someone else already built, so
    # MASTER_ADDR must name its head. --join-ray-workers makes the launcher
    # build the cluster itself, ssh-ing every host of --ray-hostfile into it.
    ray_hostfile: str = "/root/mpi_rack_hostfile"
    join_ray_workers: bool = False

    def __post_init__(self):
        if self.rollout_batch_size * self.n_samples_per_prompt % self.global_batch_size:
            raise ValueError("global-batch-size must divide the rollout's responses so each is used once")


def _flags(group: dict) -> list[str]:
    return [token for key, value in group.items() for token in ("--" + key, str(value))]


def _latest_adapter(directory: Path) -> Path:
    # LoRA checkpoints are iter_*/adapter directories; Megatron's iteration tracker is not written for them.
    adapters = sorted(directory.glob("iter_*/adapter"))
    if not adapters:
        raise ValueError(f"Resume requires an adapter checkpoint (iter_*/adapter) in {directory}")
    return adapters[-1]


def _checkpoint_args(args: ScriptArgs) -> list[str]:
    # The frozen base always loads from the HF checkpoint; the adapter carries the training state.
    values = _flags(
        {
            "hf-checkpoint": args.model_dir,
            "load": args.model_dir,
            "save": args.output_dir,
            "save-interval": args.save_interval,
            "ckpt-format": "torch_dist",
        }
    )
    if args.resume:
        values += ["--lora-adapter-path", str(_latest_adapter(Path(args.output_dir)))]
    return values + ["--use-checkpoint-opt-param-scheduler"]


def _lora_args(args: ScriptArgs) -> list[str]:
    # Targets come from the beam entry of Miles' HF LoRA registry (all Beam linear layers).
    return _flags({"lora-rank": args.lora_rank, "lora-alpha": args.lora_alpha, "lora-dropout": 0.0})


def _rollout_args(args: ScriptArgs) -> list[str]:
    values = _flags(
        {
            "prompt-data": f"{args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl",
            "input-key": "prompt",
            "label-key": "label",
            # Boxed integer accuracy plus DAPO's soft overlong penalty.
            "custom-rm-path": "examples.beam.reward.reward",
            "reward-key": "score",
            "num-rollout": args.num_rollout,
            "rollout-batch-size": args.rollout_batch_size,
            "n-samples-per-prompt": args.n_samples_per_prompt,
            "global-batch-size": args.global_batch_size,
            "rollout-max-response-len": RESPONSE_LENGTH,
            "rollout-max-prompt-len": PROMPT_LENGTH,
            "rollout-max-context-len": CONTEXT_LENGTH,
            "rollout-temperature": 1.0,
            "rollout-top-p": 1.0,
            "rollout-top-k": -1,
            "rollout-seed": 1234,
        }
    )
    values += ["--apply-chat-template", "--rollout-shuffle", "--balance-data"]
    if args.enable_eval:
        values += ["--eval-prompt-data", "aime25", f"{args.data_dir}/{EVAL_FILE}"]
        values += _flags(
            {
                "eval-interval": 5,
                "eval-input-key": "prompt",
                "eval-label-key": "label",
                # Report plain accuracy, without the +/-1 scale and overlong penalty.
                "eval-reward-key": "acc",
                "n-samples-per-eval-prompt": 8,
                "eval-temperature": 1.0,
                "eval-max-response-len": RESPONSE_LENGTH,
            }
        )
    return values


def _parallel_args(args: ScriptArgs) -> list[str]:
    values = _flags(
        {
            "train-backend": "megatron",
            "megatron-to-hf-mode": "bridge",
            "actor-num-nodes": TRAIN_NODES,
            "actor-num-gpus-per-node": NUM_GPUS_PER_NODE,
            "num-gpus-per-node": NUM_GPUS_PER_NODE,
            "rollout-num-gpus": (NUM_NODES - TRAIN_NODES) * NUM_GPUS_PER_NODE,
            "rollout-num-gpus-per-engine": ROLLOUT_GPUS_PER_ENGINE,
            "tensor-model-parallel-size": 8,
            "pipeline-model-parallel-size": 1,
            # Experts shard over all 16 trainers; expert LoRA requires expert-TP 1.
            "expert-model-parallel-size": 16,
            "expert-tensor-parallel-size": 1,
            "context-parallel-size": 1,
            "micro-batch-size": 1,
            # The validated Beam attention path: unfused core attention on padded bshd.
            "qkv-format": "bshd",
            "attention-backend": "unfused",
            "seq-length": CONTEXT_LENGTH,
            "data-pad-size-multiplier": 1,
            "recompute-granularity": "full",
            "recompute-method": "uniform",
            "recompute-num-layers": 1,
            "moe-token-dispatcher-type": "alltoall",
            "moe-router-bias-update-rate": 0,
            "moe-aux-loss-coeff": 0,
            "attention-dropout": 0.0,
            "hidden-dropout": 0.0,
        }
    )
    if args.use_rollout_routing_replay:
        values += ["--use-rollout-routing-replay", "--use-miles-router"]
    return values + ["--bf16", "--sequence-parallel", "--data-parallel-random-init"]


def _algorithm_args(args: ScriptArgs) -> list[str]:
    values = _flags(
        {
            "advantage-estimator": "grpo",
            "eps-clip": 0.2,
            "eps-clip-high": 0.28,
            "eps-clip-c": 10.0,
            "kl-coef": 0.0,
            "kl-loss-coef": 0.0,
            "entropy-coef": 0.0,
        }
    )
    return values + ["--calculate-per-token-loss", "--use-tis"]


def _optimizer_args(args: ScriptArgs) -> list[str]:
    values = _flags(
        {
            "optimizer": "adam",
            "lr": 1e-4,
            "min-lr": 1e-5,
            "lr-decay-style": "cosine",
            "lr-warmup-iters": 10,
            "lr-decay-iters": 400,
            "weight-decay": 0.0,
            "adam-beta1": 0.9,
            "adam-beta2": 0.999,
            "adam-eps": 1e-8,
            "clip-grad": 1.0,
        }
    )
    return values + [
        "--use-distributed-optimizer",
        "--grad-reduce-in-bf16",
        "--no-gradient-accumulation-fusion",
    ]


def _sglang_args(args: ScriptArgs) -> list[str]:
    engines = (NUM_NODES - TRAIN_NODES) * NUM_GPUS_PER_NODE // ROLLOUT_GPUS_PER_ENGINE
    concurrency = args.rollout_batch_size * args.n_samples_per_prompt // engines
    values = _flags(
        {
            "sglang-dtype": "bfloat16",
            "sglang-attention-backend": "triton",
            "sglang-moe-runner-backend": "triton",
            "sglang-moe-a2a-backend": "none",
            "sglang-ep-size": 1,
            "sglang-pp-size": 1,
            "sglang-context-length": CONTEXT_LENGTH,
            "sglang-chunked-prefill-size": 8192,
            "sglang-max-running-requests": concurrency,
            "sglang-mem-fraction-static": 0.85,
            "update-weight-transfer-mode": "broadcast",
            "update-weight-buffer-size": 268435456,
            "update-weights-interval": 1,
            "seed": 1234,
        }
    )
    return values + ["--sglang-cuda-graph-max-bs-decode", str(concurrency)]


def _debug_args(args: ScriptArgs) -> list[str]:
    # Rollout dumps also cover eval (as eval_<id>); train dumps are one file per DP rank.
    return _flags(
        {
            "save-debug-rollout-data": f"{args.output_dir}/dumps/rollout/{{rollout_id}}.pt",
            "save-debug-train-data": f"{args.output_dir}/dumps/train/{{rollout_id}}_{{rank}}.pt",
        }
    )


def _train_args(args: ScriptArgs) -> list[str]:
    return (
        _checkpoint_args(args)
        + _lora_args(args)
        + _rollout_args(args)
        + _parallel_args(args)
        + _algorithm_args(args)
        + _optimizer_args(args)
        + _sglang_args(args)
        + _debug_args(args)
        # Engages only when WANDB_API_KEY is set.
        + shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
        # Last, so they override anything above.
        + shlex.split(args.extra_args)
    )


def _prepare(args: ScriptArgs) -> None:
    backend = args.create_backend()
    backend.exec_command_cpu(shlex.join(["hf", "download", args.model_id, "--local-dir", args.model_dir]))
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    backend.hf_download_dataset("zhuzilin/aime-2025", data_dir=args.data_dir)
    _write_answer_format_eval(Path(args.data_dir))


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the HF checkpoint into model-dir and both datasets into data-dir."""
    _prepare(args)


def _write_answer_format_eval(data_dir: Path) -> None:
    """AIME25 with the training prompts' answer-format instruction around each problem."""
    rows = [json.loads(line) for line in (data_dir / "aime-2025/aime-2025.jsonl").read_text().splitlines() if line]
    with open(data_dir / EVAL_FILE, "w") as f:
        for row in rows:
            *history, last = row["prompt"]
            last = {**last, "content": ANSWER_FORMAT_PREFIX + last["content"] + ANSWER_FORMAT_SUFFIX}
            f.write(json.dumps({**row, "prompt": [*history, last]}) + "\n")


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Train. prepare must already have placed the checkpoint and both datasets."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the checkpoint and datasets, then train, in one command."""
    _prepare(args)
    _train(args)


def _train(args: ScriptArgs) -> None:
    if not (master := os.environ.get("MASTER_ADDR")):
        raise ValueError("Set MASTER_ADDR for the 4-node Ray cluster")
    os.environ.setdefault("RAY_ADDRESS", f"http://{master}:8265")
    backend = args.create_backend()
    if not args.join_ray_workers:
        # The cluster is already up and someone else owns it; only submit.
        os.environ["MILES_SCRIPT_EXTERNAL_RAY"] = "1"
    backend.execute_train(
        train_args=shlex.join(_train_args(args)),
        num_gpus_per_node=NUM_GPUS_PER_NODE,
        megatron_model_type="beam",
        train_script="train_async.py" if args.async_training else "train.py",
        config=args,
        megatron_path=args.megatron_path,
        before_ray_job_submit=(
            (
                lambda: backend.ssh_start_ray_workers(
                    master_addr=master,
                    num_gpus_per_node=NUM_GPUS_PER_NODE,
                    hostfile=args.ray_hostfile,
                    head_host=os.environ.get("MLP_WORKER_0_HOST"),
                )
            )
            if args.join_ray_workers
            else None
        ),
        extra_env_vars={
            "NCCL_CUMEM_ENABLE": "1",
            "NCCL_NVLS_ENABLE": "0",
            # Async graph replay must not reuse NCCL registrations after buffer remapping.
            "NCCL_GRAPH_REGISTER": "0",
            "NCCL_ALGO": "allreduce:tree",
            # Match the team server's deterministic NCCL channels on every rank.
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "8",
            "MILES_USE_LEGACY_ROLLOUT_V1": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RAY_USAGE_STATS_ENABLED": "0",
            "DO_NOT_TRACK": "1",
        },
    )


if __name__ == "__main__":
    app()
