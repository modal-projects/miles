"""Short paired Qwen3.8-27B score-centering runs through the real Miles launcher.

The Qwen dense recipe supplies the model definition, TP4 and CPU Adam offload.
Both arms start from the same converted checkpoint and synchronize real updates.
The default three optimizer steps use an initial synchronization and two
post-update synchronizations; the last step needs no further rollout handoff.
The text-only trainer does not update the checkpoint's vision encoder. The
checker excludes ``visual.`` tensors and checks all trained language weights;
this probe makes no claim about vision preservation across memory offload.
Rollout TP4 keeps two target replicas on an eight-GPU node to bound host backups.
Model checkpoints and DAPO data must already be downloaded; ``prepare`` converts
the target with the existing checkpoint utility. ``print`` only shows the plan.

Args:
  --stage: print (default), prepare, or run.
  --arm: regular or dflash; both use the same score-centering objective.
  --sampling: unfiltered or filtered (top-k 64, top-p 0.9, capture width 128).
  --model-dir / --checkpoint-dir / --data-dir: Existing inputs and conversion cache.
  --output-dir: Artifact root; each sampling/arm pair gets a separate directory.

Example (repeat the run with ``--arm dflash``):
  CUDA_DEVICE_MAX_CONNECTIONS=1 python -m tests.manual.score_centering_spec.train_probe --stage prepare \\
      --model-dir /models --checkpoint-dir /results --data-dir /datasets
  python -m tests.manual.score_centering_spec.train_probe --stage run \\
      --model-dir /models --checkpoint-dir /results --data-dir /datasets \\
      --output-dir /results/training --arm regular

Inspect rewards and gradient metrics after each run. All-zero group advantages
make a run inconclusive even if weight decay moves the parameters. These short
runs check integration and synchronization, not long-run training parity.
"""

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from scripts.run_qwen3_dense import ScriptArgs as QwenDenseArgs

from miles.utils.external_utils import command_utils as U
from miles.utils.external_utils.model_args_utils import load_model_args

_MODEL_NAME = "Qwen3.8-27B"
_MODEL_TYPE = "qwen3.8-27B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    stage: Literal["print", "prepare", "run"] = "print"
    arm: Literal["regular", "dflash"] = "regular"
    sampling: Literal["unfiltered", "filtered"] = "unfiltered"
    model_dir: str = "/root/models"
    checkpoint_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    prompt_data: str = ""
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 4
    num_rollout: int = 3
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 2048
    draft_tokens: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_nodes != 1 or self.num_gpus_per_node != 8:
            raise ValueError("This probe uses one eight-GPU node with trainer TP4 and DP2")
        if self.rollout_num_gpus_per_engine not in (1, 2, 4, 8):
            raise ValueError("rollout_num_gpus_per_engine must divide the eight-GPU node")
        if self.num_rollout < 2:
            raise ValueError("At least two rollouts are needed to exercise updated target weights")
        if self.rollout_batch_size <= 0 or self.n_samples_per_prompt < 2:
            raise ValueError("Use positive rollout_batch_size and at least two samples per GRPO group")

    @property
    def hf_checkpoint(self) -> Path:
        return Path(self.model_dir) / _MODEL_NAME

    @property
    def draft_checkpoint(self) -> Path:
        return Path(self.model_dir) / f"{_MODEL_NAME}-DFlash2"

    @property
    def converted_checkpoint(self) -> Path:
        return Path(self.checkpoint_dir) / f"{_MODEL_NAME}_torch_dist"

    @property
    def dataset(self) -> Path:
        return (
            Path(self.prompt_data) if self.prompt_data else Path(self.data_dir) / "dapo-math-17k/dapo-math-17k.jsonl"
        )

    @property
    def artifact_dir(self) -> Path:
        return Path(self.output_dir) / f"{self.sampling}-{self.arm}"


def _training_args(args: ScriptArgs) -> str:
    recipe = QwenDenseArgs(model_name=_MODEL_NAME).recipe
    checkpoint = (
        f"--hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} "
        f"--ref-load {shlex.quote(str(args.converted_checkpoint))} "
        # A fresh, absent load path starts the actor from ref-load without resuming
        # an earlier arm's optimizer or RNG state. _execute rejects reused output.
        f"--load {shlex.quote(str(args.artifact_dir / 'checkpoints'))} "
    )
    rollout = (
        f"--prompt-data {shlex.quote(str(args.dataset))} --input-key prompt --label-key label "
        "--apply-chat-template --apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--rollout-shuffle --rm-type dapo --reward-key score "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--num-steps-per-rollout 1 --update-weights-interval 1 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-max-prompt-len 1024 --rollout-max-context-len 4096 --balance-data "
    )
    performance = (
        f"--tensor-model-parallel-size {recipe.tensor_model_parallel_size} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {recipe.max_tokens_per_gpu} "
    )
    algorithm = (
        "--loss-type score_centering --advantage-estimator grpo --score-centering-is none "
        "--rollout-top-logprobs-num 128 --use-rollout-logprobs --disable-grpo-std-normalization "
        "--calculate-per-token-loss --rollout-temperature 1.0 --entropy-coef 0.0 "
        + (
            "--rollout-top-k 64 --rollout-top-p 0.9 "
            if args.sampling == "filtered"
            else "--rollout-top-k -1 --rollout-top-p 1.0 "
        )
    )
    # Reference KL is deliberately absent: support replay cannot use it. Both
    # arms use the same objective, and no PPO clipping or separate TIS is applied.
    optimizer = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    sglang = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {recipe.sglang_mem_fraction_static} "
        "--sglang-context-length 4096 --sglang-sampling-filter-order top_k_first "
        "--sglang-attention-backend fa3 --sglang-mamba-ssm-dtype float32 "
        "--sglang-mamba-radix-cache-strategy extra_buffer --sglang-kv-cache-dtype bfloat16 "
        "--sglang-sampling-mask-max-tokens 128 --sglang-max-running-requests 16 "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 "
    )
    if args.arm == "dflash":
        sglang += (
            "--sglang-speculative-algorithm DFLASH "
            f"--sglang-speculative-draft-model-path {shlex.quote(str(args.draft_checkpoint))} "
            f"--sglang-speculative-num-draft-tokens {args.draft_tokens} "
            "--sglang-speculative-accept-threshold-single 1.0 --sglang-speculative-accept-threshold-acc 1.0 "
        )
    miscellaneous = (
        "--bf16 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --colocate --offload-train-target cpu "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --seed {args.seed} "
        "--ci-test --check-weight-update-selector target --check-weight-update-skip-list visual. "
        "--use-tensorboard "
        f"--tb-project-name {shlex.quote(str(args.artifact_dir))} --tb-experiment-name probe "
        f"--save-debug-rollout-data {shlex.quote(str(args.artifact_dir / 'rollouts/{rollout_id}.pt'))} "
        f"--save-debug-train-data {shlex.quote(str(args.artifact_dir / 'train/{rollout_id}_{rank}.pt'))} "
        f"--save-debug-event-data {shlex.quote(str(args.artifact_dir / 'events'))} "
    )
    return " ".join((checkpoint, rollout, performance, algorithm, optimizer, sglang, miscellaneous))


def _manifest(args: ScriptArgs) -> dict:
    return {
        "model": _MODEL_NAME,
        "arm": args.arm,
        "sampling": args.sampling,
        "train_argv": shlex.split(_training_args(args)),
        "model_argv": shlex.split(load_model_args(_MODEL_TYPE)),
        "runtime_env": {"TENSORBOARD_DIR": str(args.artifact_dir / "tensorboard")},
        "note": "Integration smoke test; inspect nonzero advantages and gradients before claiming useful updates.",
    }


def _prepare(args: ScriptArgs) -> None:
    args.create_backend().convert_checkpoint(
        model_name=_MODEL_NAME,
        megatron_model_type=_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        hf_checkpoint=str(args.hf_checkpoint),
        dir_dst=args.checkpoint_dir,
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1",
    )


def _execute(args: ScriptArgs) -> None:
    required = [args.hf_checkpoint, args.converted_checkpoint / "latest_checkpointed_iteration.txt", args.dataset]
    if args.arm == "dflash":
        required.append(args.draft_checkpoint)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(args)
    (args.artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.create_backend().execute_train(
        train_args=_training_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MODEL_TYPE,
        megatron_path=args.megatron_path,
        extra_env_vars=manifest["runtime_env"],
        job_lifetime="launcher",
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    if args.stage == "prepare":
        _prepare(args)
    elif args.stage == "run":
        _execute(args)
    else:
        print(json.dumps(_manifest(args), indent=2))


if __name__ == "__main__":
    typer.run(main)
