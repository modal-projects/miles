"""Locate repeat/tree differences in a small native GLM, including TP and MoE.

No weight updates: repeat expanded execution before comparing compact execution.
Use two GPUs and the pinned GLM image, e.g. torchrun --nproc-per-node=2
tests/manual/prefix_tree/glm53_diagnose.py --model /path/to/GLM --output /tmp/check.json.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_sft import _check_replicated_weights, evaluate, make_model
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def _difference(actual, expected):
    a, b = actual.double(), expected.double()
    difference = a - b
    return {
        "relative_l2": float(difference.norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float(difference.abs().max()),
        "bitwise_equal": torch.equal(actual, expected),
    }


def _comparison(actual, expected):
    loss, logits, gradients, elapsed, traces = actual
    ref_loss, ref_logits, ref_gradients, _, ref_traces = expected
    names = sorted(ref_gradients)
    if set(gradients) != set(names):
        raise AssertionError("Gradient inventory changed")
    numerator = sum(float((gradients[n].double() - ref_gradients[n].double()).square().sum()) for n in names)
    denominator = sum(float(ref_gradients[n].double().square().sum()) for n in names)
    parameters = {name: _difference(gradients[name], ref_gradients[name]) for name in names}
    return {
        "loss": loss,
        "reference_loss": ref_loss,
        "forward_backward_s": elapsed,
        "logits": _difference(logits, ref_logits),
        "gradient_relative_l2": (numerator / max(denominator, 1e-30)) ** 0.5,
        "parameters": parameters,
        "activations": {name: _difference(traces[name], ref_traces[name]) for name in ref_traces},
        "routing": {
            name: {"changed_rows": int((traces[name] != value).any(-1).sum()), "rows": len(value)}
            for name, value in ref_traces.items()
            if name.endswith(".routing")
        },
    }


def _fp64_projection(x, weight, eps):
    # Test-only oracle for a shape-sensitive FP32 GEMM; not a provider change.
    x = x.double()
    return (x @ weight.double().t()).float(), (
        x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1]) + eps
    ).reciprocal().float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 12, 45])
    parser.add_argument("--precise-moe", action="store_true")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--index-topk", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint-mode", choices=("none", "full", "both"), default="both")
    parser.add_argument("--mhc-projection-fp64", action="store_true")
    parser.add_argument("--trace-operators", action="store_true")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
    os.environ["GLM_TREE_TRACE"] = "1"
    os.environ["GLM_TREE_TRACE_OPERATORS"] = "1" if args.trace_operators else "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(args.deterministic)
    generator = torch.Generator().manual_seed(71)
    prefix = torch.randint(1, 511, (129,), generator=generator).tolist()
    sequences = [prefix + [20 + i] + torch.randint(1, 511, (16,), generator=generator).tolist() for i in range(4)]
    targets = [[-100] * 128 + sequence[129:] + [-100] for sequence in sequences]
    report = []
    for depth in args.depths:
        modes = {"none": (False,), "full": (True,), "both": (False, True)}[args.checkpoint_mode]
        for recompute in modes:
            torch.manual_seed(42)
            model_parallel_cuda_manual_seed(42)
            model = make_model(
                args.model,
                lora=True,
                recompute=recompute,
                reduced_layers=depth,
                precise_moe=args.precise_moe,
                reduced_topk=args.topk,
                reduced_index_topk=args.index_topk,
            )
            _check_replicated_weights(model)
            if args.mhc_projection_fp64:
                for module in model.modules():
                    if hasattr(module, "_proj_rms_op"):
                        if module._proj_rms_compute_h_op is not None:
                            raise ValueError("The projection oracle requires unfused mHC")
                        module._proj_rms_op = _fp64_projection
            cold_expanded = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            expanded = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            repeated = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            cold_tree = evaluate(model, sequences, shared=True, lora=True, targets=targets)
            tree = evaluate(model, sequences, shared=True, lora=True, targets=targets)
            repeated_tree = evaluate(model, sequences, shared=True, lora=True, targets=targets)
            row = {
                "rank": rank,
                "depth": depth,
                "recompute": recompute,
                "topk": args.topk,
                "index_topk": args.index_topk,
                "deterministic": args.deterministic,
                "precise_moe": args.precise_moe,
                "mhc_projection_fp64": args.mhc_projection_fp64,
                "cold_expanded": _comparison(cold_expanded, expanded),
                "repeat": _comparison(repeated, expanded),
                "cold_tree": _comparison(cold_tree, expanded),
                "tree": _comparison(tree, expanded),
                "tree_repeat": _comparison(repeated_tree, tree),
            }
            report.append(row)
            print(
                "DIAGNOSTIC "
                + json.dumps(
                    {
                        "rank": rank,
                        "depth": depth,
                        "recompute": recompute,
                        "repeat_gradient_relative_l2": row["repeat"]["gradient_relative_l2"],
                        "tree_gradient_relative_l2": row["tree"]["gradient_relative_l2"],
                        "tree_repeat_gradient_relative_l2": row["tree_repeat"]["gradient_relative_l2"],
                        "repeat_logits": row["repeat"]["logits"],
                        "tree_logits": row["tree"]["logits"],
                    }
                ),
                flush=True,
            )
            args.output.with_name(args.output.stem + f"-rank{rank}.json").write_text(json.dumps(report))
            del model, cold_expanded, expanded, repeated, cold_tree, tree, repeated_tree
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
