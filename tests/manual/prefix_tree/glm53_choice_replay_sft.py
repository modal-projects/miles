"""Conditional expert/key-choice diagnosis on real DAPO, not a speed benchmark.

Record the expanded model's native choices, fork prefixes whose recorded choices
are incompatible, then compare expanded/tree gradients with neither, expert,
and expert plus DSA key replay. All comparisons use fixed post-SFT weights.
No upstream source or production replay defaults are modified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_choice_replay import ChoiceReplay, compatible_data
from glm53_dapo_sft import _persist, _phase, _RowMath, _sum_dp
from glm53_forward_trace import ForwardTrace, _data, _load_adapters
from glm53_full_sft import _emit, _expanded_loss, _gradient_error, _maximum, _prepared
from glm53_fused_norm_control import unfuse_rmsnorm_linears
from glm53_sft import make_model
from math_sft_optimizer import MasterOptimizer
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-all", action="store_true", help="Run only expert plus key replay")
    parser.add_argument(
        "--strict-reductions",
        action="store_true",
        help="Disable BF16 GEMM reduced-precision reductions after recording choices",
    )
    parser.add_argument(
        "--trace", action="store_true", help="Trace fixed-choice expanded/tree forwards before backward"
    )
    parser.add_argument("--unfused-norm", choices=("none", "fp32", "fp64"), default="none")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_tensor_parallel_size=1, expert_model_parallel_size=8
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    rollouts = json.loads(args.rollouts.read_text())
    if rollouts["config_sha256"] != hashlib.sha256(Path(args.model, "config.json").read_bytes()).hexdigest():
        raise ValueError("Sampler and trainer checkpoint configs differ")
    model = make_model(
        args.model,
        lora=True,
        full_width=True,
        pretrained=True,
        pretrained_layers=45,
        sequence_length=16384,
        recompute=True,
    )
    steps = _load_adapters(model, args.adapter_state)
    policy = _RowMath(model, "fp32")
    policy.activate(True)
    unfused = unfuse_rmsnorm_linears(model, args.unfused_norm) if args.unfused_norm != "none" else []
    optimizer = MasterOptimizer(model, 1e-5)
    data = _data(rollouts, 0, None)
    original_expanded, _ = _prepared(data, shared=False)
    replay = ChoiceReplay(model, original_expanded)
    _emit("CHOICE_RECORD_BEGIN", {})
    with torch.no_grad():
        _, batch = _prepared(data, shared=False)
        recorded_loss = float(_expanded_loss(model, original_expanded, batch))
    choices = replay.finish_record()
    data, eligibility = compatible_data(data, choices)
    del choices
    expanded, compact = _prepared(data, shared=False), _prepared(data, shared=True)
    if expanded[0].tokens != original_expanded.tokens or expanded[0].positions != original_expanded.positions:
        raise ValueError("Replay eligibility changed expanded reference tokens or positions")
    replay.prepare_compact(expanded[0], compact[0])
    supervised = _sum_dp(data.supervised_tokens)
    inventories = [None] * dist.get_world_size()
    dist.all_gather_object(inventories, {"rank": dist.get_rank(), **eligibility})
    report = {
        "status": "diagnosing",
        "saved_optimizer_steps": steps,
        "strict_reductions": args.strict_reductions,
        "moe_router_fusion": False,
        "row_math": "fp32",
        "unfused_norm": args.unfused_norm,
        "unfused_modules": [name for name, _, _ in unfused],
        "eligibility": inventories,
        "recorded_local_loss": recorded_loss,
        "phases": [],
        "scope": "Fixed-weight first real DAPO microbatch, full 45 layers; conditional replay diagnostic",
    }
    _persist(args.output_dir, report)
    # Keep recorded choices independent of the reduction-precision experiment.
    if args.strict_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.trace:
        trace = ForwardTrace(data, expanded[0], compact[0], (0,))
        report["forward_trace"] = []
        for shared in (False, True):
            replay.enable("all", shared=shared)
            plan, batch = compact if shared else expanded
            trace.begin("fixed_choices_tree" if shared else "fixed_choices_expanded", plan, not shared)
            handles, originals = trace.install(model)
            try:
                with torch.no_grad():
                    loss = batch.loss(model) if shared else _expanded_loss(model, plan, batch)
                torch.cuda.synchronize()
            finally:
                for handle in handles:
                    handle.remove()
                for module, attribute, original in originals:
                    setattr(module, attribute, original)
            if dist.get_rank() == 0:
                report["forward_trace"].append({"shared": shared, "local_loss": float(loss), **trace.report()})
                _emit("CHOICE_FORWARD_TRACE", report["forward_trace"][-1])
            _persist(args.output_dir, report)
        del trace
    native_reference = None
    for mode in ("all",) if args.only_all else ("none", "experts", "all"):
        reference, tree_reference = None, None
        cases = (False, False, True, True) if mode == "all" else (False, False, True)
        for iteration, shared in enumerate(cases):
            replay.enable(mode, shared=shared)
            _emit("CHOICE_PHASE_BEGIN", {"mode": mode, "shared": shared, "iteration": iteration})
            stats = _phase(
                model,
                optimizer,
                (compact if shared else expanded,),
                (data,),
                shared=shared,
                supervised_global=supervised,
            )
            gradients = optimizer.snapshot()
            row = {"mode": mode, "shared": shared, "iteration": iteration, **stats}
            if reference is None:
                reference = gradients
                if native_reference is None:
                    native_reference = gradients
                row["expanded_vs_unreplayed_gradient_relative_l2_max"] = _maximum(
                    _gradient_error(native_reference, gradients)
                )
            else:
                row["expanded_reference_gradient_relative_l2_max"] = _maximum(_gradient_error(reference, gradients))
            if shared:
                # Error energy by layer localizes the residual; rank-zero is
                # one declared shard, whereas the scalar metric is max-rank.
                if dist.get_rank() == 0:
                    groups = {}
                    for name, expected in reference.items():
                        group = (
                            ".".join(name.split(".")[:3]) if name.startswith("decoder.layers.") else name.split(".")[0]
                        )
                        numerator, denominator = groups.get(group, (0.0, 0.0))
                        numerator += float((gradients[name] - expected).double().square().sum())
                        denominator += float(expected.double().square().sum())
                        groups[group] = numerator, denominator
                    total = sum(value[1] for value in groups.values())
                    row["rank_zero_layer_errors"] = {
                        name: {
                            "relative_l2": (error / max(norm, 1e-30)) ** 0.5,
                            "fraction_of_total_squared_error": error / max(sum(v[0] for v in groups.values()), 1e-30),
                            "error_over_total_gradient_norm": (error / max(total, 1e-30)) ** 0.5,
                        }
                        for name, (error, norm) in groups.items()
                    }
                if tree_reference is not None:
                    row["tree_repeat_gradient_relative_l2_max"] = _maximum(_gradient_error(tree_reference, gradients))
                tree_reference = gradients
            report["phases"].append(row)
            _emit("CHOICE_PHASE_END", row)
            _persist(args.output_dir, report)
    report["status"] = "complete"
    _persist(args.output_dir, report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
