"""Full pretrained GLM SFT on recorded DAPO prompts and sampled completions.

Run with torchrun on eight GPUs in the pinned GLM trainer image. The input JSON
comes from real sampling, never padded prompts or invented reference solutions.
One ordinary expanded optimizer update precedes three paired fixed-weight
forward/backward timings, then four compact updates. FP32 master weights,
optimizer states and microbatch-gradient accumulation match mixed-precision
training practice. No metric threshold kills the run.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_diagnose import _bmm_projection, _normalization_oracle
from glm53_full_sft import _emit, _expanded_loss, _gradient_error, _maximum, _prepared
from glm53_fused_norm_control import unfuse_rmsnorm_linears
from glm53_sft import _check_replicated_weights, make_model
from math_sft_data import global_token_scale, math_microbatches
from math_sft_optimizer import MasterOptimizer
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


class _RowMath:
    def __init__(self, model, mode):
        self.operators = []
        import transformer_engine.pytorch as te

        norms = [(m, m.forward) for m in model.modules() if isinstance(m, te.RMSNorm)]
        if mode != "native":
            _normalization_oracle(model, mode)
        for module, native in norms:
            self.operators.append((module, "forward", native, module.forward))
        for module in model.modules():
            if mode != "native" and hasattr(module, "_proj_rms_op"):
                if module._proj_rms_compute_h_op is not None:
                    raise ValueError("Rowwise projection currently requires native unfused mHC")
                self.operators.append(
                    (module, "_proj_rms_op", module._proj_rms_op, _bmm_projection(module._proj_rms_op))
                )
        self.activate(False)

    def activate(self, stable):
        for module, name, native, replacement in self.operators:
            setattr(module, name, replacement if stable else native)


def _persist(output, report):
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / "report.json")


def _sum_dp(value):
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, group=parallel_state.get_data_parallel_group())
    return float(tensor)


def _phase(model, optimizer, prepared, data, *, shared, supervised_global):
    optimizer.clear(model)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    dp = parallel_state.get_data_parallel_world_size()
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    started = time.perf_counter()
    for microstep, (microbatch, (plan, batch)) in enumerate(zip(data, prepared, strict=True), 1):
        _emit("DAPO_MICROSTEP_BEGIN", {"shared": shared, "microstep": microstep, "total": len(data)})
        loss = batch.loss(model) if shared else _expanded_loss(model, plan, batch)
        # Each local microbatch returns a mean. This scaling followed by the
        # existing DP average gives the global original-occurrence token mean,
        # including unequal completion lengths and unequal DP token totals.
        weighted = loss * global_token_scale(microbatch.supervised_tokens, supervised_global, dp)
        if not torch.isfinite(weighted):
            raise FloatingPointError("Nonfinite SFT loss")
        weighted.backward()
        optimizer.accumulate(model)
        loss_sum += weighted.detach().double()
        _emit("DAPO_MICROSTEP_END", {"shared": shared, "microstep": microstep})
    optimizer.synchronize()
    norm = torch.nn.utils.get_total_norm(optimizer.buffers, foreach=True, error_if_nonfinite=True)
    torch.cuda.synchronize()
    elapsed = _maximum(time.perf_counter() - started)
    dist.all_reduce(loss_sum, group=parallel_state.get_data_parallel_group())
    loss_sum /= dp
    real_tokens = _sum_dp(sum(batch.real_input_tokens for batch in data))
    physical_tokens = _sum_dp(sum(len(plan.tokens) for plan, _ in prepared))
    return {
        "loss": float(loss_sum),
        "forward_backward_s": elapsed,
        "real_original_input_tokens": int(real_tokens),
        "physical_input_rows": int(physical_tokens),
        "supervised_tokens": int(supervised_global),
        "real_input_tokens_per_s": real_tokens / elapsed,
        "supervised_tokens_per_s": supervised_global / elapsed,
        "microsteps": len(data),
        "local_gradient_norm": float(norm),
        "peak_allocated_gib_max": _maximum(torch.cuda.max_memory_allocated() / 1024**3),
    }


@torch.no_grad()
def _paired_optimizer_step(optimizer, reference_gradients):
    shadows = optimizer.reference_step(reference_gradients)
    update_norm = sum(float((a - b).double().square().sum()) for a, b in zip(shadows, optimizer.masters, strict=True))
    optimizer.step()
    error = sum(float((a - b).double().square().sum()) for a, b in zip(shadows, optimizer.masters, strict=True))
    denominator = sum(float(p.double().square().sum()) for p in shadows)
    return {
        "update_relative_l2_max": _maximum((error / max(update_norm, 1e-30)) ** 0.5),
        "master_parameter_relative_l2_max": _maximum((error / max(denominator, 1e-30)) ** 0.5),
    }


@torch.no_grad()
def _repeat_optimizer_comparison(optimizer, first_gradients, repeated_gradients):
    first = optimizer.reference_step(first_gradients)
    repeated = optimizer.reference_step(repeated_gradients)
    error = sum(float((a - b).double().square().sum()) for a, b in zip(first, repeated, strict=True))
    step = sum(float((a - b).double().square().sum()) for a, b in zip(first, optimizer.masters, strict=True))
    return {"update_relative_l2_max": _maximum((error / max(step, 1e-30)) ** 0.5)}


def _save(model, optimizer, output, report):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "parameters": {name: p.detach().cpu() for name, p in optimizer.parameters},
            "fp32_master_parameters": {
                name: master.detach().cpu()
                for (name, _), master in zip(optimizer.parameters, optimizer.masters, strict=True)
            },
            "optimizer": optimizer.optimizer.state_dict(),
            "steps": len(report["updates"]),
            "tp": 2,
            "ep": 8,
            "sp": True,
            "adapter_slots": 2,
            "rank": 16,
        },
        output / f"state-rank-{dist.get_rank()}.pt",
    )
    dist.barrier()
    if dist.get_rank() == 0:
        (output / "report.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--row-math", choices=("native", "fp32", "fp64"), default="fp32")
    parser.add_argument("--unfused-norm", choices=("none", "fp32", "fp64"), default="none")
    parser.add_argument("--strict-reductions", action="store_true")
    parser.add_argument("--unfused-router", action="store_true", help="Disable router fusion in controlled phases")
    args = parser.parse_args()
    if args.steps < 2 or args.pairs < 2:
        parser.error("Use at least two optimizer updates and paired timings")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    if "summary" not in rollouts or len(rollouts["samples"]) != 128:
        raise ValueError("Require all 128 recorded real trajectories before training")
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
    # Standard LoRA initialization for real SFT. The first ordinary expanded
    # update makes B nonzero before comparing both A and B gradients.
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "adapters"):
                for adapter in module.adapters:
                    adapter.linear_out.weight.zero_()
    _check_replicated_weights(model)
    policy = _RowMath(model, args.row_math)
    unfused = unfuse_rmsnorm_linears(model, args.unfused_norm) if args.unfused_norm != "none" else []
    for _, module, native in unfused:
        policy.operators.append((module, "forward", native, module.forward))
    if args.unfused_router:
        seen = set()
        for name, module in model.named_modules():
            if name.endswith(".router") and id(module.config) not in seen:
                seen.add(id(module.config))
                policy.operators.append((module.config, "moe_router_fusion", module.config.moe_router_fusion, False))
    policy.activate(False)
    optimizer = MasterOptimizer(model, args.lr)
    data = math_microbatches(
        rollouts,
        dp_rank=parallel_state.get_data_parallel_rank(),
        dp_size=parallel_state.get_data_parallel_world_size(),
    )
    supervised_global = _sum_dp(sum(batch.supervised_tokens for batch in data))
    preparation = time.perf_counter()
    expanded = tuple(_prepared(batch, shared=False) for batch in data)
    compact = tuple(_prepared(batch, shared=True) for batch in data)
    report = {
        "status": "training",
        "settings": vars(args) | {"rollouts": str(args.rollouts), "output_dir": str(args.output_dir)},
        "data_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "rollout_summary": rollouts["summary"],
        "dataset": rollouts["dataset"],
        "dataset_revision": rollouts["revision"],
        "updates": [],
        "pairs": [],
        "native_baseline": [],
        "optimizer_master_dtype": "float32",
        "unfused_norm_modules": [name for name, _, _ in unfused],
        "gradient_accumulation": "float32 from native backward gradients",
        "frozen_kv_up_adapter": True,
        "planning_s_max": _maximum(time.perf_counter() - preparation),
        "padding_rows": int(_sum_dp(sum(batch.padding_tokens for batch in data))),
    }
    _emit("DAPO_MODEL_READY", {"layers": 45, "microsteps": len(data), "supervised_tokens": supervised_global})
    _persist(args.output_dir, report)
    warm = _phase(model, optimizer, expanded, data, shared=False, supervised_global=supervised_global)
    optimizer.step()
    report["updates"].append({"step": 1, "execution": "expanded", **warm})
    _emit("DAPO_SFT_UPDATE", report["updates"][-1])
    for iteration in range(args.pairs):
        native = _phase(model, optimizer, expanded, data, shared=False, supervised_global=supervised_global)
        report["native_baseline"].append(native)
        _emit("DAPO_NATIVE_BASELINE", {"iteration": iteration + 1, **native})
        _persist(args.output_dir, report)
    native_gradients = optimizer.snapshot()
    policy.activate(True)
    if args.strict_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    stable_warm = _phase(model, optimizer, expanded, data, shared=False, supervised_global=supervised_global)
    report["operator_policy_gradient_difference_max"] = _maximum(
        _gradient_error(native_gradients, optimizer.snapshot())
    )
    report["operator_policy_expanded_loss"] = stable_warm["loss"]
    del native_gradients
    _phase(model, optimizer, compact, data, shared=True, supervised_global=supervised_global)
    first_reference, first_tree = None, None
    for pair in range(args.pairs):
        ordinary = _phase(model, optimizer, expanded, data, shared=False, supervised_global=supervised_global)
        reference = optimizer.snapshot()
        expanded_repeat = _maximum(_gradient_error(first_reference, reference)) if first_reference else 0.0
        if first_reference is None:
            first_reference = reference
        tree = _phase(model, optimizer, compact, data, shared=True, supervised_global=supervised_global)
        gradients = optimizer.snapshot()
        tree_repeat = _maximum(_gradient_error(first_tree, gradients)) if first_tree else 0.0
        if first_tree is None:
            first_tree = gradients
        row = {
            "pair": pair + 1,
            "expanded": ordinary,
            "tree": tree,
            "speedup": ordinary["forward_backward_s"] / tree["forward_backward_s"],
            "gradient_relative_l2_max": _maximum(_gradient_error(reference, gradients)),
            "expanded_repeat_gradient_relative_l2_max": expanded_repeat,
            "tree_repeat_gradient_relative_l2_max": tree_repeat,
        }
        report["pairs"].append(row)
        _emit("DAPO_PAIRED_BENCHMARK", row)
        _persist(args.output_dir, report)
    report["optimizer_repeat_comparison"] = _repeat_optimizer_comparison(optimizer, first_reference, reference)
    report["optimizer_comparison"] = _paired_optimizer_step(optimizer, reference)
    report["updates"].append({"step": 2, "execution": "tree", **tree})
    _emit("DAPO_SFT_UPDATE", report["updates"][-1] | report["optimizer_comparison"])
    _persist(args.output_dir, report)
    del reference, gradients, first_reference, first_tree
    gc.collect()
    for step in range(3, args.steps + 1):
        stats = _phase(model, optimizer, compact, data, shared=True, supervised_global=supervised_global)
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        stats["optimizer_s"] = _maximum(time.perf_counter() - started)
        report["updates"].append({"step": step, "execution": "tree", **stats})
        _emit("DAPO_SFT_UPDATE", report["updates"][-1])
        _persist(args.output_dir, report)
    report["median_speedup"] = statistics.median(row["speedup"] for row in report["pairs"])
    report["speedup_over_native_miles_forward_backward"] = statistics.median(
        row["forward_backward_s"] for row in report["native_baseline"]
    ) / statistics.median(row["tree"]["forward_backward_s"] for row in report["pairs"])
    report["status"] = "complete"
    _save(model, optimizer, args.output_dir, report)
    _emit("DAPO_SFT_COMPLETE", {"updates": args.steps, "median_speedup": report["median_speedup"]})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
