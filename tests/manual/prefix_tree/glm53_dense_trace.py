"""Isolate the first dense MLP on the same real first-microbatch trajectories.

The embedding/first pretrained block and saved adapters are unchanged. Later
blocks are omitted because they cannot affect block-zero forward values. TP=2,
SP enabled, EP=1 (there are no routed experts in this one-block slice). This is
forward localization, not a full-model gradient or throughput measurement.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_dapo_sft import _persist, _RowMath
from glm53_forward_trace import ForwardTrace, _load_adapters
from glm53_full_sft import _emit, _expanded_loss, _prepared
from glm53_fused_norm_control import unfuse_rmsnorm_linears
from glm53_sft import make_model
from glm53_trace_values import map_tensors
from math_sft_data import math_microbatches
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp32-base", choices=("none", "up", "down", "both"), default="none")
    parser.add_argument("--unfused-norm", choices=("none", "fp32", "fp64"), default="none")
    parser.add_argument("--gradient-check", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    model = make_model(
        args.model,
        lora=True,
        full_width=True,
        pretrained=True,
        pretrained_layers=1,
        sequence_length=16384,
        full_width_expert_parallel_size=1,
    )
    steps = _load_adapters(model, args.adapter_state, allow_deeper_layers=True)
    policy = _RowMath(model, "fp32")
    policy.activate(True)
    unfused = unfuse_rmsnorm_linears(model, args.unfused_norm) if args.unfused_norm != "none" else []
    controlled = []
    choices = {"none": (), "up": ("linear_fc1",), "down": ("linear_fc2",), "both": ("linear_fc1", "linear_fc2")}
    for name in choices[args.fp32_base]:
        base = getattr(model.decoder.layers[0].mlp, name).to_wrap
        base.to(dtype=torch.float32)

        def promote(module, values):
            return (values[0].float(), *values[1:])

        def cast_output(module, values, output):
            return map_tensors(output, lambda item: item.to(torch.bfloat16))

        base.register_forward_pre_hook(promote)
        base.register_forward_hook(cast_output)
        controlled.append(name)
    # Preserve the original DP-zero batch, rather than repartition the dataset
    # after reducing the diagnostic's world size.
    data = math_microbatches(json.loads(args.rollouts.read_text()), dp_rank=0, dp_size=4)[0]
    expanded, compact = _prepared(data, shared=False), _prepared(data, shared=True)
    trace = ForwardTrace(data, expanded[0], compact[0], (0,))
    report = {
        "status": "tracing",
        "saved_optimizer_steps": steps,
        "fp32_base": controlled,
        "unfused_norm": args.unfused_norm,
        "unfused_modules": [name for name, _, _ in unfused],
        "precision_scope": "Frozen dense base weight/input (including any fused norm) promoted to FP32; nested outputs return to BF16. Native TP/SP communication retained.",
        "phases": [],
        "scope": "First pretrained dense block, original real DP-zero microbatch, TP2/SP/EP1",
    }
    for name, shared, replace in [("expanded", False, True), ("expanded_repeat", False, False), ("tree", True, False)]:
        plan, batch = compact if shared else expanded
        trace.begin(name, plan, replace)
        handles, originals = trace.install(model)
        _emit("DENSE_TRACE_BEGIN", {"phase": name})
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
            report["phases"].append({"phase": name, "local_loss": float(loss), **trace.report()})
            _emit("DENSE_TRACE_END", report["phases"][-1])
        _persist(args.output_dir, report)
    del trace
    if args.gradient_check:
        from glm53_dapo_sft import _phase, _sum_dp
        from glm53_full_sft import _gradient_error, _maximum
        from math_sft_optimizer import MasterOptimizer

        optimizer = MasterOptimizer(model, 1e-5)
        report["gradient_checks"] = []
        reference, tree_reference = None, None
        for shared in (False, False, True, True):
            stats = _phase(
                model,
                optimizer,
                (compact if shared else expanded,),
                (data,),
                shared=shared,
                supervised_global=_sum_dp(data.supervised_tokens),
            )
            gradients = optimizer.snapshot()
            row = {"shared": shared, **stats}
            if reference is None:
                reference = gradients
            else:
                row["expanded_reference_gradient_relative_l2_max"] = _maximum(_gradient_error(reference, gradients))
            if shared:
                if tree_reference is not None:
                    row["tree_repeat_gradient_relative_l2_max"] = _maximum(_gradient_error(tree_reference, gradients))
                tree_reference = gradients
            report["gradient_checks"].append(row)
            _emit("DENSE_GRADIENT_CHECK", row)
            _persist(args.output_dir, report)
    report["status"] = "complete"
    _persist(args.output_dir, report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
