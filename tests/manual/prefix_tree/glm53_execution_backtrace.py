"""Trace native, tree-without-sharing, and compact execution at identical weights.

The middle arm separates architecture-adapter correctness from changing shapes
and shared gradient fan-in. Run with torchrun on eight GPUs using a recorded
batch and saved adapters. This is diagnosis, not a throughput benchmark.
"""

import argparse
import gc
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_attention_trace import install_attention_trace
from glm53_dapo_sft import _phase, _RowMath, _sum_dp
from glm53_forward_trace import ForwardTrace, _load_adapters
from glm53_full_sft import _emit, _gradient_error, _maximum, _prepared
from glm53_fused_norm_control import unfuse_rmsnorm_linears
from glm53_lora_kernel_control import LoraKernelControl
from glm53_sft import make_model
from math_sft_optimizer import MasterOptimizer
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from miles.backends.training_utils.prefix_tree.data import sft_microbatches


def _persist(output, report):
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output / "report.json")


def _sources(output):
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.kda import chunk_kda
    from megatron.bridge.models.glm5_next.dsa import Glm5NextDSAAttention
    from megatron.bridge.models.glm5_next.kda import Glm5NextKDAAttention
    from megatron.bridge.models.glm5_next.kpool_indexer import build_pooled_keys

    functions = [
        Glm5NextKDAAttention.forward,
        Glm5NextDSAAttention._kpool_select,
        causal_conv1d,
        chunk_kda,
        build_pooled_keys,
    ]
    records = {
        f"{fn.__module__}.{fn.__qualname__}": {"file": inspect.getsourcefile(fn), "source": inspect.getsource(fn)}
        for fn in functions
    }
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "installed_sources.json").write_text(json.dumps(records, indent=2))


class LayerKernelTrace:
    """Profile the first block's actual CPU dispatch and CUDA kernels."""

    def __init__(self, model, output, label):
        self.output, self.label, self.profile = output, label, None
        self.handles = [
            model.decoder.layers[0].register_forward_pre_hook(self._begin),
            model.decoder.layers[0].register_forward_hook(self._end),
        ]

    def _begin(self, module, inputs):
        if dist.get_rank() == 0:
            self.profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            )
            self.profile.__enter__()

    def _end(self, module, inputs, outputs):
        if self.profile is not None:
            self.profile.__exit__(None, None, None)
            rows = [
                {
                    "operator": event.name,
                    "input_shapes": event.input_shapes,
                    "kernels": [kernel.name for kernel in event.kernels],
                }
                for event in self.profile.events()
                if event.kernels
            ]
            (self.output / f"kernels-{self.label}.json").write_text(json.dumps(rows, indent=2))
        for handle in self.handles:
            handle.remove()


def _forward(model, plan, batch, trace, *, label, attach, replace, output):
    trace.begin(label, plan, replace)
    handles, originals = trace.install(model)
    attention_handles, attention_originals = install_attention_trace(model, trace, attach=attach)
    handles.extend(attention_handles)
    originals.extend(attention_originals)
    kernel_trace = LayerKernelTrace(model, output, label)
    try:
        with torch.no_grad():
            loss = batch.loss(model, shared=attach)
        torch.cuda.synchronize()
        row = {
            "phase": label,
            "local_loss": float(loss),
            "physical_rows": len(plan.tokens),
            "tree_metadata": attach,
            **trace.report(),
        }
    finally:
        for handle in handles + kernel_trace.handles:
            handle.remove()
        for module, attr, original in originals:
            setattr(module, attr, original)
    return row


def _gradients(model, data, arms, output, report, policy, lora_control=None):
    optimizer = MasterOptimizer(model, 1e-5)
    total = _sum_dp(data.supervised_tokens)
    for controlled in (False, True):
        policy.activate(controlled)
        if lora_control is not None:
            lora_control.activate("rowwise_all" if controlled else "native")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not controlled
        reference = None
        for name, plan, batch, attach in arms:
            # _phase dispatch is the same execution flag as forward: the middle
            # arm attaches tree metadata to an expanded, unshared plan.
            stats = _phase(model, optimizer, ((plan, batch),), (data,), shared=attach, supervised_global=total)
            gradients = optimizer.snapshot()
            row = {"policy": "controlled" if controlled else "native", "execution": name, **stats}
            if reference is None:
                reference = gradients
            else:
                row["gradient_relative_l2_max"] = _maximum(_gradient_error(reference, gradients))
            report.setdefault("backward", []).append(row)
            _persist(output, report)
            _emit("EXECUTION_BACKWARD", row)
        del reference
    del optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--adapter-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--microbatch-index", type=int, default=0)
    parser.add_argument("--trace-layers", default="0,1,2,3")
    parser.add_argument("--layers", type=int, default=45)
    parser.add_argument("--expert-parallel", type=int, default=8)
    parser.add_argument("--data-dp-size", type=int)
    parser.add_argument("--rowwise-lora", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_tensor_parallel_size=1, expert_model_parallel_size=args.expert_parallel
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    _sources(args.output_dir)
    model = make_model(
        args.model,
        lora=True,
        full_width=True,
        pretrained=True,
        pretrained_layers=args.layers,
        full_width_expert_parallel_size=args.expert_parallel,
        sequence_length=16384,
        recompute=True,
    )
    steps = _load_adapters(model, args.adapter_state, allow_deeper_layers=args.layers < 45)
    data = sft_microbatches(
        json.loads(args.rollouts.read_text()),
        dp_rank=parallel_state.get_data_parallel_rank(),
        dp_size=args.data_dp_size or parallel_state.get_data_parallel_world_size(),
    )[args.microbatch_index]
    expanded, compact = _prepared(data, shared=False), _prepared(data, shared=True)
    arms = [
        ("native_expanded", *expanded, False),
        ("native_repeat", *expanded, False),
        ("tree_without_sharing", *expanded, True),
        ("compact_tree", *compact, True),
    ]
    trace = ForwardTrace(data, expanded[0], compact[0], map(int, args.trace_layers.split(",")))
    policy = _RowMath(model, "fp32")
    unfused = unfuse_rmsnorm_linears(model, "fp32")
    for _, module, native in unfused:
        policy.operators.append((module, "forward", native, module.forward))
    seen = set()
    for name, module in model.named_modules():
        if name.endswith(".router") and id(module.config) not in seen:
            seen.add(id(module.config))
            policy.operators.append((module.config, "moe_router_fusion", module.config.moe_router_fusion, False))
    lora_control = LoraKernelControl(model) if args.rowwise_lora else None
    report = {
        "status": "tracing",
        "saved_optimizer_steps": steps,
        "layers": args.layers,
        "rowwise_lora": args.rowwise_lora,
        "forward": [],
    }
    _persist(args.output_dir, report)
    for controlled in (False, True):
        policy.activate(controlled)
        if lora_control is not None:
            lora_control.activate("rowwise_all" if controlled else "native")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not controlled
        for index, (name, plan, batch, attach) in enumerate(arms):
            label = ("controlled_" if controlled else "native_") + name
            _emit("EXECUTION_FORWARD_BEGIN", {"phase": label})
            row = _forward(
                model, plan, batch, trace, label=label, attach=attach, replace=index == 0, output=args.output_dir
            )
            report["forward"].append(row)
            _persist(args.output_dir, report)
            _emit("EXECUTION_FORWARD_END", {k: v for k, v in row.items() if k != "modules"})
    del trace
    gc.collect()
    _gradients(model, data, arms, args.output_dir, report, policy, lora_control)
    report["status"] = "complete"
    _persist(args.output_dir, report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
