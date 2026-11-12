"""Locate the first full-width GLM layout difference on recorded math samples.

Forward localization with optional gradient checks, not a throughput benchmark. Reuse one recorded real
microbatch and saved adapter shards; compare native and controlled operators.
Only rank zero retains CPU activations. Selected layers use full tensors; other
layer outputs use two int64 row fingerprints for screening, not as a proof of
bitwise equality. No upstream module source is edited.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_dapo_sft import _phase, _RowMath, _sum_dp
from glm53_full_sft import _emit, _expanded_loss, _gradient_error, _maximum, _prepared
from glm53_sft import make_model
from glm53_trace_values import first_tensor as _first_tensor
from math_sft_data import _make_batch, math_microbatches
from math_sft_optimizer import MasterOptimizer
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def _compare(actual, reference, positions, samples, shared):
    if actual.shape != reference.shape:
        raise ValueError(f"Trace shape differs: {actual.shape} != {reference.shape}")
    equal = actual.dtype == reference.dtype and torch.equal(actual, reference)
    result = {
        "bitwise_equal": equal,
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "shape": list(actual.shape),
        "changed_rows": 0,
    }
    if equal:
        return result
    changed = (actual != reference).reshape(len(actual), -1).any(-1)
    first = int(changed.nonzero()[0]) if changed.any() else 0
    result.update(
        changed_rows=int(changed.sum()),
        first_row=first,
        first_position=int(positions[first]),
        first_sample=int(samples[first]),
        first_in_shared_prefix=bool(shared[first]),
    )
    if actual.is_floating_point():
        delta = actual.float() - reference.float()
        numerator = float(delta.square().sum(dtype=torch.float64))
        denominator = float(reference.float().square().sum(dtype=torch.float64))
        result.update(relative_l2=(numerator / max(denominator, 1e-30)) ** 0.5, max_abs=float(delta.abs().max()))
    return result


class ForwardTrace:
    def __init__(self, data, expanded_plan, compact_plan, layers):
        self.layers = frozenset(layers)
        self.active = parallel_state.get_data_parallel_rank() == 0
        self.owner = dist.get_rank() == 0
        self.references = {}
        self.summaries = {}
        self.plan = None
        self.replace = False
        self.positions = torch.tensor([p for seq in data.sequences for p in range(len(seq))])
        self.samples = torch.tensor([i for i, seq in enumerate(data.sequences) for _ in seq])
        mapping = torch.tensor([row for path in compact_plan.sample_nodes for row in path])
        self.shared = torch.bincount(mapping, minlength=len(compact_plan.tokens))[mapping] > 1
        if len(self.positions) != expanded_plan.expanded_tokens:
            raise ValueError("Original trace occurrence inventory differs")

    def begin(self, name, plan, replace):
        self.name, self.plan, self.replace = name, plan, replace
        self.rows = torch.tensor([row for path in plan.sample_nodes for row in path])
        self.device_positions = torch.tensor(plan.positions, device="cuda", dtype=torch.int32)
        self.summaries = {}

    def capture(self, name, value, *, fingerprint=False):
        if not self.active or not isinstance(value, torch.Tensor) or value.ndim < 2:
            return
        value = value.detach()
        count = len(self.plan.tokens)
        tp = parallel_state.get_tensor_model_parallel_world_size()
        if ".o_norm." in name and value.shape[0] % count == 0:
            value = value.reshape(count, -1)
        if value.shape[0] not in (count, count // tp):
            raise ValueError(f"{name}: cannot map shape {tuple(value.shape)} onto {count} rows")
        value = value.reshape(value.shape[0], -1).contiguous()
        pieces = [torch.empty_like(value) for _ in range(tp)]
        dist.all_gather(pieces, value, group=parallel_state.get_tensor_model_parallel_group())
        # SP tensors partition rows. Column-parallel full-row outputs partition
        # channels; retain both ranks so a difference on rank one is visible.
        value = torch.cat(pieces, dim=0 if value.shape[0] != count else 1)
        if not self.owner:
            return
        source_dtype = str(value.dtype)
        if fingerprint:
            word_type = {
                torch.bfloat16: torch.int16,
                torch.float16: torch.int16,
                torch.float32: torch.int32,
                torch.float64: torch.int64,
            }[value.dtype]
            words = value.contiguous().view(word_type).to(torch.int64)
            index = torch.arange(words.shape[1], device=words.device, dtype=torch.int64)
            a = (index + 1) * 6364136223846793005 + 1442695040888963407
            b = ((index + 11) * 2862933555777941757) ^ (index << 32)
            first = (words * a).sum(-1)
            second = (words * (words + 3) * b).sum(-1)
            value = torch.stack((first, second), dim=-1)
        canonical = value.cpu().index_select(0, self.rows)
        if name in self.references:
            summary = _compare(canonical, self.references[name], self.positions, self.samples, self.shared)
            summary["comparison"] = "two_int64_row_fingerprints" if fingerprint else "full_tensor"
            summary["source_dtype"] = source_dtype
            self.summaries[name] = summary
            if not summary["bitwise_equal"]:
                _emit("FULL_FORWARD_DIFFERENCE", {"phase": self.name, "module": name, **summary})
        if self.replace or name not in self.references:
            self.references[name] = canonical

    def module_hook(self, name, *, inputs=False, routes=False, fingerprint=False):
        def hook(module, args, kwargs, output):
            if inputs:
                value = args[0] if args else kwargs.get("hidden_states")
                self.capture(name + ".input", value)
            if inputs and name.endswith(".o_norm"):
                for index, value in enumerate(args[1:], 1):
                    self.capture(name + f".input{index}", value)
            # AdapterWrapper supports ((projection, normalized_input), bias)
            # as well as (projection, bias, normalized_input). Inspect the norm
            # before its consumer projection rather than silently skipping it.
            if name.endswith(".to_wrap") and isinstance(output, tuple):
                if len(output) == 2 and isinstance(output[0], tuple) and len(output[0]) == 2:
                    self.capture(name + ".normalized_output", output[0][1])
                elif len(output) == 3:
                    self.capture(name + ".normalized_output", output[2])
            self.capture(name + ".output", _first_tensor(output), fingerprint=fingerprint)
            if name.endswith("hyper_connection") and isinstance(output, tuple):
                for index, value in enumerate(output[1:], 1):
                    self.capture(name + f".output{index}", value)
            if routes and isinstance(output, tuple):
                self.capture(name + ".routes", output[1])

        return hook

    def install(self, model):
        handles, originals = [], []
        if not self.active:
            return handles, originals
        for name, module in model.named_modules():
            if name.startswith("decoder.layers.") and int(name.split(".")[2]) not in self.layers:
                if name.count(".") == 2:
                    handles.append(
                        module.register_forward_hook(self.module_hook(name, fingerprint=True), with_kwargs=True)
                    )
                continue
            leaf = name.rsplit(".", 1)[-1]
            if name == "embedding" or name.endswith((".self_attention", ".mlp", ".router")):
                handles.append(
                    module.register_forward_hook(
                        self.module_hook(name, routes=name.endswith(".router")), with_kwargs=True
                    )
                )
            elif name.startswith("decoder.layers.") and name.count(".") == 2 and leaf.isdigit():
                handles.append(module.register_forward_hook(self.module_hook(name), with_kwargs=True))
            elif (name.rsplit(".", 1)[0].endswith((".self_attention", ".mlp")) and leaf.startswith("linear_")) or (
                name.endswith(".to_wrap") and any(part in name for part in (".self_attention.linear_", ".mlp.linear_"))
            ):
                handles.append(module.register_forward_hook(self.module_hook(name, inputs=True), with_kwargs=True))
            elif leaf.endswith(("layernorm", "hyper_connection", "o_norm")):
                handles.append(module.register_forward_hook(self.module_hook(name, inputs=True), with_kwargs=True))
            if hasattr(module, "_proj_rms_op"):
                original = module._proj_rms_op

                def projection(*args, fn=original, label=name, **kwargs):
                    projected, reciprocal = fn(*args, **kwargs)
                    self.capture(label + ".projection", projected)
                    self.capture(label + ".rms_reciprocal", reciprocal)
                    return projected, reciprocal

                originals.append((module, "_proj_rms_op", original))
                module._proj_rms_op = projection
            if hasattr(module, "_kpool_select"):
                original = module._kpool_select

                def select(*args, fn=original, label=name, **kwargs):
                    selected = fn(*args, **kwargs)
                    logical = self.device_positions[selected.clamp_min(0).long()]
                    logical = torch.where(selected >= 0, logical, -1)
                    self.capture(label + ".selected_key_order", logical)
                    self.capture(label + ".selected_key_set", logical.sort(dim=-1).values)
                    return selected

                originals.append((module, "_kpool_select", original))
                module._kpool_select = select
        return handles, originals

    def report(self):
        changed = [(name, row) for name, row in self.summaries.items() if not row["bitwise_equal"]]
        low = [(name, row) for name, row in changed if row["actual_dtype"] in ("torch.bfloat16", "torch.float16")]
        return {
            "first_difference": changed[0] if changed else None,
            "first_low_precision_difference": low[0] if low else None,
            "modules": self.summaries,
        }


def _data(rollouts, index, completion_cap):
    data = math_microbatches(
        rollouts,
        dp_rank=parallel_state.get_data_parallel_rank(),
        dp_size=parallel_state.get_data_parallel_world_size(),
    )[index]
    if completion_cap is None:
        return data
    prompts = {row["index"]: row for row in rollouts["prompts"]}
    groups = {index: [] for index in prompts}
    for sample in rollouts["samples"]:
        groups[sample["prompt_index"]].append(sample | {"completion_ids": sample["completion_ids"][:completion_cap]})
    slots = {keys[0][0]: slot for keys, slot in zip(data.computation_keys, data.slots, strict=True)}
    return _make_batch(prompts, groups, [(i, slots[i]) for i in data.prompt_indices], sp_multiple=2, pad_id=0)


def _load_adapters(model, directory, *, allow_deeper_layers=False):
    state = torch.load(
        directory / f"state-rank-{dist.get_rank()}.pt", map_location="cpu", weights_only=True, mmap=True
    )
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    saved = state["parameters"]
    extras = saved.keys() - trainable.keys()
    permitted = allow_deeper_layers and all(
        name.startswith("decoder.layers.") and int(name.split(".")[2]) >= len(model.decoder.layers) for name in extras
    )
    if not trainable.keys() <= saved.keys() or (extras and not permitted):
        raise ValueError("Saved adapter inventory differs from the trace model")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state["parameters"][name])
    steps = state["steps"]
    del state
    gc.collect()
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--adapter-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch-index", type=int, default=0)
    parser.add_argument("--completion-cap", type=int)
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--trace-layers", default="0", help="Comma-separated zero-based layers")
    args = parser.parse_args()
    if args.completion_cap is not None and args.completion_cap < 1:
        parser.error("Completion cap must be positive")
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
    if rollouts["config_sha256"] != hashlib.sha256(Path(args.model, "config.json").read_bytes()).hexdigest():
        raise ValueError("Teacher/trainer checkpoint config differs")
    model = make_model(
        args.model,
        lora=True,
        full_width=True,
        pretrained=True,
        pretrained_layers=45,
        sequence_length=16384,
        recompute=True,
    )
    saved_steps = _load_adapters(model, args.adapter_state)
    data = _data(rollouts, args.microbatch_index, args.completion_cap)
    expanded, compact = _prepared(data, shared=False), _prepared(data, shared=True)
    trace = ForwardTrace(data, expanded[0], compact[0], map(int, args.trace_layers.split(",")))
    policy = _RowMath(model, "fp32")
    report = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "saved_optimizer_steps": saved_steps,
        "phases": [],
        "scope": "Full-width forward localization and optional backward checks, not a performance measurement",
    }
    cases = [
        ("native_expanded", False, False, False, True),
        ("native_expanded_repeat", False, False, False, False),
        ("native_tree", False, True, False, False),
        ("stable_expanded_vs_native", True, False, False, True),
        ("stable_expanded_repeat", True, False, False, False),
        ("stable_tree", True, True, False, False),
        ("strict_expanded_vs_stable", True, False, True, True),
        ("strict_expanded_repeat", True, False, True, False),
        ("strict_tree", True, True, True, False),
    ]
    for name, stable, shared, strict, replace in cases:
        policy.activate(stable)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not strict
        plan, batch = compact if shared else expanded
        trace.begin(name, plan, replace)
        handles, originals = trace.install(model)
        _emit("FULL_FORWARD_BEGIN", {"phase": name})
        try:
            with torch.no_grad():
                loss = batch.loss(model) if shared else _expanded_loss(model, plan, batch)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
            for module, attr, original in originals:
                setattr(module, attr, original)
        row = {
            "name": name,
            "local_loss": float(loss),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            **trace.report(),
        }
        report["phases"].append(row)
        if dist.get_rank() == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "trace.json").write_text(json.dumps(report, indent=2))
        _emit("FULL_FORWARD_END", {k: v for k, v in row.items() if k != "modules"})
        dist.barrier()
    if args.gradient_check:
        trace.references.clear()
        gc.collect()
        optimizer = MasterOptimizer(model, 1e-5)
        supervised = _sum_dp(data.supervised_tokens)
        gradients = []
        for shared in (False, False, True, True):
            _emit("FULL_GRADIENT_BEGIN", {"shared": shared, "pass": len(gradients)})
            stats = _phase(
                model,
                optimizer,
                (compact if shared else expanded,),
                (data,),
                shared=shared,
                supervised_global=supervised,
            )
            snapshot = optimizer.snapshot()
            row = {"shared": shared, "stats": stats}
            if gradients:
                row["expanded_reference_gradient_relative_l2_max"] = _maximum(_gradient_error(gradients[0], snapshot))
            if shared and len(gradients) == 3:
                row["tree_repeat_gradient_relative_l2_max"] = _maximum(_gradient_error(gradients[2], snapshot))
            report.setdefault("strict_gradient_checks", []).append(row)
            gradients.append(snapshot)
            _emit("FULL_GRADIENT_END", row)
            if dist.get_rank() == 0:
                (args.output_dir / "trace.json").write_text(json.dumps(report, indent=2))
        del gradients, optimizer
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
