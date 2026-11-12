"""Full-depth pretrained GLM LoRA SFT, with paired expanded/tree preflight.

Run with torchrun --nproc-per-node=8 in the GLM trainer image. The default
45-layer run trains two rank-16 slots on synthetic, tokenized code completions.
All prompts are masked; reference execution and training use vocabulary-parallel
CE. There is no sampler or RL controller. Only nonfinite loss/gradients stop the
training loop; numerical comparison metrics are reported without killing it.
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from glm53_sft import _check_replicated_weights, _synchronize_gradients, make_model
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoTokenizer

from miles.backends.training_utils.prefix_tree.glm53.sft import GlmTreeSFTBatch
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


@dataclass(frozen=True)
class Batch:
    sequences: tuple
    targets: tuple
    weights: tuple
    slots: tuple
    computation_keys: tuple


def _emit(tag, data):
    if dist.get_rank() == 0:
        print(tag + " " + json.dumps(data), flush=True)


def _data(tokenizer, args, step):
    dp = parallel_state.get_data_parallel_rank()
    sequences, targets, weights, slots, keys = [], [], [], [], []
    for sample in range(args.branches):
        slot = sample // (args.branches // 2)
        task = ("stable_unique", "insertion_sort")[slot]
        header = tokenizer.encode(
            f"### Instruction\nWrite a Python {task} function with tests. Example group {dp}, step {step}.\n### Reference context\n",
            add_special_tokens=False,
        )
        context = tokenizer.encode(
            "".join(f"def helper_{i}(x):\n    return x + {i}\n\n" for i in range(args.prefix)),
            add_special_tokens=False,
        )
        separator = tokenizer.encode("\n### Response\n", add_special_tokens=False)
        remaining = args.prefix - len(header) - len(separator)
        if remaining < 1:
            raise ValueError("Prompt budget is too short for the instruction")
        prompt = header + context[:remaining] + separator
        if slot == 0:
            code = "def stable_unique(values):\n    seen = set()\n    result = []\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            result.append(value)\n    return result\n"
        else:
            code = "def insertion_sort(values):\n    result = list(values)\n    for i in range(1, len(result)):\n        value = result[i]\n        j = i - 1\n        while j >= 0 and result[j] > value:\n            result[j + 1] = result[j]\n            j -= 1\n        result[j + 1] = value\n    return result\n"
        tests = "\n".join(
            (
                f"assert {task}([{i}, {i}, {i+1}]) == [{i}, {i+1}]"
                if slot == 0
                else f"assert {task}([{i+1}, {i}]) == [{i}, {i+1}]"
            )
            for i in range(sample, sample + 100)
        )
        suffix = tokenizer.encode(
            f"# Reference implementation {sample}\n" + code + "\n" + tests + "\n", add_special_tokens=False
        )
        if len(suffix) < args.suffix + 1:
            raise ValueError("Completion budget exceeds the generated reference")
        full = prompt + suffix[: args.suffix + 1]
        sequences.append(tuple(full[:-1]))
        targets.append(tuple(full[1:]))
        weights.append(tuple([0.0] * (args.prefix - 1) + [1.0] * (args.suffix + 1)))
        slots.append(slot)
        # Conservative sharing: keep response forks explicit even when two
        # completions start with the same token, so SP row counts stay stable.
        keys.append(tuple([None] * args.prefix + [sample] * args.suffix))
    return Batch(tuple(sequences), tuple(targets), tuple(weights), tuple(slots), tuple(keys))


def _prepared(data, *, shared):
    plan = build_prefix_tree(
        data.sequences, sharing_keys=data.slots, computation_keys=data.computation_keys, share_prefixes=shared
    )
    batch = GlmTreeSFTBatch.from_plan(
        plan, data.targets, device="cuda", weights=data.weights, adapter_slots=data.slots, num_adapter_slots=2
    )
    return plan, batch


def _expanded_loss(model, plan, batch):
    return batch.loss(model, shared=False)


def _maximum(value):
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def _phase(model, data, *, shared, capture=False):
    started = time.perf_counter()
    plan, batch = _prepared(data, shared=shared)
    planning = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = batch.loss(model) if shared else _expanded_loss(model, plan, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite SFT loss")
    loss.backward()
    _synchronize_gradients(model)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.nn.utils.get_total_norm(grads, foreach=True, error_if_nonfinite=True)
    torch.cuda.synchronize()
    elapsed = _maximum(time.perf_counter() - started)
    mean_loss = loss.detach().double()
    group = parallel_state.get_data_parallel_group()
    dist.all_reduce(mean_loss, group=group)
    mean_loss /= dist.get_world_size(group)
    stats = {
        "loss": float(mean_loss),
        "forward_backward_s": elapsed,
        "planning_s": _maximum(planning),
        "input_tokens_local": plan.expanded_tokens,
        "compact_tokens_local": len(plan.tokens),
        "supervised_tokens_local": batch.losses.rows.numel(),
        "grad_norm_local": float(norm),
        "peak_allocated_gib": _maximum(torch.cuda.max_memory_allocated() / 1024**3),
    }
    gradients = (
        {name: p.grad.detach().float().cpu() for name, p in model.named_parameters() if p.grad is not None}
        if capture
        else None
    )
    return stats, gradients


def _gradient_error(reference, actual):
    if reference.keys() != actual.keys():
        raise ValueError("Expanded/tree gradient inventory differs")
    numerator = sum(float((reference[n] - actual[n]).square().sum()) for n in reference)
    denominator = sum(float(g.square().sum()) for g in reference.values())
    return (numerator / max(denominator, 1e-30)) ** 0.5


@torch.no_grad()
def _optimizer_comparison(model, optimizer, reference_gradients, lr):
    parameters = dict(model.named_parameters())
    shadows = {name: torch.nn.Parameter(parameters[name].detach().clone()) for name in reference_gradients}
    for name, p in shadows.items():
        p.grad = reference_gradients[name].to(p.device, p.dtype)
    reference_optimizer = torch.optim.AdamW(shadows.values(), lr=lr, fused=True)
    reference_optimizer.step()
    update_norm = sum(float((parameters[name].float() - p.float()).square().sum()) for name, p in shadows.items())
    optimizer.step()
    torch.cuda.synchronize()
    error = sum(float((parameters[name].float() - p.float()).square().sum()) for name, p in shadows.items())
    parameter_norm = sum(float(p.float().square().sum()) for p in shadows.values())
    return {
        "update_relative_l2": (error / max(update_norm, 1e-30)) ** 0.5,
        "parameter_relative_l2": (error / max(parameter_norm, 1e-30)) ** 0.5,
    }


def _ce_preflight():
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    torch.manual_seed(12)
    tp = parallel_state.get_tensor_model_parallel_rank()
    full = torch.randn(9, 32, device="cuda", dtype=torch.float32)
    local = full[:, tp * 16 : (tp + 1) * 16].detach().clone().requires_grad_()
    plan = build_prefix_tree([[1, 2, 3], [1, 2, 4], [9, 10, 11]], share_prefixes=False)
    layout = SFTLossLayout.from_plan(
        plan,
        [[1, 19, -100], [22, 3, 4], [31, 16, 2]],
        device="cuda",
        weights=[[1.0, 0.5, 0.0], [0.25, 1.0, 2.0], [1.0, 0.75, 1.0]],
    )
    reference = layout.loss(gather_from_tensor_model_parallel_region(local))
    reference.backward()
    expected = local.grad.detach().clone()
    local.grad = None
    actual = layout.vocab_parallel_loss(local)
    actual.backward()
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(local.grad, expected, rtol=1e-5, atol=1e-6)
    _emit("VOCAB_PARALLEL_CE_PREFLIGHT", {"loss": float(actual.detach()), "passed": True})


def _save(model, optimizer, args, report):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "parameters": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        "step": args.steps,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 8,
        "lora_rank": 16,
        "adapter_slots": 2,
        "optimizer_saved": args.save_optimizer,
    }
    if args.save_optimizer:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, root / f"adapters-rank-{dist.get_rank()}.pt")
    dist.barrier()
    if dist.get_rank() == 0:
        (root / "report.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--layers", type=int, default=45)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prefix", type=int, default=1025)
    parser.add_argument("--suffix", type=int, default=255)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--recompute", choices=("full", "none"), default="full")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-optimizer", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.branches < 4 or args.branches % 2:
        parser.error("Need positive steps and an even branch count >=4")
    import os

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_tensor_parallel_size=1, expert_model_parallel_size=8
    )
    _ce_preflight()
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    _emit("MODEL_LOADING", {"layers": args.layers, "checkpoint": args.model})
    model = make_model(
        args.model,
        lora=True,
        full_width=True,
        pretrained=True,
        pretrained_layers=args.layers,
        sequence_length=args.prefix + args.suffix,
        recompute=args.recompute == "full",
    )
    _check_replicated_weights(model)
    assert model.config.num_layers == args.layers and len(model.decoder.layers) == args.layers
    _emit(
        "MODEL_READY_FULL_DEPTH",
        {
            "layers": args.layers,
            "local_parameter_elements": sum(p.numel() for p in model.parameters()),
            "kda_layers": len(model.config.kda_layers),
            "dsa_layers": args.layers - len(model.config.kda_layers),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, fused=True)
    report = {"settings": vars(args), "preflight": {}, "steps": []}
    data = _data(tokenizer, args, 0)
    expanded, reference = _phase(model, data, shared=False, capture=True)
    repeated, repeat_grads = _phase(model, data, shared=False, capture=True)
    repeat_error = _maximum(_gradient_error(reference, repeat_grads))
    del repeat_grads
    tree, tree_grads = _phase(model, data, shared=True, capture=True)
    error = _maximum(_gradient_error(reference, tree_grads))
    del tree_grads
    comparison = _optimizer_comparison(model, optimizer, reference, args.lr)
    report["preflight"] = {
        "expanded": expanded,
        "expanded_repeat": repeated,
        "tree": tree,
        "repeat_gradient_relative_l2_max": repeat_error,
        "tree_gradient_relative_l2_max": error,
        "optimizer": comparison,
    }
    _emit("FULL_SFT_PREFLIGHT", report["preflight"])
    del reference
    gc.collect()
    dp = parallel_state.get_data_parallel_world_size()
    report["steps"].append({"step": 1, **tree, "optimizer_applied": True})
    for step in range(1, args.steps):
        data = _data(tokenizer, args, step)
        stats, _ = _phase(model, data, shared=True)
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        stats["optimizer_s"] = _maximum(time.perf_counter() - started)
        stats["step_s"] = stats["forward_backward_s"] + stats["optimizer_s"]
        stats["global_expanded_input_tokens_per_s"] = stats["input_tokens_local"] * dp / stats["step_s"]
        stats["global_supervised_tokens_per_s"] = stats["supervised_tokens_local"] * dp / stats["step_s"]
        report["steps"].append({"step": step + 1, **stats, "optimizer_applied": True})
        _emit("FULL_SFT_STEP", report["steps"][-1])
    _save(model, optimizer, args, report)
    _emit("FULL_SFT_COMPLETE", {"steps": args.steps, "layers": args.layers, "output_dir": args.output_dir})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
