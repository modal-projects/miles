"""Full pretrained hybrid-model LoRA parity and warm F/B throughput.

Standalone Transformers backend; these numbers are not distributed Miles API
throughput. Uses recorded QASPER human-answer SFT and native FLA/FA kernels.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from flash_attn import flash_attn_varlen_func
from hybrid_sft_trace import HybridTrace
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from miles.backends.training_utils.prefix_tree.data import sft_microbatches
from miles.backends.training_utils.prefix_tree.gated_delta import enable_gated_delta_prefix_tree
from miles.backends.training_utils.prefix_tree.operators import rowwise_linear
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree


def _save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


@contextmanager
def _linear_policy(model, fixed, *, scope="all"):
    originals = []
    if fixed:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and (scope == "all" or module.weight.requires_grad):
                originals.append((module, module.forward))

                def forward(value, owner=module):
                    output = rowwise_linear(value.reshape(-1, value.shape[-1]), owner.weight)
                    if owner.bias is not None:
                        output = output + owner.bias
                    return output.reshape(*value.shape[:-1], owner.weight.shape[0])

                module.forward = forward
    try:
        yield
    finally:
        for module, forward in originals:
            module.forward = forward


def _load_model(checkpoint, *, layers=None):
    config = AutoConfig.from_pretrained(checkpoint).get_text_config()
    if layers is not None:
        config.num_hidden_layers = layers
        config.layer_types = config.layer_types[:layers]
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="eager",
        key_mapping={r"^model.language_model\.": "model."},
        output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete pretrained text weights: {info}")
    enable_transformers_prefix_tree(model, recurrent_adapter=enable_gated_delta_prefix_tree)
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"),
        autocast_adapter_dtype=False,
    )
    generator = torch.Generator(device="cuda").manual_seed(923)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.normal_(0, 0.001 if "lora_B" in name else 0.02, generator=generator)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model.train(), info


def _batch(data, shared, *, convolution_kernel_size):
    plan = build_prefix_tree(data.sequences, share_prefixes=shared)
    batch = TreeSFTBatch.from_plan(
        plan,
        data.targets,
        weights=data.weights,
        device="cuda",
        deterministic_reduction=True,
        recurrent_chunk_size=64,
        convolution_kernel_size=convolution_kernel_size,
    )
    return batch


def _loss(model, batch, *, shared, observer=None):
    # Apply the frozen vocabulary head only at supervised occurrences on both
    # paths. This preserves the common occurrence loss while avoiding huge
    # unsupervised vocabulary tensors; report this optimization explicitly.
    output = model(
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=batch.supervision.rows,
        prefix_tree_layout=batch.attention,
        prefix_tree_varlen=flash_attn_varlen_func,
        prefix_tree_shared=shared,
        prefix_tree_recurrent=batch.recurrent,
        seq_idx=batch.sequence_ids,
        cu_seq_lens_q=batch.attention.cu_q,
        prefix_tree_observer=observer,
    )
    selected = replace(
        batch.supervision, rows=torch.arange(len(batch.supervision.rows), device="cuda"), reduction=None
    )
    return selected.loss(output.logits)


def _phase(model, batch, *, shared, gradients=False):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = _loss(model, batch, shared=shared)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite SFT loss")
    loss.backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise ValueError(f"Missing trainable gradients: {missing}")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    stats = dict(
        loss=float(loss.detach()),
        seconds=seconds,
        original_input_tokens=batch.attention.plan.expanded_tokens,
        physical_input_tokens=len(batch.attention.plan.tokens),
        original_input_tps=batch.attention.plan.expanded_tokens / seconds,
        answer_tps=len(batch.supervision.rows) / seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    snapshot = (
        {
            name: p.grad.detach().float().cpu()
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        if gradients
        else None
    )
    return stats, snapshot


def _difference(reference, actual):
    if reference.keys() != actual.keys():
        raise ValueError("Missing trainable parameter gradients")
    numerator, denominator, per_parameter = 0.0, 0.0, []
    for name, expected in reference.items():
        value = actual[name]
        error = float((value.double() - expected.double()).square().sum())
        norm = float(expected.double().square().sum())
        numerator += error
        denominator += norm
        per_parameter.append((name, (error / max(norm, 1e-30)) ** 0.5))
    return dict(
        relative_l2=(numerator / max(denominator, 1e-30)) ** 0.5,
        largest_parameters=sorted(per_parameter, key=lambda row: row[1], reverse=True)[:10],
    )


def _policy_run(model, expanded, compact, *, pairs, output, report, policy):
    current = {"parity": [], "timings": [], "forward": []}
    report["policies"][policy] = current
    model.zero_grad(set_to_none=True)
    trace = HybridTrace(model, output)
    try:
        for index, (name, batch, shared) in enumerate(
            [("expanded", expanded, False), ("tree_without_sharing", expanded, True), ("compact", compact, True)]
        ):
            trace.begin(policy + "-" + name, batch, replace=index == 0)
            print("TRACE_BEGIN", policy, name, flush=True)
            with torch.no_grad():
                loss = _loss(model, batch, shared=shared, observer=trace.capture)
            current["forward"].append(dict(execution=name, loss=float(loss), **trace.report()))
            _save(output, report)
            print("TRACE_END", policy, name, current["forward"][-1]["first_difference"], flush=True)
    finally:
        trace.close()
    del trace
    gc.collect()
    reference = None
    arms = [
        ("expanded", expanded, False),
        ("expanded_repeat", expanded, False),
        ("tree_without_sharing", expanded, True),
        ("compact", compact, True),
    ]
    for name, batch, shared in arms:
        print("PARITY_BEGIN", policy, name, flush=True)
        stats, gradients = _phase(model, batch, shared=shared, gradients=True)
        if reference is None:
            reference = gradients
        else:
            stats["gradients"] = _difference(reference, gradients)
        del gradients
        current["parity"].append({"execution": name, **stats})
        _save(output, report)
        print("PARITY_END", json.dumps(current["parity"][-1]), flush=True)
    del reference
    gc.collect()
    # Warm both shapes before interleaved measurements; never time planning,
    # gradient snapshots, optimizer/checkpoint I/O, or activation tracing.
    for batch, shared in [(expanded, False), (compact, True)]:
        _phase(model, batch, shared=shared)
    for index in range(pairs):
        order = [("expanded", expanded, False), ("compact", compact, True)]
        if index % 2:
            order.reverse()
        pair = {}
        for name, batch, shared in order:
            pair[name], _ = _phase(model, batch, shared=shared)
        current["timings"].append(pair)
        _save(output, report)
        print("TIMING", policy, json.dumps(pair), flush=True)
    current["median_seconds"] = {
        name: statistics.median(pair[name]["seconds"] for pair in current["timings"])
        for name in ("expanded", "compact")
    }
    current["speedup"] = current["median_seconds"]["expanded"] / current["median_seconds"]["compact"]
    _save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--linear-scope", choices=["all", "trainable"], default="all")
    parser.add_argument(
        "--policies", nargs="+", choices=["native_linear", "fixed_linear"], default=["native_linear", "fixed_linear"]
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    data_json = args.rollouts.read_text()
    data = sft_microbatches(json.loads(data_json), dp_rank=0, dp_size=1, sp_multiple=1)[0]
    convolution_kernel_size = AutoConfig.from_pretrained(args.model).get_text_config().linear_conv_kernel_dim
    expanded = _batch(data, False, convolution_kernel_size=convolution_kernel_size)
    compact = _batch(data, True, convolution_kernel_size=convolution_kernel_size)
    report = dict(
        status="loading",
        backend="standalone Transformers + PEFT, one GPU; not Miles API TPS",
        model=args.model,
        gpu=torch.cuda.get_device_name(),
        layers=args.layers,
        versions={name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "flash-attn"]},
        recorded_data_sha256=hashlib.sha256(data_json.encode()).hexdigest(),
        input_rows=dict(expanded=len(expanded.attention.plan.tokens), compact=len(compact.attention.plan.tokens)),
        supervised_tokens=len(expanded.supervision.rows),
        lora="rank 16, alpha 16, BF16, all PEFT linear targets; seeded nonzero A and B, no optimizer updates",
        vocabulary_projection="only supervised occurrences, same optimization in both paths",
        linear_scope=args.linear_scope,
        policies={},
    )
    _save(args.output, report)
    model, info = _load_model(args.model, layers=args.layers)
    report.update(
        status="running",
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        pretrained_missing_keys=sorted(info["missing_keys"]),
        full_text_layers=model.config.num_hidden_layers,
    )
    _save(args.output, report)
    for policy in args.policies:
        fixed = policy == "fixed_linear"
        with _linear_policy(model, fixed, scope=args.linear_scope):
            _policy_run(
                model,
                expanded,
                compact,
                pairs=args.pairs,
                output=args.output,
                report=report,
                policy=(
                    ("fixed_trainable_linear" if args.linear_scope == "trainable" else "fixed_linear")
                    if fixed
                    else "native_linear"
                ),
            )
    report["status"] = "complete"
    _save(args.output, report)


if __name__ == "__main__":
    main()
