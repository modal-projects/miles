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

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


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


def _bmm_projection(x, weight, eps):
    if weight.requires_grad:
        raise ValueError("The rowwise BMM control requires frozen projection weights")
    # A fixed 1xK by KxN GEMM per token avoids changing the reduction algorithm
    # with the tree's physical row count. expand() is a zero-stride weight view.
    projected = torch.bmm(x.unsqueeze(1), weight.t().unsqueeze(0).expand(len(x), -1, -1)).squeeze(1)
    return projected, (x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1]) + eps).reciprocal()


def _normalization_oracle(model, precision):
    import transformer_engine.pytorch as te

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, te.RMSNorm):
            continue
        names.append(name)
        if precision == "native":
            continue
        dtype = {"fp32": torch.float32, "fp64": torch.float64}[precision]

        def forward(x, norm=module, compute_dtype=dtype):
            # Keep the original parameter objects and SP metadata. This is a
            # diagnostic operator substitution, never an upstream source edit.
            value = x.to(compute_dtype)
            gamma = norm.weight.to(compute_dtype)
            if norm.zero_centered_gamma:
                gamma = gamma + 1
            result = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.eps)
            return (result * gamma).to(x.dtype)

        module.forward = forward
    return names


def _record_routes(model):
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

    RouterReplay.clear_global_router_replay_instances()
    routers = {}
    for name, module in model.named_modules():
        if name.endswith(".router"):
            module.config.moe_router_fusion = False
            module.router_replay = RouterReplay()
            module.router_replay.set_router_replay_action(RouterReplayAction.RECORD)
            routers[name] = module.router_replay
    return routers


def _canonical_routes(routers, sequences):
    slots = [0] * (len(sequences) - 1) + [1]
    plan = build_prefix_tree(sequences, sharing_keys=slots)
    occurrences = [node for path in plan.sample_nodes for node in path]
    first = [-1] * len(plan.tokens)
    for i, node in enumerate(occurrences):
        if first[node] < 0:
            first[node] = i
    first = torch.tensor(first, device="cuda")
    occurrences = torch.tensor(occurrences, device="cuda")
    records, changed = {}, {}
    group = parallel_state.get_tensor_model_parallel_group()
    world = dist.get_world_size(group)
    for name, replay in routers.items():
        value = replay.get_recorded_indices().detach().clone()
        sharded = value.shape[0] != len(occurrences)
        if sharded:
            assert value.shape[0] * world == len(occurrences)
            parts = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(parts, value, group=group)
            value = torch.cat(parts)
        compact = value.index_select(0, first)
        expanded = compact.index_select(0, occurrences)
        changed[name] = int((expanded != value).any(-1).sum())
        records[name] = (expanded, compact, sharded)
    return records, changed


def _replay_routes(routers, records, *, shared):
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    rank = parallel_state.get_tensor_model_parallel_rank()
    world = parallel_state.get_tensor_model_parallel_world_size()
    for name, replay in routers.items():
        expanded, compact, sharded = records[name]
        value = compact if shared else expanded
        if sharded:
            assert len(value) % world == 0
            value = value.chunk(world)[rank]
        replay.set_target_indices(value)
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


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
    parser.add_argument("--mhc-projection-bmm", action="store_true")
    parser.add_argument("--trace-operators", action="store_true")
    parser.add_argument("--norm-oracle", choices=("native", "fp32", "fp64"), default="native")
    parser.add_argument("--fixed-routes", action="store_true")
    parser.add_argument("--trace-linears", action="store_true")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
    os.environ["GLM_TREE_TRACE"] = "1"
    os.environ["GLM_TREE_TRACE_OPERATORS"] = "1" if args.trace_operators else "0"
    os.environ["GLM_TREE_TRACE_LINEARS"] = "1" if args.trace_linears else "0"
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
            norm_names = _normalization_oracle(model, args.norm_oracle)
            if args.mhc_projection_fp64 or args.mhc_projection_bmm:
                for module in model.modules():
                    if hasattr(module, "_proj_rms_op"):
                        if module._proj_rms_compute_h_op is not None:
                            raise ValueError("The projection oracle requires unfused mHC")
                        module._proj_rms_op = _bmm_projection if args.mhc_projection_bmm else _fp64_projection
            routers = _record_routes(model) if args.fixed_routes else {}
            cold_expanded = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            changed_routes = {}
            if routers:
                routes, changed_routes = _canonical_routes(routers, sequences)
                _replay_routes(routers, routes, shared=False)
            expanded = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            repeated = evaluate(model, sequences, shared=False, lora=True, targets=targets)
            if routers:
                _replay_routes(routers, routes, shared=True)
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
                "mhc_projection_bmm": args.mhc_projection_bmm,
                "normalization_oracle": args.norm_oracle,
                "normalization_modules": norm_names,
                "fixed_routes": args.fixed_routes,
                "canonicalized_reference_route_rows": changed_routes,
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
            if routers:
                from megatron.core.transformer.moe.router_replay import RouterReplay

                RouterReplay.clear_global_router_replay_instances()
                del routers, routes
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
