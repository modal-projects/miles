"""Small random GLM-5.3 provider SFT oracle; run with torchrun on two GPUs.

Uses the real pinned Bridge DSA/KDA/mHC/MoE implementation at reduced model
width/depth. It is not a pretrained full-model convergence or throughput claim.

Run in a Miles trainer image with the pinned GLM provider installed::

    torchrun --standalone --nproc-per-node=2 tests/manual/prefix_tree/glm53_sft.py --model /path/to/GLM --lora
    torchrun --standalone --nproc-per-node=8 tests/manual/prefix_tree/glm53_sft.py --model /path/to/GLM --lora --full-width --pretrained

For warmed reduced-model timings, add --prefix 4098 --suffix 514 --branches 8
--repetitions 4, without GLM_TREE_TRACE=1. The full-width mode retains the real
attention/MLP/expert widths, but uses three random layers and a small vocabulary.
--pretrained instead loads the first five matching checkpoint layers and full
vocabulary. The random full-width case exposes routing-sensitive BF16 differences
that can exceed the checks; --precise-moe is an optional numerical diagnostic.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

import miles_plugins.megatron_bridge  # noqa: F401  # register the isolated provider
from miles.backends.training_utils.prefix_tree.glm53.bridge import glm53_tree_layer_spec
from miles.backends.training_utils.prefix_tree.glm53.sft import GlmTreeSFTBatch
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


def make_model(
    checkpoint,
    *,
    lora=False,
    full_width=False,
    precise_moe=False,
    pretrained=False,
    pretrained_layers=5,
    sequence_length=1024,
    recompute=False,
    reduced_layers=3,
    reduced_topk=2,
    reduced_index_topk=64,
):
    bridge = AutoBridge.from_hf_pretrained(checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    changes = dict(
        num_layers=3,
        hidden_size=512,
        ffn_hidden_size=1024,
        num_attention_heads=8,
        num_query_groups=8,
        num_moe_experts=max(4, reduced_topk * 2),
        moe_ffn_hidden_size=256,
        moe_router_topk=reduced_topk,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_layer_freq=[0, 1, 1],
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=0,
        v_head_dim=128,
        kda_layers=(0, 2),
        kda_num_heads=4,
        kda_head_dim=128,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=reduced_index_topk,
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        vocab_size=512,
        seq_length=sequence_length,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        moe_router_bias_update_rate=0.0,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_permute_fusion=False,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        parallel_output=True,
        transformer_layer_spec=glm53_tree_layer_spec,
    )
    if recompute:
        changes.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    if not full_width and reduced_layers != 3:
        changes.update(
            num_layers=reduced_layers,
            kda_layers=tuple(i for i in range(reduced_layers) if i % 4 != 1),
            moe_layer_freq=[0] + [1] * (reduced_layers - 1),
        )
    if full_width:
        # Preserve real GLM attention/MLP/MoE widths and expert count. The slice
        # remains three layers with a small vocabulary, initialized randomly.
        for name in (
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "num_moe_experts",
            "moe_ffn_hidden_size",
            "moe_router_topk",
            "moe_router_num_groups",
            "moe_router_group_topk",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
            "kda_num_heads",
            "kda_head_dim",
            "dsa_indexer_n_heads",
            "dsa_indexer_head_dim",
            "dsa_indexer_topk",
        ):
            changes.pop(name)
        changes["expert_model_parallel_size"] = 8
    if pretrained:
        if not full_width:
            raise ValueError("Pretrained validation requires --full-width")
        changes.pop("vocab_size")
        if not 1 <= pretrained_layers <= provider.num_layers:
            raise ValueError("Requested pretrained depth is outside the checkpoint")
        changes["num_layers"] = pretrained_layers
        changes["kda_layers"] = tuple(i for i in provider.kda_layers if i < pretrained_layers)
        changes["moe_layer_freq"] = provider.moe_layer_freq[:pretrained_layers]
        changes["perform_initialization"] = False
    if precise_moe:
        for field in ("moe_router_use_torch_mm", "moe_activation_in_fp32", "moe_combine_in_fp32"):
            if not hasattr(provider, field):
                raise ValueError(f"Pinned Megatron runtime lacks {field}")
            changes[field] = True
    for key, value in changes.items():
        if not hasattr(provider, key):
            print("ADDITIONAL_PROVIDER_FIELD", key, flush=True)
        setattr(provider, key, value)
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    provider.finalize()
    model = provider.provide().cuda().train()
    if pretrained:
        print("PRETRAINED_LOAD_BEGIN", flush=True)
        bridge._model_bridge.hf_pretrained = bridge.hf_pretrained
        bridge._model_bridge.load_weights_hf_to_megatron(bridge.hf_pretrained, [model])
        print("PRETRAINED_LOAD_COMPLETE", flush=True)
    print("MODEL_READY", sum(p.numel() for p in model.parameters()), flush=True)
    return _apply_multilora(model) if lora else model


def _apply_multilora(model):
    from megatron.bridge.peft.multi_lora import MultiLoRA
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot

    targets = [
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_b",
        "linear_f_a",
        "linear_f_b",
        "linear_g_a",
        "linear_g_b",
        "linear_proj",
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_fc1",
        "linear_fc2",
        "output_layer",
    ]
    # The pinned absorbed-MLA provider does not fold multiple KV-up adapter
    # weights. Keep that frozen, matching the supported native target set.
    model = MultiLoRA(target_modules=targets, n_adapters=2, dim=16, alpha=16)(model)
    for slot in (0, 1):
        init_adapter_slot(model, slot, 16, 16, seed=42 + slot)
    for module in model.modules():
        if hasattr(module, "adapters"):
            for adapter in module.adapters:
                torch.nn.init.normal_(adapter.linear_out.weight, std=0.002)
    print("NATIVE_MULTILORA_READY", flush=True)
    return model


def _check_replicated_weights(model):
    # Random provider construction skips checkpoint loading and DDP's replica
    # broadcasts. Initialize both DP replicas and unsharded TP replicas.
    mismatches = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            dense = getattr(p, "allreduce", True)
            groups = [
                parallel_state.get_data_parallel_group() if dense else parallel_state.get_expert_data_parallel_group()
            ]
            if dense and not getattr(p, "tensor_model_parallel", False):
                groups.append(parallel_state.get_tensor_model_parallel_group())
            for group in groups:
                if dist.get_world_size(group) == 1:
                    continue
                reference = p.detach().clone()
                dist.broadcast(reference, src=dist.get_global_rank(group, 0), group=group)
                if not torch.equal(p, reference):
                    mismatches.append(name)
                    p.copy_(reference)
    print("INITIAL_REPLICA_MISMATCHES " + json.dumps({"rank": dist.get_rank(), "parameters": mismatches}), flush=True)


def _synchronize_gradients(model):
    # This standalone oracle does not wrap DDP or run finalize_model_grads.
    # Match those replica reductions before comparing gradients or taking AdamW.
    tp_group = parallel_state.get_tensor_model_parallel_group()
    for p in model.parameters():
        if p.grad is None:
            continue
        if getattr(p, "sequence_parallel", False):
            dist.all_reduce(p.grad, group=tp_group)
        if getattr(p, "allreduce", True):
            group = parallel_state.get_data_parallel_group()
        else:
            group = parallel_state.get_expert_data_parallel_group()
        if dist.get_world_size(group) > 1:
            dist.all_reduce(p.grad, group=group)
        # Megatron normalizes both dense and expert reductions by dense DP,
        # not expert DP (which can additionally contain former TP ranks).
        p.grad.div_(parallel_state.get_data_parallel_world_size())


def evaluate(model, sequences, *, shared, lora=False, targets=None):
    slots = [0] * (len(sequences) - 1) + [1] if lora else None
    plan = build_prefix_tree(sequences, share_prefixes=shared, sharing_keys=slots)
    if lora:
        from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

        counts = [
            len({row for slot, path in zip(slots, plan.sample_nodes, strict=True) if slot == s for row in path})
            for s in (0, 1)
        ]
        set_tokens_per_adapter_slot(model, torch.tensor(counts, device="cuda", dtype=torch.int64))
    cu = torch.tensor(plan.segment_offsets, device="cuda", dtype=torch.int32)
    maximum = max(b - a for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True))
    packed = PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, qkv_format="thd", max_seqlen_q=maximum, max_seqlen_kv=maximum
    )
    tokens = torch.tensor(plan.tokens, device="cuda")[None]
    positions = torch.tensor(plan.positions, device="cuda")[None]
    if targets is None:
        targets = [[(token + 1) % 512 for token in sequence] for sequence in sequences]
    loss_layout = SFTLossLayout.from_plan(plan, targets, device="cuda")
    batch = (
        GlmTreeSFTBatch.from_plan(
            plan, targets, device="cuda", adapter_slots=slots, num_adapter_slots=2 if lora else None
        )
        if shared
        else None
    )
    model.zero_grad(set_to_none=True)
    traces = {}
    handles = []
    operators = []

    def capture(name):
        def hook(module, inputs, kwargs, output):
            values = (
                ("input", inputs[0] if inputs else kwargs.get("hidden_states")),
                ("output", output[0] if isinstance(output, tuple) else output),
            )
            if isinstance(output, tuple) and os.environ.get("GLM_TREE_TRACE_OPERATORS") == "1":
                values += tuple((f"output{i}", value) for i, value in enumerate(output[1:], 1))
            if name.endswith(".router") and isinstance(output, tuple):
                values += (("routing", output[1]),)
            for kind, value in values:
                if not isinstance(value, torch.Tensor) or value.ndim < 2:
                    continue
                if value.shape[0] * 2 == len(plan.tokens):
                    parts = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(parts, value.contiguous(), group=parallel_state.get_tensor_model_parallel_group())
                    value = torch.cat(parts, dim=0)
                if value.shape[0] == len(plan.tokens):
                    traces[name + "." + kind] = value.detach().float().reshape(len(plan.tokens), -1).cpu()

        return hook

    if os.environ.get("GLM_TREE_TRACE") == "1":
        for name, module in model.named_modules():
            if name.endswith(("self_attention", ".mlp", ".router")) or name == "embedding":
                handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
            if os.environ.get("GLM_TREE_TRACE_LINEARS") == "1" and any(
                name.startswith(f"decoder.layers.{i}.self_attention.linear_") for i in range(4)
            ) and (name.rsplit(".", 1)[-1].startswith("linear_") or name.endswith(".to_wrap")):
                handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
            if os.environ.get("GLM_TREE_TRACE_OPERATORS") == "1" and name.endswith(
                ("input_layernorm", "pre_mlp_layernorm", "q_layernorm", "kv_layernorm")
            ):
                handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
            if os.environ.get("GLM_TREE_TRACE_OPERATORS") == "1" and name.endswith("hyper_connection"):
                handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
                if hasattr(module, "_proj_rms_op"):
                    original = module._proj_rms_op
                    projection_hook = capture(name + ".projection")

                    def projection(*args, fn=original, hook=projection_hook, **kwargs):
                        output = fn(*args, **kwargs)
                        hook(None, args, kwargs, output)
                        return output

                    operators.append((module, original))
                    module._proj_rms_op = projection
    print("SFT_PHASE", "tree" if shared else "expanded", "begin", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    if shared:
        logits = batch.logits(model)
    else:
        logits = model(tokens, position_ids=positions, attention_mask=None, packed_seq_params=packed)
        if model.parallel_output:
            logits = gather_from_tensor_model_parallel_region(logits)
    for handle in handles:
        handle.remove()
    for module, original in operators:
        module._proj_rms_op = original
    loss = loss_layout.loss(logits)
    loss.backward()
    _synchronize_gradients(model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters() if p.grad is not None}
    rows = torch.tensor([node for path in plan.sample_nodes for node in path], device="cuda")
    return (
        loss.item(),
        logits.reshape(-1, logits.shape[-1]).index_select(0, rows).detach().float().cpu(),
        gradients,
        elapsed,
        {name: value[[node for path in plan.sample_nodes for node in path]] for name, value in traces.items()},
    )


def rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/assets/model")
    parser.add_argument("--full-width", action="store_true")
    parser.add_argument("--precise-moe", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--prefix", type=int, default=129)
    parser.add_argument("--suffix", type=int, default=17)
    parser.add_argument("--branches", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8 if args.full_width else 1,
    )
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    lora = args.lora or os.environ.get("GLM_TREE_LORA") == "1"
    args.branches = args.branches or (4 if lora else 3)
    model = make_model(
        args.model, lora=lora, full_width=args.full_width, precise_moe=args.precise_moe, pretrained=args.pretrained
    )
    _check_replicated_weights(model)
    sequences = [
        [1 + i % 128 for i in range(args.prefix)] + [200 + b] + [201 + (b + i) % 100 for i in range(args.suffix - 1)]
        for b in range(args.branches)
    ]
    expanded = evaluate(model, sequences, shared=False, lora=lora)
    if args.pretrained:
        repeated = evaluate(model, sequences, shared=False, lora=lora)
        print("EXPANDED_REPEAT_LOGITS_RELATIVE_L2", rel(expanded[1], repeated[1]), flush=True)
    tree = evaluate(model, sequences, shared=True, lora=lora)
    for name, ref in expanded[4].items():
        value = tree[4][name]
        if name.endswith(".routing") and rank == 0:
            print(
                "ROUTING_TRACE "
                + json.dumps({"name": name, "changed_rows": int((ref != value).any(-1).sum()), "rows": ref.shape[0]}),
                flush=True,
            )
        print(
            "FORWARD_TRACE",
            json.dumps(
                {
                    "name": name,
                    "relative_l2": rel(ref, value),
                    "prefix_repeat_error": rel(
                        ref[: args.prefix], ref[len(sequences[0]) : len(sequences[0]) + args.prefix]
                    ),
                }
            ),
            flush=True,
        )
    assert expanded[2].keys() == tree[2].keys()
    squared = sum(float((expanded[2][n] - tree[2][n]).square().sum()) for n in expanded[2])
    baseline = sum(float(g.square().sum()) for g in expanded[2].values())
    report = {
        "rank": rank,
        "expanded_loss": expanded[0],
        "tree_loss": tree[0],
        "logits_relative_l2": rel(expanded[1], tree[1]),
        "gradient_relative_l2": (squared / baseline) ** 0.5,
        "expanded_seconds": expanded[3],
        "tree_seconds": tree[3],
        "gradient_tensors": len(expanded[2]),
    }
    print("GLM_TREE_SFT", json.dumps(report), flush=True)
    assert report["logits_relative_l2"] < 0.02, report
    assert report["gradient_relative_l2"] < 0.02, report
    assert all(torch.isfinite(g).all() for g in tree[2].values())
    parameters = dict(model.named_parameters())
    shadows = {n: torch.nn.Parameter(parameters[n].detach().clone()) for n in expanded[2]}
    initial = {n: p.detach().float().clone() for n, p in shadows.items()}
    for n, p in shadows.items():
        p.grad = expanded[2][n].to(device=p.device, dtype=p.dtype)
    reference_optimizer = torch.optim.AdamW(shadows.values(), lr=1e-4)
    reference_optimizer.step()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    update_error = sum(float((parameters[n].float() - p.float()).square().sum()) for n, p in shadows.items())
    update_norm = sum(float((p.float() - initial[n]).square().sum()) for n, p in shadows.items())
    print("OPTIMIZER_UPDATE_RELATIVE_L2", (update_error / max(update_norm, 1e-30)) ** 0.5, flush=True)
    parameter_norm = sum(float(p.float().square().sum()) for p in shadows.values())
    parameter_error = (update_error / max(parameter_norm, 1e-30)) ** 0.5
    print("OPTIMIZER_PARAMETER_RELATIVE_L2", parameter_error, flush=True)
    assert parameter_error < 0.002, parameter_error
    print("OPTIMIZER_STEP_PASSED", rank, flush=True)
    for repetition in range(args.repetitions - 1):
        a = evaluate(model, sequences, shared=False, lora=lora)
        b = evaluate(model, sequences, shared=True, lora=lora)
        print(
            "WARM_SFT_TIMING",
            json.dumps(
                {
                    "rank": rank,
                    "repetition": repetition,
                    "expanded_seconds": a[3],
                    "tree_seconds": b[3],
                    "speedup": a[3] / b[3],
                    "expanded_input_tokens": sum(map(len, sequences)),
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
