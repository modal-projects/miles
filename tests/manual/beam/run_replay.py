"""Verify real SGLang routes through Miles packing and Core recomputation."""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from run_model import _build


def _init_parallel(args):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
    from miles.utils.ft_utils.process_group_utils import GroupInfo

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
    )
    model_parallel_cuda_manual_seed(1234)
    process_groups = {
        "intra_dp": parallel_state.get_data_parallel_group(),
        "intra_dp_cp": parallel_state.get_data_parallel_group(with_context_parallel=True),
        "cp": parallel_state.get_context_parallel_group(),
        "tp": parallel_state.get_tensor_model_parallel_group(),
        "pp": parallel_state.get_pipeline_model_parallel_group(),
        "ep": parallel_state.get_expert_model_parallel_group(),
        "etp": parallel_state.get_expert_tensor_parallel_group(),
    }
    groups = {
        name: GroupInfo(rank=dist.get_rank(group), size=dist.get_world_size(group), group=group)
        for name, group in process_groups.items()
    }
    set_parallel_state(ParallelState(**groups, indep_dp=GroupInfo(rank=0, size=1, group=None)))


def _router_oracle():
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    from miles.utils.replay_base import routing_replay_manager as manager

    generator = torch.Generator(device="cuda").manual_seed(911)
    logits = torch.randn(7, 16, generator=generator, device="cuda", requires_grad=True)
    forced = torch.arange(8, 16, device="cuda").expand(7, -1).clone()
    bias = torch.cat((torch.full((8,), 10.0), torch.full((8,), -10.0))).cuda()
    kwargs = dict(
        topk=8,
        num_groups=8,
        group_topk=4,
        scaling_factor=2.5,
        score_function="sigmoid",
        expert_bias=bias,
        fused=False,
        dense_output=True,
    )
    manager.enabled = False
    _, natural = topk_routing_with_score_function(logits, **kwargs)
    assert not torch.equal(natural.sort(-1).values, forced)
    manager.enabled = True
    stream = manager.create_replay()
    manager.set_current(stream)
    stream.record(forced)
    manager.stage = "replay_forward"
    probs, selected = topk_routing_with_score_function(logits, **kwargs)
    scores = logits.sigmoid().gather(1, forced)
    expected = scores / scores.sum(-1, keepdim=True) * 2.5
    weights = torch.randn(probs.shape, generator=generator, device="cuda")
    actual_grad = torch.autograd.grad((probs * weights).sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(probs, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)
    assert torch.equal(selected, forced)
    result = dict(
        forced_ids_equal=True,
        natural_ids_differ=True,
        probability_max_abs=(probs - expected).abs().max().item(),
        gradient_max_abs=(actual_grad - expected_grad).abs().max().item(),
    )
    manager.clear_all()
    manager.replays.clear()
    manager.current = None
    return result


def _load_payload(args, model, manager):
    from miles.backends.megatron_utils.replay_utils import register_replay_list_moe
    from miles.backends.training_utils.data.rollout import DataIterator
    from miles.backends.training_utils.parallel import get_parallel_state
    from miles.backends.training_utils.replay.data import fill_replay_data

    sample = json.loads(args.sample.read_text())
    token_ids = sample["prompt_ids"] + sample["response_ids"]
    routes = torch.load(args.routes, weights_only=True).cuda()
    assert routes.shape[0] == len(token_ids) - 1
    padded_length = ((len(token_ids) + args.tp - 1) // args.tp) * args.tp
    data = dict(tokens=[torch.tensor(token_ids, device="cuda")], routed_experts=[routes], max_seq_lens=[padded_length])
    fill_replay_data(
        args=SimpleNamespace(
            qkv_format="bshd", allgather_cp=False, sequence_parallel=args.sp, data_pad_size_multiplier=1
        ),
        models=[SimpleNamespace(module=model)],
        data_iterator=[DataIterator(data, micro_batch_size=1)],
        num_microbatches=[1],
        rollout_data=data,
        data_key="routed_experts",
        replay_list=manager.replays,
        register_replay_list_func=register_replay_list_moe,
    )
    expected = torch.full((padded_length, routes.shape[1], routes.shape[2]), -1, dtype=torch.long)
    expected[: routes.shape[0]] = routes.cpu()
    if args.sp:
        expected = expected.chunk(args.tp, dim=0)[get_parallel_state().tp.rank]
    ids = torch.zeros((1, padded_length), dtype=torch.long, device="cuda")
    ids[0, : len(token_ids)] = torch.tensor(token_ids, device="cuda")
    positions = torch.arange(padded_length, device="cuda").unsqueeze(0)
    return ids, positions, sample, expected


def _check_router(module, inputs, output, *, expected, calls):
    ids = expected.to(output[1].device)
    valid = (ids != -1).all(-1)
    expected_mask = torch.zeros_like(output[1])
    expected_mask[valid] = expected_mask[valid].scatter(1, ids[valid], True)
    assert torch.equal(output[1][valid], expected_mask[valid]), "Core did not use the rollout expert IDs"
    calls.append(int(valid.sum()))


def _model_run(args, recompute):
    from functools import partial

    from miles.utils.replay_base import routing_replay_manager as manager

    manager.replays.clear()
    manager.current = None
    manager.stage = "fallthrough"
    manager.enabled = True
    args.recompute = recompute
    _, model = _build(args)
    model.train()
    ids, positions, sample, expected = _load_payload(args, model, manager)
    calls = {}
    for layer in model.decoder.layers:
        if not hasattr(layer.mlp, "router"):
            continue
        layer_id = layer.layer_number - 1
        router = layer.mlp.router
        assert torch.equal(router.routing_replay.top_indices_list[0], expected[:, layer_id])
        calls[layer_id] = []
        router.register_forward_hook(partial(_check_router, expected=expected[:, layer_id], calls=calls[layer_id]))
    manager.stage = "replay_forward"
    logits = model(input_ids=ids, position_ids=positions, attention_mask=None)
    start = len(sample["prompt_ids"]) - 1
    end = start + len(sample["response_ids"])
    response = torch.tensor(sample["response_ids"], device="cuda")
    lp = logits[0, start:end].float().log_softmax(-1).gather(-1, response[:, None]).squeeze(-1)
    loss = -lp.mean()
    manager.stage = "replay_backward"
    loss.backward()
    grads = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(value).all() for value in grads.values())
    assert all(len(value) == (2 if recompute else 1) for value in calls.values()), calls
    assert all(stream.forward_index == 1 for stream in manager.replays)
    assert all(stream.backward_index == int(recompute) for stream in manager.replays)
    serving_lp = torch.tensor([x[0] for x in sample["meta_info"]["output_token_logprobs"]])
    metrics = dict(
        loss=loss.item(),
        router_calls=calls,
        streams=len(manager.replays),
        logprob_mean_abs=(lp.detach().cpu() - serving_lp).abs().mean().item(),
        logprob_max_abs=(lp.detach().cpu() - serving_lp).abs().max().item(),
    )
    result = dict(logits=logits.detach().cpu(), logprobs=lp.detach().cpu(), gradients=grads, metrics=metrics)
    del model, logits, loss, lp
    manager.clear_all()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    args = parser.parse_args()
    args.attention_backend = "unfused"
    _init_parallel(args)
    oracle = _router_oracle()
    plain = _model_run(args, False)
    recomputed = _model_run(args, True)
    assert torch.equal(plain["logprobs"], recomputed["logprobs"])
    assert plain["gradients"].keys() == recomputed["gradients"].keys()
    mismatches = [
        name for name in plain["gradients"] if not torch.equal(plain["gradients"][name], recomputed["gradients"][name])
    ]
    assert not mismatches, mismatches
    report = dict(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        oracle=oracle,
        plain=plain["metrics"],
        recomputed=recomputed["metrics"],
        gradients_equal=True,
        gradient_tensors=len(plain["gradients"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2))
    torch.save(plain, args.output / f"rank{dist.get_rank()}.pt")
    if dist.get_rank() == 0:
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
