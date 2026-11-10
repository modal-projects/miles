"""Compare Beam HF conversion, logits and gradients across pinned source trees."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


def _hf_state(path):
    index = path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    )
    result = {}
    for name in files:
        with safe_open(str(path / name), framework="pt") as source:
            result.update({key: source.get_tensor(key) for key in source.keys()})
    return result


def _configure(config, args):
    from megatron.core.transformer.enums import AttnBackend

    values = dict(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        context_parallel_size=1,
        sequence_parallel=args.sp,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend[args.attention_backend],
        attention_softmax_in_fp32=False,
        parallel_output=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        moe_router_bias_update_rate=0.0,
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
    )
    if args.recompute:
        values.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    for name, value in values.items():
        if not hasattr(config, name):
            raise ValueError(f"Unknown model setting: {name}")
        setattr(config, name, value)


def _build(args):
    from megatron.bridge import AutoBridge
    from megatron.core.process_groups_config import ProcessGroupCollection

    from miles.utils.hf_utils.config import register_hf_config_aliases
    from miles_plugins.megatron_bridge.beam import BeamModelProvider

    register_hf_config_aliases()
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    config = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(config, BeamModelProvider)
    _configure(config, args)
    config.finalize()
    config._pg_collection = groups
    model = config.provide(pre_process=True, post_process=True)
    model = model.cuda()
    bridge.load_hf_weights([model])
    return bridge, model


def _compare_weights(expected, exported):
    missing, extra = sorted(expected.keys() - exported.keys()), sorted(exported.keys() - expected.keys())
    mismatches = {}
    for name in expected.keys() & exported.keys():
        actual, target = exported[name], expected[name]
        # Compare dtypes too: casting first would hide checkpoint precision lost in Megatron.
        if actual.dtype != target.dtype or actual.shape != target.shape or not torch.equal(actual, target):
            mismatches[name] = {
                "expected_dtype": str(target.dtype),
                "actual_dtype": str(actual.dtype),
                "expected_shape": list(target.shape),
                "actual_shape": list(actual.shape),
                "max_abs": (
                    (actual.float() - target.float()).abs().max().item() if actual.shape == target.shape else None
                ),
            }
    return dict(tensors=len(exported), missing=missing, extra=extra, mismatches=mismatches)


def _roundtrip(bridge, model, args):
    return _compare_weights(_hf_state(Path(args.model)), dict(bridge.export_hf_weights([model], cpu=True)))


def _iterator_roundtrip(model, args):
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    iterator_args = argparse.Namespace(
        hf_checkpoint=args.model,
        vocab_size=model.vocab_size,
        sglang_speculative_algorithm=None,
        custom_model_provider_path=None,
        q_lora_rank=None,
        update_weight_buffer_size=256 * 1024,
    )
    iterator = HfWeightIteratorBridge(
        iterator_args,
        [model],
        placement=WeightUpdatePlacement(gather_pp=True),
        model_name="beam",
        quantization_config=None,
    )
    weights = {
        name: weight.detach().cpu().clone()
        for name, weight in named_params_and_buffers(iterator_args, [model], convert_to_global_name=False)
    }
    expected = _hf_state(Path(args.model))
    # Poison a live parameter: export must use the supplied actor backup instead.
    embedding = model.embedding.word_embeddings.weight
    saved_embedding = embedding.detach().clone()
    results = {}
    try:
        with torch.no_grad():
            embedding.zero_()
        for mode in ("all_ranks", "rank0_only"):
            materialize = mode == "all_ranks" or dist.get_rank() == 0
            exported = {}
            buckets = 0
            for bucket in iterator.iter_hf_weights(weights, materialize=materialize):
                buckets += 1
                for name, tensor in bucket:
                    assert tensor.is_cuda and name not in exported, name
                    exported[name] = tensor.detach().cpu().clone()
            if materialize:
                comparison = _compare_weights(expected, exported)
                assert (
                    not comparison["missing"] and not comparison["extra"] and not comparison["mismatches"]
                ), comparison
                results[mode] = comparison | {"buckets": buckets}
            else:
                assert not exported and buckets == 0
                results[mode] = {"tensors": 0, "buckets": 0}
    finally:
        with torch.no_grad():
            embedding.copy_(saved_embedding)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--attention-backend", default="unfused")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--packed-lengths", type=int, nargs="+")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--weight-iterator", action="store_true", help="Also validate Miles weight-sync buckets and CPU backups"
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
    )
    model_parallel_cuda_manual_seed(1234)
    bridge, model = _build(args)
    model.train()
    conversion = _roundtrip(bridge, model, args)
    iterator_conversion = _iterator_roundtrip(model, args) if args.weight_iterator else None
    calls = {}
    for name, module in model.named_modules():
        if name.endswith(("post_self_attn_layernorm", "post_mlp_layernorm")):
            calls[name] = 0

            def count(_module, _inputs, _output, key=name):
                calls[key] += 1

            module.register_forward_hook(count)
            assert hasattr(module, "weight"), f"Missing sandwich norm: {name}"
    packed = None
    if args.packed_lengths:
        from megatron.core.packed_seq_params import PackedSeqParams

        assert args.attention_backend != "unfused", "Packed attention requires flash or fused attention"
        assert all(length > 0 for length in args.packed_lengths)
        args.seq_len = sum(args.packed_lengths)
        cu = torch.tensor([0, *args.packed_lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(args.packed_lengths),
            max_seqlen_kv=max(args.packed_lengths),
        )
    ids = (torch.arange(args.seq_len, device="cuda") * 17 + 3) % model.vocab_size
    ids = ids.unsqueeze(0)
    positions = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    with torch.set_grad_enabled(not args.forward_only):
        logits = model(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)
        labels = ids.roll(-1, dims=1)
        logprobs = logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        loss = -logprobs[:, :-1].mean()
        if not args.forward_only:
            loss.backward()
    gradients = (
        {}
        if args.summary_only
        else {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    )
    gradient_hashes = {}
    if args.summary_only and not args.forward_only:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                tensor = parameter.grad.detach().cpu().contiguous()
                gradient_hashes[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                }
    result = dict(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        args=vars(args) | {"output": str(args.output)},
        loss=loss.item(),
        roundtrip=conversion,
        weight_iterator=iterator_conversion,
        norm_calls=calls,
        gradient_tensors=sum(p.grad is not None for p in model.parameters()),
        gradients_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(result, indent=2))
    torch.save(
        dict(
            logits=logits.detach().cpu(),
            logprobs=logprobs.detach().cpu(),
            gradients=gradients,
            gradient_hashes=gradient_hashes,
        ),
        args.output / f"rank{dist.get_rank()}.pt",
    )
    assert not conversion["missing"] and not conversion["extra"] and not conversion["mismatches"], conversion
    assert result["gradients_finite"]
    assert len(calls) == 2 * model.config.num_layers and all(
        value == (2 if args.recompute else 1) for value in calls.values()
    ), calls
    if dist.get_rank() == 0:
        print(json.dumps(result, default=str), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
