"""Check Beam LoRA training against Reflection SGLang's Beam LoRA serving tests.

The oracle is SGLang's (sglang.test.beam_lora_fixture / test_beam_lora.py):
eager_beam_logits on merge_beam_lora(state, adapter), with its logprob and KL bounds.

    torchrun --nproc-per-node 2 tests/manual/beam/run_lora.py train --model DIR --output DIR --tp 2 --ep 2 --sp
    python tests/manual/beam/run_lora.py serve --model DIR --output DIR

train builds the full-model and LoRA Bridge models from scripts/run_beam.py's CLI and checks:
  1. with the initial B=0 the LoRA model's logits equal the full model's bitwise;
  2. for seeded adapters per SGLang target group, Megatron's logprobs match the oracle and move off the base;
  3. the weight-sync export (HfWeightIteratorBridge) returns exactly the seeded HF factors.
serve loads the exported adapters into an SGLang engine: base requests must meet the oracle's bounds, and
every adapter must move the output onto the oracle (KL to the oracle < 0.1 x KL to the base). With
--sglang-adapters it runs SGLang's own fixture adapters through the same checks.

Use the fixture make_beam_fixture writes: its routing has no near-ties, so Megatron and the eager
oracle pick the same experts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
GROUPS = ("attention", "dense_mlp", "experts", "all")
# The same paths as SGLang fixture target groups, for control adapters written by SGLang's own fixture.
SGLANG_GROUPS = {
    "attention": ("attention", "attention_gate"),
    "dense_mlp": ("dense_mlp",),
    "experts": ("experts",),
    "all": ("attention", "attention_gate", "dense_mlp", "experts"),
}
PEFT_PREFIX = "base_model.model."
RANK, ALPHA = 8, 16
# SGLang's group strengths; the attention output gate gets twice the delta (sigmoid slope <= 1/4).
STRENGTH, GATE_STRENGTH = 0.5, 1.0
# sglang test_beam_lora.py: GENERATED_TOKENS, LORA_LOGPROB_ATOL, and the "adapter reached the output" margin.
GENERATED_TOKENS, LOGPROB_ATOL, BASE_KL_MARGIN = 4, 0.4, 10


def prompts(window):
    # Lengths stay divisible by TP for sequence parallelism; the long one straddles the sliding window.
    return [
        [1 + (i * 17 + shift) % 127 for i in range(length)] for shift, length in ((0, 8), (1, 64), (2, window + 8))
    ]


def _hf_config(model):
    from miles.utils.hf_utils.config import load_hf_config

    return load_hf_config(model)


def _hf_state(model):
    from safetensors import safe_open

    path = Path(model)
    index = path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    )
    state = {}
    for name in files:
        with safe_open(str(path / name), framework="pt") as source:
            state.update({key: source.get_tensor(key) for key in source.keys()})
    return state


def _recipe_argv(cli, config):
    sys.path.insert(0, str(REPO))
    import scripts.run_beam as recipe

    layers = config.num_hidden_layers
    model_args = (
        f"--num-layers {layers} --hidden-size {config.hidden_size} --ffn-hidden-size {config.intermediate_size} "
        f"--num-attention-heads {config.num_attention_heads} --group-query-attention "
        f"--num-query-groups {config.num_key_value_heads} --kv-channels {config.head_dim} "
        "--disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6 --swiglu "
        "--position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0 "
        f"--untie-embeddings-and-output-weights --vocab-size {config.vocab_size} "
        f"--padded-vocab-size {config.vocab_size} --make-vocab-size-divisible-by 8 "
        f"--num-experts {config.num_experts} --moe-ffn-hidden-size {config.moe_intermediate_size} "
        f"--moe-router-topk {config.num_experts_per_tok} --moe-router-score-function sigmoid "
        "--moe-router-dtype fp32 "
        f"--moe-layer-freq [{','.join(str(int(kind == 'sparse')) for kind in config.mlp_layer_types)}]"
    ).split()
    world = int(os.environ["WORLD_SIZE"])
    script = recipe.ScriptArgs(run_id="lora-check", model_dir=cli.model, output_dir=str(cli.output / "unused"))
    overrides = {
        "actor-num-nodes": "1",
        "actor-num-gpus-per-node": str(world),
        "num-gpus-per-node": str(world),
        "rollout-num-gpus": str(world),
        "rollout-num-gpus-per-engine": str(world),
        "tensor-model-parallel-size": str(cli.tp),
        "expert-model-parallel-size": str(cli.ep),
        "expert-tensor-parallel-size": str(cli.etp),
        "rollout-max-response-len": str(cli.window),
        "rollout-max-prompt-len": str(cli.window),
        "rollout-max-context-len": str(2 * cli.window),
        "seq-length": str(2 * cli.window),
        "sglang-context-length": str(2 * cli.window),
    }
    # The recipe streams the full-model optimizer to disk, which LoRA does not support.
    disk_optimizer = {
        "--stream-optimizer-state-to-disk",
        "--offload-train-disk-dir",
        "--offload-train-disk-chunk-mb",
        "--stream-optimizer-state-moment-dtype",
    }
    tokens, argv = recipe._train_args(script, colocate=False), []
    for index, token in enumerate(tokens):
        key = tokens[index - 1][2:] if index and tokens[index - 1].startswith("--") else None
        if token in disk_optimizer or (f"--{key}" in disk_optimizer and not token.startswith("--")):
            continue
        argv.append(overrides.get(key, token) if key in overrides and not token.startswith("--") else token)
    if not cli.sp:
        argv.remove("--sequence-parallel")
    return ["train.py", *model_args, *argv, "--lora-rank", str(RANK), "--lora-alpha", str(ALPHA)]


def _init(cli, config):
    import torch.distributed as dist

    from miles.backends.megatron_utils.initialize import init
    from miles.utils.arguments import parse_args

    sys.argv = _recipe_argv(cli, config)
    args = parse_args()
    args.rank, args.world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", args.local_rank))
    init(args)
    assert dist.get_world_size() == args.world_size
    return args


def _build(args, lora):
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
    from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
    from miles.backends.megatron_utils.lora.bridge import _setup_lora_model_via_bridge
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    # The same branches as setup_model_and_optimizer.
    if lora:
        model = _setup_lora_model_via_bridge(args)
        enforce_marked_param_dtypes(model)
    else:
        model = get_model(get_model_provider_func(args, "actor"), ModelType.encoder_or_decoder)
    _load_checkpoint_hf(model, None, args, args.hf_checkpoint)
    for chunk in model:
        chunk.eval()
    return model


def _logits(model, tokens):
    from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region

    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda").unsqueeze(0)
    with torch.no_grad():
        logits = model[0](input_ids=ids, position_ids=positions, attention_mask=None)
    return gather_from_tensor_model_parallel_region(logits.float())[0]


def _adapters(model):
    """Megatron LoRA modules by unwrapped name."""
    from megatron.bridge.peft.lora_layers import LoRALinear

    return {
        name.split("decoder.", 1)[1]: module.adapter
        for name, module in model[0].named_modules()
        if isinstance(module, LoRALinear)
    }


def _hf_factors(config, state, group, seed):
    """Seed HF/PEFT factors for one target group. A factor shared by a fused Megatron module is
    shared by its HF projections, so every adapter here is exactly representable in Megatron."""
    generator = torch.Generator().manual_seed(seed)
    scaling = ALPHA / RANK
    factors = {}

    def add(modules, in_features, strengths):
        a = (torch.randn(RANK, in_features, generator=generator) * RANK**-0.5).bfloat16()
        for module, strength in zip(modules, strengths, strict=True):
            weight = state[f"{module}.weight"]
            scale = strength * weight.float().std().item() / scaling if strength else 0.0
            factors[module] = (a, (torch.randn(weight.shape[0], RANK, generator=generator) * scale).bfloat16())

    for layer, kind in enumerate(config.mlp_layer_types):
        prefix = f"model.layers.{layer}"
        attention = group in ("attention", "all")
        qkvg = [f"{prefix}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "gate_proj")]
        add(qkvg, config.hidden_size, [STRENGTH * attention] * 3 + [GATE_STRENGTH * attention])
        add([f"{prefix}.self_attn.o_proj"], config.num_attention_heads * config.head_dim, [STRENGTH * attention])
        if kind == "dense":
            on = STRENGTH * (group in ("dense_mlp", "all"))
            add([f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"], config.hidden_size, [on, on])
            add([f"{prefix}.mlp.down_proj"], config.intermediate_size, [on])
        else:
            on = STRENGTH * (group in ("experts", "all"))
            for expert in range(config.num_experts):
                e = f"{prefix}.mlp.experts.{expert}"
                add([f"{e}.gate_proj", f"{e}.up_proj"], config.hidden_size, [on, on])
                add([f"{e}.down_proj"], config.moe_intermediate_size, [on])
    return factors


def _peft(factors):
    return {
        f"{PEFT_PREFIX}{module}.lora_{side}.weight": tensor
        for module, (a, b) in factors.items()
        for side, tensor in (("A", a), ("B", b))
    }


def _megatron_factor(factors, config, name, module, side, tp_size):
    """The full Megatron tensor of one adapter side; gate/up rows are already ordered per TP rank."""
    layer = f"model.layers.{name.split('.')[1]}"
    if name.endswith("linear_qkv"):
        if side == "A":
            return factors[f"{layer}.self_attn.q_proj"][0]
        rows = [factors[f"{layer}.self_attn.{p}"][1] for p in ("q_proj", "gate_proj", "k_proj", "v_proj")]
        return torch.cat([r.view(config.num_key_value_heads, -1, RANK) for r in rows], dim=1).reshape(-1, RANK)
    if name.endswith("linear_proj"):
        return factors[f"{layer}.self_attn.o_proj"][side == "B"]
    if name.endswith("linear_fc2"):
        return factors[f"{module}.down_proj"][side == "B"]
    gate, up = factors[f"{module}.gate_proj"], factors[f"{module}.up_proj"]
    if side == "A":
        return gate[0]
    return torch.cat([torch.cat(pair) for pair in zip(gate[1].chunk(tp_size), up[1].chunk(tp_size), strict=True)])


def _shard(full, shape, tp_size, tp_rank):
    """Slice the one axis a local shard splits across (expert-)TP ranks."""
    for axis, (have, want) in enumerate(zip(shape, full.shape, strict=True)):
        if have != want:
            assert have * tp_size == want, (shape, full.shape)
            return full.chunk(tp_size, dim=axis)[tp_rank]
    return full


def _install(model, config, factors):
    """Shard HF factors into Megatron's adapters: linear_qkv rows per query group as [q, g, k, v],
    linear_fc1 rows per (expert-)TP rank as [gate_r; up_r], experts by EP rank."""
    from megatron.core import parallel_state as ps

    for name, adapter in _adapters(model).items():
        layer = f"model.layers.{name.split('.')[1]}"
        expert = ".experts." in name
        tp_size = ps.get_expert_tensor_parallel_world_size() if expert else ps.get_tensor_model_parallel_world_size()
        tp_rank = ps.get_expert_tensor_parallel_rank() if expert else ps.get_tensor_model_parallel_rank()
        with torch.no_grad():
            for side, param in (("A", adapter.linear_in.weight), ("B", adapter.linear_out.weight)):
                if expert:
                    first = ps.get_expert_model_parallel_rank() * param.shape[0]
                    shards = [
                        _shard(
                            _megatron_factor(factors, config, name, f"{layer}.mlp.experts.{first + i}", side, tp_size),
                            param.shape[1:],
                            tp_size,
                            tp_rank,
                        )
                        for i in range(param.shape[0])
                    ]
                    param.copy_(torch.stack(shards))
                else:
                    full = _megatron_factor(factors, config, name, f"{layer}.mlp", side, tp_size)
                    param.copy_(_shard(full, param.shape, tp_size, tp_rank))


def _export(model, args):
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

    iterator = HfWeightIteratorBridge(
        args,
        model,
        placement=WeightUpdatePlacement(gather_pp=True),
        model_name="beam",
        quantization_config=None,
    )
    exported = {}
    for unit in iterator._iter_hf_adapter_units(None, materialize=True):
        for name, tensor in unit:
            assert name not in exported, name
            exported[name] = tensor.detach().cpu()
    return exported


def _check_close(tag, actual, expected, base):
    from sglang.test.beam_fixture import BEAM_FIXTURE_MAX_KL

    delta = (actual - expected).abs()
    kl = (expected.exp() * (expected - actual)).sum(-1).max().item()
    kl_base = None if base is None else (base.exp() * (base - actual)).sum(-1).max().item()
    close = bool((delta <= LOGPROB_ATOL + 0.002 * expected.abs()).all())
    moved = base is None or kl_base > BASE_KL_MARGIN * BEAM_FIXTURE_MAX_KL
    ok = close and kl <= BEAM_FIXTURE_MAX_KL and moved
    return dict(
        tag=tag, ok=ok, max_abs=delta.max().item(), mean_abs=delta.mean().item(), max_kl=kl, kl_vs_base=kl_base
    )


def train(cli):
    import torch.distributed as dist
    from safetensors.torch import save_file
    from sglang.test.beam_fixture import eager_beam_logits
    from sglang.test.beam_lora_fixture import BeamLoRAAdapter, merge_beam_lora

    from miles.utils.lora.utils import build_lora_config

    config = _hf_config(cli.model)
    args = _init(cli, config)
    rank0 = dist.get_rank() == 0
    sequences = prompts(cli.window)
    state = {name: tensor.cuda() for name, tensor in _hf_state(cli.model).items()}
    results = dict(layout=dict(tp=cli.tp, ep=cli.ep, etp=cli.etp, sp=cli.sp), targets=args.lora_adapter_targets)

    full = _build(args, lora=False)
    base_logits = [_logits(full, tokens) for tokens in sequences]
    del full
    model = _build(args, lora=True)
    assert all(float(a.linear_out.weight.abs().max()) == 0 for a in _adapters(model).values())
    zero = [_logits(model, tokens) for tokens in sequences]
    results["zero_b_bitwise"] = all(torch.equal(a, b) for a, b in zip(zero, base_logits, strict=True))
    results["zero_b_max_abs"] = max((a - b).abs().max().item() for a, b in zip(zero, base_logits, strict=True))

    base_ref = [eager_beam_logits(config, state, tokens).log_softmax(-1) for tokens in sequences]
    results["base"] = [
        _check_close(f"base len={len(tokens)}", logits.log_softmax(-1), ref, None)
        for tokens, logits, ref in zip(sequences, base_logits, base_ref, strict=True)
    ]
    results["groups"] = {}
    for seed, group in enumerate(GROUPS):
        factors = _hf_factors(config, state, group, seed)
        _install(model, config, factors)
        adapter = BeamLoRAAdapter(path="", weights=_peft(factors), scaling=ALPHA / RANK)
        merged = merge_beam_lora(state, adapter)
        checks = []
        logprobs = []
        for tokens, base in zip(sequences, base_ref, strict=True):
            actual = _logits(model, tokens).log_softmax(-1)
            expected = eager_beam_logits(config, merged, tokens).log_softmax(-1)
            checks.append(_check_close(f"{group} len={len(tokens)}", actual, expected, base))
            logprobs.append(actual.cpu())
        exported = _export(model, args)
        expected = {name: tensor for name, tensor in adapter.weights.items()}
        names = {name.removeprefix(PEFT_PREFIX) for name in expected}
        missing = sorted(names - exported.keys())
        extra = sorted(exported.keys() - names)
        unequal = sorted(
            name
            for name in names & exported.keys()
            if exported[name].dtype != torch.bfloat16 or not torch.equal(exported[name], expected[PEFT_PREFIX + name])
        )
        results["groups"][group] = dict(
            logprobs=checks,
            export_exact=not missing and not extra and not unequal,
            exported=len(exported),
            missing=missing[:5],
            extra=extra[:5],
            unequal=unequal[:5],
        )
        if rank0:
            out = cli.output / group
            out.mkdir(parents=True, exist_ok=True)
            save_file(
                {PEFT_PREFIX + n: t.contiguous() for n, t in exported.items()}, str(out / "adapter_model.safetensors")
            )
            (out / "adapter_config.json").write_text(
                json.dumps(build_lora_config(args, target_modules=args.lora_adapter_targets), indent=2)
            )
            torch.save(dict(sequences=sequences, logprobs=logprobs), out / "megatron_logprobs.pt")
    if rank0:
        cli.output.mkdir(parents=True, exist_ok=True)
        (cli.output / "train.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    checks = results["base"] + [check for group in results["groups"].values() for check in group["logprobs"]]
    assert results["zero_b_bitwise"], results["zero_b_max_abs"]
    assert all(group["export_exact"] for group in results["groups"].values())
    assert all(check["ok"] for check in checks), [check["tag"] for check in checks if not check["ok"]]


def serve(cli):
    import sglang as sgl
    from safetensors.torch import load_file
    from sglang.test.beam_fixture import eager_beam_logits
    from sglang.test.beam_lora_fixture import BeamLoRAAdapter, merge_beam_lora

    config = _hf_config(cli.model)
    state = {name: tensor.cuda() for name, tensor in _hf_state(cli.model).items()}
    paths = {group: str(cli.output / group) for group in GROUPS}
    if cli.sglang_adapters:
        from sglang.test.beam_lora_fixture import make_beam_lora_adapter

        paths = {group: str(cli.output / "sglang" / group) for group in GROUPS}
        for seed, group in enumerate(GROUPS):
            make_beam_lora_adapter(
                paths[group], config, state, SGLANG_GROUPS[group], rank=RANK, alpha=ALPHA, seed=seed
            )
    # test_beam_lora.py's engine settings, with the Beam recipe's attention backend.
    engine = sgl.Engine(
        model_path=cli.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        attention_backend=cli.attention_backend,
        moe_runner_backend="triton",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        max_total_tokens=8192,
        mem_fraction_static=cli.mem_fraction,
        enable_lora=True,
        lora_backend=cli.lora_backend,
        max_loras_per_batch=len(GROUPS) + 1,
        lora_paths=paths,
        tp_size=cli.tp,
    )
    results = {}
    try:
        sequences = prompts(cli.window)
        requests = [(tokens, group) for group in (*GROUPS, None) for tokens in sequences]
        # Greedy-generate like test_beam_lora.py check_batch, with prompt logprobs for every position.
        outputs = engine.generate(
            input_ids=[tokens for tokens, _ in requests],
            sampling_params={"temperature": 0, "max_new_tokens": GENERATED_TOKENS, "ignore_eos": True},
            return_logprob=True,
            logprob_start_len=0,
            token_ids_logprob=list(range(config.vocab_size)),
            lora_path=[group for _, group in requests],
        )
        for (tokens, group), output in zip(requests, outputs, strict=True):
            meta = output["meta_info"]

            def logprobs(rows):
                return torch.tensor([[entry[0] for entry in row] for row in rows], device="cuda")

            if group is None:
                weights = state
            else:
                factors = load_file(str(Path(paths[group]) / "adapter_model.safetensors"))
                weights = merge_beam_lora(
                    state, BeamLoRAAdapter(path=paths[group], weights=factors, scaling=ALPHA / RANK)
                )
            index = sequences.index(tokens)
            generated = output["output_ids"]
            # check_batch: the generated tokens' distributions against the eager oracle on merged weights.
            expected = eager_beam_logits(config, weights, tokens + generated[:-1]).log_softmax(-1)
            base = None
            if group is not None:
                base = eager_beam_logits(config, state, tokens + generated[:-1]).log_softmax(-1)[len(tokens) - 1 :]
            check = _check_close(
                f"serve {group} len={len(tokens)}",
                logprobs(meta["output_token_ids_logprobs"]),
                expected[len(tokens) - 1 :],
                base,
            )
            # Teacher-forced prompt positions avoid SGLang's LoRA decode drift, which grows with each
            # decoded token for SGLang's own fixture adapters too.
            prompt = logprobs(meta["input_token_ids_logprobs"][1:] + meta["output_token_ids_logprobs"][:1])
            scored = expected[: len(tokens)]
            check["prompt_max_kl"] = (scored.exp() * (scored - prompt)).sum(-1).max().item()
            if group is not None:
                # A factor on the wrong module or with the wrong scale leaves the output about as far
                # from the oracle as from the base.
                prompt_base = eager_beam_logits(config, state, tokens).log_softmax(-1)
                check["prompt_kl_vs_base"] = (prompt_base.exp() * (prompt_base - prompt)).sum(-1).max().item()
                check["applied"] = max(check["max_kl"], check["prompt_max_kl"]) < 0.1 * min(
                    check["kl_vs_base"], check["prompt_kl_vs_base"]
                )
            if group is not None and not cli.sglang_adapters:
                megatron = torch.load(Path(paths[group]) / "megatron_logprobs.pt")["logprobs"][index]
                check["prompt_vs_megatron_max_abs"] = (prompt.cpu() - megatron).abs().max().item()
            results.setdefault(str(group), []).append(check)
    finally:
        engine.shutdown()
    source = "sglang" if cli.sglang_adapters else "megatron"
    (cli.output / f"serve_{source}_tp{cli.tp}_{cli.lora_backend}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    # SGLang's own LoRA numerics are compared with --sglang-adapters at the same settings.
    assert all(check["ok"] for check in results["None"])
    assert all(check["applied"] for group in GROUPS for check in results[group])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "serve"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=512, help="sliding window; the long prompt exceeds it")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--lora-backend", default="csgmv")
    parser.add_argument("--mem-fraction", type=float, default=0.3)
    parser.add_argument("--sglang-adapters", action="store_true", help="serve SGLang fixture adapters (control)")
    cli = parser.parse_args()
    (train if cli.mode == "train" else serve)(cli)


if __name__ == "__main__":
    main()
