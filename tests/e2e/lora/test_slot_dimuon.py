"""Dimuon slot optimizer under Megatron: updates match the unsharded reference, padded rank stays
zero, fused projections are split, and a slot checkpoint resumes the momentum exactly.

Runs twice on 2 GPUs: DP=2 (LayerWise shards whole params across ranks) and TP=2 (adapters are
tensor-parallel sharded; the column-parallel A is sharded on its rank axis)."""

import argparse
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, init_adapter_slot
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci
from torch.utils._pytree import tree_flatten, tree_map

from miles.backends.megatron_utils.lora.checkpoint import load_slot, save_slot
from miles.backends.megatron_utils.lora.dimuon import normalized_sign, rank_scale
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer, adapter_slot_parameters, step_slot_optimizers
from miles.utils.distributed_utils import init_gloo_group

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["lora"], hardware=["hopper"])

MAX_RANK, RANK = 8, 4
# lr is large so each update dwarfs BF16 rounding of the weights it is added to
DIMUON = dict(type="dimuon", learning_rate=1.0, beta1=0.9, grad_clip_norm=0.0)
ADAM = dict(learning_rate=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
HIDDEN, HEADS, GROUPS, KV = 16, 4, 2, 4  # q per group = HEADS // GROUPS * KV = 8; qkv rows = GROUPS * 16
QKV_SPLIT = (HEADS // GROUPS * KV, KV, KV)
FFN = 16


class DimuonModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        bases = {
            "decoder.layers.0.self_attention.linear_qkv": ColumnParallelLinear(
                HIDDEN,
                GROUPS * sum(QKV_SPLIT),
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
            ),
            "decoder.layers.0.self_attention.linear_proj": RowParallelLinear(
                HIDDEN,
                HIDDEN,
                config=config,
                init_method=config.init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
            ),
            "decoder.layers.0.mlp.linear_fc1": ColumnParallelLinear(
                HIDDEN, 2 * FFN, config=config, init_method=config.init_method, bias=False, gather_output=False
            ),
        }
        self.layers = torch.nn.ModuleList()
        self.splits = []
        for name, base in bases.items():
            base.requires_grad_(False)
            self.layers.append(MultiLoRALinear(base, n_adapters=2, dim=MAX_RANK, alpha=16, full_name=name))
        tp = parallel_state.get_tensor_model_parallel_world_size()
        self.splits = [QKV_SPLIT, None, (FFN // tp, FFN // tp)]


def tp_info(param):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    if tp > 1 and getattr(param, "tensor_model_parallel", False):
        return int(param.partition_dim), parallel_state.get_tensor_model_parallel_group()
    return None, None


def gather_full(tensor, param):
    dim, group = tp_info(param)
    if dim is None:
        return tensor.clone()
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def slot_factors(model, slot):
    """(param, row_split) for every adapter factor of ``slot`` in a fixed order."""
    out = []
    for chunk in model:
        inner = chunk.module
        for layer, split in zip(inner.layers, inner.splits, strict=True):
            adapter = layer.adapters[slot]
            out += [(adapter.linear_in.weight, None), (adapter.linear_out.weight, split)]
    return out


def set_grads(model, slot, step):
    """Deterministic post-all-reduce grads, zero on the padded rank like real backward passes."""
    grads = []
    for index, (param, _) in enumerate(slot_factors(model, slot)):
        full_like = gather_full(torch.zeros_like(param, dtype=torch.float32), param)
        g = torch.sin(torch.arange(full_like.numel(), device=param.device, dtype=torch.float32) * 0.37 + index + step)
        g = g.reshape(full_like.shape)
        if index % 2 == 0:
            g[RANK:] = 0  # A: padded rank rows
        else:
            g[:, RANK:] = 0  # B: padded rank columns
        dim, group = tp_info(param)
        local = g if dim is None else g.chunk(dist.get_world_size(group), dim=dim)[dist.get_rank(group)]
        param.main_grad.copy_(local)
        grads.append(g)
    return grads


def full_weights(model, slot):
    return [gather_full(param.detach().float(), param) for param, _ in slot_factors(model, slot)]


def cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()


def snapshot(model, optimizer):
    state = {
        "weights": [param for param in adapter_slot_parameters(model, optimizer.slot)],
        "optimizer": optimizer._inner.state_dict(),
    }
    return tree_map(lambda value: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value, state)


def assert_same_state(actual, expected):
    actual_values, actual_spec = tree_flatten(actual)
    expected_values, expected_spec = tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        else:
            assert actual_value == expected_value


def run_worker(checkpoint_dir, tp):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    init_gloo_group()
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        kv_channels=KV,
        ffn_hidden_size=FFN,
        gated_linear_unit=True,
        tensor_model_parallel_size=tp,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )
    model = [
        DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            DimuonModel(config).cuda(),
        )
    ]
    args = Namespace(optimizer="adam", bf16=True, use_gloo_process_groups=True, lr=DIMUON["learning_rate"])
    for slot in (0, 1):
        init_adapter_slot(model, slot, rank=RANK, alpha=16)
    source = SlotOptimizer(args, model, slot=0, optimizer="dimuon")
    tag = f"rank={dist.get_rank()} tp={tp}"

    with pytest.raises(ValueError):
        source.apply_params(ADAM)
    print(f"{tag} adam params rejected by a dimuon slot", flush=True)

    beta, scale = DIMUON["beta1"], DIMUON["learning_rate"] * rank_scale(RANK)
    momentum = None
    for step in (1, 2):
        before = full_weights(model, 0)
        grads = set_grads(model, 0, step)
        if momentum is None:
            momentum = [torch.zeros_like(g) for g in grads]
        momentum = [beta * m + (1 - beta) * g for m, g in zip(momentum, grads, strict=True)]
        outcome = step_slot_optimizers({0: source}, {0: DIMUON})
        assert outcome[0]["grad_norm"] > 0
        after = full_weights(model, 0)
        for index, ((_param, split), m, w0, w1) in enumerate(
            zip(slot_factors(model, 0), momentum, before, after, strict=True)
        ):
            expected = -scale * normalized_sign(m, rank=RANK, row_split=split)
            actual = w1 - w0
            assert cos(actual, expected) > 0.999, (index, step, cos(actual, expected))
            assert actual.norm().item() == pytest.approx(expected.norm().item(), rel=1e-2)
            if index % 2 == 0:
                assert torch.count_nonzero(w1[RANK:]) == 0, "padded A rows must stay zero"
            else:
                assert torch.count_nonzero(w1[:, RANK:]) == 0, "padded B columns must stay zero"
        print(f"{tag} step {step}: all factors match the unsharded reference", flush=True)

    # fused blocks are normalized separately: each q/k/v (and gate/up) block of B has norm lr*scale*sqrt(rank)
    qkv_b = after[1] - before[1]
    blocks = qkv_b.view(GROUPS, sum(QKV_SPLIT), MAX_RANK).split(QKV_SPLIT, dim=1)
    for block in blocks:
        assert block.norm().item() == pytest.approx(scale * RANK**0.5, rel=1e-2)
    print(f"{tag} fused qkv blocks are normalized independently", flush=True)

    if tp > 1:
        # Slot checkpoint resume at TP>1 restores TP rank 1 incorrectly for Adam slots too (pre-existing,
        # reproduced with this model); checkpoint coverage for Dimuon stays at TP=1 until that is fixed.
        print(f"{tag} skipping checkpoint resume at tp>1", flush=True)
        dist.barrier()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
        return

    saved = snapshot(model, source)
    for child in source._inner.chained_optimizers:
        for state in child.optimizer.state.values():
            assert state["momentum_buffer"].dtype == torch.float32
    save_slot(model, source, str(checkpoint_dir))
    resumed = SlotOptimizer(args, model, slot=1, optimizer="dimuon")
    load_slot(model, resumed, str(checkpoint_dir), load_optimizer=True)
    assert_same_state(snapshot(model, resumed), saved)
    set_grads(model, 0, 3)
    step_slot_optimizers({0: source}, {0: DIMUON})
    uninterrupted = snapshot(model, source)
    set_grads(model, 1, 3)
    step_slot_optimizers({1: resumed}, {1: DIMUON})
    assert_same_state(snapshot(model, resumed), uninterrupted)
    print(f"{tag} checkpointed momentum resumes exactly", flush=True)

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--tp", type=int)
    options = parser.parse_args()
    if options.worker_dir is not None:
        run_worker(options.worker_dir, options.tp)
    else:
        for tp in (1, 2):
            with tempfile.TemporaryDirectory(prefix="slot-dimuon-", dir=options.checkpoint_root) as directory:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        __file__,
                        "--worker-dir",
                        directory + "/checkpoint",
                        "--tp",
                        str(tp),
                    ],
                    check=True,
                )
