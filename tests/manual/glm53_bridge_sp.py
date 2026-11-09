"""Check GLM's replicated multi-LoRA projections against a dense reference.

Run in the trainer image:
    torchrun --standalone --nproc-per-node=2 tests/manual/glm53_bridge_sp.py

The GLM provider branch's older Bridge core mixes sequence-parallel tokens in
this path. Keep the release core when loading the pinned GLM model dependency.
"""

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, init_adapter_slot, set_tokens_per_adapter_slot
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            params_dtype=torch.float32,
        )
        base = TELinear(
            8,
            8,
            parallel_mode="duplicated",
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        layer = MultiLoRALinear(base, n_adapters=2, dim=4, alpha=4, full_name="probe.linear_q_down_proj").cuda()
        a = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 100
        b = torch.arange(32, device="cuda", dtype=torch.float32).reshape(8, 4) / 100
        for slot, adapter in enumerate(layer.adapters):
            init_adapter_slot(layer, slot, 4, 4)
            wa, wb = adapter.linear_in.weight, adapter.linear_out.weight
            with torch.no_grad():
                wa.copy_((a if wa.shape == a.shape else a.chunk(2, dim=0)[rank]) * (slot + 1))
                wb.copy_((b if wb.shape == b.shape else b.chunk(2, dim=0)[rank]) * (slot + 1))

        full_input = torch.arange(128, device="cuda", dtype=torch.float32).reshape(16, 1, 8) / 100
        for counts in ((16, 0), (6, 10)):
            layer.zero_grad(set_to_none=True)
            x = full_input.chunk(2, dim=0)[rank].detach().clone().requires_grad_()
            set_tokens_per_adapter_slot(layer, torch.tensor(counts, dtype=torch.int64, device="cuda"))
            actual, _ = layer(x)
            reference = full_input.detach().clone().requires_grad_()
            expected = torch.cat(
                [
                    F.linear(F.linear(part, a * (slot + 1)), b * (slot + 1))
                    for slot, part in enumerate(reference.split(counts, dim=0))
                ]
            ).chunk(2, dim=0)[rank]
            actual.sum().backward()
            expected.sum().backward()
            forward_error = float((actual.detach() - expected.detach()).abs().max())
            gradient_error = float((x.grad - reference.grad.chunk(2, dim=0)[rank]).abs().max())
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "counts": counts,
                        "forward_max_abs": forward_error,
                        "input_gradient_max_abs": gradient_error,
                    }
                ),
                flush=True,
            )
            assert forward_error < 1e-5 and gradient_error < 1e-5, (forward_error, gradient_error)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
