"""Slot mechanics for multi-LoRA: model instrumentation, shard addressing, and
per-slot optimizer state hygiene. Kept deliberately free of any control-plane
or data-plane logic."""

import logging
from argparse import Namespace
from contextlib import contextmanager
from types import MethodType

import torch

logger = logging.getLogger(__name__)


def create_multi_lora_instance(args: Namespace, *, target_modules):
    from megatron.bridge.peft.multi_lora import MultiLoRA

    return MultiLoRA(
        target_modules=target_modules,
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis, addressed
    from the end so packed grouped-expert exports are not sliced on the expert axis."""
    if "lora_A" in hf_name:
        rank_dim = tensor.ndim - 2
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_A padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    if "lora_B" in hf_name:
        rank_dim = tensor.ndim - 1
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_B padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    return tensor


@contextmanager
def unwrap_multi_lora_linears(model):
    """Swap every multi-LoRA wrapper for its base linear, so loaders that walk the model
    (mbridge's HF load, torch_dist) see exactly the base model's modules and names."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    swapped = []
    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for parent in list(chunk.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, MultiLoRALinear):
                    setattr(parent, name, child.to_wrap)
                    swapped.append((parent, name, child))
    try:
        yield
    finally:
        for parent, name, wrapper in swapped:
            setattr(parent, name, wrapper)


def _tp_exact_multi_lora_forward(self, x, *args, **kwargs):
    """``MultiLoRALinear.forward`` with the gradients of TP-sharded adapters summed over TP,
    as Megatron's own parallel linears do.

    The upstream fused forward multiplies raw weight shards, so the gradient reaching a
    TP-sharded ``lora_B`` input (and, for a column-parallel base without sequence parallelism,
    the adapter input) holds only this rank's share; ``lora_A`` and everything upstream then
    train on partial gradients."""
    from megatron.bridge.peft.multi_lora_layers import _dense_multi_lora_mm, _narrow_token_counts_to_window
    from megatron.bridge.peft.utils import all2all_hp2sp
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
        reduce_from_tensor_model_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
    if not self._adapter_enabled:
        return linear_output, bias

    tokens_per_adapter = self.tokens_per_adapter
    token_splits = self.tokens_per_adapter_splits
    assert tokens_per_adapter is not None and token_splits is not None
    x = layernorm_output.contiguous()
    sharded = not self.replicate_adapter
    if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
        x = gather_from_sequence_parallel_region(x)
    elif sharded and not self.input_is_parallel:
        x = copy_to_tensor_model_parallel_region(x)
    x_flat = x.reshape(-1, x.shape[-1])

    total = self.tokens_per_adapter_total
    if self.replicate_adapter and total is not None and x_flat.shape[0] != total:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if x_flat.shape[0] * tp_size != total:
            raise RuntimeError(
                f"{self.base_linear_name}: adapter token spans cover {total} tokens but the base linear "
                f"received {x_flat.shape[0]} rows"
            )
        start = parallel_state.get_tensor_model_parallel_rank() * x_flat.shape[0]
        token_splits = _narrow_token_counts_to_window(token_splits, start, x_flat.shape[0])
        tokens_per_adapter = tokens_per_adapter.new_tensor(token_splits)

    offsets = tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)
    stacked_a = torch.stack([adapter.linear_in.weight for adapter in self.adapters])
    stacked_b = torch.stack([adapter.linear_out.weight for adapter in self.adapters])
    mid = _dense_multi_lora_mm(x_flat, stacked_a, token_splits=token_splits, offsets=offsets)

    if self._external_output_reduce:
        stacked_b = gather_from_sequence_parallel_region(
            stacked_b.movedim(1, 0).contiguous(), tensor_parallel_output_grad=True
        ).movedim(0, 1)
    elif sharded:
        if self.input_is_parallel:
            mid = reduce_from_tensor_model_parallel_region(mid)
        else:
            mid = gather_from_tensor_model_parallel_region(mid)
        mid = copy_to_tensor_model_parallel_region(mid)

    out = _dense_multi_lora_mm(mid, stacked_b, token_splits=token_splits, offsets=offsets)
    scaling = self.alpha_values / self.rank_values
    out = out * torch.repeat_interleave(scaling, tokens_per_adapter).unsqueeze(-1)

    if self._gather_output and not self._external_output_reduce:
        out = gather_from_tensor_model_parallel_region(out)
    if not self.disable_sequence_parallel_comm and self.input_is_parallel:
        out = all2all_hp2sp(out) if self.use_a2a else scatter_to_sequence_parallel_region(out)
    return linear_output + out.reshape(linear_output.shape), bias


def use_tp_exact_multi_lora_forward(model) -> int:
    """Switch every TP-sharded dense multi-LoRA wrapper in ``model`` to the TP-exact forward."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    switched = 0
    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for module in chunk.modules():
            if type(module) is MultiLoRALinear and not module.replicate_adapter:
                module.forward = MethodType(_tp_exact_multi_lora_forward, module)
                switched += 1
    return switched
