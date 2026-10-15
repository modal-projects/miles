"""Slot mechanics for multi-LoRA: model instrumentation, shard addressing, and
per-slot optimizer state hygiene. Kept deliberately free of any control-plane
or data-plane logic."""

import logging
from argparse import Namespace
from contextlib import contextmanager

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
