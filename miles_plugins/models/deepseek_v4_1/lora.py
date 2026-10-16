"""Multi-LoRA for DeepSeek-V4.1 on the Miles model definition (``--megatron-to-hf-mode raw``)."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from contextlib import contextmanager
from types import MethodType

import torch
from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRALinear,
    _dense_multi_lora_mm,
    _narrow_token_counts_to_window,
)
from megatron.bridge.peft.utils import all2all_hp2sp
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.router import TopKRouter

from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance
from miles_plugins.models.deepseek_v4_1.lora_export import ATTENTION_ADAPTERS, adapter_modules, megatron_template
from miles_plugins.models.deepseek_v4_1.lora_targets import deepseek_v41_megatron_targets

logger = logging.getLogger(__name__)


def apply_deepseek_v41_multi_lora(model, args):
    if os.environ.get("CUDA_MODULE_LOADING") != "EAGER":
        logger.warning(
            "CUDA_MODULE_LOADING=%s: lazy kernel loading can deadlock cross-node expert all-to-all; "
            "launch DeepSeek-V4.1 multi-LoRA trainers with CUDA_MODULE_LOADING=EAGER",
            os.environ.get("CUDA_MODULE_LOADING"),
        )
    targets = deepseek_v41_megatron_targets(args.hf_lora_targets)
    lora = create_multi_lora_instance(args, target_modules=targets)
    model = lora(model, training=True)
    use_tp_exact_multi_lora_forward(model)
    # TE drops the gradient through a gathered layernorm output, which would cut the adapter's input gradient.
    gathered = [
        module.base_linear_name
        for module in model.modules()
        if isinstance(module, MultiLoRALinear) and getattr(module.to_wrap, "return_layernorm_output_gathered", False)
    ]
    assert not gathered, f"DeepSeek-V4.1 multi-LoRA does not support layernorm-fused targets: {gathered}"

    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.frozen_expert_bias = True

    for layer in model.decoder.layers:
        attention = layer.self_attention
        for attribute, _ in ATTENTION_ADAPTERS:
            wrapper = getattr(attention, attribute)
            if not isinstance(wrapper, MultiLoRALinear):
                continue
            # V4.1 attention gathers its sequence-parallel input before every projection and
            # scatters only after linear_proj, so its adapters see and return full sequences.
            wrapper.disable_sequence_parallel_comm = True
            if wrapper.replicate_adapter:
                # Full-sequence inputs give every TP rank the complete gradient; do not TP-sum it.
                for param in wrapper.adapters.parameters():
                    param.sequence_parallel = False

    _check_layout(model, targets)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(
        "DeepSeek-V4.1 multi-LoRA applied: slots=%d max_rank=%d targets=%s trainable=%d total=%d",
        args.multi_lora_n_adapters,
        args.lora_rank,
        targets,
        trainable,
        total,
    )
    return model


def wrap_model_provider_with_deepseek_v41_multi_lora(provider_func, args):
    def wrapped(*provider_args, **provider_kwargs):
        return apply_deepseek_v41_multi_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped


def _check_layout(model, megatron_targets: Sequence[str]) -> None:
    missing = [
        f"layer {layer.layer_number - 1}: {template}"
        for layer in model.decoder.layers
        for hf_suffix, module in adapter_modules(layer)
        if (template := megatron_template(hf_suffix)) in megatron_targets and not hasattr(module, "adapters")
    ]
    if missing:
        raise RuntimeError(f"DeepSeek-V4.1 multi-LoRA did not wrap every target: {missing[:5]}")


@contextmanager
def unwrap_multi_lora_linears(model):
    """Swap every multi-LoRA wrapper for its base linear, so loaders that walk the model
    (mbridge's HF load, torch_dist) see exactly the base model's modules and names."""
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
    """``MultiLoRALinear.forward`` with the gradients of TP-sharded adapters summed over TP."""
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
    switched = 0
    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for module in chunk.modules():
            if type(module) is MultiLoRALinear and not module.replicate_adapter:
                module.forward = MethodType(_tp_exact_multi_lora_forward, module)
                switched += 1
    return switched
