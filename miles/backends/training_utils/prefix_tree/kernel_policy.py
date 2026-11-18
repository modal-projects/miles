"""Optional frozen-expert kernel matching for Megatron CUDA workers."""

import torch

from miles.backends.training_utils.prefix_tree.operators import rowwise_linear


def frozen_expert_forward(module, value, splits):
    sizes = [int(n) for n in splits]
    weights = [
        getattr(module, f"weight{i}") for i in range(module.num_gemms)
    ]  # config-access-exempt: TE numbers its expert parameters
    if len(sizes) != len(weights) or sum(sizes) != len(value):
        raise ValueError("Frozen expert split inventory differs")
    if module.use_bias or module.tp_size != 1 or any(w.requires_grad for w in weights):
        raise ValueError("Frozen expert matching requires bias-free frozen weights and ETP=1")
    return torch.cat([rowwise_linear(part, w) for part, w in zip(value.split(sizes), weights, strict=True)]), None


def install_frozen_expert_math(model):
    # Transformer Engine is an optional dependency outside CUDA workers.
    from transformer_engine.pytorch import GroupedLinear

    for name, module in model.named_modules():
        if ".experts." in name and isinstance(module, GroupedLinear):

            def forward(value, splits, owner=module):
                return frozen_expert_forward(owner, value, splits)

            module.forward = forward
