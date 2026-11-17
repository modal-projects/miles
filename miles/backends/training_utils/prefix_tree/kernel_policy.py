"""Opt-in shape-stable kernel policy shared by packed-prefix trainer adapters.

These substitutions preserve parameter objects and native routing/communication.
They apply equally to native expanded and compact execution. They are expensive
numerical controls, not a promise of gradient equivalence or faster execution.
"""

import torch

from miles.backends.training_utils.prefix_tree.operators import rowwise_linear


def grouped_rowwise(x, weights, offsets):
    ends = offsets.detach().cpu().tolist()
    if len(ends) != len(weights) or (ends and ends[-1] != len(x)):
        raise ValueError("Grouped row-wise offsets do not cover the rows/weights")
    starts = [0, *ends[:-1]]
    return torch.cat([rowwise_linear(x[a:b], w.t()) for a, b, w in zip(starts, ends, weights, strict=True)])


def frozen_expert_forward(module, x, splits):
    """Control only local frozen GEMMs; dispatcher/slot routing stay native."""
    sizes = [int(n) for n in splits]
    weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
    if len(sizes) != len(weights) or sum(sizes) != len(x):
        raise ValueError("Frozen expert split inventory differs")
    if module.use_bias or module.tp_size != 1 or any(w.requires_grad for w in weights):
        raise ValueError("Frozen expert control requires bias-free frozen weights and ETP=1")
    return torch.cat([rowwise_linear(part, w) for part, w in zip(x.split(sizes), weights, strict=True)]), None


class FrozenExpertKernelControl:
    def __init__(self, model):
        self.originals = []
        for name, module in model.named_modules():
            if ".experts.linear_" in name and name.endswith(".to_wrap"):
                self.originals.append((module, module.forward))

                def forward(x, splits, owner=module):
                    return frozen_expert_forward(owner, x, splits)

                module.forward = forward

    def close(self):
        for module, forward in self.originals:
            module.forward = forward


class LoraKernelControl:
    def __init__(self):
        from megatron.bridge.peft import multi_lora_layers

        self.module = multi_lora_layers
        self.grouped = torch._grouped_mm
        self.dense = self.module._dense_multi_lora_mm

        def dense(x, weights, *, token_splits, offsets):
            return grouped_rowwise(x, weights.transpose(-2, -1), offsets)

        def grouped(x, weights, offsets=None, *args, **kwargs):
            if x.ndim == 2 and weights.ndim == 3 and offsets is not None and not args and not kwargs:
                return grouped_rowwise(x, weights, offsets)
            return self.grouped(x, weights, offsets, *args, **kwargs)

        self.module._dense_multi_lora_mm = dense
        torch._grouped_mm = grouped

    def close(self):
        self.module._dense_multi_lora_mm = self.dense
        torch._grouped_mm = self.grouped
