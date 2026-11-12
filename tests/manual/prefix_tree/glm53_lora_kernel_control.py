"""Test-only native/per-slot grouped-LoRA kernel ablations; no source edits."""

import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.operators import rowwise_linear


def grouped_reference(x, weights, offsets):
    ends = offsets.detach().cpu().tolist()
    if len(ends) != len(weights) or (ends and ends[-1] != len(x)):
        raise ValueError("Grouped reference offsets do not cover the rows/weights")
    starts = [0, *ends[:-1]]
    return torch.cat([F.linear(x[a:b], w.t()) for a, b, w in zip(starts, ends, weights, strict=True)])


def grouped_rowwise(x, weights, offsets):
    ends = offsets.detach().cpu().tolist()
    if len(ends) != len(weights) or (ends and ends[-1] != len(x)):
        raise ValueError("Grouped row-wise offsets do not cover the rows/weights")
    starts = [0, *ends[:-1]]
    return torch.cat([rowwise_linear(x[a:b], w.t()) for a, b, w in zip(starts, ends, weights, strict=True)])


class LoraKernelControl:
    def __init__(self, model):
        from megatron.bridge.peft import multi_lora_layers

        self.module = multi_lora_layers
        self.gate = self.module._can_use_grouped_mm
        self.grouped = torch._grouped_mm
        self.dense = self.module._dense_multi_lora_mm
        self.configs = []
        seen = set()
        fields = ("moe_router_use_torch_mm", "moe_activation_in_fp32", "moe_combine_in_fp32")
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and id(config) not in seen and all(hasattr(config, field) for field in fields):
                seen.add(id(config))
                self.configs.append((config, {field: getattr(config, field) for field in fields}))
        self.calls = 0

    def activate(self, mode):
        self.close()
        self.calls = 0
        if mode in ("rowwise_dense", "rowwise_all"):

            def dense(x, weights, *, token_splits, offsets):
                return grouped_rowwise(x, weights.transpose(-2, -1), offsets)

            self.module._dense_multi_lora_mm = dense
        if mode == "rowwise_all":

            def grouped_row(x, weights, offsets=None, *args, **kwargs):
                if x.ndim == 2 and weights.ndim == 3 and offsets is not None and not args and not kwargs:
                    self.calls += 1
                    return grouped_rowwise(x, weights, offsets)
                return self.grouped(x, weights, offsets, *args, **kwargs)

            torch._grouped_mm = grouped_row
        if mode == "dense_per_slot":
            self.module._can_use_grouped_mm = lambda *args, **kwargs: False
        if mode.startswith("all_per_group"):

            def grouped(x, weights, offsets=None, *args, **kwargs):
                if x.ndim == 2 and weights.ndim == 3 and offsets is not None and not args and not kwargs:
                    self.calls += 1
                    return grouped_reference(x, weights, offsets)
                return self.grouped(x, weights, offsets, *args, **kwargs)

            torch._grouped_mm = grouped
        if mode == "all_per_group_precise":
            for config, values in self.configs:
                for field in values:
                    setattr(config, field, True)

    def close(self):
        self.module._can_use_grouped_mm = self.gate
        self.module._dense_multi_lora_mm = self.dense
        torch._grouped_mm = self.grouped
        for config, values in self.configs:
            for field, value in values.items():
                setattr(config, field, value)
