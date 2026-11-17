"""Grouped expert host contracts for native adapter attachment tests."""

import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params_dtype = torch.float32
        for index in range(num_experts):
            self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(out_features, in_features)))
