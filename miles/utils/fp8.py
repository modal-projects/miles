"""Block FP8 as the trainer exports it, shared with tools/convert_hf_to_fp8.py."""

import torch

from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast
from miles.utils.environ import fp8_block_scaling_fp32_scales

try:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
except ImportError:
    per_block_cast_to_fp8 = None


def quantize_fp8_block(weight: torch.Tensor, block_size) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: this [128, 128] is hacky. need improve
    if fp8_block_scaling_fp32_scales() == "0" and per_block_cast_to_fp8 is not None and list(block_size) == [128, 128]:
        return per_block_cast_to_fp8(weight)
    return fp8_blockwise_cast(weight, block_size)
