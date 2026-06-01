import os

import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16


def nvfp4_weight_e4m3_max() -> int:
    if os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all") and os.getenv(
        "NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"
    ).strip().lower() in ("weights", "all"):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    fp4_max = torch.tensor(FP4_E2M1_MAX, device=global_amax.device, dtype=torch.float32)
    fp8_max = torch.tensor(float(nvfp4_e4m3_max), device=global_amax.device, dtype=torch.float32)
    global_encode_scale = torch.div(fp8_max * fp4_max, global_amax.to(torch.float32))
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale.numel() == 1:
        if global_encode_scale == torch.tensor(0.0, device=global_amax.device, dtype=torch.float32):
            global_encode_scale = torch.tensor(1.0, device=global_amax.device, dtype=torch.float32)
    else:
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
    return global_encode_scale


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max))


def _nvfp4_4over6_enabled() -> bool:
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all")


def _pad_rows_for_te_quantizer(weight: torch.Tensor) -> torch.Tensor:
    pad_rows = (-weight.shape[0]) % TE_NVFP4_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


def nvfp4_quantize_1d(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    row_scaled_nvfp4 = global_amax is not None and global_amax.ndim > 0 and global_amax.numel() == num_rows
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=_nvfp4_4over6_enabled(),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        with_random_sign_mask=False,
    )

    quant_input = weight
    if global_amax is not None and not row_scaled_nvfp4:
        amax_row = torch.zeros((1, num_cols), device=weight.device, dtype=weight.dtype)
        amax_row[0, 0] = global_amax.to(device=weight.device, dtype=weight.dtype).reshape(())
        quant_input = torch.cat((quant_input, amax_row), dim=0)

    quantized = quantizer.quantize(_pad_rows_for_te_quantizer(quant_input))
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[:num_rows, : num_cols // NVFP4_GROUP_SIZE].contiguous()
    if row_scaled_nvfp4:
        amax = quantized._amax_rowwise[:num_rows].contiguous()
    else:
        amax = quantized._amax_rowwise.reshape(-1)[0]
    return qweight, block_scale.view(torch.float8_e4m3fn), nvfp4_global_decode_scale_te(amax, nvfp4_e4m3_max)
