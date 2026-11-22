import functools
import hashlib
import logging
import os
from contextlib import contextmanager

import torch

from miles.utils.nvfp4_fake_qat import NVFP4_FAKE_QAT_FLAG

logger = logging.getLogger(__name__)

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16

NVFP4_STOCHASTIC_ROUNDING_FLAG = "OPEN_TRAINING_NVFP4_STOCHASTIC_ROUNDING_FLAG"
# (seed, weight version) of the weights being exported.
_rounding_point: tuple[int, int] | None = None


def is_nvfp4_stochastic_rounding_enabled() -> bool:
    return os.getenv(NVFP4_STOCHASTIC_ROUNDING_FLAG, "0") == "1"


@contextmanager
def nvfp4_rounding_point(seed: int, weight_version: int):
    """Exports inside quantize the weights of ``weight_version``."""
    global _rounding_point
    previous, _rounding_point = _rounding_point, (seed, weight_version)
    try:
        yield
    finally:
        _rounding_point = previous


@functools.cache
def _validate_nvfp4_stochastic_rounding() -> None:
    if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") == "1":
        raise ValueError(
            f"{NVFP4_STOCHASTIC_ROUNDING_FLAG}=1 with {NVFP4_FAKE_QAT_FLAG}=1: fake QAT trains on the "
            "round-to-nearest weights, which rollouts would no longer serve"
        )
    if _nvfp4_4over6_enabled():
        raise ValueError(
            f"{NVFP4_STOCHASTIC_ROUNDING_FLAG}=1 with NVTE_NVFP4_4OVER6={os.environ['NVTE_NVFP4_4OVER6']}: TE's "
            "NVFP4 quantizer does not round 4over6 blocks stochastically"
        )
    logger.info(f"{NVFP4_STOCHASTIC_ROUNDING_FLAG}=1: NVFP4 exports are rounded stochastically")


def nvfp4_philox_seed(name: str) -> int | None:
    """The Philox seed that rounds tensor ``name`` stochastically at the current point; None rounds to nearest.

    A function of (seed, weight version, name) alone: a resumed trainer exports a version's bytes again, and
    every layout rounds a tensor alike.
    """
    if not is_nvfp4_stochastic_rounding_enabled():
        return None
    _validate_nvfp4_stochastic_rounding()
    if _rounding_point is None:
        raise RuntimeError(f"{NVFP4_STOCHASTIC_ROUNDING_FLAG}=1: an export must run inside nvfp4_rounding_point")
    seed, weight_version = _rounding_point
    digest = hashlib.blake2b(f"{seed}/{weight_version}/{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


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
    # Python scalar operands are passed to device kernels without allocating
    # and copying scalar tensors. Both supported numerators (1536 and 2688)
    # are exactly representable in FP32.
    global_encode_scale = torch.div(float(nvfp4_e4m3_max) * FP4_E2M1_MAX, global_amax.to(torch.float32)).clamp_max(
        torch.finfo(torch.float32).max
    )
    # A Python conditional on a CUDA scalar synchronizes once per expert
    # quantization during weight publication. Keep the TE zero-scale rule on
    # device for scalar and batched amax alike.
    return torch.where(global_encode_scale == 0.0, 1.0, global_encode_scale)


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max))


def _nvfp4_4over6_enabled() -> bool:
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all")


def _clear_negative_zero(qweight: torch.Tensor) -> torch.Tensor:
    # E2M1 has +0 (0x0) and -0 (0x8), and TE gives a value that rounds to zero that value's sign: a weight
    # near zero would flip between them with no change in value.
    low, high = qweight & 0x0F, qweight & 0xF0
    return torch.where(high == 0x80, 0, high) | torch.where(low == 0x08, 0, low)


def _pad_rows_for_te_quantizer(weight: torch.Tensor) -> torch.Tensor:
    pad_rows = (-weight.shape[0]) % TE_NVFP4_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


def _quantize_seeded(quantizer, weight: torch.Tensor, philox_seed: int):
    # TE draws the rounding bits from the current device's default generator: seed it for this call only.
    generator = torch.cuda.default_generators[torch.cuda.current_device()]
    state = generator.get_state()
    generator.manual_seed(philox_seed)
    try:
        return quantizer.quantize(weight)
    finally:
        generator.set_state(state)


def nvfp4_quantize_1d(
    weight: torch.Tensor,
    *,
    clear_negative_zero: bool = True,
    philox_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=philox_seed is not None,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=_nvfp4_4over6_enabled(),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        with_random_sign_mask=False,
    )

    padded = _pad_rows_for_te_quantizer(weight)
    if philox_seed is None:
        quantized = quantizer.quantize(padded)
    else:
        quantized = _quantize_seeded(quantizer, padded, philox_seed)
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2]
    if clear_negative_zero:
        qweight = _clear_negative_zero(qweight)
    qweight = qweight.contiguous()
    block_scale = quantized._rowwise_scale_inv[:num_rows, : num_cols // NVFP4_GROUP_SIZE].contiguous()
    amax = quantized._amax_rowwise.reshape(-1)[0]
    return qweight, block_scale.view(torch.float8_e4m3fn), nvfp4_global_decode_scale_te(amax, nvfp4_e4m3_max)


def nvfp4_quantize_1d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    clear_negative_zero: bool = True,
    philox_seed: int | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError("nvfp4_quantize_1d_pair expects two 2D tensors.")
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"NVFP4 paired quantization requires matching K dimensions, got {first.shape[1]} and {second.shape[1]}."
        )

    first_rows = first.shape[0]
    combined = _contiguous_pair_view(first, second)
    if combined is None:
        combined = torch.cat((first.contiguous(), second.contiguous()), dim=0)
    combined_qweight, combined_block_scale, global_scale = nvfp4_quantize_1d(
        combined, clear_negative_zero=clear_negative_zero, philox_seed=philox_seed
    )
    first_result = (
        combined_qweight[:first_rows].contiguous(),
        combined_block_scale[:first_rows].contiguous(),
        global_scale.clone(),
    )
    second_result = (
        combined_qweight[first_rows:].contiguous(),
        combined_block_scale[first_rows:].contiguous(),
        global_scale.clone(),
    )
    return first_result, second_result


def _contiguous_pair_view(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor | None:
    if not first.is_contiguous() or not second.is_contiguous():
        return None
    if first.device != second.device or first.dtype != second.dtype or first.stride() != second.stride():
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if first.storage_offset() + first.numel() != second.storage_offset():
        return None

    try:
        return first.as_strided(
            (first.shape[0] + second.shape[0], first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )
    except RuntimeError:
        return None
