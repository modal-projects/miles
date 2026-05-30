import logging
import os
from contextlib import contextmanager

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16

logger = logging.getLogger(__name__)
FLASHINFER_NVFP4_ENV_KEYS = (
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
)


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def nvfp4_4over6_weight_scope_enabled(value) -> bool:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("weights", "all"):
            return True
        if value in ("none", "activations"):
            return False
    return str_to_bool(value)


def nvfp4_4over6_enabled() -> bool:
    return nvfp4_4over6_weight_scope_enabled(os.getenv("NVTE_NVFP4_4OVER6"))


def nvfp4_4over6_weight_scope() -> str:
    return "weights" if nvfp4_4over6_enabled() else "none"


def nvfp4_weight_e4m3_max() -> int:
    if nvfp4_4over6_enabled() and nvfp4_4over6_weight_scope_enabled(
        os.getenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    ):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_4over6_err_mode() -> str:
    err_mode = os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()
    if err_mode not in ("MAE", "MSE"):
        raise ValueError("NVTE_NVFP4_4OVER6_ERR_MODE must be one of: 'MAE', 'MSE'.")
    return err_mode


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


def sync_flashinfer_nvfp4_env_from_nvte() -> dict[str, str]:
    weight_4over6_enabled = nvfp4_4over6_enabled()
    flashinfer_env = {
        "FLASHINFER_NVFP4_4OVER6": "1" if weight_4over6_enabled else "0",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": (
            "1"
            if weight_4over6_enabled
            and nvfp4_4over6_weight_scope_enabled(os.getenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"))
            else "0"
        ),
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": nvfp4_4over6_err_mode(),
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": (
            "1" if str_to_bool(os.getenv("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "0")) else "0"
        ),
    }
    os.environ.update(flashinfer_env)
    os.environ.setdefault("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH", "1")
    return {**flashinfer_env, "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": os.environ["TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"]}


@contextmanager
def flashinfer_nvfp4_env_from_nvte():
    original_env = {key: os.environ.get(key) for key in FLASHINFER_NVFP4_ENV_KEYS}
    try:
        sync_flashinfer_nvfp4_env_from_nvte()
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


try:
    from flashinfer import nvfp4_quantize as _flashinfer_nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout
except ImportError:
    _flashinfer_nvfp4_quantize = None
    SfLayout = None
    logger.warning("FlashInfer nvfp4_quantize not available; falling back to TransformerEngine reference.")


def _te_nvfp4_quantize_1d(
    weight: torch.Tensor,
    global_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef
    except ImportError:
        from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef

    weight_4over6_enabled = nvfp4_4over6_enabled()
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()
    try:
        qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
            weight,
            global_amax,
            NVFP4_GROUP_SIZE,
            1,
            pow_2_scales=False,
            nvfp4_use_4over6=weight_4over6_enabled,
            nvfp4_e4m3_max=nvfp4_e4m3_max,
            nvfp4_4over6_err_mode=nvfp4_4over6_err_mode(),
            eps=0.0,
        )
    except TypeError:
        if weight_4over6_enabled:
            raise
        qweight, block_scale = NVFP4QuantizerRef._quantize_blockwise_reference(
            weight,
            global_amax,
            NVFP4_GROUP_SIZE,
            1,
            pow_2_scales=False,
            eps=0.0,
        )
    return qweight, block_scale, nvfp4_global_decode_scale_te(global_amax, nvfp4_e4m3_max)


def nvfp4_quantize_1d(
    weight: torch.Tensor,
    global_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _flashinfer_nvfp4_quantize is None or weight.dtype == torch.float32 or not weight.is_cuda:
        return _te_nvfp4_quantize_1d(weight, global_amax)

    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()
    global_encode_scale = nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max)
    with flashinfer_nvfp4_env_from_nvte():
        qweight, block_scale = _flashinfer_nvfp4_quantize(
            weight,
            global_encode_scale.reshape(1).contiguous(),
            sfLayout=SfLayout.layout_linear,
            do_shuffle=False,
            sf_vec_size=NVFP4_GROUP_SIZE,
            backend="cuda",
        )
    return qweight, block_scale.view(torch.float8_e4m3fn), nvfp4_global_decode_scale_te(global_amax, nvfp4_e4m3_max)
