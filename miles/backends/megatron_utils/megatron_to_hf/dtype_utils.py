import torch

from miles.backends.megatron_utils.hf_checkpoint_metadata import load_tensor_spec


_SAFETENSORS_TO_TORCH_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}
if hasattr(torch, "float8_e4m3fn"):
    _SAFETENSORS_TO_TORCH_DTYPE["F8_E4M3"] = torch.float8_e4m3fn
    _SAFETENSORS_TO_TORCH_DTYPE["F8_E4M3FN"] = torch.float8_e4m3fn


def to_model_dtype(args, param, hf_name: str | None = None):
    """Cast a router param back to the model dtype before export.

    The MoE router runs in fp32 (--moe-router-dtype fp32), so Megatron can hold its weight /
    expert_bias buffer in fp32 even when the model dtype is bf16/fp16. Disk-delta XORs each freshly
    converted tensor against the rollout base bytes, so match that checkpoint's dtype when available
    and otherwise fall back to the configured model dtype.
    """
    if hf_name is not None:
        spec = load_tensor_spec(args.hf_checkpoint, hf_name)
        if spec is not None and spec.dtype in _SAFETENSORS_TO_TORCH_DTYPE:
            return param.to(_SAFETENSORS_TO_TORCH_DTYPE[spec.dtype])
    if getattr(args, "bf16", False):
        return param.to(torch.bfloat16)
    if getattr(args, "fp16", False):
        return param.to(torch.float16)
    return param
