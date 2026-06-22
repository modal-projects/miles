import torch


def to_model_dtype(args, param):
    """Cast a router param back to the model dtype before export.

    The MoE router runs in fp32 (--moe-router-dtype fp32), so Megatron can hold its weight /
    expert_bias buffer in fp32 even when the model dtype is bf16/fp16. The HF base checkpoint
    stores those buffers in the model dtype, and update_weight_from_disk_delta XORs each freshly
    converted tensor against the base HF bytes — so a leftover fp32 router is a byte-width mismatch
    against a bf16/fp16 base. Cast back so the exported byte shape matches the base on disk.
    """
    if getattr(args, "bf16", False):
        return param.to(torch.bfloat16)
    if getattr(args, "fp16", False):
        return param.to(torch.float16)
    return param
