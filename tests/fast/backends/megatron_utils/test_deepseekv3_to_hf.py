from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf


def _args():
    return SimpleNamespace(
        update_weight_transfer_mode="disk-delta",
        bf16=True,
        fp16=False,
        kv_channels=128,
        num_attention_heads=64,
        num_query_groups=1,
        indexer_rope_interleave=False,
    )


def test_disk_delta_emits_canonical_router_dtypes():
    gate = torch.zeros((256, 16), dtype=torch.float32)
    bias = torch.zeros(256, dtype=torch.float32)

    [(gate_name, converted_gate)] = convert_deepseekv3_to_hf(
        _args(),
        "module.module.decoder.layers.3.mlp.router.weight",
        gate,
    )
    [(bias_name, converted_bias)] = convert_deepseekv3_to_hf(
        _args(),
        "module.module.decoder.layers.3.mlp.router.expert_bias",
        bias,
    )

    assert gate_name == "model.layers.3.mlp.gate.weight"
    assert converted_gate.dtype == torch.bfloat16
    assert bias_name == "model.layers.3.mlp.gate.e_score_correction_bias"
    assert converted_bias.dtype == torch.float32
