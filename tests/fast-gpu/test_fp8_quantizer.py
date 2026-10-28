from tests.ci.ci_register import register_cuda_ci

# Hopper only: on Blackwell with DeepGEMM the export re-quantizes non-expert weights to UE8M0, which the converter does not follow.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper"], num_gpus=1)


import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from tools.convert_hf_to_fp8 import convert_fp8

from miles.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8

HIDDEN = 320
# A GLM-5 layer's Megatron parameters; sizes off the 128 grid leave partial blocks.
LAYER = {
    "input_layernorm.weight": (HIDDEN,),
    "self_attention.linear_q_down_proj.weight": (200, HIDDEN),
    "self_attention.linear_q_up_proj.layer_norm_weight": (200,),
    "self_attention.linear_q_up_proj.weight": (384, 200),
    "self_attention.linear_kv_down_proj.weight": (192, HIDDEN),
    "self_attention.linear_kv_up_proj.weight": (512, 136),
    "self_attention.linear_proj.weight": (HIDDEN, 384),
    "self_attention.wq_b.weight": (512, 200),
    "self_attention.wk.weight": (128, HIDDEN),
    "self_attention.k_norm.weight": (128,),
    "self_attention.weights_proj.weight": (32, HIDDEN),
    "pre_mlp_layernorm.weight": (HIDDEN,),
    "mlp.router.weight": (4, HIDDEN),
    "mlp.experts.linear_fc1.weight0": (2 * 96, HIDDEN),
    "mlp.experts.linear_fc2.weight0": (HIDDEN, 96),
    "mlp.shared_experts.linear_fc1.weight": (2 * 96, HIDDEN),
    "mlp.shared_experts.linear_fc2.weight": (HIDDEN, 96),
}
MTP = {"eh_proj.weight": (HIDDEN, 2 * HIDDEN), "enorm.weight": (HIDDEN,), "hnorm.weight": (HIDDEN,)}


def _megatron_params() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = {f"module.module.decoder.layers.0.{name}": shape for name, shape in LAYER.items()}
    shapes |= {f"module.module.mtp.layers.0.transformer_layer.{name}": shape for name, shape in LAYER.items()}
    shapes |= {f"module.module.mtp.layers.0.{name}": shape for name, shape in MTP.items()}
    params = {
        name: torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        for name, shape in shapes.items()
    }
    params["module.module.decoder.layers.0.self_attention.linear_proj.weight"][:128, :128] = 0
    return params


@pytest.mark.parametrize("fp32_scales", ["1", "0"])
@pytest.mark.parametrize("indexer_rope_interleave", [True, False])
def test_the_converter_writes_the_trainers_fp8_export(tmp_path, monkeypatch, fp32_scales, indexer_rope_interleave):
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", fp32_scales)
    params = _megatron_params()
    convert_args = SimpleNamespace(
        kv_channels=128,
        hidden_size=HIDDEN,
        num_attention_heads=1,
        num_query_groups=1,
        num_layers=1,
        indexer_rope_interleave=indexer_rope_interleave,
    )

    def hf(name):
        return convert_deepseekv3_to_hf(convert_args, name, params[name])

    source, target = tmp_path / "bf16", tmp_path / "fp8"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["GlmMoeDsaForCausalLM"], "indexer_rope_interleave": indexer_rope_interleave})
    )
    bf16 = {hf_name: tensor.contiguous().cpu() for name in params for hf_name, tensor in hf(name)}
    safetensors.torch.save_file(bf16, source / "model.safetensors")
    convert_fp8(str(source), str(target), "block", block_size=[128, 128])
    converted = safetensors.torch.load_file(target / "model.safetensors")
    quantization_config = json.loads((target / "config.json").read_text())["quantization_config"]

    export_args = SimpleNamespace(backend=SimpleNamespace(indexer_rope_interleave=indexer_rope_interleave))
    exported = {
        hf_name: tensor.cpu()
        for name in params
        for hf_name, tensor in quantize_params_fp8(export_args, name, hf(name), quantization_config)
    }

    assert exported.keys() == converted.keys()
    for name, tensor in exported.items():
        assert tensor.dtype == converted[name].dtype, name
        assert torch.equal(tensor.view(torch.uint8), converted[name].view(torch.uint8)), name
    wk = "model.layers.0.self_attn.indexer.wk.weight"
    assert converted[wk].dtype == (torch.bfloat16 if indexer_rope_interleave else torch.float8_e4m3fn)
    scale = converted["model.layers.0.self_attn.o_proj.weight_scale_inv"]
    assert torch.equal(scale.log2().round().exp2(), scale) == (fp32_scales == "0")
