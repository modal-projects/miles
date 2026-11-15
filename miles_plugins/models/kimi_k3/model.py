import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.backends.megatron_utils.megatron_config import has_hf_checkpoint
from miles.utils.lora.utils import exports_hf_snapshots, is_lora_enabled

from .layers import KimiK3Attention, KimiK3KDAAttention, KimiK3TransformerLayer
from .mxfp4_experts import (
    MXFP4ColumnParallelGroupedLinear,
    MXFP4RowParallelGroupedLinear,
    hf_checkpoint_has_mxfp4_experts,
)
from .ops import situ_and_mul

KIMI_K3_KDA_LAYERS = tuple(
    layer_number for layer_number in range(1, 94) if layer_number % 4 != 0 and layer_number != 93
)


def configure_kimi_k3(config) -> None:
    assert config.num_layers <= 93
    assert config.num_attention_heads == 96
    assert config.moe_latent_size == 3584
    config.kimi_kda_layers = tuple(
        layer_number for layer_number in KIMI_K3_KDA_LAYERS if layer_number <= config.num_layers
    )
    config.kimi_linear_num_heads = 96
    config.kimi_linear_head_dim = 128
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    config.kimi_attn_res_block_size = 12
    config.gated_activation_func = situ_and_mul
    config.moe_latent_use_norm = True
    config.bias_activation_fusion = False
    config.use_te_activation_func = False


def build_kimi_k3_spec(config, vp_stage=None, *, mxfp4_experts: bool = False):
    assert config.virtual_pipeline_model_parallel_size is None, "Kimi K3 does not support VPP yet"

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    # layer_specs holds only this stage's layers; moe_layer_freq is indexed globally
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_number = layer_offset + len(layer_specs) + 1
        attention = KimiK3KDAAttention if layer_number in config.kimi_kda_layers else KimiK3Attention
        layer_spec.submodules.self_attention = ModuleSpec(module=attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm

        if not config.moe_layer_freq[layer_offset + len(layer_specs)]:
            # The dense MLP spec is partial(MLP.as_mlp_submodule, submodules=...).
            layer_spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        elif mxfp4_experts:
            # partial(MoELayer, submodules=MoESubmodules(experts=partial(TEGroupedMLP, submodules=...)))
            experts = layer_spec.submodules.mlp.keywords["submodules"].experts.keywords["submodules"]
            experts.linear_fc1 = MXFP4ColumnParallelGroupedLinear
            experts.linear_fc2 = MXFP4RowParallelGroupedLinear

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec


def frozen_experts_are_mxfp4(args) -> bool:
    """The trainer keeps the routed experts as the checkpoint's MXFP4 bytes when its base is a compressed-tensors
    MXFP4 checkpoint. Only a LoRA run can: the frozen experts have no BF16 master to update."""
    # the bridge loads --load when it is an HF checkpoint; a resume or a peer recovery points it elsewhere, and
    # --ref-load still names the base
    base = args.backend.load if has_hf_checkpoint(args.backend.load) else args.ref_load
    mxfp4 = hf_checkpoint_has_mxfp4_experts(base)
    assert not mxfp4 or is_lora_enabled(args), (
        f"{base} keeps its routed experts as MXFP4, which only a LoRA run (frozen experts) trains from; "
        "convert it with tools/convert_mxfp4_to_bf16.py for full-parameter training"
    )
    assert not mxfp4 or not exports_hf_snapshots(args), (
        "MXFP4 routed experts cannot be written as HF weights; drop --save-hf / eval snapshots and export the adapter"
    )
    return mxfp4


def get_kimi_k3_spec(args, config, vp_stage):
    configure_kimi_k3(config)
    return build_kimi_k3_spec(config, vp_stage=vp_stage, mxfp4_experts=frozen_experts_are_mxfp4(args))
