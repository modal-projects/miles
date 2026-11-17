"""Raw-mode Nemotron Nano checkpoint conversion without Megatron-Bridge."""

from megatron.core.activations import squared_relu
from mbridge.core import register_model
from mbridge.models import Qwen2Bridge

from miles_plugins.models.nemotron_h.weights import hf_weight_names, pack_mamba_components, unpack_mamba_components


@register_model("nemotron_h")
class NemotronHBridge(Qwen2Bridge):
    _CONFIG_MAPPING = {**Qwen2Bridge._CONFIG_MAPPING, "layernorm_epsilon": "layer_norm_epsilon"}

    def _build_config(self):
        hf = self.hf_config
        assert hf.mlp_hidden_act == "relu2", "native Nemotron-H currently supports the Nano ReLU-squared MLP"
        latent = getattr(
            hf, "moe_latent_size", None
        )  # config-access-exempt: only Super and larger checkpoints declare a latent expert width
        assert not latent, "native Nemotron-H does not yet support latent experts"
        return self._build_base_config(
            activation_func=squared_relu,
            gated_linear_unit=False,
            bias_activation_fusion=False,
            add_qkv_bias=False,
            qk_layernorm=False,
            mamba_head_dim=hf.mamba_head_dim,
            mamba_num_heads=hf.mamba_num_heads,
            mamba_num_groups=hf.n_groups,
            mamba_state_dim=hf.ssm_state_size,
            num_moe_experts=hf.n_routed_experts,
            moe_ffn_hidden_size=hf.moe_intermediate_size,
            moe_shared_expert_intermediate_size=hf.moe_shared_expert_intermediate_size,
            moe_router_topk=hf.num_experts_per_tok,
            moe_router_topk_scaling_factor=hf.routed_scaling_factor,
            moe_router_num_groups=hf.n_group,
            moe_router_group_topk=hf.topk_group,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_router_pre_softmax=True,
            moe_grouped_gemm=True,
            moe_aux_loss_coeff=0.0,
            moe_router_bias_update_rate=0.0,
            moe_shared_expert_overlap=False,
            moe_layer_freq=[int(symbol == "E") for symbol in hf.hybrid_override_pattern],
        )

    def _weight_name_mapping_mcore_to_hf(self, name):
        return hf_weight_names(name)

    def _weight_to_mcore_format(self, name, hf_weights):
        weight = super()._weight_to_mcore_format(name, hf_weights)
        return pack_mamba_components(self.hf_config, name, weight, self.mpu.tp_size)

    def _weight_to_hf_format(self, name, weight):
        weight = unpack_mamba_components(self.hf_config, name, weight, self.mpu.tp_size)
        return super()._weight_to_hf_format(name, weight)
