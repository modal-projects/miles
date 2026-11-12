from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVGMapping
from megatron.bridge.models.conversion.utils import remove_non_pickleables
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.moe.router import TopKRouter

from miles_plugins.models.beam import BeamTopKRouter, beam_layer_spec


class BeamOutputLayer(torch.nn.Module):
    """Softcap the BF16 head projection in FP32, as Reflection SGLang does before logprobs."""

    def forward(self, *args, **kwargs):
        output, bias = super().forward(*args, **kwargs)
        cap = self.config.final_logit_softcapping
        return cap * torch.tanh(output.float() / cap), bias


@dataclass
class BeamModelProvider(GPTModelProvider):
    transformer_layer_spec: ModuleSpec | Callable = field(default_factory=lambda: beam_layer_spec)
    final_logit_softcapping: float | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        assert self.bf16 and not self.fp16 and self.fp8 is None, "Beam training requires BF16"
        assert self.num_query_groups % self.tensor_model_parallel_size == 0, "Beam TP must divide num_key_value_heads"
        # Beam is validated with unfused RoPE and SwiGLU. The LoRA path copies the CLI fusion flags,
        # which would change the base model's numerics.
        self.apply_rope_fusion = False
        self.bias_activation_fusion = False
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        routers = [module for module in model.modules() if isinstance(module, TopKRouter)]
        assert all(isinstance(router, BeamTopKRouter) for router in routers), "Beam routers must keep FP32 weights"
        if model.post_process and self.final_logit_softcapping:
            extend_instance(model.output_layer, BeamOutputLayer)
        return model


def _split_qkvg(packed, config):
    """Split linear_qkv rows, stored per query group as [q, g, k, v], into the four HF projections."""
    groups, head_dim = config.num_query_groups, config.kv_channels
    q_rows = config.num_attention_heads // groups * head_dim
    q, g, k, v = packed.view(groups, 2 * q_rows + 2 * head_dim, -1).split([q_rows, q_rows, head_dim, head_dim], dim=1)
    return {name: tensor.reshape(-1, packed.shape[-1]) for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g))}


class BeamQKVMapping(QKVGMapping):
    def hf_to_megatron(self, hf_weights, megatron_module):
        merged = None
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            groups = config.num_query_groups
            tensors = [hf_weights[name].reshape(groups, -1, config.hidden_size) for name in ("q", "g", "k", "v")]
            merged = torch.cat(tensors, dim=1).flatten(0, 1)
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
        config = (
            None if megatron_module is None else remove_non_pickleables(self._get_config(megatron_module), max_depth=3)
        )
        config = self.broadcast_obj_from_pp_rank(config, "qkvg_config")
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}
        packed = next(iter(packed_dict.values()))
        return {self.hf_param[name]: tensor for name, tensor in _split_qkvg(packed, config).items()}


AutoMapping.register_module_type("BeamGateBias", "column")
AutoMapping.register_module_type("BeamTopKRouter", "replicated")


@MegatronModelBridge.register_bridge(
    source="BeamForCausalLM", target=GPTModel, provider=BeamModelProvider, model_type="beam"
)
class BeamBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        hf = hf_pretrained.config
        if hf.to_dict().get("quantization_config") is not None:
            raise ValueError("Beam requires a BF16 HF checkpoint")
        if hf.attention_dropout != 0.0:
            raise ValueError("Beam BF16 training requires attention_dropout=0")
        if hf.num_experts_per_tok // hf.topk_group != 2:
            raise ValueError("Beam grouped routing requires num_experts_per_tok // topk_group == 2")
        provider = super().provider_bridge(hf_pretrained)
        provider.bf16 = True
        provider.fp16 = False
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.pipeline_dtype = torch.bfloat16
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.attention_softmax_in_fp32 = True
        provider.position_embedding_type = "rope"
        provider.rotary_percent = 1.0
        provider.attention_output_gate = True
        provider.share_embeddings_and_output_weights = hf.tie_word_embeddings
        provider.num_moe_experts = hf.num_experts
        provider.moe_ffn_hidden_size = hf.moe_intermediate_size
        provider.moe_router_topk = hf.num_experts_per_tok
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_pre_softmax = False
        provider.moe_router_dtype = "fp32"
        # BeamTopKRouter keeps an FP32 weight, which TE's router GEMM rejects with BF16 inputs.
        provider.moe_router_use_torch_mm = True
        provider.moe_router_fusion = False
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_layer_freq = [int(kind == "sparse") for kind in hf.mlp_layer_types]
        provider.moe_router_num_groups = hf.n_group
        provider.moe_router_group_topk = hf.topk_group
        provider.moe_router_topk_scaling_factor = hf.routed_scaling_factor
        provider.window_size = (hf.sliding_window - 1, 0)
        provider.window_attn_skip_freq = [int(kind == "sliding_attention") for kind in hf.layer_types]
        provider.no_rope_freq = [int(kind == "full_attention") for kind in hf.layer_types]
        provider.rotary_base = hf.rope_parameters["sliding_attention"]["rope_theta"]
        provider.final_logit_softcapping = hf.final_logit_softcapping
        return provider

    def _get_fused_adapter_linear_out_slices(
        self, megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=False
    ):
        # The linear_qkv adapter covers four HF projections; the base class splits only q/k/v.
        projections = {name.rsplit(".", 2)[-2]: name for name in base_hf_weight_names}
        if is_expert or projections.keys() != {"q_proj", "k_proj", "v_proj", "gate_proj"}:
            return super()._get_fused_adapter_linear_out_slices(
                megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=is_expert
            )
        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        hf_names = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "g": "gate_proj"}
        return {projections[hf_names[key]]: rows for key, rows in _split_qkvg(linear_out_tensor, model.config).items()}

    def mapping_registry(self):
        direct = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.output_gate_bias.bias": "model.layers.*.self_attn.gate_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.post_self_attn_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.post_mlp_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in direct.items()]
        mappings.append(
            BeamQKVMapping(
                "decoder.layers.*.self_attention.linear_qkv.weight",
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
                "model.layers.*.self_attn.gate_proj.weight",
            )
        )
        for layer, kind in enumerate(self.hf_config.mlp_layer_types):
            m, h = f"decoder.layers.{layer}", f"model.layers.{layer}"
            if kind == "sparse":
                mappings.extend(
                    [
                        AutoMapping(f"{m}.mlp.router.weight", f"{h}.mlp.gate.weight"),
                        AutoMapping(f"{m}.mlp.router.expert_bias", f"{h}.mlp.gate.e_score_correction_bias"),
                        GatedMLPMapping(
                            f"{m}.mlp.experts.linear_fc1.weight*",
                            f"{h}.mlp.experts.*.gate_proj.weight",
                            f"{h}.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(f"{m}.mlp.experts.linear_fc2.weight*", f"{h}.mlp.experts.*.down_proj.weight"),
                    ]
                )
            else:
                mappings.extend(
                    [
                        GatedMLPMapping(
                            f"{m}.mlp.linear_fc1.weight", f"{h}.mlp.gate_proj.weight", f"{h}.mlp.up_proj.weight"
                        ),
                        AutoMapping(f"{m}.mlp.linear_fc2.weight", f"{h}.mlp.down_proj.weight"),
                        AutoMapping(f"{m}.mlp.linear_fc1.layer_norm_weight", f"{h}.pre_feedforward_layernorm.weight"),
                    ]
                )
        return MegatronMappingRegistry(*mappings)
