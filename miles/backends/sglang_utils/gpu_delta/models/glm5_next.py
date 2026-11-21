"""GPU-delta bindings for SGLang's GLM-5.3-Flash model implementation (CUDA receiver only).

Mirrors ``sglang/srt/models/glm5_next.py``: ``Glm5NextForConditionalGeneration.load_weights`` for the text
model and the ``DeepseekV2WeightLoaderMixin.post_load_weights`` it runs:

- KDA layers (``Glm5NextLinearAttention``) in the layout SGLang builds for unquantized projections: q, k, v and
  b as the column-parallel and f_a, g_a as the repeated shards of ``fused_qkvbfg_a_proj``, f_b and g_b as the
  two batches of ``fused_fg_b_proj``, the q, k and v convolutions as shards of ``qkv_conv1d``, and the
  head-sharded ``A_log`` and ``dt_bias``;
- DSA layers (``DeepseekV2AttentionMLA``) through the DeepSeek-V2 MLA helpers, with the k-pool indexer
  (``IndexerKPool``), which keeps ``wk`` and ``weights_proj`` separate;
- each decoder layer's mHC parameters, its dense MLP or DeepSeek-V2 MoE (routed experts, the shared expert and
  the router, its correction bias pinned for TopK), the norms, the embedding and the head.

The loader widens BF16 checkpoint matrices into FP32 parameters: the KDA convolutions, the mHC ``hc_*_fn``
mappings, and the indexer's ``weights_proj`` and ``index_kpool_compress_ape``. Each binds a canonical shadow, and a
computed image recasts the parameter after every apply.

The checkpoint's vision tower is frozen and its MTP layer a static draft (the checkpoint policy in
``miles/utils/gpu_delta/publication.py``): Miles exports the text decoder only. Live classes match by exact
type; SGLang is imported lazily. The admitted architecture is listed only in ``models/__init__.py``.
"""

from __future__ import annotations

import re
from functools import cache
from types import MappingProxyType
from typing import NamedTuple

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ComputedImage,
    ConsumerSnapshot,
    ParameterBindings,
    _full_slices,
    _tensor_identity,
    batched_column_target,
    merged_column_target,
    repeated_column_target,
    router_bias_consumer,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _DtypeMismatch
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import (
    fused_qkv_a_view,
    mla_derived_images,
    mla_projection_binding,
)

_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
# A routed expert, or the shared expert once load_weights renames it into its fused routed slot.
_EXPERT = re.compile(r"mlp\.experts\.\d+\.(?:gate|up|down)_proj\.[^.]+")

# load_weights' stacked_params_mapping shards of the fused KDA projections and convolution.
_QKVBFG_A_SHARDS = MappingProxyType({"q_proj": 0, "k_proj": 1, "v_proj": 2, "b_proj": 3, "f_a_proj": 4, "g_a_proj": 5})
_FG_B_BATCHES = MappingProxyType({"f_b_proj": 0, "g_b_proj": 1})
_QKV_CONV_SHARDS = MappingProxyType({"q_conv1d": 0, "k_conv1d": 1, "v_conv1d": 2})

# Routes of the served names: how bind() serves each canonical tensor. Outside the decoder:
_MODEL_ROUTES = MappingProxyType(
    {
        "model.embed_tokens.weight": "dense",
        "model.norm.weight": "dense",
        "lm_head.weight": "dense",
    }
)
# In every decoder layer, by its name below the layer: the norms, the mHC parameters and o_proj.
_LAYER_ROUTES = MappingProxyType(
    {
        "input_layernorm.weight": "dense",
        "post_attention_layernorm.weight": "dense",
        **{f"hc_{site}_{leaf}": "dense" for site in ("attn", "ffn") for leaf in ("base", "scale")},
        **{f"hc_{site}_fn": "widened" for site in ("attn", "ffn")},
        "self_attn.o_proj.weight": "dense",
    }
)
_ATTENTION_ROUTES = MappingProxyType(
    {
        "kda": MappingProxyType(
            {
                **{f"self_attn.{leaf}.weight": "kda_qkvbfg_a" for leaf in _QKVBFG_A_SHARDS},
                **{f"self_attn.{leaf}.weight": "kda_fg_b" for leaf in _FG_B_BATCHES},
                **{f"self_attn.{leaf}.weight": "kda_conv" for leaf in _QKV_CONV_SHARDS},
                "self_attn.A_log": "kda_a_log",
                "self_attn.dt_bias": "kda_dt_bias",
                "self_attn.o_norm.weight": "dense",
            }
        ),
        "mla": MappingProxyType(
            {
                "self_attn.q_a_proj.weight": "mla_qkv_a",
                "self_attn.kv_a_proj_with_mqa.weight": "mla_qkv_a",
                "self_attn.q_a_layernorm.weight": "dense",
                "self_attn.kv_a_layernorm.weight": "dense",
                "self_attn.q_b_proj.weight": "mla_projection",
                "self_attn.kv_b_proj.weight": "mla_projection",
                **{
                    f"self_attn.indexer.{leaf}": "dense"
                    for leaf in (
                        "wq_b.weight",
                        "wk.weight",
                        "k_norm.weight",
                        "k_norm.bias",
                        "index_kpool_compress_gate",
                    )
                },
                "self_attn.indexer.index_kpool_compress_ape": "widened",
                "self_attn.indexer.weights_proj.weight": "indexer_weights",
            }
        ),
    }
)
_MLP_ROUTES = MappingProxyType(
    {
        "dense": MappingProxyType(
            {
                "mlp.gate_proj.weight": "gate_up",
                "mlp.up_proj.weight": "gate_up",
                "mlp.down_proj.weight": "dense",
            }
        ),
        "moe": MappingProxyType(
            {
                "mlp.gate.weight": "dense",
                "mlp.gate.e_score_correction_bias": "router_bias",
                "mlp.shared_experts.gate_proj.weight": "gate_up",
                "mlp.shared_experts.up_proj.weight": "gate_up",
                "mlp.shared_experts.down_proj.weight": "dense",
            }
        ),
    }
)


def served_name(name, shared_expert_slot=None):
    """The parameter ``load_weights`` loads checkpoint tensor ``name`` into, before its stacked-shard renames.

    The text stack drops ``language_model.``; with a fused shared expert, ``mlp.shared_experts`` becomes routed
    slot ``shared_expert_slot`` (``n_routed_experts``).
    """
    name = name.replace("language_model.", "")
    if shared_expert_slot is not None and "mlp.shared_experts" in name:
        name = name.replace("mlp.shared_experts", f"mlp.experts.{shared_expert_slot}")
    return name


def route(live, attention, mlp):
    """How the mapping binds one canonical GLM-5.3-Flash tensor, from its served name; raises for anything else.

    ``live`` is ``served_name``'s result. ``attention`` (``"kda"`` or ``"mla"``) and ``mlp`` (``"dense"`` or
    ``"moe"``) are the kinds of the decoder layer that holds the tensor, both None outside the decoder layers.
    Every name the static policy excludes is unclassified. Pure.
    """
    match = _LAYER.fullmatch(live)
    if match is None:
        if (attention, mlp) != (None, None):
            raise ValueError(f"a GLM-5.3-Flash tensor outside the decoder layers has no layer kinds: {live}")
        kind = _MODEL_ROUTES.get(live)
    elif attention not in _ATTENTION_ROUTES or mlp not in _MLP_ROUTES:
        raise ValueError(f"unsupported GLM-5.3-Flash layer kinds {attention!r}, {mlp!r}: {live}")
    else:
        rest = match[2]
        kind = _LAYER_ROUTES.get(rest) or _ATTENTION_ROUTES[attention].get(rest) or _MLP_ROUTES[mlp].get(rest)
        if kind is None and mlp == "moe" and _EXPERT.fullmatch(rest):
            kind = "expert"
    if kind is None:
        raise ValueError(f"unclassified GLM-5.3-Flash canonical tensor: {live}")
    return kind


def _leaf(live):
    # The module of a parameter name: ``...self_attn.q_proj.weight`` -> ``q_proj``.
    return live.rsplit(".", 2)[-2]


@cache
def _classes():
    """The SGLang classes the mapping admits, by exact type."""
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP, DeepseekV2MoE
    from sglang.srt.models.glm5_next import (
        Glm5NextDecoderLayer,
        Glm5NextForConditionalGeneration,
        Glm5NextLinearAttention,
    )

    return {
        cls.__name__: cls
        for cls in (
            DeepseekV2AttentionMLA,
            DeepseekV2MLP,
            DeepseekV2MoE,
            Glm5NextDecoderLayer,
            Glm5NextForConditionalGeneration,
            Glm5NextLinearAttention,
            IndexerKPool,
        )
    }


class _Layer(NamedTuple):
    """One admitted decoder layer of the live model."""

    prefix: str
    attention: str  # "kda" or "mla"
    mlp_kind: str  # "dense" or "moe"
    attn: torch.nn.Module
    mlp: torch.nn.Module
    heads: tuple[int, int] | None  # a KDA layer's (head-group rank, size)


def _require_glm5_next(model):
    """The identity gate: the exact wrapper class, serving its language model."""
    if type(model) is not _classes()["Glm5NextForConditionalGeneration"]:
        raise ValueError(f"unsupported GLM-5.3-Flash runtime class: {type(model).__name__}")
    if model.model is None:
        raise ValueError(
            "GPU delta requires the GLM-5.3-Flash language model; encoder-only serving has no text weights"
        )


def _attention_kind(attn):
    classes = _classes()
    if type(attn) is classes["Glm5NextLinearAttention"]:
        return "kda"
    elif type(attn) is classes["DeepseekV2AttentionMLA"]:
        return "mla"
    else:
        raise ValueError(f"unsupported GLM-5.3-Flash attention {type(attn).__name__}")


def _mlp_kind(mlp):
    classes = _classes()
    if type(mlp) is classes["DeepseekV2MoE"]:
        return "moe"
    elif type(mlp) is classes["DeepseekV2MLP"]:
        return "dense"
    else:
        raise ValueError(f"unsupported GLM-5.3-Flash MLP {type(mlp).__name__}")


def _kda_heads(attn):
    """This rank's KDA head group, ``(rank, size)``: the partition of every head-sharded linear, and the rank the
    ``A_log`` and ``dt_bias`` loaders resolved from the same group at construction."""
    from sglang.srt.distributed.utils import get_group_rank_size

    heads = get_group_rank_size(attn.fused_qkvbfg_a_proj.tp_group)
    if any(
        get_group_rank_size(module.tp_group) != heads
        for module in (attn.fused_fg_b_proj, attn.qkv_conv1d, attn.o_proj)
    ):
        raise ValueError("GLM-5.3-Flash KDA head group differs between its linears")
    return heads


def _kda_snapshot(attn):
    """Pin what the KDA kernels read through RadixLinearAttention: its view of ``qkv_conv1d``, ``A_log`` and
    ``dt_bias`` in place, and no convolution bias (the checkpoint has none)."""
    linear = attn.attn
    if (
        linear.bias is not None
        or _tensor_identity(linear.conv_weights) != _tensor_identity(attn.qkv_conv1d.weight.squeeze(1))
        or linear.A_log is not attn.A_log
        or linear.dt_bias is not attn.dt_bias
    ):
        raise ValueError("GLM-5.3-Flash KDA must read its convolution, A_log and dt_bias parameters in place")
    return ConsumerSnapshot(lambda: (linear.conv_weights, linear.bias, linear.A_log, linear.dt_bias))


def _widen(value):
    # The loader's copy_ into an FP32 parameter: exact BF16 widening.
    return (value.to(torch.float32),)


class Glm5NextMapping:
    """GLM-5.3-Flash's KDA/DSA hybrid with mHC, served by ``Glm5NextForConditionalGeneration``."""

    # load_weights' gate/up shards of the dense MLPs and shared experts; ParameterBindings binds them.
    stacked_params = (("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1))

    def __init__(self, model, shadows):
        _require_glm5_next(model)
        self.parameters = ParameterBindings(model, self.stacked_params, shadows)
        self.consumers = []
        self.derived = []
        # load_weights serves a fused shared expert from routed slot n_routed_experts.
        self._shared_expert_slot = model.config.n_routed_experts if model.num_fused_shared_experts > 0 else None
        self._layers = {}  # layer index -> _Layer
        self._widened = {}  # canonical name -> (its shadow, the FP32 view the loader widens it into)
        self._mla_layers = {}  # MLA layer prefix -> attention

    def bind(self, name, meta):
        live = served_name(name, self._shared_expert_slot)
        match = _LAYER.fullmatch(live)
        layer = None if match is None else self._layer(int(match[1]))
        kind = route(live, *((None, None) if layer is None else (layer.attention, layer.mlp_kind)))
        if kind in ("dense", "gate_up", "expert"):
            return self.parameters.bind(name, meta, live)
        elif kind == "router_bias":
            binding = self.parameters.bind(name, meta, live)
            mlp = layer.mlp
            self.consumers.append(router_bias_consumer(name, mlp.topk, mlp.gate.e_score_correction_bias))
            return binding
        elif kind == "kda_qkvbfg_a":
            module = layer.attn.fused_qkvbfg_a_proj
            view, slices = repeated_column_target(module.weight, module, _QKVBFG_A_SHARDS[_leaf(live)], meta["shape"])
            return self._bind_fused(name, meta, f"{layer.prefix}.self_attn.fused_qkvbfg_a_proj", module, view, slices)
        elif kind == "kda_fg_b":
            module = layer.attn.fused_fg_b_proj
            view, slices = batched_column_target(module, _FG_B_BATCHES[_leaf(live)], meta["shape"])
            return self._bind_fused(name, meta, f"{layer.prefix}.self_attn.fused_fg_b_proj", module, view, slices)
        elif kind == "kda_conv":
            module = layer.attn.qkv_conv1d
            view, slices = merged_column_target(module.weight, module, _QKV_CONV_SHARDS[_leaf(live)], meta["shape"])
            return self._bind_widened(name, meta, module, view, slices)
        elif kind == "kda_a_log":
            return self._bind_a_log(name, meta, layer)
        elif kind == "kda_dt_bias":
            return self._bind_dt_bias(name, meta, layer)
        elif kind == "mla_qkv_a":
            attn = layer.attn
            module = attn.fused_qkv_a_proj_with_mqa
            view = fused_qkv_a_view(
                module.weight,
                _leaf(live),
                q_rows=attn.q_lora_rank,
                kv_rows=attn.kv_lora_rank + attn.qk_rope_head_dim,
            )
            fused = f"{layer.prefix}.self_attn.fused_qkv_a_proj_with_mqa"
            return self._bind_fused(name, meta, fused, module, view, _full_slices(meta["shape"]))
        elif kind == "mla_projection":
            binding = self.parameters.bind(name, meta, live)
            binding, kv_layer = mla_projection_binding(self.parameters.modules, name, live, binding)
            if kv_layer is not None:
                self._mla_layers[kv_layer.prefix] = kv_layer.attn
                self.consumers.append(kv_layer.snapshot)
            return binding
        elif kind == "indexer_weights":
            module = self.parameters.module(live.removesuffix(".weight"))
            return self._bind_widened(name, meta, module, module.weight, _full_slices(meta["shape"]))
        elif kind == "widened":
            module_name, leaf = live.rsplit(".", 1)
            module = self.parameters.module(module_name)
            return self._bind_widened(name, meta, module, module._parameters[leaf], _full_slices(meta["shape"]))
        else:
            raise ValueError(f"unsupported GLM-5.3-Flash route {kind}: {name}")

    def finish(self):
        for name, (shadow, served) in self._widened.items():
            self.derived.append(ComputedImage(name, (shadow,), (served,), _widen))
        for prefix, attn in self._mla_layers.items():
            self.derived.extend(mla_derived_images(prefix, attn))

    def _layer(self, index):
        """Admit one decoder layer at its first canonical tensor and pin its layer-wide consumers."""
        if index not in self._layers:
            prefix = f"model.layers.{index}"
            decoder = self.parameters.module(prefix)
            classes = _classes()
            if type(decoder) is not classes["Glm5NextDecoderLayer"]:
                raise ValueError(f"unsupported GLM-5.3-Flash decoder layer {type(decoder).__name__}: {prefix}")
            attn, mlp = decoder.self_attn, decoder.mlp
            attention, mlp_kind, heads = _attention_kind(attn), _mlp_kind(mlp), None
            if attention == "kda":
                # Glm5NextLinearAttention._can_fuse_proj: the fused layout needs unquantized projections and no LoRA.
                if not attn.do_fuse_qkvbfg:
                    raise ValueError("GPU delta maps GLM-5.3-Flash's KDA only in its fused qkvbfg layout")
                heads = _kda_heads(attn)
                self.consumers.append(_kda_snapshot(attn))
            elif attn.indexer is not None and type(attn.indexer) is not classes["IndexerKPool"]:
                # The kpool-less Indexer fuses wk and weights_proj on CUDA; this mapping binds them separately.
                raise ValueError(f"unsupported GLM-5.3-Flash DSA indexer {type(attn.indexer).__name__}: {prefix}")
            self._layers[index] = _Layer(prefix, attention, mlp_kind, attn, mlp, heads)
        return self._layers[index]

    def _bind_fused(self, name, meta, fused, module, view, slices):
        """A loader shard of the fused ``module``'s weight, declared in its alias group."""
        binding = self.parameters.bind_view(name, meta, module, view, slices)
        self.parameters.aliases.declare(name, f"{fused}.weight", binding, module.weight)
        return binding

    def _bind_widened(self, name, meta, module, served, slices):
        """A BF16 canonical tensor whose loader ``copy_`` widens ``slices`` of it into the FP32 ``served`` view:
        bound to a canonical shadow of those rows, which a computed image recasts into ``served``."""
        if (meta["dtype"], served.dtype) != ("BF16", torch.float32):
            raise _DtypeMismatch(f"GLM-5.3-Flash widens a BF16 checkpoint into FP32 here, served={served.dtype}")
        if tuple(served.shape) != tuple(stop - start for start, stop in slices):
            raise ValueError(f"GLM-5.3-Flash widened tensor geometry differs from its loader: {name}")
        shadow = self.parameters.shadow(name, meta, slices[0], served.device)
        self._widened[name] = shadow, served
        return self.parameters.bind_view(name, meta, module, shadow, slices)

    def _bind_a_log(self, name, meta, layer):
        # load_weights views the [heads] vector as (1, 1, heads, 1); sharded_weight_loader(2) takes this rank's heads.
        (rank, size), attn = layer.heads, layer.attn
        heads = attn.local_num_heads
        if (
            tuple(attn.A_log.shape) != (1, 1, heads, 1)
            or heads * size != attn.num_heads
            or list(meta["shape"]) != [attn.num_heads]
        ):
            raise ValueError(f"unsupported GLM-5.3-Flash KDA head-vector geometry: {name}")
        return self.parameters.bind_view(name, meta, attn, attn.A_log, [[rank * heads, (rank + 1) * heads]])

    def _bind_dt_bias(self, name, meta, layer):
        # sharded_weight_loader(0) on the head group: rows [rank * S, (rank + 1) * S).
        (rank, size), bias = layer.heads, layer.attn.dt_bias
        rows = bias.shape[0]
        if bias.ndim != 1 or list(meta["shape"]) != [rows * size]:
            raise ValueError(f"unsupported GLM-5.3-Flash KDA head-vector geometry: {name}")
        return self.parameters.bind_view(name, meta, layer.attn, bias, [[rank * rows, (rank + 1) * rows]])
