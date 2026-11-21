"""GPU-delta bindings for SGLang's Muse Spark model implementation (CUDA receiver only).

Mirrors ``sglang/srt/models/muse_spark.py`` as ``muse_spark_omni.MuseSparkForConditionalGeneration`` serves its
language model: ``MuseSparkTextForCausalLM.load_weights`` and ``post_load_weights``, for BF16 checkpoints:

- attention: q, k, v and the output gate (``self_attn.gate_proj``) as the four shards of ``qkv_proj``
  (``QKVOGateParallelLinear.weight_loader``); the loader interleaves each q and k head's rotary halves first, so
  those two bind through a permuted view of their rows;
- the routed experts (``MuseSpark1xBf16Experts``): gate and up fill the halves of ``gate_up_proj`` and down fills
  ``down_proj``. On FlashInfer's TRT-LLM BF16 MoE (SM10x), ``_shuffle_weights`` rewrites them in FlashInfer's
  blocked BF16 layout with the up half first, so they bind through that layout's views
  (``kernel_backends/flashinfer.py``); the Triton and PyTorch paths keep the checkpoint layout. MegaMoE is refused;
- the FP32 router and correction bias, the latent expert projections, o_proj, the dense MLP, embedding and head;
- the values ``post_load_weights`` computes, as computed images: each norm's effective weight (its zero-centered
  weight plus its gain center), the residual gates' coefficients, the fused norm boundaries' copies of both, and
  the router's BF16 split once built (SM90 only). The TRT-LLM and Triton expert finalizes read the post-expert
  norm's effective weight in place.

The vision and audio towers are frozen (the checkpoint policy in ``miles/utils/gpu_delta/publication.py``):
Miles exports the language model only. NVFP4 and int4 checkpoints are refused. Live classes match by exact type;
SGLang is imported lazily. The admitted architecture is listed only in ``models/__init__.py``.
"""

from __future__ import annotations

import re
from functools import cache
from types import MappingProxyType
from typing import NamedTuple

import torch

from miles.backends.sglang_utils.gpu_delta.bindings import (
    REMOTE_EXPERT,
    ComputedImage,
    ConsumerSnapshot,
    ParameterBindings,
    TensorBinding,
    _full_slices,
    _output_dim,
    _require_unquantized,
    _tensor_identity,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _storage_key
from miles.backends.sglang_utils.gpu_delta.kernel_backends.flashinfer import FlashInferLayout

_LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight")
# load_weights' shards of the fused attention projection, by checkpoint leaf.
_QKVG_SHARDS = MappingProxyType({"q_proj": "q", "k_proj": "k", "v_proj": "v", "gate_proj": "g"})
# MuseSpark1xBf16Experts._shuffle_weights: FlashInfer's TRT-LLM BF16 layout of each expert's [up; gate] rows (it
# swaps the checkpoint's [gate; up] before the gated row interleave) and of its down projection.
_TRTLLM_BF16_UP_FIRST = FlashInferLayout("bf16", "trtllm", gated=True, up_first=True)

# Routes of the canonical names, which the loader serves under the same names: how bind() serves each tensor.
# Outside the decoder:
_MODEL_ROUTES = MappingProxyType(
    {
        "model.language_model.embed_tokens.weight": "dense",
        "model.language_model.norm.weight": "norm",
        "lm_head.weight": "dense",
    }
)
# In every decoder layer, by its name below the layer:
_LAYER_ROUTES = MappingProxyType(
    {
        "input_layernorm.weight": "norm",
        "pre_feedforward_layernorm.weight": "norm",
        "post_feedforward_layernorm.weight": "norm",
        "post_attention_residual_gate.gate": "residual_gate",
        "post_feedforward_residual_gate.gate": "residual_gate",
        **{f"self_attn.{leaf}.weight": "qkvg" for leaf in _QKVG_SHARDS},
        "self_attn.o_proj.weight": "dense",
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
                "mlp.gate.weight": "router",
                "mlp.gate.e_score_correction_bias": "router_bias",
                "mlp.pre_expert_proj.weight": "dense",
                "mlp.post_expert_proj.weight": "dense",
                "mlp.pre_expert_norm.weight": "norm",
                "mlp.experts.post_expert_norm.weight": "norm",
            }
        ),
    }
)


def route(name, mlp):
    """How the mapping binds one canonical Muse Spark tensor; raises for anything else.

    ``mlp`` (``"dense"`` or ``"moe"``) is the MLP kind of the decoder layer that holds the tensor, None outside the
    decoder layers. Every name the static policy excludes is unclassified. Pure.
    """
    match = _LAYER.fullmatch(name)
    if match is None:
        if mlp is not None:
            raise ValueError(f"a Muse Spark tensor outside the decoder layers has no MLP kind: {name}")
        kind = _MODEL_ROUTES.get(name)
    elif mlp not in _MLP_ROUTES:
        raise ValueError(f"unsupported Muse Spark MLP kind {mlp!r}: {name}")
    else:
        rest = match[2]
        kind = _LAYER_ROUTES.get(rest) or _MLP_ROUTES[mlp].get(rest)
        if kind is None and mlp == "moe" and _EXPERT.fullmatch(rest):
            kind = "expert"
    if kind is None:
        raise ValueError(f"unclassified Muse Spark canonical tensor: {name}")
    return kind


@cache
def _classes():
    """The SGLang classes the mapping admits, by exact type."""
    from sglang.kernels.ops.layernorm.muse_spark_norm_residual import FusedPostNormResidualPreNorm
    from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
    from sglang.srt.layers.muse_spark.muse_spark_layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
        QKVOGateParallelLinear,
    )
    from sglang.srt.models.muse_spark import MuseSpark1xDecoderLayer, MuseSpark1xMLP, MuseSpark1xMoE, MuseSpark1xRouter
    from sglang.srt.models.muse_spark_omni import MuseSparkForConditionalGeneration

    return {
        cls.__name__: cls
        for cls in (
            FusedPostNormResidualPreNorm,
            MuseSpark1xBf16Experts,
            MuseSpark1xDecoderLayer,
            MuseSpark1xMLP,
            MuseSpark1xMoE,
            MuseSpark1xNormPreservingResidualConnection,
            MuseSpark1xRMSNorm,
            MuseSpark1xRMSNormWithInputScale,
            MuseSpark1xRouter,
            MuseSparkForConditionalGeneration,
            QKVOGateParallelLinear,
        )
    }


class _Layer(NamedTuple):
    """One admitted decoder layer of the live model."""

    prefix: str
    mlp_kind: str  # "dense" or "moe"
    decoder: torch.nn.Module
    expert_layout: str | None  # a MoE layer's "trtllm" or "checkpoint"


class _FusedExpertWeights(NamedTuple):
    """MuseSpark1xBf16Experts' fused weights under the names FlashInfer's layouts read them by."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


def _require_muse_spark(model):
    """The identity gate: the exact wrapper class, serving a BF16 language model."""
    if type(model) is not _classes()["MuseSparkForConditionalGeneration"]:
        raise ValueError(f"unsupported Muse Spark runtime class: {type(model).__name__}")
    if model.encoder_only:
        raise ValueError("GPU delta requires the Muse Spark language model; encoder-only serving has no text weights")
    if model.nvfp4_checkpoint or model.int4_checkpoint:
        raise ValueError("GPU delta maps BF16 Muse Spark checkpoints only, not NVFP4 or int4 ones")


def _expert_layout(experts):
    """How ``prepare_trtllm_weights`` left the experts: ``"trtllm"``, shuffled for FlashInfer's TRT-LLM BF16 MoE,
    or ``"checkpoint"``, the [gate; up] rows the Triton and PyTorch paths read."""
    if type(experts) is not _classes()["MuseSpark1xBf16Experts"]:
        raise ValueError(f"GPU delta maps Muse Spark's BF16 experts only, not {type(experts).__name__}")
    if experts.use_megamoe:
        raise ValueError("GPU delta does not map Muse Spark's MegaMoE expert weights")
    # The finalize ops take the effective weight .to(torch.bfloat16), which returns a BF16 norm's own tensor.
    if experts.use_trtllm:
        finalize_weight = experts._post_expert_norm_weight
    elif "_finalize_norm_cache" in experts.__dict__:
        # The Triton path's, cached at its first forward.
        finalize_weight = experts._finalize_norm_cache[0]
    else:
        finalize_weight = None
    effective = experts.post_expert_norm._frozen_weight
    if finalize_weight is not None and _tensor_identity(finalize_weight) != _tensor_identity(effective):
        raise ValueError("Muse Spark's expert finalize must read the post-expert norm's effective weight in place")
    return "trtllm" if experts.use_trtllm else "checkpoint"


def _effective_weights(norm, dtypes):
    """MuseSpark1xRMSNorm.precompute_effective_weight, then each fused boundary's copy in its branch dtype."""
    gain_center = norm.gain_center

    def compute(weight):
        effective = weight + gain_center
        return (effective, *(effective.to(dtype) for dtype in dtypes))

    return compute


def _residual_coefficients(residual, boundary_copies):
    """MuseSpark1xNormPreservingResidualConnection.precompute_coefficients, then the flat copies of
    ``boundary_copies`` fused boundaries."""

    def compute(gate):
        alpha, beta = residual._coefficients(gate.view(1, -1).float() / residual.temperature)
        return (alpha, beta, *(value.reshape(-1) for _ in range(boundary_copies) for value in (alpha, beta)))

    return compute


def _router_split(weight):
    # MuseSpark1xRouter._split2_parts, as refresh_split2 copies them into its contiguous transposes.
    high = weight.to(torch.bfloat16)
    low = (weight - high.to(torch.float32)).to(torch.bfloat16)
    return high.t().contiguous(), low.t().contiguous()


class MuseSparkMapping:
    """Muse Spark's language model with BF16 experts, served by ``MuseSparkForConditionalGeneration``."""

    # load_weights' gate/up shards of the dense MLPs; ParameterBindings binds them.
    stacked_params = (("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1))

    def __init__(self, model, shadows):
        _require_muse_spark(model)
        # The loader serves every language-model tensor in its checkpoint dtype, so no binding uses shadows.
        self.parameters = ParameterBindings(model, self.stacked_params, shadows)
        self.consumers = []
        self.derived = []
        self._config = model.config
        boundary_class = _classes()["FusedPostNormResidualPreNorm"]
        # The fused post-norm, residual and pre-norm boundaries, which keep copies of their norms' and gate's values.
        self._boundaries = []
        for decoder in model.model.language_model.layers:
            for boundary in (decoder._fused_attn_boundary, decoder._fused_ffn_boundary):
                if boundary is None:
                    continue
                if type(boundary) is not boundary_class:
                    raise ValueError(f"unsupported Muse Spark norm boundary {type(boundary).__name__}")
                self._boundaries.append(boundary)
        self._layers = {}  # layer index -> _Layer
        self._images = []  # (canonical name, its live source, the snapshot of the values computed from it, compute)

    def bind(self, name, meta):
        match = _LAYER.fullmatch(name)
        layer = None if match is None else self._layer(int(match[1]))
        kind = route(name, None if layer is None else layer.mlp_kind)
        if kind in ("dense", "gate_up"):
            return self.parameters.bind(name, meta, name)
        elif kind == "norm":
            return self._bind_norm(name, meta)
        elif kind == "residual_gate":
            return self._bind_residual_gate(name, meta)
        elif kind == "qkvg":
            return self._bind_qkvg(name, meta, layer, name.rsplit(".", 2)[-2])
        elif kind == "router":
            return self._bind_router(name, meta)
        elif kind == "router_bias":
            router = self._module(name, "MuseSpark1xRouter")
            bias = router.e_score_correction_bias
            # FusedMuseSparkRouting reads the buffer itself and rebuilds when the module's buffer is replaced.
            self.consumers.append(ConsumerSnapshot(lambda: (router.e_score_correction_bias,)))
            return self.parameters.bind_view(name, meta, router, bias, _full_slices(meta["shape"]))
        elif kind == "expert":
            expert = _EXPERT.fullmatch(match[2])
            return self._bind_expert(name, meta, layer, int(expert[1]), expert[2])
        else:
            raise ValueError(f"unsupported Muse Spark route {kind}: {name}")

    def finish(self):
        for name, source, snapshot, compute in self._images:
            self.derived.append(ComputedImage(name, (source,), tuple(snapshot.objects), compute))

    def _layer(self, index):
        """Admit one decoder layer at its first canonical tensor and pin its experts."""
        if index not in self._layers:
            prefix = f"model.language_model.layers.{index}"
            decoder = self.parameters.module(prefix)
            classes = _classes()
            if type(decoder) is not classes["MuseSpark1xDecoderLayer"]:
                raise ValueError(f"unsupported Muse Spark decoder layer {type(decoder).__name__}: {prefix}")
            mlp, expert_layout = decoder.mlp, None
            if type(mlp) is classes["MuseSpark1xMoE"]:
                mlp_kind, experts = "moe", mlp.experts
                expert_layout = _expert_layout(experts)
                self.consumers.append(
                    ConsumerSnapshot(
                        lambda: (experts.use_trtllm, experts.use_triton, experts._post_expert_norm_weight)
                    )
                )
            elif type(mlp) is classes["MuseSpark1xMLP"]:
                mlp_kind = "dense"
            else:
                raise ValueError(f"unsupported Muse Spark MLP {type(mlp).__name__}: {prefix}")
            self._layers[index] = _Layer(prefix, mlp_kind, decoder, expert_layout)
        return self._layers[index]

    def _module(self, name, *class_names):
        """The module owning canonical ``name``, of one of the exact ``class_names``."""
        module = self.parameters.module(name.rsplit(".", 1)[0])
        if not any(type(module) is _classes()[class_name] for class_name in class_names):
            raise ValueError(f"unsupported Muse Spark module {type(module).__name__}: {name}")
        return module

    def _bind_norm(self, name, meta):
        """A norm weight, and its effective weight and boundary copies as one computed image."""
        norm = self._module(name, "MuseSpark1xRMSNorm", "MuseSpark1xRMSNormWithInputScale")
        binding = self.parameters.bind(name, meta, name)
        post_boundaries = [boundary for boundary in self._boundaries if boundary._post_norm is norm]
        pre_boundaries = [boundary for boundary in self._boundaries if boundary._pre_norm is norm]
        snapshot = ConsumerSnapshot(
            lambda: (
                norm._frozen_weight,
                *(boundary._cc_post_weight for boundary in post_boundaries),
                *(boundary._cc_pre_weight for boundary in pre_boundaries),
            )
        )
        self.consumers.append(snapshot)
        dtypes = [copy.dtype for copy in snapshot.objects[1:]]
        self._images.append((name, norm.weight, snapshot, _effective_weights(norm, dtypes)))
        return binding

    def _bind_residual_gate(self, name, meta):
        """A residual gate, and its coefficients and boundary copies as one computed image."""
        residual = self._module(name, "MuseSpark1xNormPreservingResidualConnection")
        binding = self.parameters.bind(name, meta, name)
        boundaries = [boundary for boundary in self._boundaries if boundary._residual_add is residual]
        snapshot = ConsumerSnapshot(
            lambda: (
                residual._alpha,
                residual._beta,
                *(copy for boundary in boundaries for copy in (boundary._cc_alpha, boundary._cc_beta)),
            )
        )
        self.consumers.append(snapshot)
        self._images.append((name, residual.gate, snapshot, _residual_coefficients(residual, len(boundaries))))
        return binding

    def _bind_router(self, name, meta):
        """The FP32 router, which the router GEMMs read in place; SM90's split copies are a computed image."""
        router = self._module(name, "MuseSpark1xRouter")
        binding = self.parameters.bind(name, meta, name)
        # MuseSpark1xRouter builds its split on the first large SM90 prefill; one built later forces readmission.
        snapshot = ConsumerSnapshot(lambda: (router._split2_hi, router._split2_lo))
        self.consumers.append(snapshot)
        if router._split2_hi is not None:
            self._images.append((name, router.weight, snapshot, _router_split))
        return binding

    def _bind_qkvg(self, name, meta, layer, leaf):
        """``QKVOGateParallelLinear.weight_loader`` for one checkpoint shard, after load_weights interleaves the
        rotary halves of each q and k head: canonical row (head, half, i) serves at row (head, i, half)."""
        from sglang.srt.distributed.utils import get_group_rank_size

        module = layer.decoder.self_attn.qkv_proj
        if type(module) is not _classes()["QKVOGateParallelLinear"]:
            raise ValueError(f"unsupported Muse Spark attention projection {type(module).__name__}: {name}")
        _require_unquantized(name, module)
        param = module.weight
        if _output_dim(param) != 0:
            raise ValueError("Muse Spark attention projection is not sharded along its rows")
        shard = _QKVG_SHARDS[leaf]
        q_rows, kv_rows = module.q_proj_shard_size, module.kv_proj_shard_size
        offset, rows = {
            "q": (0, q_rows),
            "k": (q_rows, kv_rows),
            "v": (q_rows + kv_rows, kv_rows),
            "g": (q_rows + 2 * kv_rows, q_rows),
        }[shard]
        rank, _ = get_group_rank_size(module.tp_group)
        start = (rank if shard in ("q", "g") else rank // module.num_kv_head_replicas) * rows
        view = param.narrow(0, offset, rows)
        if shard in ("q", "k"):
            heads = self._config.num_attention_heads if shard == "q" else self._config.num_key_value_heads
            half = meta["shape"][0] // heads // 2
            if rows % (2 * half):
                raise ValueError(f"Muse Spark attention shard does not hold whole heads: {name}")
            view = view.view(rows // (2 * half), half, 2, -1).transpose(1, 2)
        slices = _full_slices(meta["shape"])
        slices[0] = [start, start + rows]
        binding = self.parameters.bind_view(name, meta, module, view, slices)
        self.parameters.aliases.declare(name, f"{layer.prefix}.self_attn.qkv_proj.weight", binding, param)
        return binding

    def _bind_expert(self, name, meta, layer, expert, projection):
        """``MuseSpark1xBf16Experts.weight_loader`` for one local expert's projection, in the layout
        ``prepare_trtllm_weights`` left it."""
        experts = layer.decoder.mlp.experts
        local = expert - experts.local_expert_offset
        if not 0 <= local < experts.num_local_experts:
            self.parameters.excluded[name] = REMOTE_EXPERT
            return None
        intermediate, hidden = experts.intermediate_size, experts.expert_hidden_size
        shape = (hidden, intermediate) if projection == "down" else (intermediate, hidden)
        if tuple(meta["shape"]) != shape:
            raise ValueError(f"Muse Spark expert geometry differs from its loader: {name}")
        if layer.expert_layout == "trtllm":
            weights = _FusedExpertWeights(experts.gate_up_proj, experts.down_proj)
            segments = _TRTLLM_BF16_UP_FIRST.expert_destinations(
                weights, local, projection, "weight", shape, meta["dtype"]
            )
            storage = {_storage_key(view): view for segment in segments for view in segment.destinations}
            return TensorBinding(
                name,
                meta["dtype"],
                shape,
                _full_slices(shape),
                tuple(storage.values()),
                segments=segments,
                format=_TRTLLM_BF16_UP_FIRST.format,
            )
        # w1 (gate) fills rows [0, I) of gate_up_proj[local], w3 (up) rows [I, 2I); w2 (down) fills down_proj[local].
        if projection == "down":
            view = experts.down_proj[local]
        else:
            view = experts.gate_up_proj[local].narrow(0, 0 if projection == "gate" else intermediate, intermediate)
        return self.parameters.bind_view(name, meta, experts, view, _full_slices(shape))
