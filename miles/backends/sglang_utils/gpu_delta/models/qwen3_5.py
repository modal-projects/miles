"""GPU-delta bindings for SGLang's Qwen3.5 model implementation (CUDA receiver only).

Mirrors ``sglang/srt/models/qwen3_5.py``: ``Qwen3_5ForConditionalGeneration`` and
``Qwen3_5MoeForConditionalGeneration``, their text model's ``load_weights`` and the Gated DeltaNet's
post-load packing:

- full attention: ``q_proj`` (query rows with their output gate), ``k_proj`` and ``v_proj`` as the
  shards of ``qkv_proj`` (``QKVParallelLinear.weight_loader``);
- Gated DeltaNet: ``in_proj_qkv`` as the first three shards of ``in_proj_qkvz`` and ``in_proj_z`` as
  its fourth, ``in_proj_b``/``in_proj_a`` as the two of ``in_proj_ba`` (both views of one packed
  buffer once loaded in BF16 on CUDA), ``conv1d`` sliced per q/k/v group
  (``mamba_v2_sharded_weight_loader``), and the head-sharded ``A_log`` and ``dt_bias``;
- dense and MoE MLPs: gate/up shards of ``gate_up_proj``, routed experts, the shared expert and its gate;
- each ``GemmaRMSNorm``'s ``gemma_weight`` (its weight plus one, in the weight's dtype), recomputed
  after every apply.

The checkpoint's vision tower and bundled MTP layer are static (the checkpoint policy in
``miles/utils/gpu_delta/publication.py``): Miles trains and exports the text model only. Admitted
architectures are listed only in ``models/__init__.py``.
"""

from __future__ import annotations

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ComputedImage,
    ConsumerSnapshot,
    ParameterBindings,
    TensorBinding,
    _direct_binding,
    _full_slices,
    _merged_partition,
    _require_canonical_dtype,
    _require_slices,
    _require_unquantized,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import DestinationSegment, _byte_view

_LANGUAGE_MODEL = "model.language_model."
# load_weights' shards of the fused Gated DeltaNet projections: checkpoint leaf -> (fused leaf, shards).
_GDN_SHARDS = {
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", (3,)),
    "in_proj_b": ("in_proj_ba", (0,)),
    "in_proj_a": ("in_proj_ba", (1,)),
}
_QKV_SHARDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}


def served_name(name):
    """The text model's parameter for checkpoint tensor ``name``, as ``load_weights`` renames it."""
    if name.startswith(_LANGUAGE_MODEL):
        name = "model." + name.removeprefix(_LANGUAGE_MODEL)
    return name.replace(".self_attn.", ".")


def _group_rank(module):
    from sglang.srt.distributed.utils import get_group_rank_size

    return get_group_rank_size(module.tp_group)


def _segmented_binding(name, meta, param, pieces):
    """One canonical tensor whose row ranges a loader writes to separate rows of ``param``.

    ``pieces`` are ``(canonical rows, local rows)`` pairs, each ``(start, stop)`` along dim 0.
    """
    _require_canonical_dtype(meta)
    segments = []
    for (start, stop), (local_start, local_stop) in pieces:
        slices = _full_slices(meta["shape"])
        slices[0] = [start, stop]
        _require_slices(name, meta, slices)
        view = param[local_start:local_stop]
        if tuple(view.shape) != tuple(high - low for low, high in slices):
            raise ValueError(f"loader geometry differs for {name}")
        segments.append(DestinationSegment(tuple(tuple(axis) for axis in slices), (_byte_view(view),)))
    return TensorBinding(
        name,
        meta["dtype"],
        tuple(meta["shape"]),
        _full_slices(meta["shape"]),
        (param,),
        encoding=meta["encoding"],
        segments=tuple(segments),
        format=meta.get("format"),
    )


class Qwen3_5Mapping:
    """The Qwen3.5 family (Qwen3.6 MoE, Qwen3.8 dense) on SGLang's Qwen3.5 implementation."""

    # load_weights' gate/up shards of dense and shared-expert MLPs; ParameterBindings binds them.
    stacked_params = (("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1))

    def __init__(self, model, shadows):
        # The loader serves every text-model tensor in its checkpoint dtype, so no binding uses shadows.
        self.parameters = ParameterBindings(model, self.stacked_params, shadows)
        self.consumers = []
        self.derived = []
        self._gemma_norms = {}
        self._packed_in_proj = set()

    def bind(self, name, meta):
        target_name = served_name(name)
        module_name, leaf = target_name.rsplit(".", 1)
        prefix, _, stem = module_name.rpartition(".")
        if stem in _QKV_SHARDS:
            return self._bind_qkv(name, meta, f"{prefix}.qkv_proj", _QKV_SHARDS[stem], leaf)
        if stem in _GDN_SHARDS:
            fused, shards = _GDN_SHARDS[stem]
            return self._bind_gdn_projection(name, meta, prefix, fused, shards, leaf)
        if stem == "conv1d":
            return self._bind_conv1d(name, meta, prefix, leaf)
        if leaf in ("A_log", "dt_bias"):
            return self._bind_heads(name, meta, module_name, leaf)
        binding = self.parameters.bind(name, meta, target_name)
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        module = self.parameters.modules.get(module_name)
        if type(module) is GemmaRMSNorm and leaf == "weight":
            self._gemma_norms[module_name] = module
            self.consumers.append(ConsumerSnapshot(lambda norm=module: (norm.gemma_weight,)))
        return binding

    def finish(self):
        for prefix, norm in self._gemma_norms.items():
            # GemmaRMSNorm.refresh_gemma_weight: copy_(weight).add_(1.0), in the weight's dtype.
            self.derived.append(
                ComputedImage(f"{prefix}.gemma_weight", (norm.weight,), (norm.gemma_weight,), lambda w: (w + 1.0,))
            )

    def _bind_qkv(self, name, meta, target_name, shard, leaf):
        """``QKVParallelLinear.weight_loader`` for one checkpoint shard: query rows by the TP rank, key and value
        rows by the KV rank over its replicas."""
        if leaf != "weight":
            raise ValueError(f"unclassified attention projection tensor: {name}")
        from sglang.srt.distributed.utils import get_group_rank_size
        from sglang.srt.layers.linear import QKVParallelLinear

        module = self.parameters.module(target_name)
        if type(module) is not QKVParallelLinear or module.use_presharded_weights:
            raise ValueError(f"unclassified attention projection: {type(module).__name__}")
        _require_unquantized(name, module)
        param = module.weight
        if (
            getattr(param, "output_dim", None) != 0
        ):  # config-access-exempt: SGLang parameter loaders attach the column-parallel sharding axis.
            raise ValueError("attention projection is not sharded along its rows")
        heads, kv_heads, head = module.num_heads, module.num_kv_heads, module.head_size
        offset, rows = {
            "q": (0, heads * head),
            "k": (heads * head, kv_heads * head),
            "v": ((heads + kv_heads) * head, kv_heads * module.v_head_size),
        }[shard]
        if shard == "q":
            rank, _ = get_group_rank_size(module.tp_group)
        else:
            kv_rank, _ = get_group_rank_size(module.kv_tp_group)
            rank = kv_rank // module.num_kv_head_replicas
        binding = _segmented_binding(name, meta, param, [((rank * rows, (rank + 1) * rows), (offset, offset + rows))])
        self.parameters.aliases.declare(name, target_name, binding, param)
        return binding

    def _bind_gdn_projection(self, name, meta, prefix, fused, shards, leaf):
        """``MergedColumnParallelLinear`` shards of a fused Gated DeltaNet projection; ``in_proj_qkv`` fills three
        consecutive ones from its rows, as the packed-checkpoint loader splits it."""
        if leaf != "weight":
            raise ValueError(f"unclassified Gated DeltaNet projection tensor: {name}")
        attn = self.parameters.module(prefix)
        module = {"in_proj_qkvz": attn.in_proj_qkvz, "in_proj_ba": attn.in_proj_ba}[fused]
        _require_unquantized(name, module)
        if (
            getattr(module.weight, "output_dim", None) != 0
        ):  # config-access-exempt: SGLang parameter loaders attach the column-parallel sharding axis.
            raise ValueError("Gated DeltaNet projection is not sharded along its rows")
        # MergedColumnParallelLinear.weight_loader: shard s fills local rows from sum(sizes[:s]) // tp with its
        # canonical rows [rank * width, (rank + 1) * width), width = sizes[s] // tp.
        rank, size, sizes = _merged_partition(module)
        pieces, start = [], 0
        for shard in shards:
            width = sizes[shard] // size
            local = sum(sizes[:shard]) // size
            pieces.append(((start + rank * width, start + (rank + 1) * width), (local, local + width)))
            start += sizes[shard]
        if start != meta["shape"][0]:
            raise ValueError(f"Gated DeltaNet projection geometry differs from its loader: {name}")
        binding = _segmented_binding(name, meta, module.weight, pieces)
        packed = attn._fused_in_proj_weight
        if packed is None:
            self.parameters.aliases.declare(name, f"{prefix}.{fused}", binding, module.weight)
        else:
            # Once loaded in BF16 on CUDA, in_proj_qkvz and in_proj_ba are row views of one packed GEMM weight.
            if prefix not in self._packed_in_proj:
                self._packed_in_proj.add(prefix)
                self.consumers.append(
                    ConsumerSnapshot(
                        lambda attn=attn: (
                            attn._fused_in_proj_weight,
                            attn.in_proj_qkvz.weight,
                            attn.in_proj_ba.weight,
                        )
                    )
                )
            self.parameters.aliases.declare(name, f"{prefix}._fused_in_proj_weight", binding, packed)
        return binding

    def _bind_conv1d(self, name, meta, prefix, leaf):
        """``mamba_v2_sharded_weight_loader`` over the query, key and value channel groups."""
        if leaf != "weight":
            raise ValueError(f"unclassified Gated DeltaNet convolution tensor: {name}")
        attn = self.parameters.module(prefix)
        conv = attn.conv1d
        _require_unquantized(name, conv)
        rank, size = _group_rank(conv)
        groups = (attn.key_dim, attn.key_dim, attn.value_dim)
        if any(group % size for group in groups) or sum(groups) != meta["shape"][0]:
            raise ValueError(f"Gated DeltaNet convolution geometry differs from its loader: {name}")
        pieces, start, local = [], 0, 0
        for group in groups:
            width = group // size
            pieces.append(((start + rank * width, start + (rank + 1) * width), (local, local + width)))
            start, local = start + group, local + width
        return _segmented_binding(name, meta, conv.weight, pieces)

    def _bind_heads(self, name, meta, prefix, leaf):
        """``A_log`` and ``dt_bias``: one value per value head, sharded with the projections' TP group."""
        attn = self.parameters.module(prefix)
        param = {"A_log": attn.A_log, "dt_bias": attn.dt_bias}[leaf]
        rank, size = _group_rank(attn.conv1d)
        heads = param.shape[0]
        if heads * size != meta["shape"][0]:
            raise ValueError(f"Gated DeltaNet head geometry differs from its loader: {name}")
        slices = _full_slices(meta["shape"])
        slices[0] = [rank * heads, (rank + 1) * heads]
        return _direct_binding(name, meta, param, slices, live_name=f"{prefix}.{leaf}")
