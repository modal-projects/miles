"""CPU checks of the Qwen3.5 mapping against SGLang's own loaders.

Each binding is applied to a sentinel-filled parameter and compared with what the real SGLang loader writes there
from the same checkpoint tensor, on one rank of a TP group, as the shared loader-view tests do.
"""

from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers import linear as sglang_linear
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear

from miles.backends.sglang_utils.gpu_delta.models.qwen3_5 import Qwen3_5Mapping, served_name
from miles.utils.gpu_delta import publication

_DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32"}


def _sglang_linear(cls, tp_size, tp_rank, *args, **kwargs):
    # A real SGLang linear on one TP rank without a distributed runtime, as SGLang resolves an
    # unsharded partition before distributed initialization.
    group = None if tp_size == 1 else sglang_linear._LogicalGroup(tp_rank, tp_size)
    with patch.object(sglang_linear, "_resolve_linear_group", lambda parallel_group: group):
        return cls(*args, params_dtype=torch.bfloat16, **kwargs)


def _meta(tensor):
    shape = list(tensor.shape)
    return {"dtype": _DTYPES[tensor.dtype], "shape": shape, "encoding": publication.canonical_encoding(shape)}


@torch.no_grad()
def _bound_image(param, binding, canonical):
    """What applying ``binding`` writes into a sentinel-filled ``param`` from ``canonical``."""
    param.data.fill_(-7)
    for piece in binding.apply_bindings():
        selected = canonical[tuple(slice(start, stop) for start, stop in piece.slices)]
        if piece.is_raw:
            piece.storage[0].copy_(selected)
        else:
            (destination,) = piece.destinations
            destination.copy_(selected.contiguous().view(torch.uint8).reshape(destination.shape))
    return param.data.clone()


@torch.no_grad()
def _loader_image(param, load, *args):
    param.data.fill_(-7)
    load(*args)
    return param.data.clone()


def _model(**layers):
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for index in range(len(layers)):
        root.model.layers.append(layers[f"layer{index}"])
    return root


_REAL_NAMES = {
    # One tensor of each kind in Qwen3.6-35B-A3B NVFP4 and Qwen3.8-27B BF16.
    "model.language_model.layers.3.self_attn.q_proj.weight": "model.layers.3.q_proj.weight",
    "model.language_model.layers.3.self_attn.q_norm.weight": "model.layers.3.q_norm.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.A_log": "model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.mlp.experts.7.down_proj.weight_scale_2": (
        "model.layers.0.mlp.experts.7.down_proj.weight_scale_2"
    ),
    "model.language_model.layers.0.mlp.shared_expert_gate.weight": "model.layers.0.mlp.shared_expert_gate.weight",
    "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.language_model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def test_text_tensors_take_the_loaders_names_and_the_policy_keeps_vision_and_mtp_static():
    for name, served in _REAL_NAMES.items():
        assert served_name(name) == served
    for architecture in ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"):
        policy = publication.static_policy({"architectures": [architecture], "text_config": {"num_hidden_layers": 40}})
        for name in _REAL_NAMES:
            assert publication.static_tensor_reason(name, policy) is None
        for name, reason in {
            "model.visual.blocks.3.attn.qkv.weight": publication.FROZEN_MODULE,
            "model.visual.merger.linear_fc1.bias": publication.FROZEN_MODULE,
            "model.visual.pos_embed.weight": publication.FROZEN_MODULE,
            "mtp.layers.0.mlp.experts.3.up_proj.weight": publication.STATIC_DRAFT,
            "mtp.fc.weight": publication.STATIC_DRAFT,
            "mtp.pre_fc_norm_hidden.weight": publication.STATIC_DRAFT,
        }.items():
            assert publication.static_tensor_reason(name, policy) == reason


@pytest.mark.parametrize(
    "tp_size,tp_rank,kv_heads", [(1, 0, 2), (2, 1, 2), (4, 3, 2), (4, 2, 4)], ids=["tp1", "tp2", "kv_rep", "kv_split"]
)
def test_query_key_and_value_bind_the_rows_the_qkv_loader_fills(tp_size, tp_rank, kv_heads):
    # Gated attention: SGLang declares twice the query heads, the output gate's rows beside the query's.
    heads, head_dim, hidden = 4, 4, 8
    qkv = _sglang_linear(QKVParallelLinear, tp_size, tp_rank, hidden, head_dim, heads * 2, kv_heads, bias=False)
    layer = torch.nn.Module()
    layer.qkv_proj = qkv
    mapping = Qwen3_5Mapping(_model(layer0=layer), None)
    for leaf, shard, rows in (
        ("q_proj", "q", heads * 2 * head_dim),
        ("k_proj", "k", kv_heads * head_dim),
        ("v_proj", "v", kv_heads * head_dim),
    ):
        canonical = torch.randn(rows, hidden).bfloat16()
        binding = mapping.bind(f"model.language_model.layers.0.self_attn.{leaf}.weight", _meta(canonical))
        expected = _loader_image(qkv.weight, qkv.weight_loader, qkv.weight, canonical, shard)
        assert torch.equal(_bound_image(qkv.weight, binding, canonical).view(torch.int16), expected.view(torch.int16))


def _gdn(tp_size, tp_rank, key_dim=8, value_dim=16, heads=4, hidden=6, kernel=4):
    attn = torch.nn.Module()
    attn.key_dim, attn.value_dim = key_dim, value_dim
    attn.in_proj_qkvz = _sglang_linear(
        MergedColumnParallelLinear, tp_size, tp_rank, hidden, [key_dim, key_dim, value_dim, value_dim], bias=False
    )
    attn.in_proj_ba = _sglang_linear(MergedColumnParallelLinear, tp_size, tp_rank, hidden, [heads, heads], bias=False)
    attn.conv1d = _sglang_linear(ColumnParallelLinear, tp_size, tp_rank, kernel, key_dim * 2 + value_dim, bias=False)
    attn.conv1d.weight.data = attn.conv1d.weight.data.unsqueeze(1)
    attn.A_log = torch.nn.Parameter(torch.empty(heads // tp_size))
    attn.dt_bias = torch.nn.Parameter(torch.empty(heads // tp_size, dtype=torch.bfloat16))
    attn._fused_in_proj_weight = None
    layer = torch.nn.Module()
    layer.linear_attn = attn
    return attn, Qwen3_5Mapping(_model(layer0=layer), None)


def _packed_loader(module, canonical, shards):
    # Qwen3_5GatedDeltaNet's packed-checkpoint loader: split the rows by the shards' full sizes, then
    # hand each part to the merged projection's own int-shard loader.
    sizes = [module.output_sizes[shard] for shard in shards]
    for shard, part in zip(shards, canonical.split(sizes), strict=True):
        module.weight_loader(module.weight, part, shard)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 2)])
@pytest.mark.parametrize("packed", [False, True], ids=["separate", "packed"])
def test_gated_deltanet_projections_bind_the_rows_the_packed_loader_fills(tp_size, tp_rank, packed):
    attn, mapping = _gdn(tp_size, tp_rank)
    if packed:
        # The post-load packing on CUDA in BF16: both projections become row views of one GEMM weight.
        fused = torch.cat([attn.in_proj_qkvz.weight.data, attn.in_proj_ba.weight.data]).contiguous()
        width = attn.in_proj_qkvz.weight.shape[0]
        attn.in_proj_qkvz.weight.data, attn.in_proj_ba.weight.data = fused[:width], fused[width:]
        attn._fused_in_proj_weight = fused
    for leaf, fused_leaf, shards in (
        ("in_proj_qkv", "in_proj_qkvz", (0, 1, 2)),
        ("in_proj_z", "in_proj_qkvz", (3,)),
        ("in_proj_b", "in_proj_ba", (0,)),
        ("in_proj_a", "in_proj_ba", (1,)),
    ):
        module = getattr(attn, fused_leaf)
        canonical = torch.randn(sum(module.output_sizes[shard] for shard in shards), 6).bfloat16()
        binding = mapping.bind(f"model.language_model.layers.0.linear_attn.{leaf}.weight", _meta(canonical))
        expected = _loader_image(module.weight, _packed_loader, module, canonical, shards)
        actual = _bound_image(module.weight, binding, canonical)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), leaf
    # The interface pins consumers while binding; finish() only adds derived images.
    consumers = len(mapping.consumers)
    mapping.finish()
    assert consumers == len(mapping.consumers) == (1 if packed else 0)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 3)])
def test_convolution_binds_each_query_key_and_value_group_as_its_loader_slices_it(tp_size, tp_rank):
    attn, mapping = _gdn(tp_size, tp_rank)
    conv = attn.conv1d
    canonical = torch.randn(attn.key_dim * 2 + attn.value_dim, 1, 4).bfloat16()
    loader = mamba_v2_sharded_weight_loader(
        [(attn.key_dim, 0, False), (attn.key_dim, 0, False), (attn.value_dim, 0, False)], tp_size, tp_rank
    )
    binding = mapping.bind("model.language_model.layers.0.linear_attn.conv1d.weight", _meta(canonical))
    expected = _loader_image(conv.weight, loader, conv.weight, canonical)
    assert torch.equal(_bound_image(conv.weight, binding, canonical).view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (4, 1)])
def test_head_vectors_bind_this_ranks_heads(tp_size, tp_rank):
    attn, mapping = _gdn(tp_size, tp_rank)
    for leaf, dtype in (("A_log", torch.float32), ("dt_bias", torch.bfloat16)):
        param = getattr(attn, leaf)
        canonical = torch.randn(4, dtype=dtype)
        binding = mapping.bind(f"model.language_model.layers.0.linear_attn.{leaf}", _meta(canonical))
        heads = param.shape[0]
        # sharded_weight_loader(0): this rank's heads.
        expected = _loader_image(param, param.copy_, canonical[tp_rank * heads : (tp_rank + 1) * heads])
        assert torch.equal(_bound_image(param, binding, canonical), expected)


def test_a_gemma_norm_serves_its_weight_plus_one_recomputed_like_its_loader():
    layer = torch.nn.Module()
    layer.input_layernorm = GemmaRMSNorm(8).to(torch.bfloat16)
    norm = layer.input_layernorm
    mapping = Qwen3_5Mapping(_model(layer0=layer), None)
    canonical = torch.randn(8).bfloat16()
    norm.weight.weight_loader(norm.weight, canonical)
    binding = mapping.bind("model.language_model.layers.0.input_layernorm.weight", _meta(canonical))
    assert binding.storage == (norm.weight,) and len(mapping.consumers) == 1
    mapping.finish()
    (image,) = mapping.derived
    image.verify()
    updated = torch.randn(8).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(updated)
    image.refresh(torch.tensor(True))
    reference = GemmaRMSNorm(8).to(torch.bfloat16)
    reference.weight.weight_loader(reference.weight, updated)
    assert torch.equal(norm.gemma_weight.view(torch.int16), reference.gemma_weight.view(torch.int16))
    (consumer,) = mapping.consumers
    assert consumer.objects[0] is norm.gemma_weight


def test_an_unknown_text_tensor_is_unclassified():
    layer = torch.nn.Module()
    layer.qkv_proj = _sglang_linear(QKVParallelLinear, 1, 0, 8, 4, 8, 2, bias=False)
    mapping = Qwen3_5Mapping(_model(layer0=layer), None)
    with pytest.raises(ValueError, match="unclassified attention projection tensor"):
        mapping.bind("model.language_model.layers.0.self_attn.q_proj.bias", _meta(torch.zeros(8)))
