"""CPU checks of the GLM-5.3-Flash mapping against SGLang's own loaders.

A two-layer model of real SGLang modules on one rank of a TP group (a KDA layer with a dense MLP, a DSA layer
with the MoE router and shared expert) loads a base checkpoint through ``Glm5NextForConditionalGeneration
.load_weights``. Its bindings then XOR the base into a second checkpoint, as an update does, and every served
parameter, absorbed cache and canonical shadow must equal what the loader writes from that checkpoint.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.layers import linear as sglang_linear
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import glm5_next as sglang_glm5_next
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP, DeepseekV2MoE, MoEGate
from sglang.srt.true_on_policy import config as true_on_policy

from miles.backends.sglang_utils.gpu_delta import layout
from miles.backends.sglang_utils.gpu_delta.bindings import (
    CanonicalShadows,
    ComputedImage,
    batched_column_target,
    prepare_raw_xor,
    repeated_column_target,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _storage_key
from miles.backends.sglang_utils.gpu_delta.checkpoint import CanonicalCheckpoint
from miles.backends.sglang_utils.gpu_delta.models.glm5_next import Glm5NextMapping, route, served_name
from miles.utils.gpu_delta import publication

_ARCHITECTURE = "Glm5NextForConditionalGeneration"
_HIDDEN = 32
# KDA: heads, head and projection widths, convolution kernel.
_HEADS, _HEAD_DIM, _CONV = 4, 8, 4
_PROJECTION = _HEADS * _HEAD_DIM
# DSA: MLA heads without RoPE (GLM-5.3-Flash's qk_rope_head_dim is 0), LoRA ranks, k-pool indexer.
_MLA_HEADS, _Q_LORA, _KV_LORA, _NOPE, _V = 4, 16, 8, 8, 8
_INDEX_HEADS, _INDEX_DIM, _KPOOL = 4, 8, 4
_INTERMEDIATE, _EXPERTS, _SHARED, _HC_MULT = 16, 8, 16, 2
_KDA, _DSA = "model.language_model.layers.0", "model.language_model.layers.1"


def _text_config():
    return Glm5NextTextConfig(
        architectures=[_ARCHITECTURE],
        vocab_size=64,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        moe_intermediate_size=_SHARED,
        num_hidden_layers=2,
        num_attention_heads=_MLA_HEADS,
        n_routed_experts=_EXPERTS,
        first_k_dense_replace=1,
        q_lora_rank=_Q_LORA,
        kv_lora_rank=_KV_LORA,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=0,
        v_head_dim=_V,
        mhc=True,
        hc_mult=_HC_MULT,
        linear_attn_config={
            "num_heads": _HEADS,
            "head_dim": _HEAD_DIM,
            "short_conv_kernel_size": _CONV,
            "kda_layers": [0],
            "full_attn_layers": [1],
        },
        index_head_dim=_INDEX_DIM,
        index_topk=8,
        index_n_heads=_INDEX_HEADS,
        index_kpool=_KPOOL,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
    )


@contextmanager
def _rank(tp_size, tp_rank, lora=False):
    """SGLang building its BF16 model on one rank of a TP group, before distributed initialization: every
    partition group resolves to a logical group, as SGLang resolves it without a distributed runtime, and the
    server runs without true-on-policy norms, its default."""
    group = None if tp_size == 1 else sglang_linear._LogicalGroup(tp_rank, tp_size)
    kernels = SimpleNamespace(kernel=SimpleNamespace(dsa_paged_mqa_logits_backend="auto"))
    with ExitStack() as stack:
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        stack.enter_context(patch.object(sglang_linear, "_resolve_linear_group", lambda parallel_group: group))
        stack.enter_context(patch.object(sglang_glm5_next, "linear_attn_tp_size", lambda: tp_size))
        stack.enter_context(patch.object(sglang_glm5_next, "linear_attn_parallel_group", lambda: "attn_tp"))
        lora_config = SimpleNamespace(enable_lora=lora, lora_paths=None)
        stack.enter_context(patch.object(sglang_glm5_next, "get_lora", lambda: lora_config))
        stack.enter_context(patch.object(dsa_indexer_kpool, "get_exec", lambda: kernels))
        stack.enter_context(patch.object(true_on_policy, "is_true_on_policy_enabled", lambda: False))
        yield


def _bare(cls, **children):
    # An exact SGLang class without its runtime-bound constructor; the children are real SGLang modules.
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    for name, child in children.items():
        setattr(module, name, child)
    return module


def _kda_attention(config):
    return sglang_glm5_next.Glm5NextLinearAttention(
        layer_idx=0, hidden_size=_HIDDEN, config=config, prefix="model.layers.0.self_attn"
    )


def _dsa_attention(config):
    """DeepseekV2AttentionMLA's weights as its constructor builds them for GLM-5.3-Flash, with the real indexer."""
    attn = _bare(
        DeepseekV2AttentionMLA,
        fused_qkv_a_proj_with_mqa=ReplicatedLinear(_HIDDEN, _Q_LORA + _KV_LORA, bias=False),
        q_a_layernorm=RMSNorm(_Q_LORA),
        kv_a_layernorm=RMSNorm(_KV_LORA),
        q_b_proj=ColumnParallelLinear(_Q_LORA, _MLA_HEADS * _NOPE, bias=False, parallel_group="attn_tp"),
        kv_b_proj=ColumnParallelLinear(_KV_LORA, _MLA_HEADS * (_NOPE + _V), bias=False, parallel_group="attn_tp"),
        o_proj=RowParallelLinear(_MLA_HEADS * _V, _HIDDEN, bias=False, parallel_group="attn_tp"),
        indexer=IndexerKPool(
            hidden_size=_HIDDEN,
            index_n_heads=_INDEX_HEADS,
            index_head_dim=_INDEX_DIM,
            rope_head_dim=0,
            index_topk=8,
            q_lora_rank=_Q_LORA,
            max_position_embeddings=64,
            rope_theta=10000.0,
            layer_id=1,
            scale_fmt="ue8m0",
            prefix="model.layers.1.self_attn.indexer",
            skip_rope=True,
            config=config,
        ),
    )
    attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim = _Q_LORA, _KV_LORA, 0
    attn.qk_nope_head_dim, attn.v_head_dim, attn.qk_head_dim = _NOPE, _V, _NOPE
    attn.w_kc = attn.w_vc = None
    attn.w_kc_qrep = attn.q_b_proj_qrep = attn.q_b_proj_qrep_weight = None
    return attn


def _moe(config):
    """The DeepSeek-V2 MoE's router and shared expert; TopK reads the router's correction bias."""
    gate = MoEGate(config=config, quant_config=None, prefix="model.layers.1.mlp.gate")
    moe = _bare(DeepseekV2MoE, gate=gate, shared_experts=DeepseekV2MLP(_HIDDEN, _SHARED, "silu", reduce_results=False))
    moe.topk = SimpleNamespace(topk_config=SimpleNamespace(correction_bias=gate.e_score_correction_bias))
    return moe


def _decoder_layer(attn, mlp):
    layer = _bare(
        sglang_glm5_next.Glm5NextDecoderLayer,
        self_attn=attn,
        mlp=mlp,
        input_layernorm=RMSNorm(_HIDDEN),
        post_attention_layernorm=RMSNorm(_HIDDEN),
    )
    # The decoder's FP32 mHC parameters, loaded under their checkpoint names.
    mix, width = (2 + _HC_MULT) * _HC_MULT, _HC_MULT * _HIDDEN
    for site in ("attn", "ffn"):
        setattr(layer, f"hc_{site}_base", torch.nn.Parameter(torch.empty(mix, dtype=torch.float32)))
        setattr(layer, f"hc_{site}_scale", torch.nn.Parameter(torch.empty(3, dtype=torch.float32)))
        setattr(layer, f"hc_{site}_fn", torch.nn.Parameter(torch.empty(mix, width, dtype=torch.float32)))
    return layer


def _model(tp_size, tp_rank):
    config = _text_config()
    with _rank(tp_size, tp_rank):
        layers = torch.nn.ModuleList(
            [
                _decoder_layer(_kda_attention(config), DeepseekV2MLP(_HIDDEN, _INTERMEDIATE, "silu")),
                _decoder_layer(_dsa_attention(config), _moe(config)),
            ]
        )
        text = _bare(torch.nn.Module, layers=layers, norm=RMSNorm(_HIDDEN))
    text.start_layer, text.end_layer = 0, len(layers)
    model = _bare(sglang_glm5_next.Glm5NextForConditionalGeneration, model=text)
    model.config, model.quant_config, model.num_fused_shared_experts = config, None, 0
    model.fuse_qkv_a_proj, model.encoder_only, model.language_only = True, False, False
    return model


def _checkpoint(seed):
    """Every text tensor of the model's checkpoint, in checkpoint names, dtypes and shapes."""
    generator = torch.Generator().manual_seed(seed)

    def bf16(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    def f32(*shape):
        return torch.randn(*shape, generator=generator)

    tensors = {"model.language_model.norm.weight": bf16(_HIDDEN)}
    tensors.update({f"{_KDA}.self_attn.{leaf}_proj.weight": bf16(_PROJECTION, _HIDDEN) for leaf in "qkv"})
    tensors.update({f"{_KDA}.self_attn.{leaf}_a_proj.weight": bf16(_HEAD_DIM, _HIDDEN) for leaf in "fg"})
    tensors.update({f"{_KDA}.self_attn.{leaf}_b_proj.weight": bf16(_PROJECTION, _HEAD_DIM) for leaf in "fg"})
    tensors.update({f"{_KDA}.self_attn.{leaf}_conv1d.weight": bf16(_PROJECTION, 1, _CONV) for leaf in "qkv"})
    tensors.update(
        {
            f"{_KDA}.self_attn.b_proj.weight": bf16(_HEADS, _HIDDEN),
            f"{_KDA}.self_attn.A_log": f32(_HEADS),
            f"{_KDA}.self_attn.dt_bias": f32(_PROJECTION),
            f"{_KDA}.self_attn.o_norm.weight": bf16(_HEAD_DIM),
            f"{_KDA}.self_attn.o_proj.weight": bf16(_HIDDEN, _PROJECTION),
            f"{_KDA}.mlp.gate_proj.weight": bf16(_INTERMEDIATE, _HIDDEN),
            f"{_KDA}.mlp.up_proj.weight": bf16(_INTERMEDIATE, _HIDDEN),
            f"{_KDA}.mlp.down_proj.weight": bf16(_HIDDEN, _INTERMEDIATE),
            f"{_DSA}.self_attn.q_a_proj.weight": bf16(_Q_LORA, _HIDDEN),
            f"{_DSA}.self_attn.kv_a_proj_with_mqa.weight": bf16(_KV_LORA, _HIDDEN),
            f"{_DSA}.self_attn.q_a_layernorm.weight": bf16(_Q_LORA),
            f"{_DSA}.self_attn.kv_a_layernorm.weight": bf16(_KV_LORA),
            f"{_DSA}.self_attn.q_b_proj.weight": bf16(_MLA_HEADS * _NOPE, _Q_LORA),
            f"{_DSA}.self_attn.kv_b_proj.weight": bf16(_MLA_HEADS * (_NOPE + _V), _KV_LORA),
            f"{_DSA}.self_attn.o_proj.weight": bf16(_HIDDEN, _MLA_HEADS * _V),
            f"{_DSA}.self_attn.indexer.wq_b.weight": bf16(_INDEX_HEADS * _INDEX_DIM, _Q_LORA),
            f"{_DSA}.self_attn.indexer.wk.weight": bf16(_INDEX_DIM, _HIDDEN),
            f"{_DSA}.self_attn.indexer.weights_proj.weight": bf16(_INDEX_HEADS, _HIDDEN),
            f"{_DSA}.self_attn.indexer.k_norm.weight": bf16(_INDEX_DIM),
            f"{_DSA}.self_attn.indexer.k_norm.bias": bf16(_INDEX_DIM),
            f"{_DSA}.self_attn.indexer.index_kpool_compress_gate": bf16(_INDEX_DIM, _HIDDEN),
            f"{_DSA}.self_attn.indexer.index_kpool_compress_ape": bf16(_KPOOL, _INDEX_DIM),
            f"{_DSA}.mlp.gate.weight": bf16(_EXPERTS, _HIDDEN),
            f"{_DSA}.mlp.gate.e_score_correction_bias": f32(_EXPERTS),
            f"{_DSA}.mlp.shared_experts.gate_proj.weight": bf16(_SHARED, _HIDDEN),
            f"{_DSA}.mlp.shared_experts.up_proj.weight": bf16(_SHARED, _HIDDEN),
            f"{_DSA}.mlp.shared_experts.down_proj.weight": bf16(_HIDDEN, _SHARED),
        }
    )
    mix, width = (2 + _HC_MULT) * _HC_MULT, _HC_MULT * _HIDDEN
    for layer in (_KDA, _DSA):
        tensors[f"{layer}.input_layernorm.weight"] = bf16(_HIDDEN)
        tensors[f"{layer}.post_attention_layernorm.weight"] = bf16(_HIDDEN)
        for site in ("attn", "ffn"):
            tensors[f"{layer}.hc_{site}_base"] = f32(mix)
            tensors[f"{layer}.hc_{site}_scale"] = f32(3)
            tensors[f"{layer}.hc_{site}_fn"] = bf16(mix, width)
    return tensors


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


def _served(model):
    """Every parameter and absorbed MLA cache the model serves, by name."""
    served = dict(model.named_parameters())
    for index, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, DeepseekV2AttentionMLA):
            served[f"model.layers.{index}.self_attn.w_kc"] = layer.self_attn.w_kc
            served[f"model.layers.{index}.self_attn.w_vc"] = layer.self_attn.w_vc
    return {name: _bits(value).clone() for name, value in served.items()}


@torch.no_grad()
def _loaded(tp_size, tp_rank, tensors):
    model = _model(tp_size, tp_rank)
    for param in model.parameters():
        param.fill_(-7)
    model.load_weights(list(tensors.items()))
    return model


@torch.no_grad()
def _apply(bindings, masks):
    """The receiver's update: raw masks through prepare_raw_xor's physical masks, matrices through byte views."""
    raw = [(binding, masks[binding.name]) for binding in bindings if binding.is_raw]
    for target, mask, _ in prepare_raw_xor(raw):
        target.detach().view(-1).view(torch.uint8).bitwise_xor_(mask)
    for binding in bindings:
        if binding.is_raw:
            continue
        for piece in binding.apply_bindings():
            selected = piece.selected_bytes(masks[binding.name])
            for destination in piece.destinations:
                destination.bitwise_xor_(selected.reshape(destination.shape))


def _admit(model, tmp_path, tensors):
    """Describe-time admission of ``model`` at checkpoint ``tensors``, and the first prepare's shadow fill."""
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    locations = publication.read_safetensors_headers([path])
    inventory = {name: {"shape": shape, "dtype": dtype} for name, (_, _, _, dtype, shape) in locations.items()}
    policy = publication.static_policy({"architectures": [_ARCHITECTURE], "text_config": {"num_hidden_layers": 2}})
    checkpoint = CanonicalCheckpoint(inventory, policy, locations, "a" * 64)
    plan = layout.GpuDeltaLayout(model, checkpoint, CanonicalShadows({}, create=True))
    parameters = plan._mapping.parameters
    parameters.shadows.fill(checkpoint, parameters.shadowed)
    return plan


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 3)])
def test_an_update_through_the_bindings_serves_what_the_loader_loads(tmp_path, tp_size, tp_rank):
    base, update = _checkpoint(0), _checkpoint(1)
    model = _loaded(tp_size, tp_rank, base)
    before = _served(model)
    plan = _admit(model, tmp_path, base)
    mapping = plan._mapping
    parameters = mapping.parameters
    assert isinstance(mapping, Glm5NextMapping)
    assert sorted(binding.name for binding in plan.bindings) == sorted(base) and not plan.excluded

    aliases = parameters.aliases
    kda = [f"{_KDA}.self_attn.{leaf}_proj.weight" for leaf in ("q", "k", "v", "b", "f_a", "g_a")]
    assert all(aliases(kda[0], name) for name in kda[1:])
    assert aliases(f"{_KDA}.self_attn.f_b_proj.weight", f"{_KDA}.self_attn.g_b_proj.weight")
    assert aliases(f"{_DSA}.self_attn.q_a_proj.weight", f"{_DSA}.self_attn.kv_a_proj_with_mqa.weight")
    assert aliases(f"{_KDA}.mlp.gate_proj.weight", f"{_KDA}.mlp.up_proj.weight")
    assert aliases(f"{_DSA}.mlp.shared_experts.gate_proj.weight", f"{_DSA}.mlp.shared_experts.up_proj.weight")
    assert not aliases(kda[0], f"{_KDA}.self_attn.f_b_proj.weight")
    # The layout's own check at the first prepare: live storage is shared only within a declared group.
    plan._reject_overlaps(aliases)

    plan.check_identity()
    pinned = list(mapping.consumers)
    mapping.finish()
    # Consumers are pinned while binding, at describe; finish() only adds images.
    assert mapping.consumers == pinned
    computed = [image for image in mapping.derived if type(image) is ComputedImage]
    # The KDA convolutions, the mHC mappings and the indexer's weights_proj and kpool ape, widened from BF16 shadows
    # of this rank's rows.
    widened = {f"{_KDA}.self_attn.{leaf}_conv1d.weight" for leaf in "qkv"}
    widened |= {f"{layer}.hc_{site}_fn" for layer in (_KDA, _DSA) for site in ("attn", "ffn")}
    widened |= {f"{_DSA}.self_attn.indexer.weights_proj.weight", f"{_DSA}.self_attn.indexer.index_kpool_compress_ape"}
    assert {image.name for image in computed} == set(parameters.shadowed) == widened
    bound = {pointer for binding in plan.bindings for pointer in binding.storage_pointers}
    for image in computed:
        assert image.sources == (parameters.shadowed[image.name],)
        assert all(_storage_key(destination) not in bound for destination in image.destinations)
        # Admission's check: the image reproduces the loader's widening bytewise.
        image.verify()

    masks = {name: _bits(base[name]).reshape(-1) ^ _bits(update[name]).reshape(-1) for name in base}
    _apply(plan.bindings, masks)
    for image in computed:
        with pytest.raises(ValueError, match="does not reproduce its live consumer"):
            image.verify()
    for image in mapping.derived:
        image.refresh(torch.tensor(True))

    expected = _served(_loaded(tp_size, tp_rank, update))
    actual = _served(model)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert not torch.equal(before[name], value), f"the update leaves {name} unchanged"
        assert torch.equal(actual[name], value), name
    for name, shadow in parameters.shadowed.items():
        start, stop = parameters.shadows.store[name].rows
        assert torch.equal(_bits(shadow), _bits(update[name][start:stop])), name
    for image in computed:
        image.verify()

    # The snapshots re-read the live consumers: re-pointing what a kernel reads requires readmission.
    kda_layer, dsa_layer = model.model.layers
    for consumer, field in (
        (kda_layer.self_attn.attn, "conv_weights"),
        (dsa_layer.self_attn, "w_kc"),
        (dsa_layer.mlp.topk.topk_config, "correction_bias"),
    ):
        original = getattr(consumer, field)
        setattr(consumer, field, original.clone())
        with pytest.raises(RuntimeError, match="consumer storage changed"):
            plan.check_identity()
        setattr(consumer, field, original)
    plan.check_identity()


def _meta(tensor):
    shape = list(tensor.shape)
    dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}[tensor.dtype]
    return {"dtype": dtype, "shape": shape, "encoding": publication.canonical_encoding(shape)}


@torch.no_grad()
def _loader_image(param, load, *args):
    # SGLang's loader writes into a sentinel-filled parameter; unwritten rows keep the sentinel.
    param.fill_(-7)
    load(*args)
    return _bits(param).clone()


@torch.no_grad()
def _helper_image(param, view, slices, canonical):
    param.fill_(-7)
    view.copy_(canonical[tuple(slice(start, stop) for start, stop in slices)])
    return _bits(param).clone()


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 2)])
def test_repeated_column_shards_bind_the_rows_their_loader_fills(tp_size, tp_rank):
    # Three column-parallel parts and two repeated ones, as fused_qkvbfg_a_proj's [q, k, v, b] and [f_a, g_a].
    columns, repeated = [8, 16, 4], [6, 2]
    with _rank(tp_size, tp_rank):
        module = MergedColumnParallelRepeatedLinear(5, columns, repeated)
    param = module.weight
    for shard, rows in enumerate(columns + repeated):
        canonical = torch.randn(rows, 5).bfloat16()
        expected = _loader_image(param, module.weight_loader, param, canonical, shard)
        view, slices = repeated_column_target(param, module, shard, list(canonical.shape))
        if shard < len(columns):
            assert slices[0] == [tp_rank * rows // tp_size, (tp_rank + 1) * rows // tp_size]
        else:
            assert slices[0] == [0, rows]
        assert torch.equal(_helper_image(param, view, slices, canonical), expected), shard


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 3)])
def test_batched_column_batches_bind_the_rows_their_loader_fills(tp_size, tp_rank):
    with _rank(tp_size, tp_rank):
        module = ColumnParallelBatchedLinear(2, 6, 16, dtype=torch.bfloat16)
    param = module.weight
    for batch in range(2):
        canonical = torch.randn(16, 6).bfloat16()
        expected = _loader_image(param, module.weight_loader, param, canonical, batch)
        view, slices = batched_column_target(module, batch, [16, 6])
        assert torch.equal(_helper_image(param, view, slices, canonical), expected), batch


def test_fused_kda_helpers_reject_other_classes_shards_and_geometry():
    with _rank(2, 1):
        repeated = MergedColumnParallelRepeatedLinear(5, [8, 4], [6])
        batched = ColumnParallelBatchedLinear(2, 6, 16, dtype=torch.bfloat16)
        merged = MergedColumnParallelLinear(5, [8, 4], bias=False)
    with pytest.raises(ValueError, match="unclassified repeated column projection: MergedColumnParallelLinear"):
        repeated_column_target(merged.weight, merged, 0, [8, 5])
    with pytest.raises(ValueError, match="unclassified batched column projection: MergedColumnParallelRepeatedLinear"):
        batched_column_target(repeated, 0, [16, 6])
    for shard in (3, -1, True):
        with pytest.raises(ValueError, match="unsupported repeated column shard"):
            repeated_column_target(repeated.weight, repeated, shard, [8, 5])
    for batch in (2, -1, True):
        with pytest.raises(ValueError, match="unsupported batched column shard"):
            batched_column_target(batched, batch, [16, 6])
    # A column shard's checkpoint rows span the TP group; a repeated shard's are its local rows.
    for shard, shape in ((0, [4, 5]), (2, [12, 5]), (0, [8, 5, 1])):
        with pytest.raises(ValueError, match="repeated column projection geometry differs from its loader"):
            repeated_column_target(repeated.weight, repeated, shard, shape)
    for shape in ([8, 6], [16, 3]):
        with pytest.raises(ValueError, match="batched column projection geometry differs from its loader"):
            batched_column_target(batched, 0, shape)


def test_kda_outside_its_fused_layout_is_refused():
    # With LoRA, SGLang builds separate q/k/v, b, f and g projections.
    config = _text_config()
    with _rank(1, 0, lora=True):
        attn = _kda_attention(config)
        layer = _decoder_layer(attn, DeepseekV2MLP(_HIDDEN, _INTERMEDIATE, "silu"))
    assert not attn.do_fuse_qkvbfg
    model = _bare(
        sglang_glm5_next.Glm5NextForConditionalGeneration,
        model=_bare(torch.nn.Module, layers=torch.nn.ModuleList([layer])),
    )
    model.config, model.num_fused_shared_experts = config, 0
    mapping = Glm5NextMapping(model, None)
    with pytest.raises(ValueError, match="only in its fused qkvbfg layout"):
        mapping.bind(f"{_KDA}.self_attn.q_proj.weight", _meta(torch.zeros(_PROJECTION, _HIDDEN).bfloat16()))


# One canonical tensor of each text kind in the GLM-5.3-Flash 5-layer cut (KDA layers 0-2 and 4, DSA layer 3; dense
# MLPs in layers 0-2, MoE in 3-4; ModelOpt NVFP4 routed experts), with its layer kinds and route.
_TEXT_ROUTES = {
    **{
        f"model.language_model.layers.0.hc_{site}_{leaf}": ("kda", "dense", "dense")
        for site in ("attn", "ffn")
        for leaf in ("base", "scale")
    },
    **{f"model.language_model.layers.0.hc_{site}_fn": ("kda", "dense", "widened") for site in ("attn", "ffn")},
    "model.language_model.layers.0.input_layernorm.weight": ("kda", "dense", "dense"),
    "model.language_model.layers.0.post_attention_layernorm.weight": ("kda", "dense", "dense"),
    **{
        f"model.language_model.layers.0.self_attn.{leaf}.weight": ("kda", "dense", "kda_qkvbfg_a")
        for leaf in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
    },
    "model.language_model.layers.0.self_attn.f_b_proj.weight": ("kda", "dense", "kda_fg_b"),
    "model.language_model.layers.0.self_attn.g_b_proj.weight": ("kda", "dense", "kda_fg_b"),
    **{
        f"model.language_model.layers.0.self_attn.{leaf}_conv1d.weight": ("kda", "dense", "kda_conv") for leaf in "qkv"
    },
    "model.language_model.layers.0.self_attn.A_log": ("kda", "dense", "kda_a_log"),
    "model.language_model.layers.0.self_attn.dt_bias": ("kda", "dense", "kda_dt_bias"),
    "model.language_model.layers.0.self_attn.o_norm.weight": ("kda", "dense", "dense"),
    "model.language_model.layers.0.self_attn.o_proj.weight": ("kda", "dense", "dense"),
    "model.language_model.layers.0.mlp.gate_proj.weight": ("kda", "dense", "gate_up"),
    "model.language_model.layers.0.mlp.up_proj.weight": ("kda", "dense", "gate_up"),
    "model.language_model.layers.0.mlp.down_proj.weight": ("kda", "dense", "dense"),
    "model.language_model.layers.3.self_attn.q_a_proj.weight": ("mla", "moe", "mla_qkv_a"),
    "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight": ("mla", "moe", "mla_qkv_a"),
    "model.language_model.layers.3.self_attn.q_a_layernorm.weight": ("mla", "moe", "dense"),
    "model.language_model.layers.3.self_attn.kv_a_layernorm.weight": ("mla", "moe", "dense"),
    "model.language_model.layers.3.self_attn.q_b_proj.weight": ("mla", "moe", "mla_projection"),
    "model.language_model.layers.3.self_attn.kv_b_proj.weight": ("mla", "moe", "mla_projection"),
    "model.language_model.layers.3.self_attn.o_proj.weight": ("mla", "moe", "dense"),
    **{
        f"model.language_model.layers.3.self_attn.indexer.{leaf}": ("mla", "moe", "dense")
        for leaf in (
            "wq_b.weight",
            "wk.weight",
            "k_norm.weight",
            "k_norm.bias",
            "index_kpool_compress_gate",
        )
    },
    "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape": ("mla", "moe", "widened"),
    "model.language_model.layers.3.self_attn.indexer.weights_proj.weight": ("mla", "moe", "indexer_weights"),
    "model.language_model.layers.3.mlp.gate.weight": ("mla", "moe", "dense"),
    "model.language_model.layers.3.mlp.gate.e_score_correction_bias": ("mla", "moe", "router_bias"),
    "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight": ("mla", "moe", "gate_up"),
    "model.language_model.layers.3.mlp.shared_experts.up_proj.weight": ("mla", "moe", "gate_up"),
    "model.language_model.layers.3.mlp.shared_experts.down_proj.weight": ("mla", "moe", "dense"),
    **{
        f"model.language_model.layers.3.mlp.experts.0.{projection}_proj.{suffix}": ("mla", "moe", "expert")
        for projection in ("gate", "up", "down")
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    },
    "model.language_model.layers.4.self_attn.q_proj.weight": ("kda", "moe", "kda_qkvbfg_a"),
    "model.language_model.embed_tokens.weight": (None, None, "dense"),
    "model.language_model.norm.weight": (None, None, "dense"),
    "lm_head.weight": (None, None, "dense"),
}
# The same cut's static tensors: the MTP layer (layer 5) and the vision tower.
_STATIC = {
    **{
        f"model.language_model.layers.5.{leaf}": publication.STATIC_DRAFT
        for leaf in (
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "shared_head.norm.weight",
            "input_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.indexer.weights_proj.weight",
            "mlp.gate.e_score_correction_bias",
            "mlp.experts.0.gate_proj.weight",
            "mlp.shared_experts.down_proj.weight",
        )
    },
    **{
        f"model.visual.{leaf}": publication.FROZEN_MODULE
        for leaf in (
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.mlp.gate_proj.bias",
            "blocks.0.norm1.weight",
            "downsample.weight",
            "merger.proj.weight",
            "merger.post_projection_norm.bias",
            "patch_embed.proj.weight",
            "post_layernorm.weight",
        )
    },
}


def test_the_policy_and_routes_classify_every_tensor_kind_of_the_flash_cut():
    # config.json of the cut: the MTP layer follows text_config.num_hidden_layers.
    policy = publication.static_policy({"architectures": [_ARCHITECTURE], "text_config": {"num_hidden_layers": 5}})
    assert (policy.num_hidden_layers, policy.draft) == (5, None)
    for name, (attention, mlp, kind) in _TEXT_ROUTES.items():
        assert publication.static_tensor_reason(name, policy) is None, name
        assert route(served_name(name), attention, mlp) == kind, name
    for name, reason in _STATIC.items():
        assert publication.static_tensor_reason(name, policy) == reason, name


def test_a_fused_shared_expert_takes_its_routed_slot_as_the_loader_renames_it():
    name = "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"
    assert served_name(name) == "model.layers.3.mlp.shared_experts.gate_proj.weight"
    assert served_name(name, 288) == "model.layers.3.mlp.experts.288.gate_proj.weight"
    assert route(served_name(name, 288), "mla", "moe") == "expert"


@pytest.mark.parametrize(
    "live,attention,mlp",
    [
        # What the loader would skip (hc_head) or never serves in this layout (an unfused qkv_proj).
        ("model.hc_head_fn", None, None),
        ("model.layers.0.self_attn.qkv_proj.weight", "kda", "dense"),
        # A tensor of the other attention or MLP kind.
        ("model.layers.0.self_attn.q_a_proj.weight", "kda", "dense"),
        ("model.layers.3.self_attn.q_proj.weight", "mla", "moe"),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", "kda", "dense"),
        ("model.layers.3.mlp.gate_proj.weight", "mla", "moe"),
        # Static tensors never reach the mapping.
        ("visual.blocks.0.attn.qkv_proj.weight", None, None),
        ("model.layers.5.eh_proj.weight", "mla", "moe"),
    ],
)
def test_an_unknown_tensor_is_unclassified(live, attention, mlp):
    with pytest.raises(ValueError, match="unclassified GLM-5.3-Flash canonical tensor"):
        route(live, attention, mlp)
