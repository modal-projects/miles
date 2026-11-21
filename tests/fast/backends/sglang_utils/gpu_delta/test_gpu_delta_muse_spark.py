"""Checks of the Muse Spark mapping against SGLang's own loaders.

A three-layer model of real SGLang Muse Spark modules (a dense MLP layer, then two MoE layers, the last without
RoPE) on one rank of its attention-TP, TP and EP groups loads a base checkpoint through ``load_weights`` and
``post_load_weights``, with the experts on each path ``prepare_trtllm_weights`` can take: FlashInfer's TRT-LLM BF16
shuffle, the Triton fused MoE or the PyTorch loop. Its bindings then XOR the base into a second checkpoint, as an
update does, and every served parameter, buffer and post-load copy must equal what the loader writes from that
checkpoint. On CUDA the experts take the path the device selects (TRT-LLM on SM10x) and admission runs in full.
"""

import re
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from sglang.kernels.ops.layernorm.muse_spark_norm_residual import FusedPostNormResidualPreNorm
from sglang.srt.configs.muse_spark import MuseSparkTextConfig
from sglang.srt.layers import linear as sglang_linear
from sglang.srt.layers import vocab_parallel_embedding
from sglang.srt.layers.muse_spark import muse_spark_bf16_moe
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark.muse_spark_layers import MuseSpark1xRMSNorm
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import muse_spark as sglang_muse_spark
from sglang.srt.models.muse_spark import MuseSpark1xRouter, MuseSparkModel
from sglang.srt.models.muse_spark_omni import MuseSparkForConditionalGeneration

from miles.backends.sglang_utils.gpu_delta import layout
from miles.backends.sglang_utils.gpu_delta.bindings import REMOTE_EXPERT, CanonicalShadows, prepare_raw_xor
from miles.backends.sglang_utils.gpu_delta.byte_views import _storage_key
from miles.backends.sglang_utils.gpu_delta.checkpoint import CanonicalCheckpoint
from miles.backends.sglang_utils.gpu_delta.models.muse_spark import MuseSparkMapping, route
from miles.utils.gpu_delta import publication

_ARCHITECTURE = "MuseSparkForConditionalGeneration"
_HIDDEN, _HEAD_DIM, _HEADS, _KV_HEADS, _INTERMEDIATE = 128, 16, 4, 2, 64
_EXPERT_HIDDEN, _EXPERT_INTERMEDIATE, _EXPERTS = 64, 64, 8
_VOCAB, _LAYERS = 64, 3
_TEXT = "model.language_model"
# Tower tensors the policy freezes and a language-model-only engine never loads.
_TOWERS = (
    "model.vision_tower.layers.0.attn.q_proj.weight",
    "model.vision_adapter.fc1.weight",
    "model.vision_projection.weight",
    "model.audio_tower.proj.weight",
)
# One rank of (attention TP, TP, EP), as (size, rank) each.
_RANKS = {
    # The gate engine, dp_attention(4): attention TP 1 under DP attention, dense MLPs over TP 4, EP 4.
    "dp-attention-ep4": ((1, 0), (4, 2), (4, 2)),
    "tp2-ep2": ((2, 1), (2, 1), (2, 1)),
    # Attention TP 4 over 2 KV heads: each KV head is replicated on two ranks.
    "tp4-kv-replicas": ((4, 3), (4, 3), (1, 0)),
}
_GATE_RANKS = _RANKS["dp-attention-ep4"]
_EXPERT_PATHS = ("trtllm", "triton", "torch")
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# On CPU each expert path prepare_trtllm_weights can take; on CUDA the device's own.
_CASES = [
    *(
        pytest.param(ranks, "cpu", path, id=f"{name}-{path}")
        for name, ranks in _RANKS.items()
        for path in _EXPERT_PATHS
    ),
    *(pytest.param(ranks, "cuda", "native", id=f"{name}-cuda", marks=_CUDA) for name, ranks in _RANKS.items()),
]
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")


def _text_config():
    config = MuseSparkTextConfig(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        moe_hidden_size=_EXPERT_HIDDEN,
        moe_intermediate_size=_EXPERT_INTERMEDIATE,
        num_local_experts=_EXPERTS,
        num_experts_per_tok=2,
        num_hidden_layers=_LAYERS,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 500000.0, 0],
        mlp_layer_types=["dense", "sparse", "sparse"],
        vocab_size=_VOCAB,
        max_position_embeddings=256,
        sliding_window=16,
    )
    config.dtype = torch.bfloat16
    return config


def _group(size, rank):
    return None if size == 1 else sglang_linear._LogicalGroup(rank, size)


@contextmanager
def _rank(ranks, device):
    """SGLang building its BF16 Muse Spark model on one rank, before distributed initialization: each partition group
    resolves to a logical group, the embedding follows attention TP as under DP attention, and the CUDA-only
    finalize all-reduce and LayerNorm SP stay off."""
    (attn_tp, attn_rank), (tp, tp_rank), (ep, ep_rank) = ranks
    groups = {"attn_tp": _group(attn_tp, attn_rank), "tp": _group(tp, tp_rank)}
    parallel = SimpleNamespace(
        attn_tp_size=attn_tp,
        attn_tp_rank=attn_rank,
        tp_size=tp,
        tp_rank=tp_rank,
        moe_tp_size=1,
        moe_ep_size=ep,
        moe_ep_rank=ep_rank,
        enable_megamoe_for_layernorm_sp=False,
    )
    with ExitStack() as stack:
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        stack.enter_context(torch.device(device))
        for module in (sglang_linear, vocab_parallel_embedding):
            stack.enter_context(patch.object(module, "_resolve_linear_group", lambda name: groups[name or "tp"]))
        for module in (sglang_muse_spark, vocab_parallel_embedding):
            stack.enter_context(patch.object(module, "get_parallel", lambda: parallel))
        stack.enter_context(
            patch.object(
                sglang_muse_spark, "get_embedding_tp_kwargs", lambda: {"enable_tp": True, "use_attn_tp_group": True}
            )
        )
        stack.enter_context(patch.object(sglang_muse_spark, "is_cuda", lambda: False))
        stack.enter_context(patch.object(sglang_muse_spark.layernorm_sp, "layernorm_sp_enabled", lambda: False))
        if device == "cpu":
            stack.enter_context(patch.object(torch.cuda, "Stream", lambda: None))
            # Without CUDA, SGLang's rotary embedding needs vLLM's kernel; its frequencies are computed, never loaded.
            stack.enter_context(patch.object(sglang_muse_spark, "get_rope", lambda *args, **kwargs: torch.nn.Module()))
        yield


@contextmanager
def _expert_path(path):
    """Select the path prepare_trtllm_weights takes on this device, or keep the device's own (``"native"``)."""
    with ExitStack() as stack:
        if path != "native":
            supported = {"trtllm": (True, False), "triton": (False, True), "torch": (False, False)}[path]
            for probe, value in zip(("_trtllm_bf16_moe_supported", "_triton_bf16_moe_supported"), supported):
                stack.enter_context(patch.object(muse_spark_bf16_moe, probe, lambda device, value=value: value))
        yield


def _model(ranks, device):
    config = _text_config()
    with _rank(ranks, device):
        text = sglang_muse_spark.MuseSpark1xModel(config, None, _TEXT)
        # MuseSparkTextForCausalLM's head under --enable-dp-lm-head.
        head = ParallelLMHead(_VOCAB, _HIDDEN, quant_config=None, prefix="lm_head", use_attn_tp_group=True)
    # The wrappers without their constructors, whose logits processor needs a running server's context.
    model = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    torch.nn.Module.__init__(model)
    language = MuseSparkModel.__new__(MuseSparkModel)
    torch.nn.Module.__init__(language)
    language.language_model = text
    model.model, model.lm_head = language, head
    model.config, model.quant_config, model.capture_aux_hidden_states = config, None, False
    model.nvfp4_checkpoint = model.int4_checkpoint = False
    # --language-model-only: the towers are not built and load_weights skips their tensors.
    model.encoder_only, model.language_only = False, True
    return model


def _checkpoint(seed):
    """Every tensor kind of the cut checkpoint at this model's sizes, in checkpoint names, dtypes and shapes."""
    generator = torch.Generator().manual_seed(seed)

    def bf16(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    def f32(*shape):
        return torch.randn(*shape, generator=generator)

    tensors = {
        f"{_TEXT}.embed_tokens.weight": bf16(_VOCAB, _HIDDEN),
        f"{_TEXT}.norm.weight": bf16(_HIDDEN),
        "lm_head.weight": bf16(_VOCAB, _HIDDEN),
        **{name: bf16(16, 16) for name in _TOWERS},
    }
    for index in range(_LAYERS):
        layer = f"{_TEXT}.layers.{index}"
        for leaf in ("input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            tensors[f"{layer}.{leaf}.weight"] = bf16(_HIDDEN)
        for site in ("post_attention", "post_feedforward"):
            tensors[f"{layer}.{site}_residual_gate.gate"] = bf16(_HIDDEN)
        for leaf, rows in (("q", _HEADS), ("k", _KV_HEADS), ("v", _KV_HEADS), ("gate", _HEADS)):
            tensors[f"{layer}.self_attn.{leaf}_proj.weight"] = bf16(rows * _HEAD_DIM, _HIDDEN)
        tensors[f"{layer}.self_attn.o_proj.weight"] = bf16(_HIDDEN, _HEADS * _HEAD_DIM)
        if index == 0:
            tensors[f"{layer}.mlp.gate_proj.weight"] = bf16(_INTERMEDIATE, _HIDDEN)
            tensors[f"{layer}.mlp.up_proj.weight"] = bf16(_INTERMEDIATE, _HIDDEN)
            tensors[f"{layer}.mlp.down_proj.weight"] = bf16(_HIDDEN, _INTERMEDIATE)
            continue
        tensors[f"{layer}.mlp.gate.weight"] = f32(_EXPERTS, _HIDDEN)
        tensors[f"{layer}.mlp.gate.e_score_correction_bias"] = f32(_EXPERTS)
        tensors[f"{layer}.mlp.pre_expert_proj.weight"] = bf16(_EXPERT_HIDDEN, _HIDDEN)
        tensors[f"{layer}.mlp.post_expert_proj.weight"] = bf16(_HIDDEN, _EXPERT_HIDDEN)
        tensors[f"{layer}.mlp.pre_expert_norm.weight"] = bf16(_EXPERT_HIDDEN)
        tensors[f"{layer}.mlp.experts.post_expert_norm.weight"] = bf16(_EXPERT_HIDDEN)
        for expert in range(_EXPERTS):
            for leaf in ("gate", "up"):
                tensors[f"{layer}.mlp.experts.{expert}.{leaf}_proj.weight"] = bf16(
                    _EXPERT_INTERMEDIATE, _EXPERT_HIDDEN
                )
            tensors[f"{layer}.mlp.experts.{expert}.down_proj.weight"] = bf16(_EXPERT_HIDDEN, _EXPERT_INTERMEDIATE)
    return tensors


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


def _served(model):
    """Every parameter and buffer the model serves, and the copies post_load_weights and the first forwards keep."""
    served = dict(model.named_parameters())
    served.update(model.named_buffers())
    for name, module in model.named_modules():
        if isinstance(module, MuseSpark1xRMSNorm) and module._frozen_weight is not None:
            served[f"{name}._frozen_weight"] = module._frozen_weight
        elif isinstance(module, FusedPostNormResidualPreNorm):
            for field in ("_cc_post_weight", "_cc_pre_weight", "_cc_alpha", "_cc_beta"):
                served[f"{name}.{field}"] = module.__dict__[field]
        elif isinstance(module, MuseSpark1xRouter) and module._split2_hi is not None:
            served[f"{name}._split2_hi"], served[f"{name}._split2_lo"] = module._split2_hi, module._split2_lo
        elif isinstance(module, MuseSpark1xBf16Experts) and "_finalize_norm_cache" in module.__dict__:
            served[f"{name}._finalize_norm_cache"] = module._finalize_norm_cache[0]
    return {name: _bits(value).clone() for name, value in served.items()}


@torch.no_grad()
def _loaded(ranks, device, path, tensors):
    model = _model(ranks, device)
    for param in model.parameters():
        param.fill_(-7)
    with _expert_path(path):
        model.load_weights(list(tensors.items()))
    layers = model.model.language_model.layers
    # The first forwards: the Triton experts cache their finalize norm weight, and an SM90 prefill splits a router.
    for decoder in layers[1:]:
        experts = decoder.mlp.experts
        if experts.use_triton:
            experts._finalize_norm_params(experts.post_expert_norm)
    layers[1].mlp.gate._build_split2()
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
    """Describe-time admission of ``model`` at checkpoint ``tensors``."""
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    locations = publication.read_safetensors_headers([path])
    inventory = {name: {"shape": shape, "dtype": dtype} for name, (_, _, _, dtype, shape) in locations.items()}
    policy = publication.static_policy(
        {"architectures": [_ARCHITECTURE], "text_config": {"num_hidden_layers": _LAYERS}}
    )
    checkpoint = CanonicalCheckpoint(inventory, policy, locations, "a" * 64)
    return layout.GpuDeltaLayout(model, checkpoint, CanonicalShadows({}, create=True))


@pytest.mark.parametrize("ranks,device,path", _CASES)
def test_an_update_through_the_bindings_serves_what_the_loader_loads(tmp_path, ranks, device, path):
    base, update = _checkpoint(0), _checkpoint(1)
    model = _loaded(ranks, device, path, base)
    layers = model.model.language_model.layers
    experts = [decoder.mlp.experts for decoder in layers[1:]]
    if device == "cuda":
        # SM10x serves BF16 experts on FlashInfer's TRT-LLM MoE, as the gate's B300 does.
        assert all(module.use_trtllm for module in experts) == (torch.cuda.get_device_capability()[0] == 10)
    before = _served(model)
    plan = _admit(model, tmp_path, base)
    mapping = plan._mapping
    parameters = mapping.parameters
    assert isinstance(mapping, MuseSparkMapping)

    _, _, (ep, ep_rank) = ranks
    local = range(ep_rank * _EXPERTS // ep, (ep_rank + 1) * _EXPERTS // ep)
    remote = {name for name in base if (expert := _EXPERT_INDEX.search(name)) and int(expert[1]) not in local}
    assert plan.excluded == {
        **{name: publication.FROZEN_MODULE for name in _TOWERS},
        **{name: REMOTE_EXPERT for name in remote},
    }
    assert sorted(binding.name for binding in plan.bindings) == sorted(set(base) - set(_TOWERS) - remote)

    aliases = parameters.aliases
    attention = [f"{_TEXT}.layers.2.self_attn.{leaf}_proj.weight" for leaf in ("q", "k", "v", "gate")]
    assert all(aliases(attention[0], name) for name in attention[1:])
    assert aliases(f"{_TEXT}.layers.0.mlp.gate_proj.weight", f"{_TEXT}.layers.0.mlp.up_proj.weight")
    assert not aliases(attention[0], f"{_TEXT}.layers.1.self_attn.q_proj.weight")
    plan._reject_overlaps(aliases)

    plan.check_identity()
    pinned = list(mapping.consumers)
    if device == "cuda":
        # The receiver's first-prepare admission: CUDA residency, pinned destinations, bytewise reproduction.
        plan.finish()
    else:
        mapping.finish()
    # Consumers are pinned while binding, at describe; finish() only adds images.
    assert mapping.consumers == pinned
    computed = mapping.derived
    norms = {name for name in base if name.endswith("layernorm.weight") or name.endswith("norm.weight")}
    gates = {name for name in base if name.endswith("residual_gate.gate")}
    assert {image.name for image in computed} == norms | gates | {f"{_TEXT}.layers.1.mlp.gate.weight"}
    bound = {pointer for binding in plan.bindings for pointer in binding.storage_pointers}
    for image in computed:
        assert all(_storage_key(destination) not in bound for destination in image.destinations)
        image.verify()
    # The input norm of layer 1 is also the pre-norm of layer 0's FFN boundary, which keeps a copy.
    (input_norm,) = [image for image in computed if image.name == f"{_TEXT}.layers.1.input_layernorm.weight"]
    assert input_norm.destinations[1] is layers[0]._fused_ffn_boundary._cc_pre_weight

    masks = {
        name: (_bits(base[name]).reshape(-1) ^ _bits(update[name]).reshape(-1)).to(device)
        for name in base
        if name not in _TOWERS
    }
    _apply(plan.bindings, masks)
    for image in computed:
        with pytest.raises(ValueError, match="does not reproduce its live consumer"):
            image.verify()
    for image in computed:
        image.refresh(torch.tensor(True, device=device))

    expected = _served(_loaded(ranks, device, path, update))
    actual = _served(model)
    assert actual.keys() == expected.keys()
    parameter_names = {name for name, _ in model.named_parameters()}
    for name, value in expected.items():
        if name in parameter_names:
            assert not torch.equal(before[name], value), f"the update leaves {name} unchanged"
        assert torch.equal(actual[name], value), name
    for image in computed:
        image.verify()

    # The snapshots re-read the live consumers: re-pointing what a kernel reads requires readmission.
    norm, boundary, router = layers[1].input_layernorm, layers[0]._fused_ffn_boundary, layers[1].mlp.gate
    repointed = [
        (norm, "_frozen_weight"),
        (boundary, "_cc_pre_weight"),
        (layers[1].post_attention_residual_gate, "_alpha"),
        (router, "e_score_correction_bias"),
        (router, "_split2_hi"),
    ]
    if experts[0].use_trtllm:
        repointed.append((experts[0], "_post_expert_norm_weight"))
    for owner, field in repointed:
        original = getattr(owner, field)
        replacement = original.clone()
        if field in owner._buffers:
            owner._buffers[field] = replacement
        else:
            object.__setattr__(owner, field, replacement)
        with pytest.raises(RuntimeError, match="consumer storage changed"):
            plan.check_identity()
        if field in owner._buffers:
            owner._buffers[field] = original
        else:
            object.__setattr__(owner, field, original)
    plan.check_identity()


@pytest.mark.parametrize("path", ["trtllm", "triton", "torch"])
def test_each_expert_path_binds_its_own_layout(tmp_path, path):
    ranks = _GATE_RANKS
    model = _loaded(ranks, "cpu", path, _checkpoint(0))
    plan = _admit(model, tmp_path, _checkpoint(0))
    experts = model.model.language_model.layers[1].mlp.experts
    gate = next(
        binding for binding in plan.bindings if binding.name.endswith("layers.1.mlp.experts.4.gate_proj.weight")
    )
    if path == "trtllm":
        assert experts.use_trtllm and gate.format == "bf16" and gate.segments
        # The finalize op's norm weight is the post-expert norm's effective weight, served in place.
        assert experts._post_expert_norm_weight.data_ptr() == experts.post_expert_norm._frozen_weight.data_ptr()
    else:
        assert not experts.use_trtllm and experts.use_triton == (path == "triton")
        assert gate.storage[0].data_ptr() == experts.gate_up_proj[0].data_ptr() and not gate.segments


def test_megamoe_and_other_expert_formats_are_refused(tmp_path):
    ranks = _GATE_RANKS
    base = _checkpoint(0)
    model = _loaded(ranks, "cpu", "trtllm", base)
    model.model.language_model.layers[2].mlp.experts.use_megamoe = True
    with pytest.raises(ValueError, match="does not map Muse Spark's MegaMoE expert weights"):
        _admit(model, tmp_path, base)
    model.model.language_model.layers[2].mlp.experts.use_megamoe = False
    for flag in ("nvfp4_checkpoint", "int4_checkpoint"):
        setattr(model, flag, True)
        with pytest.raises(ValueError, match="BF16 Muse Spark checkpoints only"):
            MuseSparkMapping(model, None)
        setattr(model, flag, False)
    model.encoder_only = True
    with pytest.raises(ValueError, match="encoder-only serving has no text weights"):
        MuseSparkMapping(model, None)


# One canonical tensor of each kind in the muse-spark-open-last4-partial-per-expert cut (four MoE layers,
# 256 BF16 experts per layer, one tensor per expert) with its route.
_CUT_ROUTES = {
    "model.language_model.embed_tokens.weight": "dense",
    "model.language_model.norm.weight": "norm",
    "lm_head.weight": "dense",
    "model.language_model.layers.3.input_layernorm.weight": "norm",
    "model.language_model.layers.3.pre_feedforward_layernorm.weight": "norm",
    "model.language_model.layers.3.post_feedforward_layernorm.weight": "norm",
    "model.language_model.layers.3.post_attention_residual_gate.gate": "residual_gate",
    "model.language_model.layers.3.post_feedforward_residual_gate.gate": "residual_gate",
    **{f"model.language_model.layers.3.self_attn.{leaf}_proj.weight": "qkvg" for leaf in ("q", "k", "v", "gate")},
    "model.language_model.layers.3.self_attn.o_proj.weight": "dense",
    "model.language_model.layers.3.mlp.gate.weight": "router",
    "model.language_model.layers.3.mlp.gate.e_score_correction_bias": "router_bias",
    "model.language_model.layers.3.mlp.pre_expert_proj.weight": "dense",
    "model.language_model.layers.3.mlp.post_expert_proj.weight": "dense",
    "model.language_model.layers.3.mlp.pre_expert_norm.weight": "norm",
    "model.language_model.layers.3.mlp.experts.post_expert_norm.weight": "norm",
    **{
        f"model.language_model.layers.3.mlp.experts.255.{leaf}_proj.weight": "expert"
        for leaf in ("gate", "up", "down")
    },
}
# The same cut's towers; the decoder-depth draft rule labels tower layers numbered from 4 before the frozen rule.
_CUT_STATIC = {
    "model.vision_tower.layers.3.attn.q_proj.weight": publication.FROZEN_MODULE,
    "model.vision_tower.layers.30.mlp.fc1.weight": publication.STATIC_DRAFT,
    "model.vision_tower.patch_embedder.patch_embedding.weight": publication.FROZEN_MODULE,
    "model.vision_tower.ln_post.weight": publication.FROZEN_MODULE,
    "model.vision_adapter.fc1.weight": publication.FROZEN_MODULE,
    "model.vision_projection.weight": publication.FROZEN_MODULE,
    "model.audio_tower.layers.0.conv_blocks.0.conv.weight": publication.FROZEN_MODULE,
    "model.audio_tower.proj.weight": publication.FROZEN_MODULE,
}


def test_the_policy_and_routes_classify_every_tensor_kind_of_the_cut():
    policy = publication.static_policy({"architectures": [_ARCHITECTURE], "text_config": {"num_hidden_layers": 4}})
    assert (policy.num_hidden_layers, policy.draft) == (4, None)
    for name, kind in _CUT_ROUTES.items():
        assert publication.static_tensor_reason(name, policy) is None, name
        assert route(name, None if ".layers." not in name else "moe") == kind, name
    for name, reason in _CUT_STATIC.items():
        assert publication.static_tensor_reason(name, policy) == reason, name


@pytest.mark.parametrize(
    "name,mlp",
    [
        # Tensors load_weights skips or never serves: a stacked qkv_proj, an embedding norm weight.
        ("model.language_model.layers.0.self_attn.qkv_proj.weight", "moe"),
        ("model.language_model.embed_norm.weight", None),
        # A tensor of the other MLP kind.
        ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", "dense"),
        ("model.language_model.layers.0.mlp.gate_proj.weight", "moe"),
        # NVFP4 expert tensors and the towers never reach a BF16 mapping.
        ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale", "moe"),
        ("model.vision_tower.ln_post.weight", None),
    ],
)
def test_an_unknown_tensor_is_unclassified(name, mlp):
    with pytest.raises(ValueError, match="unclassified Muse Spark canonical tensor"):
        route(name, mlp)
