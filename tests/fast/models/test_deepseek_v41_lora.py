from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.utils.lora import utils as lora_utils
from miles.utils.lora.hf_lora_targets import expand_hf_lora_targets, get_hf_lora_targets, resolve_hf_lora_targets
from miles.utils.lora.utils import AdapterSpec, get_adapter_target_modules
from miles_plugins.models.deepseek_v4_1.lora_export import export_deepseek_v41_multi_lora
from miles_plugins.models.deepseek_v4_1.lora_targets import (
    SUPPORTED_TARGETS,
    deepseek_v41_megatron_targets,
    resolve_deepseek_v41_adapter_targets,
)

_HF_CONFIG = {"model_type": "deepseek_v41", "n_shared_experts": 1}
_SUPPORTED_SELECTION = ["attn", "*.shared_experts.w1", "*.shared_experts.w2", "*.shared_experts.w3"]


def _supported_targets():
    targets = resolve_hf_lora_targets(_HF_CONFIG, target_modules=_SUPPORTED_SELECTION)
    return expand_hf_lora_targets(targets, get_hf_lora_targets(_HF_CONFIG))


def test_layout_uses_checkpoint_names():
    layout = get_hf_lora_targets(_HF_CONFIG)
    assert layout.attention == (
        "layers.*.attn.wq_a",
        "layers.*.attn.wkv",
        "layers.*.attn.wq_b",
        "layers.*.attn.wo_b",
    )
    assert "layers.*.ffn.shared_experts.w1" in layout.mlp
    assert "layers.*.ffn.experts.*.w2" in layout.mlp
    assert layout.unembed == ("head",)


def test_supported_targets_map_to_megatron_modules():
    targets = _supported_targets()
    assert set(resolve_deepseek_v41_adapter_targets(targets)) == set(SUPPORTED_TARGETS)
    assert deepseek_v41_megatron_targets(targets) == [
        "decoder.layers.*.self_attention.linear_q_down_proj",
        "decoder.layers.*.self_attention.linear_kv_proj",
        "decoder.layers.*.self_attention.linear_q_up_proj",
        "decoder.layers.*.self_attention.linear_proj",
        "decoder.layers.*.mlp.shared_experts.linear_fc1",
        "decoder.layers.*.mlp.shared_experts.linear_fc2",
    ]


@pytest.mark.parametrize(
    "selection,error,message",
    [
        (["mlp"], NotImplementedError, "routed experts"),
        (["all-linear"], NotImplementedError, "routed experts"),
        (["unembed"], NotImplementedError, "does not support"),
        (["*.shared_experts.w1"], ValueError, "select all of them or none"),
    ],
)
def test_rejects_unsupported_targets(selection, error, message):
    targets = resolve_hf_lora_targets(_HF_CONFIG, target_modules=selection)
    targets = expand_hf_lora_targets(targets, get_hf_lora_targets(_HF_CONFIG))
    with pytest.raises(error, match=message):
        resolve_deepseek_v41_adapter_targets(targets)


def _param(shape, *, offset=0.0, partition_dim=None):
    tensor = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) + offset
    param = nn.Parameter(tensor)
    param.tensor_model_parallel = partition_dim is not None
    param.partition_dim = -1 if partition_dim is None else partition_dim
    return param


def _wrapper(a_shape, b_shape, *, a_dim=None, b_dim=None, slots=2):
    wrapper = nn.Module()
    adapters = []
    for slot in range(slots):
        adapter = nn.Module()
        adapter.linear_in = nn.Module()
        adapter.linear_out = nn.Module()
        adapter.linear_in.weight = _param(a_shape, offset=1000 * slot, partition_dim=a_dim)
        adapter.linear_out.weight = _param(b_shape, offset=1000 * slot, partition_dim=b_dim)
        adapters.append(adapter)
    wrapper.adapters = nn.ModuleList(adapters)
    return wrapper


def _layer(layer_number, *, rank=4, hidden=8, ffn=6):
    """TP=1 shapes for a column (wq_b, fc1), row (wo_b, fc2) and replicated (wq_a, wkv) base."""
    attention = SimpleNamespace(
        linear_q_down_proj=_wrapper((rank, hidden), (5, rank)),
        linear_kv_proj=_wrapper((rank, hidden), (7, rank)),
        linear_q_up_proj=_wrapper((rank, 5), (12, rank), a_dim=0, b_dim=0),
        linear_proj=_wrapper((rank, 10), (hidden, rank), a_dim=1, b_dim=0),
        linear_o_group_proj=nn.Parameter(torch.zeros(10, 12)),
    )
    shared = SimpleNamespace(
        linear_fc1=_wrapper((rank, hidden), (2 * ffn, rank), a_dim=0, b_dim=0),
        linear_fc2=_wrapper((rank, ffn), (hidden, rank), a_dim=1, b_dim=0),
    )
    return SimpleNamespace(
        layer_number=layer_number, self_attention=attention, mlp=SimpleNamespace(shared_experts=shared)
    )


def _model(*layer_numbers):
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[_layer(number) for number in layer_numbers]))
    return SimpleNamespace(module=SimpleNamespace(module=model))


def _no_tp(tensor, dim):
    return tensor


def test_export_names_and_shapes_follow_the_checkpoint():
    exported = dict(
        export_deepseek_v41_multi_lora([_model(1, 2)], AdapterSpec(slot=1, rank=4, alpha=8.0), gather=_no_tp)
    )

    assert get_adapter_target_modules(exported) == sorted(
        f"layers.{layer}.{module}"
        for layer in (0, 1)
        for module in (
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        )
    )
    assert exported["layers.0.attn.wq_a.lora_A.weight"].shape == (4, 8)
    assert exported["layers.0.attn.wkv.lora_B.weight"].shape == (7, 4)
    assert exported["layers.1.attn.wo_b.lora_A.weight"].shape == (4, 10)
    assert exported["layers.1.ffn.shared_experts.w2.lora_B.weight"].shape == (8, 4)
    # slot 1, not slot 0
    assert exported["layers.0.attn.wq_a.lora_A.weight"][0, 0] == 1000


def test_export_gathers_tp_shards_along_their_partition_dim():
    """Two TP ranks hold [w1_r; w3_r] each; a plain concat would interleave w1 and w3 rows."""

    def two_ranks(tensor, dim):
        return torch.cat([tensor, tensor + 0.5], dim=dim)

    model = _model(1)
    layer = model.module.module.decoder.layers[0]
    # column-parallel lora_A is sharded on its rank dim, so two ranks of 4 make max rank 8
    exported = dict(export_deepseek_v41_multi_lora([model], AdapterSpec(slot=0, rank=8, alpha=8.0), gather=two_ranks))

    fc1 = layer.mlp.shared_experts.linear_fc1.adapters[0]
    w1_local, w3_local = fc1.linear_out.weight.detach().chunk(2)
    torch.testing.assert_close(
        exported["layers.0.ffn.shared_experts.w1.lora_B.weight"], torch.cat([w1_local, w1_local + 0.5])
    )
    torch.testing.assert_close(
        exported["layers.0.ffn.shared_experts.w3.lora_B.weight"], torch.cat([w3_local, w3_local + 0.5])
    )
    # w1 and w3 share the fused adapter's lora_A, gathered along its rank dim
    lora_a = fc1.linear_in.weight.detach()
    for leaf in ("w1", "w3"):
        torch.testing.assert_close(
            exported[f"layers.0.ffn.shared_experts.{leaf}.lora_A.weight"], torch.cat([lora_a, lora_a + 0.5])
        )
    # replicated (wq_a), column (wq_b: rank-sharded A, output-sharded B), row (wo_b: input-sharded A)
    assert exported["layers.0.attn.wq_a.lora_A.weight"].shape == (4, 8)
    assert exported["layers.0.attn.wq_b.lora_A.weight"].shape == (8, 5)
    assert exported["layers.0.attn.wq_b.lora_B.weight"].shape == (24, 4)
    assert exported["layers.0.attn.wo_b.lora_A.weight"].shape == (4, 20)
    assert exported["layers.0.attn.wo_b.lora_B.weight"].shape == (16, 4)


def test_export_trims_to_the_slot_rank():
    model = _model(1)
    for module in vars(model.module.module.decoder.layers[0].self_attention).values():
        if hasattr(module, "adapters"):
            module.adapters[0].linear_in.weight.data[2:] = 0
            module.adapters[0].linear_out.weight.data[:, 2:] = 0
    for module in vars(model.module.module.decoder.layers[0].mlp.shared_experts).values():
        module.adapters[0].linear_in.weight.data[2:] = 0
        module.adapters[0].linear_out.weight.data[:, 2:] = 0

    exported = dict(export_deepseek_v41_multi_lora([model], AdapterSpec(slot=0, rank=2, alpha=4.0), gather=_no_tp))

    assert exported["layers.0.attn.wq_a.lora_A.weight"].shape == (2, 8)
    assert exported["layers.0.attn.wq_a.lora_B.weight"].shape == (5, 2)
    assert exported["layers.0.ffn.shared_experts.w3.lora_B.weight"].shape == (6, 2)


def test_export_rejects_a_model_without_adapters():
    model = SimpleNamespace(
        decoder=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_number=1,
                    self_attention=SimpleNamespace(),
                    mlp=SimpleNamespace(),
                )
            ]
        )
    )
    with pytest.raises(RuntimeError, match="found no adapters"):
        export_deepseek_v41_multi_lora([model], AdapterSpec(slot=0, rank=4, alpha=8.0), gather=_no_tp)


@pytest.mark.parametrize(
    "multi_lora,mode,model_type,expected",
    [
        (True, "raw", "deepseek_v41", True),
        (True, "bridge", "deepseek_v41", False),
        (False, "raw", "deepseek_v41", False),
        (True, "raw", "deepseek_v3", NotImplementedError),
    ],
)
def test_raw_multi_lora_is_limited_to_supported_models(monkeypatch, multi_lora, mode, model_type, expected):
    monkeypatch.setattr(lora_utils, "_hf_model_type", lambda checkpoint: model_type)
    args = SimpleNamespace(multi_lora=multi_lora, megatron_to_hf_mode=mode, hf_checkpoint="/checkpoint")
    if expected is NotImplementedError:
        with pytest.raises(NotImplementedError, match="deepseek_v3"):
            lora_utils.is_raw_multi_lora_enabled(args)
    else:
        assert lora_utils.is_raw_multi_lora_enabled(args) is expected
