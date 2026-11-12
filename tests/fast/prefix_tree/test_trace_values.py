import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/glm53_trace_values.py"
_SPEC = importlib.util.spec_from_file_location("_glm_trace_values", _PATH)
_TRACE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRACE)


def test_nested_fused_norm_projection_is_not_silently_skipped():
    projection = torch.ones(2, 3)
    norm = torch.full((2, 4), 2.0)
    native = ((projection, norm), None)
    assert _TRACE.first_tensor(native) is projection
    converted = _TRACE.map_tensors(native, lambda item: item.to(torch.bfloat16))
    assert isinstance(converted, tuple) and isinstance(converted[0], tuple)
    assert converted[1] is None
    assert converted[0][0].dtype == converted[0][1].dtype == torch.bfloat16
    torch.testing.assert_close(converted[0][1].float(), norm)
    assert norm.dtype == projection.dtype == torch.float32


def test_tensor_traversal_preserves_bias_and_empty_returns():
    assert _TRACE.first_tensor((None, (None,))) is None
    native = [torch.ones(2), None, (torch.zeros(3), "metadata")]
    converted = _TRACE.map_tensors(native, lambda item: item.double())
    assert isinstance(converted, list) and isinstance(converted[2], tuple)
    assert converted[1] is None and converted[2][1] == "metadata"
    assert converted[0].dtype == converted[2][0].dtype == torch.float64


@pytest.mark.parametrize("attach", [False, True])
def test_attention_observer_compares_restored_boundaries_and_restores_native_functions(monkeypatch, attach):
    from types import ModuleType, SimpleNamespace

    import miles.backends.training_utils.prefix_tree.glm53 as package

    path = _PATH.with_name("glm53_attention_trace.py")
    spec = importlib.util.spec_from_file_location("_attention_trace", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    conv = ModuleType("fla.modules.conv.causal_conv1d")
    conv.causal_conv1d = lambda x: (x + 1, None)
    native = ModuleType("megatron.bridge.models.glm5_next.kda")
    native.chunk_kda = lambda x: (x * 2, None)
    tree = SimpleNamespace(_tree_convolution=lambda x: conv.causal_conv1d(x)[0])
    bridge = SimpleNamespace(tree_kda=lambda x: tree._tree_convolution(x) * 2)
    monkeypatch.setitem(sys.modules, conv.__name__, conv)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(package, "bridge", bridge, raising=False)
    monkeypatch.setattr(package, "kda", tree, raising=False)

    class Attention(torch.nn.Module):
        local_num_heads = 1
        conv1d = None

        def forward(self, x):
            return bridge.tree_kda(x) if attach else native.chunk_kda(conv.causal_conv1d(x)[0])[0]

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = Attention()
    model.decoder.layers = torch.nn.ModuleList([layer])
    captures = []
    trace = SimpleNamespace(layers={0}, capture=lambda name, value, **kwargs: captures.append((name, value)))
    handles, originals = observer.install_attention_trace(model, trace, attach=attach)
    value = torch.arange(12).reshape(1, 4, 3).float()
    try:
        actual = layer.self_attention(value)
    finally:
        for handle in handles:
            handle.remove()
        for module, name, original in originals:
            setattr(module, name, original)
    torch.testing.assert_close(actual, (value + 1) * 2)
    assert [name for name, _ in captures] == [
        "decoder.layers.0.self_attention.convolution.output",
        "decoder.layers.0.self_attention.recurrence.output",
    ]
    torch.testing.assert_close(captures[0][1], (value + 1).transpose(0, 1))
    torch.testing.assert_close(captures[1][1], ((value + 1) * 2).transpose(0, 1))
    layer.self_attention(value)
    assert len(captures) == 2
