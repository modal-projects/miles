import importlib.util
from pathlib import Path

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
