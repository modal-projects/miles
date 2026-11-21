"""DeepSeek-V4.1 bindings that do not need a served model: routes and the rounded F32 checkpoint tensors."""

import pytest
import torch

from miles.backends.sglang_utils.gpu_delta.bindings import CanonicalShadows, ParameterBindings
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v4 import DeepseekV4Mapping, route

ROUNDED = [
    "model.layers.2.self_attn.compressor.norm.weight",
    "model.layers.2.self_attn.indexer.k_norm.weight",
    "model.layers.1.engram.q_weight",
    "model.layers.1.engram.k_weight",
]


@pytest.mark.parametrize("live", ROUNDED)
def test_norms_and_engram_weights_take_the_rounded_route(live):
    assert route(live) == "rounded"


def _served(dtype):
    """The modules the four rounded tensors live in, with their parameters built in ``dtype``."""
    root = torch.nn.Module()
    for live in ROUNDED:
        module_name, leaf = live.rsplit(".", 1)
        parent = root
        for part in module_name.split("."):
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        shape = (4, 8) if "engram" in live else (16,)
        parent.register_parameter(leaf, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
    return root


def _mapping(root):
    mapping = object.__new__(DeepseekV4Mapping)
    mapping.parameters = ParameterBindings(root, (), CanonicalShadows({}, create=True))
    mapping.derived, mapping._sinks, mapping._rounded = [], {}, {}
    return mapping


def test_an_f32_checkpoint_tensor_served_in_bf16_is_rounded_from_its_shadow_as_the_loader_copies_it():
    root = _served(torch.bfloat16)
    mapping = _mapping(root)
    checkpoint = {}
    for live in ROUNDED:
        weight = root.get_parameter(live)
        meta = {"dtype": "F32", "shape": list(weight.shape), "encoding": "xor_raw_bytes"}
        binding = mapping._bind_rounded(live.removeprefix("model."), meta, live)
        shadow, served = mapping._rounded[live.removeprefix("model.")]
        assert served is weight and shadow.dtype == torch.float32 and tuple(shadow.shape) == tuple(weight.shape)
        # The binding XORs the canonical F32 bytes in the shadow, never the BF16 parameter.
        assert all(
            target.untyped_storage().data_ptr() == shadow.untyped_storage().data_ptr() for target in binding.storage
        )
        checkpoint[live] = torch.randn(weight.shape, dtype=torch.float32)
        shadow.copy_(checkpoint[live])
    mapping.finish()
    assert len(mapping.derived) == len(ROUNDED)
    for image, live in zip(mapping.derived, ROUNDED, strict=True):
        loaded = torch.empty_like(root.get_parameter(live)).copy_(checkpoint[live])
        (value,) = image.values()
        assert torch.equal(value.view(torch.int16), loaded.view(torch.int16))


def test_a_tensor_served_in_its_checkpoint_dtype_binds_plainly():
    root = _served(torch.float32)
    mapping = _mapping(root)
    live = ROUNDED[0]
    weight = root.get_parameter(live)
    mapping._bind_rounded(
        live.removeprefix("model."), {"dtype": "F32", "shape": list(weight.shape), "encoding": "xor_raw_bytes"}, live
    )
    assert mapping._rounded == {}
