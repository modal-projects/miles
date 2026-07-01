import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_disk_delta import _merge_weight_maps


def test_merge_weight_maps_ignores_empty_ranks():
    assert _merge_weight_maps([{}, {"a": "model-00000.safetensors"}, None]) == {
        "a": "model-00000.safetensors"
    }


def test_merge_weight_maps_rejects_duplicate_tensor_names():
    with pytest.raises(RuntimeError, match="duplicate disk-delta tensor names"):
        _merge_weight_maps(
            [
                {"a": "model-00000.safetensors"},
                {"a": "model-00001.safetensors"},
            ]
        )
