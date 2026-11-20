"""Native provider selection never falls back to a Bridge adapter."""

import pytest

from miles.backends.training_utils.prefix_tree.megatron_adapters import resolve_native_tree_adapter


@pytest.mark.parametrize("factory", ["get_glm5_next_spec", "get_glm5_next_tree_spec"])
def test_native_provider_resolution_is_idempotent(factory):
    module = (
        "miles_plugins.models.glm5_next.glm5_next"
        if factory == "get_glm5_next_spec"
        else "miles.backends.training_utils.prefix_tree.glm53.native"
    )
    selected = resolve_native_tree_adapter([module, factory])
    assert resolve_native_tree_adapter(selected.spec) == selected
    assert selected.spec == ("miles.backends.training_utils.prefix_tree.glm53.native", "get_glm5_next_tree_spec")


@pytest.mark.parametrize("spec", [None, [], "glm5_next", ["unknown", "factory"], ["megatron.bridge", "provider"]])
def test_unregistered_providers_fail_before_model_construction(spec):
    with pytest.raises(ValueError, match="native provider"):
        resolve_native_tree_adapter(spec)
