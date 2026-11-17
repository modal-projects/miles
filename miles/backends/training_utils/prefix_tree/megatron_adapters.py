"""Architecture capability boundary for the common packed trainer transport.

Register native provider specifications here, never checkpoint/model names.
Additional architectures reuse packing, loss restoration, and kernel policy.
"""

from functools import partial


def configure_tree_provider(provider):
    # GLM provider dependencies are optional outside the validated GPU image.
    from megatron.bridge.models.glm5_next.layer_specs import glm5_next_layer_spec

    from miles.backends.training_utils.prefix_tree.glm53.bridge import glm53_tree_layer_spec
    from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout

    native = provider.transformer_layer_spec
    while isinstance(native, partial):
        native = native.func
    if native is not glm5_next_layer_spec:
        raise ValueError("No packed-prefix adapter registered for this native provider specification")
    provider.transformer_layer_spec = glm53_tree_layer_spec
    return GlmTreeLayout
