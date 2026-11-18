"""Native provider-spec capability boundary for packed-prefix execution."""

from dataclasses import dataclass

from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout


@dataclass(frozen=True)
class NativeTreeAdapter:
    spec: tuple[str, str]
    layout_type: type


_NATIVE_GLM_SPEC = ("miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec")
_TREE_GLM_SPEC = ("miles.backends.training_utils.prefix_tree.glm53.native", "get_glm5_next_tree_spec")


def resolve_native_tree_adapter(spec) -> NativeTreeAdapter:
    if not isinstance(spec, (list, tuple)) or tuple(spec) not in (_NATIVE_GLM_SPEC, _TREE_GLM_SPEC):
        raise ValueError("No shared-prefix adapter registered for this native provider specification")
    return NativeTreeAdapter(_TREE_GLM_SPEC, GlmTreeLayout)
