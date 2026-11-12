"""GLM eligibility and compatibility names over common recurrent tree maps."""

from miles.backends.training_utils.prefix_tree.recurrent import RecurrentTreeLayout
from miles.backends.training_utils.prefix_tree.recurrent import TreeLevel as TreeLevel

KDA_CHUNK_SIZE = 64


class GlmTreeLayout(RecurrentTreeLayout):
    @classmethod
    def from_plan(cls, plan, *, device):
        return super().from_plan(plan, device=device, chunk_size=KDA_CHUNK_SIZE)

    def prepare_packed(self, model, packed) -> None:
        # Bridge/Megatron and replay are optional GPU dependencies.
        from megatron.core.utils import get_attr_wrapped_model

        from miles.backends.training_utils.prefix_tree.glm53.bridge import glm53_tree_layer_spec
        from miles.utils.replay_base import all_replay_managers

        config = get_attr_wrapped_model(model, "config", allow_none=False)
        if getattr(config, "transformer_layer_spec", None) is not glm53_tree_layer_spec:
            raise ValueError("Build the GLM provider with transformer_layer_spec=glm53_tree_layer_spec")
        if any(manager.enabled and manager.stage != "fallthrough" for manager in all_replay_managers):
            raise ValueError("GLM tree SFT does not yet map router/indexer replay occurrence streams")
        packed.glm_tree_layout = self
