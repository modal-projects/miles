"""GLM eligibility and compatibility names over common recurrent tree maps."""

from miles.backends.training_utils.prefix_tree.recurrent import RecurrentTreeLayout
from miles.backends.training_utils.prefix_tree.recurrent import TreeLevel as TreeLevel

KDA_CHUNK_SIZE = 64


class GlmTreeLayout(RecurrentTreeLayout):
    @classmethod
    def from_plan(cls, plan, *, device):
        return super().from_plan(plan, device=device, chunk_size=KDA_CHUNK_SIZE)

    def prepare_packed(self, model, packed) -> None:
        # Megatron and replay are optional GPU dependencies.
        from megatron.core.utils import get_attr_wrapped_model

        from miles.utils.replay_base import all_replay_managers

        config = get_attr_wrapped_model(model, "config", allow_none=False)
        if (
            getattr(config, "prefix_tree_layout_type", None) is not GlmTreeLayout
        ):  # config-access-exempt: only the tree provider adds this capability marker
            raise ValueError("Build the native GLM provider with the shared-prefix specification")
        if any(manager.enabled and manager.stage != "fallthrough" for manager in all_replay_managers):
            raise ValueError("GLM tree SFT does not yet map router/indexer replay occurrence streams")
        packed.glm_tree_layout = self
