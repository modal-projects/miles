"""Compatibility entrypoint; GLM supplies only its architecture execution layout."""

from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.megatron import MegatronTreeSFTBatch


class GlmTreeSFTBatch(MegatronTreeSFTBatch):
    @classmethod
    def from_plan(cls, plan, targets, *, device, **kwargs):
        return super().from_plan(
            plan, targets, device=device, layout=GlmTreeLayout.from_plan(plan, device=device), **kwargs
        )
