"""Model-independent packed Megatron SFT; architecture adapters attach tree metadata."""

from dataclasses import dataclass
from typing import Protocol

import torch

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


def validate_megatron_tree_config(config) -> None:
    """Common sharing eligibility; architecture adapters add operator constraints."""
    if getattr(config, "attention_dropout", 0.0) or getattr(config, "hidden_dropout", 0.0):
        raise ValueError("Shared-prefix SFT requires zero attention and hidden dropout")
    if (
        getattr(config, "moe_router_force_load_balancing", False)
        or getattr(config, "moe_router_force_biased", None) is not None
    ):
        raise ValueError("Shared-prefix SFT does not support randomized benchmark routing")
    if getattr(config, "moe_router_jitter_eps", None):
        raise ValueError("Shared-prefix SFT does not support router input jitter")
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("Shared-prefix SFT does not support capacity-based token dropping")
    coefficient = getattr(config, "moe_aux_loss_coeff", 0.0)
    coefficients = coefficient if isinstance(coefficient, (list, tuple)) else (coefficient,)
    if any(coefficients):
        raise ValueError(
            "Shared-prefix SFT requires zero MoE auxiliary loss until occurrence weighting is implemented"
        )
    if getattr(config, "moe_router_enable_expert_bias", False) and getattr(config, "moe_router_bias_update_rate", 0.0):
        raise ValueError("Shared-prefix SFT requires fixed router expert bias")


class TreeExecutionLayout(Protocol):
    plan: PrefixTreePlan

    def prepare_packed(self, model, packed) -> None:
        """Validate architecture capabilities and attach the execution layout."""
        ...


@dataclass(frozen=True)
class MegatronTreeSFTBatch:
    layout: TreeExecutionLayout
    losses: SFTLossLayout
    tokens: torch.Tensor
    position_ids: torch.Tensor
    slot_counts: torch.Tensor | None = None

    @classmethod
    def from_plan(
        cls,
        plan: PrefixTreePlan,
        targets,
        *,
        device,
        layout: TreeExecutionLayout,
        weights=None,
        deterministic_reduction=False,
        adapter_slots=None,
        num_adapter_slots=None
    ):
        if layout.plan is not plan:
            raise ValueError("Execution layout and supervision must use the same plan")
        return cls(
            layout=layout,
            losses=SFTLossLayout.from_plan(
                plan, targets, device=device, weights=weights, deterministic_reduction=deterministic_reduction
            ),
            slot_counts=_slot_counts(plan, adapter_slots, num_adapter_slots, device=device),
            tokens=torch.tensor(plan.tokens, device=device, dtype=torch.long).unsqueeze(0),
            position_ids=torch.tensor(plan.positions, device=device, dtype=torch.long).unsqueeze(0),
        )

    def loss(self, model, *, shared=True):
        # Keep the large vocabulary sharded during training; .logits remains
        # the gathered-output interface for numerical diagnostics.
        from megatron.core.utils import get_attr_wrapped_model

        logits = self._forward_logits(model, gather_vocab=False, shared=shared)
        if get_attr_wrapped_model(model, "parallel_output", allow_none=False):
            return self.losses.vocab_parallel_loss(logits)
        return self.losses.loss(logits)

    def logits(self, model, *, shared=True):
        return self._forward_logits(model, gather_vocab=True, shared=shared)

    def _forward_logits(self, model, *, gather_vocab, shared):
        # Optional Megatron dependency; CPU planning does not import its GPU stack.
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.utils import get_attr_wrapped_model

        if not shared and len(self.layout.plan.tokens) != self.layout.plan.expanded_tokens:
            raise ValueError("Native packed execution requires an expanded plan without shared rows")
        config = get_attr_wrapped_model(model, "config", allow_none=False)
        if getattr(config, "sequence_parallel", False):
            tp_size = config.tensor_model_parallel_size
            if self.tokens.numel() % tp_size:
                raise ValueError(
                    "Compact token count must be divisible by TP size when sequence parallelism is enabled"
                )
        cu = torch.tensor(self.layout.plan.segment_offsets, device=self.tokens.device, dtype=torch.int32)
        maximum = max(
            b - a
            for a, b in zip(self.layout.plan.segment_offsets[:-1], self.layout.plan.segment_offsets[1:], strict=True)
        )
        packed = PackedSeqParams(
            cu_seqlens_q=cu, cu_seqlens_kv=cu, qkv_format="thd", max_seqlen_q=maximum, max_seqlen_kv=maximum
        )
        if shared:
            validate_megatron_tree_config(config)
            self.layout.prepare_packed(model, packed)
        if self.slot_counts is not None:
            from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

            set_tokens_per_adapter_slot(model, self.slot_counts)
        logits = model(self.tokens, position_ids=self.position_ids, attention_mask=None, packed_seq_params=packed)
        if gather_vocab and get_attr_wrapped_model(model, "parallel_output", allow_none=False):
            # Gather after the complete output-layer wrapper. Asking the base
            # linear to gather early leaves a native multi-LoRA delta sharded.
            from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

            logits = gather_from_tensor_model_parallel_region(logits)
        return logits


def _slot_counts(plan, adapter_slots, num_adapter_slots, *, device):
    if adapter_slots is None:
        if num_adapter_slots is not None:
            raise ValueError("num_adapter_slots requires per-sample adapter_slots")
        return None
    if len(adapter_slots) != len(plan.sample_nodes) or not num_adapter_slots or num_adapter_slots < 1:
        raise ValueError("Provide one adapter slot per sample and a positive slot capacity")
    owners = [-1] * len(plan.tokens)
    for slot, path in zip(adapter_slots, plan.sample_nodes, strict=True):
        if not isinstance(slot, int) or not 0 <= slot < num_adapter_slots:
            raise ValueError("Adapter slot is outside the model's slot capacity")
        for row in path:
            if owners[row] not in (-1, slot):
                raise ValueError(
                    "A tree row cannot be shared across adapters; include adapter identity in sharing_keys"
                )
            owners[row] = slot
    if any(left > right for left, right in zip(owners[:-1], owners[1:], strict=True)):
        raise ValueError("Group samples by adapter slot before constructing the tree")
    counts = [0] * num_adapter_slots
    for slot in owners:
        counts[slot] += 1
    return torch.tensor(counts, device=device, dtype=torch.int64)
