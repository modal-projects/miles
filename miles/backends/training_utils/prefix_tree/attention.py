"""Asymmetric varlen tree attention, with a slow independent reference backend."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan


@dataclass(frozen=True)
class TreeAttentionLayout:
    plan: PrefixTreePlan
    kv_indices: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    max_q: int
    max_kv: int

    @classmethod
    def from_plan(cls, plan: PrefixTreePlan, device) -> "TreeAttentionLayout":
        return cls(
            plan=plan,
            kv_indices=torch.tensor(plan.kv_indices, device=device, dtype=torch.long),
            cu_q=torch.tensor(plan.segment_offsets, device=device, dtype=torch.int32),
            cu_kv=torch.tensor(plan.kv_offsets, device=device, dtype=torch.int32),
            max_q=max(b - a for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True)),
            max_kv=max(b - a for a, b in zip(plan.kv_offsets[:-1], plan.kv_offsets[1:], strict=True)),
        )


def tree_attention(query, key, value, layout: TreeAttentionLayout, *, scaling: float, varlen_func=None):
    """Input/output THD; Q/K must already carry their logical-position rotations.

    ``varlen_func`` follows flash_attn_varlen_func's training-capable interface.
    It must implement bottom-right causal alignment when Q is shorter than KV.
    Index-select autograd sums the gradients of repeated ancestor KV rows.
    """
    gathered_k = key.index_select(0, layout.kv_indices)
    gathered_v = value.index_select(0, layout.kv_indices)
    if varlen_func is not None:
        return varlen_func(
            query,
            gathered_k,
            gathered_v,
            cu_seqlens_q=layout.cu_q,
            cu_seqlens_k=layout.cu_kv,
            max_seqlen_q=layout.max_q,
            max_seqlen_k=layout.max_kv,
            softmax_scale=scaling,
            causal=True,
        )
    return _reference_attention(query, gathered_k, gathered_v, layout, scaling=scaling)


def _reference_attention(query, key, value, layout, *, scaling):
    outputs = []
    plan = layout.plan
    for segment in range(len(plan.segment_parents)):
        q_start, q_end = plan.segment_offsets[segment : segment + 2]
        k_start, k_end = plan.kv_offsets[segment : segment + 2]
        q = query[q_start:q_end].transpose(0, 1)
        k = key[k_start:k_end].transpose(0, 1)
        v = value[k_start:k_end].transpose(0, 1)
        repeats = q.shape[0] // k.shape[0]
        k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
        q_positions = torch.arange(q_end - q_start, device=q.device) + (k_end - k_start) - (q_end - q_start)
        allowed = torch.arange(k_end - k_start, device=q.device)[None, :] <= q_positions[:, None]
        outputs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scaling).transpose(0, 1))
    return torch.cat(outputs, dim=0)
