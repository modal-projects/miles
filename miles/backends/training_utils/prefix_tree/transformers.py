"""Opt-in causal-LM SFT using the public Transformers attention interface.

This is a standalone correctness/performance entrypoint; distributed Miles actor
integration remains separate. No model classes or upstream source are patched.
"""

from dataclasses import dataclass

import torch

from miles.backends.training_utils.prefix_tree.attention import TreeAttentionLayout, tree_attention
from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan
from miles.backends.training_utils.prefix_tree.recurrent import RecurrentTreeLayout
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


def _transformers_tree_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    scaling,
    dropout=0.0,
    sliding_window=None,
    softcap=None,
    prefix_tree_layout=None,
    prefix_tree_varlen=None,
    **kwargs,
):
    if prefix_tree_layout is None:
        raise ValueError("Miles tree attention requires an explicit prefix_tree_layout")
    if not getattr(module, "is_causal", False):
        raise ValueError("tree attention requires causal self-attention")
    if softcap is not None or any(
        kwargs.get(key) is not None for key in ("alibi", "head_mask", "sinks", "position_bias")
    ):
        raise ValueError("tree attention does not implement attention score modifiers")
    if query.shape[0] != 1 or dropout or sliding_window is not None or attention_mask is not None:
        raise ValueError("tree attention supports one compact row, full attention, and zero dropout")
    # With FP32 master weights some Q/K norms promote their outputs to FP32,
    # while the V projection remains BF16. Match HF's fused-attention autocast
    # handling before calling a kernel that requires a single Q/K/V dtype.
    if torch.is_autocast_enabled(query.device.type):
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (tensor.to(dtype) for tensor in (query, key, value))
    output = tree_attention(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        prefix_tree_layout,
        scaling=scaling,
        varlen_func=prefix_tree_varlen,
    )
    return output.unsqueeze(0), None


def enable_transformers_prefix_tree(model, *, recurrent_adapter=None) -> None:
    """Enable full causal attention by capability, without a model-name allowlist.

    The model must use Transformers' public attention interface, forward custom
    attention kwargs, and honor logical position_ids. Unsupported semantics fail
    explicitly; recurrent/hybrid models need an execution adapter of their own.
    """
    # Transformers is optional for the planner, loss, and Megatron consumers.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    config = model.config
    supports_backend = getattr(model, "_supports_attention_backend", False)
    if recurrent_adapter is not None and not supports_backend:
        # Some hybrid providers omit the legacy flag but implement the public
        # interface. An explicit adapter can use Transformers' own capability check.
        supports_backend = model._can_set_attn_implementation()
    if not supports_backend:
        raise ValueError("model must support Transformers' public attention backend interface")
    if getattr(config, "is_encoder_decoder", False) or model.get_output_embeddings() is None:
        raise ValueError("tree SFT requires a decoder-only causal language model")
    if getattr(config, "sliding_window", None) is not None or getattr(config, "use_sliding_window", False):
        raise ValueError("tree SFT currently requires full attention, without a sliding window")
    hybrid = any(kind != "full_attention" for kind in (getattr(config, "layer_types", None) or ()))
    if hybrid and recurrent_adapter is None:
        raise ValueError("hybrid/recurrent/local layers require their own tree execution adapter")
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()) or any(
        getattr(config, field, 0.0)
        for field in (
            "attention_dropout",
            "hidden_dropout",
            "hidden_dropout_prob",
            "resid_pdrop",
            "embd_pdrop",
            "attn_pdrop",
        )
    ):
        raise ValueError("tree SFT requires zero dropout")
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    _validate_rope(rope)
    if getattr(config, "alibi", False) or getattr(config, "position_embedding_type", "rotary") not in (
        "rotary",
        "absolute",
        "rope",
    ):
        raise ValueError("position-dependent attention biases require their own tree execution adapter")
    if hybrid:
        recurrent_adapter(model)
    ALL_ATTENTION_FUNCTIONS.register("miles_prefix_tree", _transformers_tree_attention)
    model.set_attn_implementation("miles_prefix_tree")


def _validate_rope(parameters):
    if parameters.get("rope_type", parameters.get("type", "default")) not in ("default", "linear", "llama3"):
        raise ValueError("length-dependent RoPE requires separate tree validation")
    for value in parameters.values():
        if isinstance(value, dict):
            _validate_rope(value)


@dataclass(frozen=True)
class TreeSFTBatch:
    """Prepare once outside timed forward/backward; no mutable global plan state."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention: TreeAttentionLayout
    supervision: SFTLossLayout
    recurrent: RecurrentTreeLayout | None = None
    sequence_ids: torch.Tensor | None = None

    @classmethod
    def from_plan(
        cls,
        plan: PrefixTreePlan,
        targets,
        *,
        device,
        weights=None,
        deterministic_reduction=False,
        recurrent_chunk_size=None,
        convolution_kernel_size=None,
    ) -> "TreeSFTBatch":
        return cls(
            input_ids=torch.tensor([plan.tokens], device=device, dtype=torch.long),
            position_ids=torch.tensor([plan.positions], device=device, dtype=torch.long),
            attention=TreeAttentionLayout.from_plan(plan, device, deterministic_reduction=deterministic_reduction),
            supervision=SFTLossLayout.from_plan(
                plan,
                targets,
                device=device,
                weights=weights,
                deterministic_reduction=deterministic_reduction,
            ),
            recurrent=(
                RecurrentTreeLayout.from_plan(
                    plan,
                    device=device,
                    chunk_size=recurrent_chunk_size,
                    convolution_kernel_size=convolution_kernel_size,
                )
                if recurrent_chunk_size is not None
                else None
            ),
            sequence_ids=(
                torch.repeat_interleave(
                    torch.arange(len(plan.segment_parents), device=device, dtype=torch.int32),
                    torch.tensor(
                        [b - a for a, b in zip(plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True)],
                        device=device,
                    ),
                ).unsqueeze(0)
                if recurrent_chunk_size is not None
                else None
            ),
        )

    def loss(self, model, *, varlen_func=None, shared=True) -> torch.Tensor:
        if not shared and len(self.attention.plan.tokens) != self.attention.plan.expanded_tokens:
            raise ValueError("Native packed execution requires a plan without shared rows")
        recurrent = (
            {}
            if self.recurrent is None
            else dict(
                prefix_tree_recurrent=self.recurrent,
                prefix_tree_shared=shared,
                seq_idx=self.sequence_ids,
                cu_seq_lens_q=self.attention.cu_q,
            )
        )
        output = model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            # The custom backend has no registered physical-order mask builder;
            # Transformers delegates causal masking to our logical tree layout.
            attention_mask=None,
            use_cache=False,
            prefix_tree_layout=self.attention,
            prefix_tree_varlen=varlen_func,
            **recurrent,
        )
        return self.supervision.loss(output.logits)
