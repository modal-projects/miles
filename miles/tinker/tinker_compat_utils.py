"""Tinker SDK names that moved between SDK releases."""

try:
    # Tinker 0.29 and later define the top-k padding logprob next to TopkLogprobs
    from tinker.types.topk_logprobs import MASK_LOGPROB
except ImportError:
    # Earlier Tinker releases define it on the sample response module
    from tinker.types.sample_response import MASK_LOGPROB

__all__ = ["MASK_LOGPROB", "set_topk_shape"]


def set_topk_shape(topk_message, num_positions: int, k: int) -> None:
    """Sets the position count and k on a top-k logprobs protobuf message.

    Tinker 0.29 renamed the position count field from ``prompt_length`` to ``length``,
    so this writes whichever one the installed SDK's message defines."""
    fields = topk_message.DESCRIPTOR.fields_by_name
    setattr(topk_message, "length" if "length" in fields else "prompt_length", num_positions)
    topk_message.k = k
