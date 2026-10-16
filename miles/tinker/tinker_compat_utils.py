"""Tinker SDK names that moved between SDK releases."""

try:
    # Tinker 0.29 and later define the top-k padding logprob next to TopkLogprobs
    from tinker.types.topk_logprobs import MASK_LOGPROB
except ImportError:
    # Earlier Tinker releases define it on the sample response module
    from tinker.types.sample_response import MASK_LOGPROB

__all__ = ["MASK_LOGPROB"]
