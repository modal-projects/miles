"""Compatibility imports for earlier recorded-math probes."""

from miles.backends.training_utils.prefix_tree.data import SFTMicrobatch as MathMicrobatch
from miles.backends.training_utils.prefix_tree.data import global_token_scale
from miles.backends.training_utils.prefix_tree.data import sft_microbatches as math_microbatches

__all__ = ["MathMicrobatch", "global_token_scale", "math_microbatches"]
