"""Configuration for multiple serialized rollout weight representations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightView
from miles.utils.hf_utils.config import load_hf_config


def get_quantized_weight_basenames(hf_checkpoint: str, quantization_config: dict) -> set[str] | None:
    """Return the checkpoint-authoritative set of quantized HF modules."""
    with open(os.path.join(hf_checkpoint, "model.safetensors.index.json")) as index_file:
        names = json.load(index_file)["weight_map"]
    method = quantization_config.get("quant_method")
    if method == "compressed-tensors":
        return {name.removesuffix(".weight_packed") for name in names if name.endswith(".weight_packed")}
    if method == "fp8":
        suffixes = (".weight_scale_inv", ".weight_scale")
        return {name[: -len(suffix)] for name in names for suffix in suffixes if name.endswith(suffix)}
    return None


def _source_only_suffixes(quantization_config: dict | None) -> tuple[str, ...]:
    if quantization_config is None:
        return ()
    if quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4":
        # W4A16 input scaling is immutable checkpoint metadata. The offline
        # converter writes one; the live exporter deliberately emits only state
        # derived from trainer weights.
        return (".input_scale",)
    return ()


def load_weight_views(config: object) -> tuple[WeightView, ...]:
    """Load each view's quantization policy from its canonical checkpoint."""
    if config is None:
        return ()
    if not isinstance(config, Mapping) or not config:
        raise ValueError("--update-weight-views must be a non-empty JSON object")

    views = []
    for name, checkpoint in config.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"weight view names must be identifiers, got {name!r}")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"weight view {name!r} must name a checkpoint directory")
        if not os.path.isdir(checkpoint):
            raise ValueError(f"weight view {name!r} checkpoint is not a local directory: {checkpoint}")
        hf_config = load_hf_config(checkpoint)
        quantization_config = getattr(hf_config, "quantization_config", None)
        quantized_weight_basenames = get_quantized_weight_basenames(checkpoint, quantization_config) if quantization_config is not None else None
        views.append(
            WeightView(
                name=name,
                checkpoint=checkpoint,
                quantization_config=quantization_config,
                quantized_weight_basenames=(frozenset(quantized_weight_basenames) if quantized_weight_basenames is not None else None),
                source_only_suffixes=_source_only_suffixes(quantization_config),
            )
        )
    return tuple(views)
