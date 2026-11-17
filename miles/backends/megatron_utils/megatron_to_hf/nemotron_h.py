"""Nemotron-H raw base-weight export, sharing names and packing with its HF loader."""

from functools import lru_cache

import torch

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.nemotron_h.weights import hf_weight_names, unpack_mamba_components


@lru_cache(maxsize=8)
def _config(checkpoint):
    return load_hf_config(checkpoint)


def convert_nemotron_h_to_hf(args, name, weight):
    name = name.removeprefix("module.module.")
    names = hf_weight_names(name)
    if len(names) == 1 and not name.endswith(("mixer.in_proj.weight", "mixer.conv1d_weight", "mixer.conv1d_bias")):
        return [(names[0], weight)]
    hf = _config(args.hf_checkpoint)
    weight = unpack_mamba_components(hf, name, weight, args.backend.tensor_model_parallel_size)
    if len(names) == 1:
        return [(names[0], weight)]
    grouped = weight.view(hf.num_key_value_heads, -1, hf.head_dim, hf.hidden_size)
    q, k, v = torch.split(grouped, [hf.num_attention_heads // hf.num_key_value_heads, 1, 1], dim=1)
    return list(zip(names, (part.reshape(-1, hf.hidden_size) for part in (q, k, v)), strict=True))
