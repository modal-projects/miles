from __future__ import annotations

import functools
import json
import os
import struct
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HfTensorSpec:
    dtype: str
    shape: tuple[int, ...]


@functools.cache
def load_weight_map(checkpoint: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}
    with open(index_path) as f:
        return dict(json.load(f).get("weight_map", {}))


@functools.cache
def load_safetensors_specs(checkpoint: str) -> dict[str, HfTensorSpec]:
    """Read safetensors header metadata without loading tensor data."""
    specs: dict[str, HfTensorSpec] = {}
    for filename in _safetensors_files(checkpoint):
        try:
            header = _read_safetensors_header(os.path.join(checkpoint, filename))
        except OSError:
            return {}
        for name, info in header.items():
            if name == "__metadata__" or "dtype" not in info or "shape" not in info:
                continue
            specs[name] = HfTensorSpec(
                dtype=str(info["dtype"]),
                shape=tuple(int(dim) for dim in info["shape"]),
            )
    return specs


def load_tensor_spec(checkpoint: str, name: str) -> HfTensorSpec | None:
    return load_safetensors_specs(checkpoint).get(name)


def load_fp8_weight_specs(checkpoint: str) -> dict[str, HfTensorSpec] | None:
    names = set(load_weight_map(checkpoint)) or set(load_safetensors_specs(checkpoint))
    specs = {
        name: spec
        for name, spec in load_safetensors_specs(checkpoint).items()
        if name.endswith(".weight")
        and spec.dtype in {"F8_E4M3", "F8_E4M3FN"}
        and name.replace(".weight", ".weight_scale") in names
    }
    return specs or None


def _safetensors_files(checkpoint: str) -> list[str]:
    weight_map = load_weight_map(checkpoint)
    if weight_map:
        return sorted({filename for filename in weight_map.values() if str(filename).endswith(".safetensors")})
    try:
        return sorted(filename for filename in os.listdir(checkpoint) if filename.endswith(".safetensors"))
    except OSError:
        return []


def _read_safetensors_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))
