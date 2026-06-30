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
def _load_weight_map(checkpoint: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return {}
    with open(index_path) as f:
        return dict(json.load(f).get("weight_map", {}))


@functools.cache
def _load_safetensors_specs(checkpoint: str) -> dict[str, HfTensorSpec]:
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
    return _load_safetensors_specs(checkpoint).get(name)


def load_fp8_weight_specs(checkpoint: str) -> dict[str, HfTensorSpec] | None:
    # A weight is native fp8-with-a-scale iff its ``.weight_scale`` sibling is
    # in the same headers, so the header specs are the only source needed.
    specs = _load_safetensors_specs(checkpoint)
    fp8 = {
        name: spec
        for name, spec in specs.items()
        if name.endswith(".weight")
        and spec.dtype in {"F8_E4M3", "F8_E4M3FN"}
        and name.replace(".weight", ".weight_scale") in specs
    }
    return fp8 or None


def _safetensors_files(checkpoint: str) -> list[str]:
    weight_map = _load_weight_map(checkpoint)
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
