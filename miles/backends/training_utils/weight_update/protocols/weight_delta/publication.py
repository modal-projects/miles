"""The files a weight-delta sync writes for each version: one safetensors file per owner, then the manifest."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

FORMAT = "weight-delta/v1"
CODECS = ("gap-zstd", "snappy-zstd")
FRAME_NBYTES = 1 << 20
ALIGNMENT = 16
_SAFETENSORS_DTYPES = {np.dtype(np.int64): "I64", np.dtype(np.uint8): "U8", np.dtype(np.bool_): "BOOL"}


def is_raw(dtype: str, shape: Sequence[int]) -> bool:
    return dtype == "F32" or len(shape) <= 1


def version_dir(delta_dir: Path, version: int) -> Path:
    return delta_dir / f"weight_v{version:06d}"


class _Tensor(NamedTuple):
    name: str
    dtype: str
    shape: list[int]
    raw: bool
    offset: int  # in the payload
    nbytes: int
    inner_nbytes: int


class OwnerWriter:
    """One trainer GPU's file for one version: the payload of its changed tensors, and as columns where
    each tensor's bytes lie in it. Ragged columns are flat values plus each row's end."""

    def __init__(self, directory: Path, owner: int) -> None:
        self._path = directory / f"owner-{owner:05d}.safetensors"
        self._tensors: list[_Tensor] = []
        self._chunks: list[Mapping] = []
        self._frames: list[Mapping] = []
        self._chunk_ends: list[int] = []
        self._frame_ends: list[int] = []
        self._payload: list = []
        self._payload_nbytes = 0

    def add_raw(self, name: str, dtype: str, shape: Sequence[int], data) -> None:
        self._add(name, dtype, shape, data, raw=True, inner_nbytes=0, chunks=(), frames=())

    def add_xor(
        self, name: str, dtype: str, shape: Sequence[int], frames: Sequence[Mapping], payload, zstd: Mapping
    ) -> None:
        self._add(
            name,
            dtype,
            shape,
            payload,
            raw=False,
            inner_nbytes=zstd["inner_nbytes"],
            chunks=zstd["chunks"],
            frames=frames,
        )

    def finish(self) -> dict | None:
        """The file's name and size, or None when none of this owner's tensors changed."""
        if not self._tensors:
            return None
        tensors = self._tensors
        names = [tensor.name.encode() for tensor in tensors]
        dtypes = list(dict.fromkeys(tensor.dtype for tensor in tensors))
        columns = {
            "tensors.name_end": np.cumsum([len(name) for name in names], dtype=np.int64),
            "tensors.shape": np.array([dim for tensor in tensors for dim in tensor.shape], dtype=np.int64),
            "tensors.shape_end": np.cumsum([len(tensor.shape) for tensor in tensors], dtype=np.int64),
            "tensors.offset": np.array([tensor.offset for tensor in tensors], dtype=np.int64),
            "tensors.nbytes": np.array([tensor.nbytes for tensor in tensors], dtype=np.int64),
            "tensors.inner_nbytes": np.array([tensor.inner_nbytes for tensor in tensors], dtype=np.int64),
            "tensors.chunk_end": np.array(self._chunk_ends, dtype=np.int64),
            "tensors.frame_end": np.array(self._frame_ends, dtype=np.int64),
            **_regions("chunks", self._chunks),
            **_regions("frames", self._frames),
            "tensors.name": np.frombuffer(b"".join(names), dtype=np.uint8),
            "tensors.dtype": np.array([dtypes.index(tensor.dtype) for tensor in tensors], dtype=np.uint8),
            "tensors.raw": np.array([tensor.raw for tensor in tensors], dtype=np.bool_),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        nbytes = _write_safetensors(
            self._path, columns, self._payload, self._payload_nbytes, {"dtypes": ",".join(dtypes)}
        )
        return {"name": self._path.name, "nbytes": nbytes}

    def _add(self, name, dtype, shape, data, *, raw, inner_nbytes, chunks, frames) -> None:
        offset = -(-self._payload_nbytes // ALIGNMENT) * ALIGNMENT
        nbytes = memoryview(data).nbytes
        self._payload += [bytes(offset - self._payload_nbytes), data]
        self._payload_nbytes = offset + nbytes
        self._tensors.append(_Tensor(name, dtype, list(shape), raw, offset, nbytes, inner_nbytes))
        self._chunks += chunks
        self._frames += frames
        self._chunk_ends.append(len(self._chunks))
        self._frame_ends.append(len(self._frames))


def seal(
    directory: Path, files: Iterable[Mapping | None], *, codec: str, base_version: int, target_version: int
) -> Path:
    manifest = {
        "format": FORMAT,
        "codec": codec,
        "frame_nbytes": FRAME_NBYTES,
        "base_version": base_version,
        "target_version": target_version,
        "files": sorted((file for file in files if file is not None), key=lambda file: file["name"]),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    with path.open("xb") as output:
        output.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        output.flush()
        os.fsync(output.fileno())
    return path


def _regions(table: str, rows: Sequence[Mapping]) -> dict[str, np.ndarray]:
    fields = ("offset", "nbytes", "inner_offset", "inner_nbytes")
    return {f"{table}.{field}": np.array([row[field] for row in rows], dtype=np.int64) for field in fields}


def _write_safetensors(
    path: Path, columns: Mapping[str, np.ndarray], payload: Sequence, payload_nbytes: int, metadata: Mapping[str, str]
) -> int:
    """The columns in order, then the payload pieces streamed as one U8 tensor, without joining them."""
    header: dict[str, dict] = {"__metadata__": dict(metadata)}
    offset = 0
    for name, column in columns.items():
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[column.dtype],
            "shape": list(column.shape),
            "data_offsets": [offset, offset + column.nbytes],
        }
        offset += column.nbytes
    header["payload"] = {"dtype": "U8", "shape": [payload_nbytes], "data_offsets": [offset, offset + payload_nbytes]}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(text)) + text)
        for column in columns.values():
            output.write(column.tobytes())
        for piece in payload:
            output.write(piece)
        output.flush()
        os.fsync(output.fileno())
        return output.tell()
