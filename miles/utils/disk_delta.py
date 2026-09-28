from __future__ import annotations

import glob
import json
import os
import struct
import zlib
from functools import cache

import numpy as np

# The delta phases (diff, zstd, checksum) are memory-bandwidth bound and release the GIL,
# so a thread pool over tensors recovers the bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

# Trainer-side (publish) helpers for disk-level delta weight sync. The receive side —
# materializing the host-local checkpoint and applying published deltas in place — lives in
# the engine behind its /pull_weights endpoint (sglang.srt.weight_sync.local_checkpoint), so it
# runs on every host of a multi-node engine while miles only talks to one endpoint.


def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values.
    Idempotent to apply, unlike xor (an involution); the trainer picks the encoding per the docs."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])


class _Adler32:
    """adler32 behind the incremental .update / .hexdigest interface the hash objects expose."""

    def __init__(self):
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()


def load_delta_lineage_checksums(delta_dir: str, target_version: int) -> dict[str, tuple[str, str]]:
    """Return each changed tensor's checksum at ``target_version``.

    Every delta index commits one transition from ``version - 1`` to ``version``.
    Walking those indexes in order reconstructs the checksum state without
    materializing the model bytes. Tensors absent from the result still have
    their canonical checkpoint bytes.
    """
    if target_version < 0:
        raise ValueError("target_version must be non-negative")

    locations: dict[str, tuple[str, str]] = {}
    for version in range(1, target_version + 1):
        version_dir = os.path.join(delta_dir, f"weight_v{version:06d}")
        index_path = os.path.join(version_dir, "model.safetensors.index.json")
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid delta lineage at version {version}: {index_path}") from error

        metadata = index.get("metadata") or {}
        try:
            published_version = int(metadata["version"])
            base_version = int(metadata["base_version"])
            checksum_format = str(metadata["checksum_format"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid delta metadata in {index_path}") from error
        if published_version != version or base_version != version - 1:
            raise ValueError(f"non-contiguous delta lineage in {index_path}: base={base_version}, version={published_version}")

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_path}")
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise ValueError(f"invalid weight_map entry in {index_path}")
            locations[name] = (os.path.join(version_dir, shard), checksum_format)

    names_by_shard: dict[tuple[str, str], list[str]] = {}
    for name, location in locations.items():
        names_by_shard.setdefault(location, []).append(name)

    state = {}
    for (shard_path, checksum_format), names in names_by_shard.items():
        try:
            with open(shard_path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
        except (OSError, struct.error, json.JSONDecodeError) as error:
            raise ValueError(f"invalid delta shard {shard_path}") from error
        checksums = header.get("__metadata__") or {}
        for name in names:
            if name not in header or name not in checksums:
                raise ValueError(
                    f"delta shard {shard_path} has no checksum for {name!r}"
                )
            state[name] = (checksum_format, str(checksums[name]))

    return state


@cache
def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int, str, tuple[int, ...]]]:
    """Index each tensor's byte range and declared safetensors layout."""
    locations: dict[str, tuple[str, int, int, str, tuple[int, ...]]] = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (
                path,
                8 + header_len + begin,
                end - begin,
                info["dtype"],
                tuple(info["shape"]),
            )
    return locations


def checkpoint_tensor_layout(ckpt_dir: str, name: str) -> tuple[str, tuple[int, ...]]:
    """Return a tensor's declared safetensors dtype and shape."""
    _, _, _, dtype, shape = _tensor_locations(ckpt_dir)[name]
    return dtype, shape


def checkpoint_tensor_names(ckpt_dir: str) -> frozenset[str]:
    """Return tensor names from sharded or single-file safetensors checkpoints."""
    return frozenset(_tensor_locations(ckpt_dir))


def make_tensor_reader(ckpt_dir: str):
    """Index headers once and return a layout-aware raw tensor reader."""
    locations = _tensor_locations(ckpt_dir)

    def read(
        name: str,
        *,
        expected_dtype: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        path, offset, nbytes, dtype, shape = locations[name]
        if (expected_dtype is not None and dtype != expected_dtype) or (
            expected_shape is not None and shape != expected_shape
        ):
            raise ValueError(
                f"Checkpoint tensor {name!r} has dtype={dtype}, shape={shape}; "
                f"trainer emitted dtype={expected_dtype}, shape={expected_shape}"
            )
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)

    return read
