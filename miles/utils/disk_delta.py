from __future__ import annotations

import fcntl
import glob
import io
import json
import logging
import mmap
import os
import shutil
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import zstandard

logger = logging.getLogger(__name__)

# The delta phases (XOR/scatter, zstd, checksum) are memory-bandwidth bound and release the GIL,
# so a thread pool over tensors recovers the bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

SYNC_DIR = ".delta_sync"  # per-checkpoint dir holding the applied-version marker and the apply lock


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


@contextmanager
def _apply_lock(local_ckpt_dir: str):
    sync = os.path.join(local_ckpt_dir, SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    with open(os.path.join(sync, "lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_applied_version(local_ckpt_dir: str) -> str | None:
    try:
        with open(os.path.join(local_ckpt_dir, SYNC_DIR, "state.json")) as f:
            return json.load(f)["version"]
    except FileNotFoundError:
        return None


def _write_applied_version(local_ckpt_dir: str, version: str) -> None:
    path = os.path.join(local_ckpt_dir, SYNC_DIR, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": version}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def drop_page_cache(path: str) -> None:
    """Evict a file from the page cache (POSIX_FADV_DONTNEED)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def init_local_checkpoint(local_ckpt_dir: str, base_dir: str) -> None:
    """Copy the base HF checkpoint into local_ckpt_dir once if absent (run at engine start). Each
    later delta is applied on top of this copy in place."""
    with _apply_lock(local_ckpt_dir):
        if _read_applied_version(local_ckpt_dir) is not None:
            return
        logger.info("Materializing base checkpoint %s -> %s", base_dir, local_ckpt_dir)
        os.makedirs(local_ckpt_dir, exist_ok=True)
        for entry in os.scandir(base_dir):
            if entry.is_file():
                shutil.copy2(entry.path, os.path.join(local_ckpt_dir, entry.name))
                drop_page_cache(entry.path)  # don't let the source evict the local copy we keep resident
        _write_applied_version(local_ckpt_dir, "000000")


def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    """Map each tensor name to (file, byte offset, nbytes) by reading every safetensors header."""
    locations: dict[str, tuple[str, int, int]] = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def make_tensor_reader(ckpt_dir: str):
    """Index the headers once, then return ``read(name) -> uint8 bytes`` that seeks straight to the
    tensor — for reading many tensors without rescanning every header. KeyError if absent."""
    locations = _tensor_locations(ckpt_dir)

    def read(name: str) -> np.ndarray:
        path, offset, nbytes = locations[name]
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)

    return read


def _apply_version(local_ckpt_dir: str, version_dir: str) -> dict | None:
    """Apply one version's delta in place: decompress + apply + checksum each tensor across a thread
    pool (each writes a distinct mmap region, so the writes don't conflict). Any mismatch raises.
    Returns a per-phase timing/byte breakdown (None when the version was already applied)."""
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        meta = json.load(f)["metadata"]
    applied = _read_applied_version(local_ckpt_dir)
    if applied == meta["version"]:
        return None
    if applied != meta["base_version"]:
        raise RuntimeError(f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}")
    if meta["compression_format"] != "zstd":
        raise NotImplementedError(f"compression {meta['compression_format']!r} not supported")
    encoding = meta["delta_encoding"]
    algorithm = meta["checksum_format"]
    locations = _tensor_locations(local_ckpt_dir)
    open_mmaps: dict[str, tuple] = {}
    mismatches: list[str] = []
    lock = threading.Lock()
    # Thread-seconds per phase, summed across workers (wall time is the pool span).
    thread_s = {"decompress": 0.0, "patch": 0.0, "checksum": 0.0}
    file_bytes: list[bytes] = []  # keep alive: items hold zero-copy views into these
    items: list[tuple] = []  # (name, compressed_view, path, offset, nbytes, want_checksum)
    try:
        t0 = time.perf_counter()
        for delta_file in sorted(glob.glob(os.path.join(version_dir, "*.safetensors"))):
            with open(delta_file, "rb") as f:
                blob = f.read()
            file_bytes.append(blob)
            (header_len,) = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8 : 8 + header_len])
            want_checksums = header.get("__metadata__", {})
            view = memoryview(blob)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                path, offset, nbytes = locations[name]
                if path not in open_mmaps:
                    fh = open(path, "r+b")
                    open_mmaps[path] = (fh, mmap.mmap(fh.fileno(), 0))
                data_start = 8 + header_len
                items.append(
                    (name, view[data_start + begin : data_start + end], path, offset, nbytes, want_checksums.get(name))
                )
        read_s = time.perf_counter() - t0

        # prefetch into page cache (evicted during the rollout) so the apply doesn't fault from cold storage
        t0 = time.perf_counter()
        for _, mm in open_mmaps.values():
            try:
                mm.madvise(mmap.MADV_WILLNEED)
            except (OSError, AttributeError, ValueError):
                pass
        madvise_s = time.perf_counter() - t0

        def apply_xor(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
            hasher = _new_hasher(algorithm)
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(bytes(compressed)))
            t_dec = t_patch = t_sum = 0.0
            pos = 0
            while pos < nbytes:  # 2 MB chunks stay L2-resident across decompress -> XOR -> checksum
                t = time.perf_counter()
                block = reader.read(min(2 << 20, nbytes - pos))
                t_dec += time.perf_counter() - t
                if not block:
                    break
                chunk = np.frombuffer(block, dtype=np.uint8)
                t = time.perf_counter()
                region[pos : pos + chunk.size] ^= chunk
                t_patch += time.perf_counter() - t
                t = time.perf_counter()
                hasher.update(region[pos : pos + chunk.size])
                t_sum += time.perf_counter() - t
                pos += chunk.size
            ok = hasher.hexdigest() == want
            with lock:
                if not ok:
                    mismatches.append(name)
                thread_s["decompress"] += t_dec
                thread_s["patch"] += t_patch
                thread_s["checksum"] += t_sum

        def apply_overwrite(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            t = time.perf_counter()
            delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(bytes(compressed)), dtype=np.uint8)
            t_dec = time.perf_counter() - t
            region = np.ndarray((nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset)
            count = int.from_bytes(delta[:4].tobytes(), "little")
            positions = np.frombuffer(delta[4 : 4 + 4 * count].tobytes(), dtype="<u4")
            t = time.perf_counter()
            region[positions] = delta[4 + 4 * count :]
            t_patch = time.perf_counter() - t
            t = time.perf_counter()
            ok = checksum(algorithm, region) == want
            t_sum = time.perf_counter() - t
            with lock:
                if not ok:
                    mismatches.append(name)
                thread_s["decompress"] += t_dec
                thread_s["patch"] += t_patch
                thread_s["checksum"] += t_sum

        if encoding == "xor":
            apply_tensor = apply_xor
        elif encoding == "overwrite":
            apply_tensor = apply_overwrite
        else:
            raise NotImplementedError(f"delta encoding {encoding!r} not supported")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            list(pool.map(apply_tensor, items))
        apply_s = time.perf_counter() - t0
        # no msync: the engine reads these pages via the shared cache; durability isn't needed
        # (a host that loses the cache rebuilds from base)
    finally:
        for fh, mm in open_mmaps.values():
            mm.close()
            fh.close()
    if mismatches:
        raise RuntimeError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {version_dir}: "
            f"{sorted(mismatches)[:20]}"
        )
    _write_applied_version(local_ckpt_dir, meta["version"])
    stats = {
        "version": meta["version"],
        "read_s": round(read_s, 3),
        "madvise_s": round(madvise_s, 3),
        "apply_s": round(apply_s, 3),
        "decompress_thread_s": round(thread_s["decompress"], 3),
        "patch_thread_s": round(thread_s["patch"], 3),
        "checksum_thread_s": round(thread_s["checksum"], 3),
        "tensors": len(items),
        "compressed_bytes": sum(len(compressed) for _, compressed, *_ in items),
        "touched_bytes": sum(nbytes for *_, nbytes, _ in items),
    }
    logger.info(
        "[disk delta apply] v=%s read=%.2fs madvise=%.2fs apply=%.2fs "
        "(thread-s: decompress=%.1f patch=%.1f checksum=%.1f) tensors=%d compressed=%.3fGB touched=%.3fGB",
        meta["version"],
        stats["read_s"],
        stats["madvise_s"],
        stats["apply_s"],
        stats["decompress_thread_s"],
        stats["patch_thread_s"],
        stats["checksum_thread_s"],
        stats["tensors"],
        stats["compressed_bytes"] / 1e9,
        stats["touched_bytes"] / 1e9,
    )
    return stats


def apply_deltas(local_ckpt_dir: str, delta_root: str, target_version: int) -> list[dict]:
    """Apply the delta chain in order to bring the local checkpoint up to target_version, in place.
    A per-tensor checksum guards every write and any mismatch raises (fail loud, never serve bad
    weights). Serialized per host by the lock (co-located actors collapse to one apply).
    Returns the per-version phase breakdowns (empty when everything was already applied)."""
    stats: list[dict] = []
    with _apply_lock(local_ckpt_dir):
        applied = _read_applied_version(local_ckpt_dir)
        if applied is None:
            raise RuntimeError("local checkpoint not materialized")
        for version in range(int(applied) + 1, target_version + 1):
            version_stats = _apply_version(local_ckpt_dir, os.path.join(delta_root, f"weight_v{version:06d}"))
            if version_stats is not None:
                stats.append(version_stats)
    return stats
