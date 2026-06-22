from __future__ import annotations

import json
import logging
import os
import queue
import shutil
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import ray
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard
from ray.actor import ActorHandle

from miles.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import load_function

from .update_weight_from_distributed.broadcast import UpdateWeightFromDistributed

logger = logging.getLogger(__name__)


class UpdateWeightFromDiskDelta(UpdateWeightFromDistributed):
    """Delta weight sync over a shared filesystem. Source ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    every rollout host applies the delta into its local checkpoint and reloads via the ordinary
    update_weights_from_disk path, so sglang needs no delta support.

    Reuses the base class's bucketed TP/EP all-gather (``_gather_and_update_*_weights``): the
    per-bucket callback feeds a snapshot-seed pass (baseline) or a diff/compress pipeline (publish)
    instead of an NCCL broadcast.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        assert not is_lora, "disk delta weight sync does not support LoRA"
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )
        self.delta_dir = args.update_weight_disk_dir
        os.makedirs(self.delta_dir, exist_ok=True)
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        # Opaque HTTP rollout: no engine handles, so publish the version to disk and let the fleet
        # pull it, instead of pushing per-engine reload RPCs.
        self._publish_only = bool(getattr(args, "rollout_endpoint_url", None))
        self._commit_hook: Callable | None = (
            load_function(args.custom_delta_pre_push_path) if args.custom_delta_pre_push_path else None
        )
        self.update_weight_metrics: dict[str, float] = {}

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        # The local checkpoint is host-local: every engine actor applies its own copy, and a per-host
        # flock + applied-version marker collapse co-located actors to a single apply, so the NCCL
        # group and the rollout_engine_lock the broadcast path needs aren't used here.
        self.rollout_engines = rollout_engines

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        # The first call only captures the baseline snapshot the next sync diffs against.
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return

        self.weight_version += 1
        if dist.get_rank() == 0 and not self._publish_only:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            if mode != "in_place":
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        self._publish()
        if self._publish_only:
            self._announce_version()
        else:
            self._reload_engines()
        self._record_metrics()

    def _capture_baseline(self) -> None:
        """Capture the baseline snapshot the first delta diffs against (no publish), and clear any
        stale stream from a prior run. Seeds from hf_checkpoint — what each host materializes its
        base from — so the invariant ``snapshot == engine base`` holds even where the megatron->HF
        round-trip trims vocab-padding rows (embed/lm_head). A tensor absent there (rare) falls back
        to the gathered value."""
        # a prior run's versions would apply against the wrong base; start the dir clean
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._commit_hook is not None:
                self._commit_hook(self.args, self.delta_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        self._baseline_reader = make_tensor_reader(self.args.hf_checkpoint) if self._is_source else None
        self._gather_and_update_non_expert_weights(self._snapshot_bucket)
        dist.barrier(group=get_gloo_group())
        self._gather_and_update_expert_weights(self._snapshot_bucket)
        dist.barrier(group=get_gloo_group())
        self._baseline_reader = None
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )

    def _snapshot_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar=None) -> None:
        """Seed the snapshot for one gathered HF bucket from hf_checkpoint (source ranks only)."""
        for name, tensor in converted_named_tensors:
            try:
                self._snapshot[name] = self._baseline_reader(name)
            except KeyError:
                self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
                logger.warning("seed: %s absent from hf_checkpoint; seeding from current weights", name)
        converted_named_tensors.clear()

    def _publish(self) -> None:
        """Encode this version's changed tensors (source ranks), then write it as a canonical HF dir."""
        self._encode_delta()
        dist.barrier(group=get_gloo_group())
        self._write_delta_files()

    def _encode_delta(self) -> None:
        """Diff each gathered HF tensor against the snapshot, keeping the changed ones (compressed)
        in self._delta with their checksums. The GPU->CPU gather is pipelined into a compute pool:
        the main loop copies one tensor to a pinned buffer and submits it; pool workers diff and
        compress in parallel (each is a few big GIL-releasing numpy/zstd calls)."""
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        self._delta: dict[str, np.ndarray] = {}  # changed tensor name -> compressed diff
        self._checksums: dict[str, str] = {}  # changed tensor name -> new-state checksum
        self.changed_bytes = self.total_bytes = 0
        self._pool: ThreadPoolExecutor | None = None
        self._inflight: deque[Future] = deque()

        if self._is_source:
            os.makedirs(self._version_dir, exist_ok=True)
            self._setup_encode_buffers()
            self._pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        try:
            self._gather_and_update_non_expert_weights(self._encode_bucket)
            dist.barrier(group=get_gloo_group())
            self._gather_and_update_expert_weights(self._encode_bucket)
            while self._inflight:
                self._collect(self._inflight.popleft())
        finally:
            if self._pool is not None:
                self._pool.shutdown()

    def _setup_encode_buffers(self) -> None:
        # A pinned non_blocking GPU->CPU copy is far faster than .cpu(); fall back to pageable if a
        # low memlock limit forbids pinning.
        self._max_bytes = max((int(v.nbytes) for v in self._snapshot.values()), default=0)
        self._free_q: queue.Queue = queue.Queue()
        self._use_pinned = True
        try:
            for _ in range(max(4, min(2 * NUM_WORKERS, (32 << 30) // max(self._max_bytes, 1)))):
                self._free_q.put(torch.empty(self._max_bytes, dtype=torch.uint8, pin_memory=True))
        except RuntimeError as e:  # low memlock limit
            logger.warning("pinned host buffers unavailable (%s); using pageable .cpu()", e)
            self._use_pinned = False

    def _encode_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar=None) -> None:
        """Copy each gathered HF tensor to host and submit it to the diff/compress pool, draining
        once enough work is in flight to backpressure the gather (source ranks only)."""
        for name, tensor in converted_named_tensors:
            flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            nbytes = int(flat.numel())
            if self._use_pinned and nbytes <= self._max_bytes:
                buf = self._free_q.get()  # blocks when all buffers are in flight -> backpressures the gather
                buf[:nbytes].copy_(flat, non_blocking=True)
                torch.cuda.current_stream().synchronize()
                payload, pinned = buf, True
            else:
                payload, pinned = flat.cpu().numpy(), False
            self.total_bytes += nbytes
            self._inflight.append(self._pool.submit(self._diff_and_compress, name, payload, nbytes, pinned))
            if len(self._inflight) >= 2 * NUM_WORKERS:
                self._collect(self._inflight.popleft())
        converted_named_tensors.clear()

    def _diff_and_compress(self, name: str, buf, nbytes: int, pinned: bool):
        if pinned:  # copy out and free the pinned buffer before the heavy diff/compress
            new = np.empty(nbytes, dtype=np.uint8)
            np.copyto(new, buf.numpy()[:nbytes])
            self._free_q.put(buf)
        else:
            new = buf
        old = self._snapshot[name]
        if self.delta_encoding == "xor":
            diff = new ^ old
            changed = int(np.count_nonzero(diff))
        elif self.delta_encoding == "overwrite":
            mask = new != old
            changed = int(np.count_nonzero(mask))
            diff = overwrite_encode(new, mask)
        else:
            raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
        if not changed:
            return name, new, None, None, 0
        compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
        return name, new, compressed, checksum(self.checksum_algorithm, new), changed

    def _collect(self, fut: Future) -> None:
        name, new, compressed, digest, changed = fut.result()
        self._snapshot[name] = new  # becomes the next sync's base
        if changed:
            self.changed_bytes += changed
            self._delta[name] = compressed
            self._checksums[name] = digest

    def _write_delta_files(self) -> None:
        """Write this rank's changed tensors as one canonical model-NNNNN.safetensors, and on rank
        0 the HF index. The sequential file numbers and the index are coordinated over gloo (small
        object gathers), not the filesystem — a shared volume may not surface one rank's writes to
        another until commit."""
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()

        # number the files sequentially across only the ranks that have one (no gaps)
        counts: list = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        offset, total = sum(counts[:rank]), sum(counts)

        fname = None
        self.wire_bytes = 0
        if self._delta:
            fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, fname), blob)

        maps: list = [None] * world
        dist.all_gather_object(maps, {name: fname for name in self._delta}, group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{self.weight_version:06d}",
                    "base_version": f"{self.weight_version - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {name: f for m in maps for name, f in m.items()},
            }
            _atomic_write(os.path.join(self._version_dir, "model.safetensors.index.json"), json.dumps(index).encode())
        dist.barrier(group=group)

    def _reload_engines(self) -> None:
        """Commit the published files, have each host apply the delta, then reload the engines."""
        if self._commit_hook is not None:
            self._commit_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.sync_local_checkpoint.remote(self.weight_version) for engine in self.rollout_engines])
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _announce_version(self) -> None:
        """Publish-only: commit the version dir and advance the latest-version pointer, so the
        external fleet pulls and applies it on its own. No engine handles, hence no reload RPCs."""
        if self._commit_hook is not None:
            self._commit_hook(self.args, self._version_dir, [])  # opaque fleet: no engine handles
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            _atomic_write(os.path.join(self.delta_dir, "latest"), f"{self.weight_version:06d}".encode())
        dist.barrier(group=get_gloo_group())

    def _record_metrics(self) -> None:
        """All-reduce the byte counts and record changed-fraction / wire size; the actor drains
        update_weight_metrics onto the step log."""
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        if dist.get_rank() == 0:  # only rank 0 logs, so only rank 0 keeps the metrics
            self.update_weight_metrics["perf/update_weights_density"] = changed / max(total, 1)
            self.update_weight_metrics["perf/update_weights_wire_bytes"] = wire
            logger.info(
                "[disk delta v=%s] density=%.2f%% wire=%.2f GB",
                self.weight_version,
                100.0 * changed / max(total, 1),
                wire / 1e9,
            )


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
