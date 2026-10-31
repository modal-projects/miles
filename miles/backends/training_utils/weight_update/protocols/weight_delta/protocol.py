"""Weight-delta weight sync: each version publishes what changed, and engines apply them in order."""

import logging
import re
import shutil
import time
from argparse import Namespace
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.delta import _safetensors_dtype
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import get_data_replica_rank_and_size
from miles.backends.training_utils.weight_update.protocols.weight_delta.encoder import (
    DeltaEncoder,
    EncodedTensor,
    changed_bits,
)
from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import (
    OwnerWriter,
    is_raw,
    seal,
    version_dir,
)
from miles.backends.training_utils.weight_update.rollout_cell_updater import (
    _RolloutCellUpdater,
    create_rollout_cell_updaters,
)
from miles.backends.training_utils.weight_update.session import (
    maybe_pause_engines,
    maybe_resume_engines,
    set_weight_version,
)
from miles.utils import async_utils
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.disk_delta import make_tensor_reader
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_VERSION_DIR_NAME = re.compile(r"weight_v(\d+)")


class UpdateWeightFromDelta(WeightTransferProtocol):
    required_placement = WeightUpdatePlacement(gather_pp=False, per_owner=True)
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._delta_dir = Path(args.update_weight_disk_dir)
        self._post_write_hook = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)
        self._version: int | None = None
        self._previous: dict[str, torch.Tensor] = {}
        self._raw_current: dict[str, torch.Tensor] = {}
        self._encoder: DeltaEncoder | None = None
        self._replay: list[int] = []
        # Only for --weight-delta-change-metrics; the sync never reads it.
        self._ever_changed: dict[str, torch.Tensor] | None = {} if args.weight_delta_change_metrics else None
        self._ever_changed_nbytes = 0
        self._reset_sync()

    def _reset_sync(self) -> None:
        self._error: BaseException | None = None
        self._exported: dict[str, tuple[str, list[int]]] = {}
        self._directory: Path | None = None
        self._writer: OwnerWriter | None = None
        # (name, dtype, shape, the encoder's result or None for a raw tensor), in the order they are written.
        self._writes: list[tuple[str, str, list[int], EncodedTensor | None]] = []
        self._changed_raw: set[str] = set()
        self._record: dict | None = None
        self._rechanged: list[torch.Tensor] = []
        self._stats = {
            "exported_nbytes": 0,
            "changed_nbytes": 0,
            "rechanged_nbytes": 0,
            "ever_changed_nbytes": 0,
            "encode_s": 0.0,
        }

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        if self._version is not None:
            # TODO: for rollout fault tolerance, add this mode to supports_partial_target_weight_update and
            # catch a restarted engine up here: it loads --hf-checkpoint, then applies every published version.
            raise RuntimeError("The rollout engines changed after the launch; weight delta cannot catch one up yet")
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.group_name = "miles-weight-delta"
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._reset_sync()
        if self._version is None:
            if dist.get_rank() == 0:
                try:
                    self._replay = self._begin_launch(weight_version)
                except Exception as error:
                    self._error = error
            _raise_if_any_rank_failed(self._error, f"launching at version {weight_version}")
            return True
        if weight_version == self._version:
            return False
        if weight_version < self._version:
            raise RuntimeError(f"weight version {weight_version} is behind the published {self._version}")
        self._encoder = DeltaEncoder(
            self.args.weight_delta_codec,
            torch.device("cuda", torch.cuda.current_device()),
            self.args.update_weight_buffer_size,
            with_changed_bits=self._ever_changed is not None,
        )
        self._directory = version_dir(self._delta_dir, weight_version)
        self._writer = OwnerWriter(self._directory, dist.get_rank())
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # Every rank keeps driving the export's collectives; an error waits for after_base_weights.
        if self._error is not None:
            return
        try:
            for name, tensor in bucket:
                self._exported[name] = (_safetensors_dtype(tensor.dtype), list(tensor.shape))
            if self._version is None:
                self._copy_to_previous(bucket)
            else:
                self._encode(bucket)
        except Exception as error:
            self._error = error

    def _copy_to_previous(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # TODO: holding previous on the GPU would skip its host round trip each sync, but keeps one owner
        # share of GPU memory through training.
        for name, tensor in bucket:
            current = _as_bytes(tensor)
            self._previous[name] = torch.empty(current.numel(), dtype=torch.uint8, pin_memory=True)
            self._previous[name].copy_(current, non_blocking=True)

    def _encode(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        pairs, xor_tensors = [], []
        for name, tensor in bucket:
            if name not in self._previous:
                raise ValueError(f"{name} is exported now but was not at the launch")
            current = _as_bytes(tensor)
            dtype, shape = self._exported[name]
            self._stats["exported_nbytes"] += current.numel()
            if is_raw(dtype, shape):
                if name not in self._raw_current:
                    self._raw_current[name] = torch.empty(current.numel(), dtype=torch.uint8, pin_memory=True)
                self._raw_current[name].copy_(current, non_blocking=True)
                self._writes.append((name, dtype, shape, None))
            else:
                # The encoder overwrites previous with current, so a failed sync must end the run.
                pairs.append((self._previous[name], current))
                xor_tensors.append((name, dtype, shape))
        started = time.monotonic()
        encoded = self._encoder.encode(pairs)
        self._stats["encode_s"] += time.monotonic() - started
        for (name, dtype, shape), result in zip(xor_tensors, encoded, strict=True):
            if result is not None:
                self._writes.append((name, dtype, shape, result))
                self._stats["changed_nbytes"] += result.changed_nbytes
                if result.changed_bits is not None:
                    self._track_changes(name, result.changed_bits)

    def _compare_raw(self) -> None:
        for name, _, _, encoded in self._writes:
            if encoded is not None:
                continue
            current, previous = self._raw_current[name], self._previous[name]
            changed_nbytes = int(torch.count_nonzero(current != previous))
            if changed_nbytes:
                self._changed_raw.add(name)
                self._stats["changed_nbytes"] += changed_nbytes
                if self._ever_changed is not None:
                    self._track_changes(name, changed_bits(current ^ previous).cuda())
                previous.copy_(current)

    def _write_owner_file(self) -> None:
        started = time.monotonic()
        compressed = iter(zip(*self._encoder.finish(), strict=True))
        self._stats["encode_s"] += time.monotonic() - started
        for name, dtype, shape, encoded in self._writes:
            if encoded is not None:
                self._writer.add_xor(name, dtype, shape, encoded.frames, *next(compressed), encoded.checksum)
            elif name in self._changed_raw:
                self._writer.add_raw(name, dtype, shape, memoryview(self._previous[name].numpy()))

    def _track_changes(self, name: str, bits: torch.Tensor) -> None:
        if name not in self._ever_changed:
            self._ever_changed[name] = torch.zeros(bits.numel(), dtype=torch.uint8, pin_memory=True)
        ever = self._ever_changed[name].to(bits.device, non_blocking=True)
        self._rechanged.append(_popcount(ever & bits))
        self._ever_changed[name].copy_(ever | bits, non_blocking=True)

    def after_base_weights(self) -> None:
        torch.cuda.current_stream().synchronize()
        if self._error is None and self._writer is not None:
            try:
                self._compare_raw()
            except Exception as error:
                self._error = error
        self._stats["rechanged_nbytes"] = int(torch.stack(self._rechanged).sum()) if self._rechanged else 0
        self._ever_changed_nbytes += self._stats["changed_nbytes"] - self._stats["rechanged_nbytes"]
        self._stats["ever_changed_nbytes"] = self._ever_changed_nbytes
        if self._error is None and self._writer is not None and set(self._exported) != set(self._previous):
            self._error = ValueError(
                f"the exported tensors changed since the launch: {len(set(self._previous) - set(self._exported))} "
                f"missing, {len(set(self._exported) - set(self._previous))} new"
            )
        if self._writer is not None:
            try:
                if self._error is None:
                    self._write_owner_file()
                self._record = self._writer.finish()
                if self._error is None:
                    self._run_post_write_hook(self._directory)
            except Exception as error:
                self._error = self._error or error
        _raise_if_any_rank_failed(self._error, "exporting this version")

    def finalize(self, weight_version: int) -> None:
        if self._version is None:
            self._finish_launch(weight_version)
            return
        records: list = [None] * dist.get_world_size()
        dist.gather_object(
            (self._record, self._stats), records if dist.get_rank() == 0 else None, dst=0, group=get_gloo_group()
        )
        error = None
        if dist.get_rank() == 0:
            try:
                seal(
                    self._directory,
                    [record for record, _ in records],
                    codec=self.args.weight_delta_codec,
                    base_version=self._version,
                    target_version=weight_version,
                )
                self._run_post_write_hook(self._directory)
            except Exception as caught:
                error = caught
        _raise_if_any_rank_failed(error, f"publishing version {weight_version}")
        self._version = weight_version
        if dist.get_rank() == 0:
            self._record_metrics(records)
            try:
                self._apply([weight_version])
            except Exception as caught:
                error = caught
        _raise_if_any_rank_failed(error, f"applying version {weight_version}")

    def _begin_launch(self, version: int) -> list[int]:
        startup_label = self.args.sglang.common_value("weight_version")
        updaters = self.cell_updaters_of_cell_id
        labels = async_utils.wait_futures(
            [updater.submit_client_call("get_weight_version") for updater in updaters.values()]
        )
        updated = [cell for cell, label in zip(updaters, labels, strict=True) if label != startup_label]
        if updated:
            raise RuntimeError(
                f"rollout engines {updated} hold updated weights; weight delta needs every engine to start from "
                f"--hf-checkpoint"
            )
        published = sorted(
            int(match[1])
            for entry in (self._delta_dir.iterdir() if self._delta_dir.is_dir() else ())
            if (match := _VERSION_DIR_NAME.fullmatch(entry.name))
        )
        history = [v for v in published if v <= version]
        if history and not (
            history[-1] == version and (version_dir(self._delta_dir, version) / "manifest.json").exists()
        ):
            raise RuntimeError(
                f"{self._delta_dir} holds no published version {version}, the trainer's: the run stopped between "
                f"saving it and publishing it, so resume from an earlier save"
            )
        abandoned = [v for v in published if v > version]
        for v in abandoned:
            shutil.rmtree(version_dir(self._delta_dir, v))
        if abandoned:
            logger.info("[weight delta] deleted %d versions published after %d", len(abandoned), version)
            self._run_post_write_hook(self._delta_dir)
        return history

    def _finish_launch(self, weight_version: int) -> None:
        exported_by_rank: list = [None] * dist.get_world_size()
        dist.gather_object(
            self._exported, exported_by_rank if dist.get_rank() == 0 else None, dst=0, group=get_gloo_group()
        )
        replaying = [bool(self._replay)]
        dist.broadcast_object_list(replaying, src=0, group=get_gloo_group())
        error = None
        try:
            if dist.get_rank() == 0:
                _check_exported(exported_by_rank, self.args.hf_checkpoint)
            # With no history to replay, the engines serve --hf-checkpoint's bytes as this version.
            # TODO: if exports turn out to differ from checkpoints in practice, publish a version-0 delta
            # from the checkpoint's bytes instead of refusing.
            if not replaying[0]:
                _check_exported_values(self._previous, self.args.hf_checkpoint)
        except Exception as caught:
            error = caught
        _raise_if_any_rank_failed(error, f"checking the export at version {weight_version} against --hf-checkpoint")
        if dist.get_rank() == 0:
            try:
                # TODO: an engine started from a newer --save-hf snapshot needs only the versions after it.
                if self._replay:
                    self._apply(self._replay)
                else:
                    set_weight_version(list(self.cell_updaters_of_cell_id.values()), weight_version)
            except Exception as caught:
                error = caught
        _raise_if_any_rank_failed(error, f"launching at version {weight_version}")
        self._version = weight_version

    def _apply(self, versions: Sequence[int]) -> None:
        updaters = list(self.cell_updaters_of_cell_id.values())
        prepare_s = apply_s = 0.0
        for version in versions:
            manifest = str(version_dir(self._delta_dir, version) / "manifest.json")
            started = time.monotonic()
            prepare_replies = _call(updaters, "prepare_weights_from_delta", manifest=manifest)
            prepared = time.monotonic()
            maybe_pause_engines(self.args, updaters)
            try:
                apply_replies = _call(
                    updaters, "update_weights_from_delta", manifest=manifest, weight_version=str(version)
                )
            finally:
                maybe_resume_engines(self.args, updaters)
            prepare_s += prepared - started
            apply_s += time.monotonic() - prepared
            self.update_weight_metrics.update(_engine_stats(prepare_replies, "prepare"))
            self.update_weight_metrics.update(_engine_stats(apply_replies, "apply"))
        self.update_weight_metrics["perf/update_weights_prepare_s"] = prepare_s
        self.update_weight_metrics["perf/update_weights_apply_s"] = apply_s

    def _run_post_write_hook(self, path: Path) -> None:
        if self._post_write_hook is not None:
            fn_args = compute_custom_function_config(self.args, self.args.custom_update_weight_post_write_path)
            self._post_write_hook(fn_args, str(path), list(self.rollout_engines))

    def _record_metrics(self, records: list[tuple[dict, dict]]) -> None:
        stats = [item for _, item in records]
        exported = sum(item["exported_nbytes"] for item in stats)
        changed = sum(item["changed_nbytes"] for item in stats)
        if self._ever_changed is not None:
            self.update_weight_metrics["perf/update_weights_ever_changed"] = sum(
                item["ever_changed_nbytes"] for item in stats
            ) / max(exported, 1)
            self.update_weight_metrics["perf/update_weights_rechanged"] = sum(
                item["rechanged_nbytes"] for item in stats
            ) / max(changed, 1)
        self.update_weight_metrics.update(
            {
                "perf/update_weights_density": changed / max(exported, 1),
                "perf/update_weights_wire_bytes": sum(record["nbytes"] for record, _ in records if record is not None),
                "perf/update_weights_encode_s": max(item["encode_s"] for item in stats),
            }
        )


def _check_exported(exported_by_rank: list[dict[str, tuple[str, list[int]]]], hf_checkpoint: str) -> None:
    counts = Counter(name for exported in exported_by_rank for name in exported)
    duplicated = sorted(name for name, count in counts.items() if count > 1)
    if duplicated:
        raise RuntimeError(f"{len(duplicated)} tensors are exported by two ranks, e.g. {duplicated[:5]}")
    in_checkpoint = {}
    for path in Path(hf_checkpoint).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                header = f.get_slice(name)
                in_checkpoint[name] = (header.get_dtype(), header.get_shape())
    mismatched = sorted(
        (name, dtype_shape, in_checkpoint.get(name))
        for exported in exported_by_rank
        for name, dtype_shape in exported.items()
        if in_checkpoint.get(name) != dtype_shape
    )
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} exported tensors differ from --hf-checkpoint, e.g. "
            + "; ".join(f"{name}: exported {ours}, checkpoint {theirs}" for name, ours, theirs in mismatched[:3])
        )


def _check_exported_values(previous: dict[str, torch.Tensor], hf_checkpoint: str) -> None:
    read = make_tensor_reader(hf_checkpoint)
    differing = sorted(name for name, exported in previous.items() if not np.array_equal(exported.numpy(), read(name)))
    if differing:
        raise ValueError(
            f"{len(differing)} exported tensors differ from --hf-checkpoint's values, e.g. {differing[:5]}; the "
            "engines start from its bytes, so it must be the trainer's own export (same quantizer and settings)"
        )


def _popcount(bits: torch.Tensor) -> torch.Tensor:
    count = bits - ((bits >> 1) & 0x55)
    count = (count & 0x33) + ((count >> 2) & 0x33)
    return ((count + (count >> 4)) & 0x0F).sum(dtype=torch.int64)


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _call(updaters: Sequence[_RolloutCellUpdater], name: str, **kwargs) -> list:
    results = async_utils.wait_futures([updater.submit_client_call(name, **kwargs) for updater in updaters])
    for updater, result in zip(updaters, results, strict=True):
        if isinstance(result, dict) and result.get("success") is False:
            updater.mark_errored(RuntimeError(f"{name} failed on a rollout engine: {result.get('message')}"))
    return results


def _engine_stats(replies: Sequence, phase: str) -> dict[str, float]:
    """Each numeric stat the engines report per GPU, at its largest: the slowest GPU sets the pace."""
    gpus = [gpu for reply in replies if isinstance(reply, dict) for gpu in reply.get("stats") or () if gpu]
    keys = {key for gpu in gpus for key, value in gpu.items() if isinstance(value, int | float)}
    return {
        f"perf/update_weights_engine_{phase}_{key}": max(gpu[key] for gpu in gpus if key in gpu)
        for key in sorted(keys)
    }


def _raise_if_any_rank_failed(error: BaseException | None, doing: str) -> None:
    group = get_gloo_group()
    messages: list = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(messages, None if error is None else f"{type(error).__name__}: {error}", group=group)
    for rank, message in enumerate(messages):
        if message is not None:
            raise RuntimeError(f"weight delta failed on rank {rank} while {doing}: {message}") from error
