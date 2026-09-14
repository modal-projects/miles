"""Collective HF export through Miles' canonical converters or the LoRA bridge."""

import json
import logging
import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import safetensors.torch
import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.megatron_utils.lora_utils import is_lora_model, save_lora_checkpoint
from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER, load_hf_config
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)
HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _collectively(operation):
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error, group=get_gloo_group())
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError("HF export failed:\n" + "\n".join(failures))
    return result


@dataclass(frozen=True)
class _HfExport:
    finalize: Callable[[], None]
    future: Future | None = None
    executor: ThreadPoolExecutor | None = None

    def finish(self) -> None:
        """Called on every rank; only a successful collective write can be finalized."""
        try:
            _collectively(self.future.result if self.future is not None else lambda: None)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
        _collectively(self.finalize)


def _prepare_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)


def _is_hf_metadata_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.name != HF_EXPORT_COMPLETE_MARKER
        and path.suffix not in HF_WEIGHT_SUFFIXES
        and not path.name.endswith(".index.json")
    )


def _copy_source_static_tensors(base: Path, path: Path, weight_map: dict[str, str]) -> int:
    """Preserve fixed calibration and rotary buffers absent from the training model."""
    index = base / "model.safetensors.index.json"
    if not index.is_file():
        return 0
    by_shard = {}
    for name, shard in json.loads(index.read_text())["weight_map"].items():
        if name.endswith((".input_scale", ".rotary_emb.inv_freq")) and name not in weight_map:
            by_shard.setdefault(shard, []).append(name)
    tensors = {}
    for shard, names in by_shard.items():
        with safetensors.safe_open(base / shard, framework="pt", device="cpu") as source:
            tensors.update({name: source.get_tensor(name).contiguous() for name in names})
    if not tensors:
        return 0
    shard = "model-static.safetensors"
    safetensors.torch.save_file(tensors, path / shard)
    weight_map.update(dict.fromkeys(tensors, shard))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _write_metadata(base: Path, path: Path, weight_map: dict[str, str], total_size: int) -> None:
    if base.is_dir():
        total_size += _copy_source_static_tensors(base, path, weight_map)
        for file in base.iterdir():
            if _is_hf_metadata_file(file):
                shutil.copy2(file, path / file.name)
    else:
        logger.warning("hf_checkpoint %s is not a local directory; metadata not copied to %s", base, path)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _buffer_shards(chunks):
    rank = torch.distributed.get_rank()
    tp_ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(tp_ranks, get_parallel_state().tp.rank, group=get_gloo_group())
    # Match the native checkpoint's TP-rank-0 copy of replicated parameters.
    loads = {rank: 0 for rank, tp_rank in enumerate(tp_ranks) if tp_rank == 0}
    weight_map, pending = {}, []
    error = None
    # The iterator performs collectives. Drain it before handling a local error;
    # owned tensors stay buffered so file I/O never gates the next collective.
    for index, tensors in enumerate(chunks, start=1):
        if error is not None:
            continue
        try:
            tensors = list(tensors)
            owner = min(loads, key=loads.__getitem__)
            shard = f"model-{index:05d}.safetensors"
            loads[owner] += sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
            for name, _ in tensors:
                if name in weight_map:
                    raise ValueError(f"duplicate HF tensor: {name}")
                weight_map[name] = shard
            if rank == owner:
                pending.append((shard, tensors))
        except Exception as exc:
            error = exc
    if error is not None:
        raise error
    if not weight_map:
        raise ValueError("HF export produced no weights")
    payloads = [
        (shard, {name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in tensors})
        for shard, tensors in pending
    ]
    return payloads, weight_map, sum(loads.values())


def _write_shards(path: Path, payloads) -> None:
    for shard, tensors in payloads:
        safetensors.torch.save_file(tensors, path / shard)


def _start_direct_export(
    args, model, path: Path, *, model_name, quantization_config, megatron_local_weights, complete: bool
) -> _HfExport:
    _collectively(lambda: _prepare_directory(path))
    iterator = HfWeightIteratorDirect(args, model, model_name=model_name, quantization_config=quantization_config)
    payloads, weight_map, total_size = _collectively(
        lambda: _buffer_shards(iterator.get_hf_weight_chunks(megatron_local_weights))
    )
    rank = torch.distributed.get_rank()

    def finalize():
        if rank == 0:
            _write_metadata(Path(args.hf_checkpoint), path, weight_map, total_size)
            if complete:
                (path / HF_EXPORT_COMPLETE_MARKER).touch()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-export")
    return _HfExport(finalize, executor.submit(_write_shards, path, payloads), executor)


def export_hf_model_direct(
    args,
    model: Sequence[DDP],
    path: str | Path,
    *,
    model_name: str,
    quantization_config,
    megatron_local_weights,
) -> None:
    """Collectively write canonical HF shards and their index; return after every writer finishes."""
    _start_direct_export(
        args,
        model,
        Path(path),
        model_name=model_name,
        quantization_config=quantization_config,
        megatron_local_weights=megatron_local_weights,
        complete=False,
    ).finish()


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Megatron-Bridge is optional for direct exports.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def start_hf_export(args, rollout_id: int, model: Sequence[DDP], *, path: str | Path | None = None) -> _HfExport:
    """Capture HF weights collectively; the caller must finish the export on every rank.

    Direct exports write immutable CPU buffers on a thread. Bridge/LoRA exports
    stay on the calling thread because they perform framework collectives.
    """
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))
    logger.info("Saving model in HuggingFace format to %s", path)
    lora = is_lora_model(model)
    if not lora and (
        args.megatron_to_hf_mode == "raw" or getattr(args, "update_weight_transfer_mode", None) == "disk-delta"
    ):
        config = load_hf_config(args.hf_checkpoint)
        return _start_direct_export(
            args,
            model,
            path,
            model_name=type(config).__name__.lower() if args.model_name is None else args.model_name,
            quantization_config=getattr(config, "quantization_config", None),
            megatron_local_weights=dict(named_params_and_buffers(args, model, convert_to_global_name=True)),
            complete=True,
        )
    _collectively(lambda: _prepare_directory(path))
    bridge = _get_hf_bridge(args.hf_checkpoint)
    with patch_megatron_model(model):
        bridge.save_hf_pretrained(model, path=path)
    if lora:
        _collectively(lambda: save_lora_checkpoint(model, args, str(path / "adapter")))

    def finalize():
        if torch.distributed.get_rank() == 0:
            if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
                raise RuntimeError(
                    f"HF export to {path} produced no weight files; the bridge may lack a model mapping"
                )
            (path / HF_EXPORT_COMPLETE_MARKER).touch()

    return _HfExport(finalize)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Collectively save a complete HF checkpoint, including merged and adapter-only LoRA weights.

    Failures are logged unless ``raise_on_error`` is set. Success means all local
    files are closed; the caller's storage publication makes them durable.
    """
    try:
        start_hf_export(args, rollout_id, model, path=path).finish()
    except Exception:
        if raise_on_error:
            raise
        logger.exception("Failed to save HuggingFace format")
