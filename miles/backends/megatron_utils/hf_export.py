"""HF-format export of the live Megatron model.

``export_hf_model_direct`` goes through miles' own megatron->HF converters (the
weight updater's machinery), so export coverage always matches weight-sync
coverage; ``save_hf_model`` picks between it and the Megatron-Bridge exporter
(LoRA needs the bridge for adapter merging) and writes a ``.complete`` marker.
Everything here is collective: all ranks must call it, global rank 0 writes.
"""

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import safetensors.torch
import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from safetensors import safe_open

from miles.backends.megatron_utils.lora.utils import is_lora_model, save_lora_checkpoint
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER, load_hf_config
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
HF_STATIC_WEIGHT_SUFFIXES = (".input_scale", ".rotary_emb.inv_freq")


def _collectively(operation):
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error, group=get_gloo_group())
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
    if failures:
        raise RuntimeError("HF export failed:\n" + "\n".join(failures))
    return result


@dataclass(frozen=True)
class _HfExport:
    finalize: Callable[[], None]
    future: Future | None = None
    executor: ThreadPoolExecutor | None = None

    def finish(self) -> None:
        try:
            _collectively(self.future.result if self.future is not None else lambda: None)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
        _collectively(self.finalize)


def _is_hf_metadata_file(path: Path) -> bool:
    """Tokenizer/config files worth copying into an export — not weights, and not the
    base checkpoint's weight index, which would clobber the one the export writes."""
    return (
        path.is_file()
        and path.name != HF_EXPORT_COMPLETE_MARKER
        and path.suffix not in HF_WEIGHT_SUFFIXES
        and not path.name.endswith(".index.json")
    )


def _prepare_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)


def _static_weight_prefixes(args) -> tuple[str, ...]:
    prefixes = getattr(args, "hf_export_static_weight_prefixes", ())
    if prefixes is None:
        return ()
    if not isinstance(prefixes, (list, tuple)) or any(
        not isinstance(prefix, str) or not prefix for prefix in prefixes
    ):
        raise ValueError("hf_export_static_weight_prefixes must be a list of nonempty strings")
    return tuple(dict.fromkeys(prefixes))


def _source_weight_map(base: Path) -> dict[str, str]:
    index = base / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or any(
            not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items()
        ):
            raise ValueError(f"Invalid safetensors index: {index}")
        return weight_map

    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        return {}
    if len(shards) != 1:
        raise ValueError(f"Sharded source checkpoint has no model.safetensors.index.json: {base}")
    with safe_open(shards[0], framework="pt", device="cpu") as source:
        return dict.fromkeys(source.keys(), shards[0].name)


def _copy_source_static_weights(
    base: Path,
    path: Path,
    weight_map: dict[str, str],
    prefixes: tuple[str, ...],
) -> int:
    """Copy immutable source tensors absent from the live training model."""
    source_weights = _source_weight_map(base)
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in source_weights):
            raise ValueError(f"HF export static prefix has no source weights: {prefix!r}")

    by_shard: dict[str, list[str]] = {}
    for name, shard in source_weights.items():
        if name in weight_map:
            continue
        if name.endswith(HF_STATIC_WEIGHT_SUFFIXES) or name.startswith(prefixes):
            by_shard.setdefault(shard, []).append(name)

    total_size = 0
    for index, (source_shard, names) in enumerate(sorted(by_shard.items()), start=1):
        with safe_open(base / source_shard, framework="pt", device="cpu") as source:
            tensors = {name: source.get_tensor(name).contiguous() for name in names}
        output_shard = f"model-static-{index:05d}.safetensors"
        safetensors.torch.save_file(tensors, path / output_shard)
        for name, tensor in tensors.items():
            weight_map[name] = output_shard
            total_size += tensor.numel() * tensor.element_size()
    return total_size


def _buffer_shards(
    chunks: Iterable[list[tuple[str, torch.Tensor]]],
) -> tuple[list[tuple[str, dict[str, torch.Tensor]]], dict[str, str], int]:
    """Assign each canonical shard to one writer while every rank drains the collectives."""
    rank = torch.distributed.get_rank()
    tp_ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(tp_ranks, get_parallel_state().tp.rank, group=get_gloo_group())
    writer_loads = {global_rank: 0 for global_rank, tp_rank in enumerate(tp_ranks) if tp_rank == 0}
    if not writer_loads:
        raise RuntimeError("HF export found no TP-rank-zero writer")

    weight_map: dict[str, str] = {}
    payloads: list[tuple[str, dict[str, torch.Tensor]]] = []
    local_error: Exception | None = None
    for index, chunk in enumerate(chunks, start=1):
        if local_error is not None:
            continue
        try:
            shard_name = f"model-{index:05d}.safetensors"
            shard_size = sum(tensor.numel() * tensor.element_size() for _, tensor in chunk)
            owner = min(writer_loads, key=writer_loads.__getitem__)
            writer_loads[owner] += shard_size
            for name, _ in chunk:
                if name in weight_map:
                    raise ValueError(f"duplicate HF tensor: {name}")
                weight_map[name] = shard_name
            if rank == owner:
                payloads.append(
                    (
                        shard_name,
                        {name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in chunk},
                    )
                )
        except Exception as exc:
            # Keep driving the iterator so peers do not block in a later gather.
            local_error = exc

    if local_error is not None:
        raise local_error
    if not weight_map:
        raise ValueError("HF export produced no weights")
    return payloads, weight_map, sum(writer_loads.values())


def _write_shards(path: Path, payloads: list[tuple[str, dict[str, torch.Tensor]]]) -> None:
    for shard_name, tensors in payloads:
        safetensors.torch.save_file(tensors, path / shard_name)


def _start_direct_export(
    args,
    model: Sequence[DDP],
    path: Path,
    *,
    model_name: str,
    quantization_config,
    megatron_local_weights,
) -> _HfExport:
    static_weight_prefixes = _static_weight_prefixes(args)
    _collectively(lambda: _prepare_directory(path))
    iterator = HfWeightIteratorDirect(
        args,
        model,
        placement=WeightUpdatePlacement(gather_pp=True),
        model_name=model_name,
        quantization_config=quantization_config,
    )
    payloads, weight_map, total_size = _collectively(
        lambda: _buffer_shards(iterator.iter_hf_weights(megatron_local_weights))
    )
    rank = torch.distributed.get_rank()

    def finalize() -> None:
        if rank != 0:
            return
        base_checkpoint = Path(args.hf_checkpoint)
        if base_checkpoint.is_dir():
            exported_size = total_size + _copy_source_static_weights(
                base_checkpoint,
                path,
                weight_map,
                static_weight_prefixes,
            )
            for meta_file in base_checkpoint.iterdir():
                if _is_hf_metadata_file(meta_file):
                    shutil.copy2(meta_file, path / meta_file.name)
        else:
            if static_weight_prefixes:
                raise ValueError("hf_export_static_weight_prefixes requires a local source checkpoint")
            logger.warning("hf_checkpoint %s is not a local dir; metadata not copied to %s", args.hf_checkpoint, path)
            exported_size = total_size
        index = {"metadata": {"total_size": exported_size}, "weight_map": weight_map}
        (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-export")
    return _HfExport(finalize=finalize, future=executor.submit(_write_shards, path, payloads), executor=executor)


def export_hf_model_direct(
    args,
    model: Sequence[DDP],
    path: str | Path,
    *,
    model_name: str,
    quantization_config,
    megatron_local_weights,
) -> None:
    """Export current weights as an HF checkpoint via miles' own megatron->HF converters.

    Same conversion machinery as the weight updater, so export coverage matches
    weight-sync coverage (the bridge silently exports zero weights for specs it has
    no mapping for, e.g. qwen3.5). Collective — all ranks must call it; rank 0 writes.
    """
    _start_direct_export(
        args,
        model,
        Path(path),
        model_name=model_name,
        quantization_config=quantization_config,
        megatron_local_weights=megatron_local_weights,
    ).finish()


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Local: megatron.bridge is only needed on the bridge export path.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Save Megatron model in HuggingFace format.

    For LoRA models this saves both:
    - A **merged** HF model (adapter weights folded into base) at ``{path}/``
      so it can be loaded directly with ``AutoModelForCausalLM.from_pretrained``.
    - An **adapter-only** HF PEFT checkpoint at ``{path}/adapter/``
      so it can be loaded with ``PeftModel.from_pretrained``.

    This function is collective — all ranks must call it. On success, global rank 0
    writes a ``.complete`` marker file.

    Args:
        args: Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
        path: Destination directory; defaults to ``args.save_hf.format(rollout_id)``.
        raise_on_error: Re-raise export failures instead of logging them.
    """
    should_log = get_parallel_state().effective_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))

    try:
        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        if args.megatron_to_hf_mode == "raw" and not is_lora_model(model):
            # LoRA keeps the bridge (adapter merging).
            hf_config = load_hf_config(args.hf_checkpoint)
            export_hf_model_direct(
                args,
                model,
                path,
                model_name=type(hf_config).__name__.lower() if args.model_name is None else args.model_name,
                quantization_config=getattr(hf_config, "quantization_config", None),
                megatron_local_weights=dict(named_params_and_buffers(args, model, convert_to_global_name=True)),
            )
        else:
            bridge = _get_hf_bridge(args.hf_checkpoint)
            path.mkdir(parents=True, exist_ok=True)
            if torch.distributed.get_rank() == 0:
                (path / HF_EXPORT_COMPLETE_MARKER).unlink(missing_ok=True)
            with patch_megatron_model(model):
                # For LoRA models, merge_adapter_weights=True (default) merges
                # adapter weights into base weights for a standalone HF model.
                bridge.save_hf_pretrained(model, path=path)

            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
                    raise RuntimeError(
                        f"HF export to {path} produced no weight files — the megatron "
                        f"bridge likely has no mapping for this model architecture."
                    )

        if should_log:
            logger.info(f"Successfully saved merged HuggingFace model to {path}")
    except Exception as e:
        if raise_on_error:
            raise
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
        return

    # Additionally save adapter-only checkpoint for LoRA models
    if is_lora_model(model):
        try:
            adapter_path = path / "adapter"
            if should_log:
                logger.info(f"Saving LoRA adapter (HF PEFT format) to {adapter_path}")
            save_lora_checkpoint(model, args, str(adapter_path))
            if should_log:
                logger.info(f"Successfully saved LoRA adapter to {adapter_path}")
        except Exception as e:
            if raise_on_error:
                raise
            if should_log:
                logger.error(f"Failed to save LoRA adapter: {e}")
            return

    if torch.distributed.get_rank() == 0:
        (path / HF_EXPORT_COMPLETE_MARKER).touch()
