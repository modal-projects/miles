from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from .parallel import get_parallel_state

logger = logging.getLogger(__name__)


def save_hf_model_to_path(args: Namespace, output_dir: str | Path, model: Sequence[torch.nn.Module]) -> None:
    """Save a Megatron model as a complete HF checkpoint at *output_dir*.

    Collective — all ranks must call it. Used by disk weight sync to publish a full checkpoint the
    rollout engines reload via ``update_weights_from_disk``; goes through Megatron-Bridge, like
    ``--save-hf``. Unlike ``model.save_hf_model`` it raises on failure (a bad publish must not
    silently leave the engines on stale weights)."""
    from megatron.bridge import AutoBridge

    from miles.utils.megatron_bridge_utils import patch_megatron_model

    path = Path(output_dir)
    should_log = get_parallel_state().intra_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    if should_log:
        logger.info("Saving HF checkpoint to %s with Megatron Bridge", path)

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    path.mkdir(parents=True, exist_ok=True)
    with patch_megatron_model(model):
        bridge.save_hf_pretrained(model, path=path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if should_log:
        logger.info("Successfully saved HF checkpoint to %s", path)
