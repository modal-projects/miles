"""Checkpoint directories: written collectively, complete at their final path."""

# TODO: isolate checkpoint IO failures; they currently terminate the trainer cell.

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


def publish_checkpoint_dir(tmp_dir: Path, final_dir: Path) -> None:
    """Atomically point ``final_dir`` at ``tmp_dir``'s contents; rank 0 only, older versions are retained."""
    if _rank() != 0:
        return
    version_dir = final_dir.parent / f"_version_{final_dir.name}_{uuid4().hex}"
    os.replace(tmp_dir, version_dir)
    tmp_dir.symlink_to(version_dir.name, target_is_directory=True)
    os.replace(tmp_dir, final_dir)


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
    publish: Callable[[Path, Path], None] | None = None,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    ``publish`` runs on every rank once all shards and ``META.json`` are in
    ``tmp_dir``; it must leave ``path`` pointing at the completed checkpoint.
    Default: rank 0 renames it into place, which assumes a shared POSIX filesystem.
    """
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"

    def make_tmp_dir():
        if _rank() == 0:
            if not overwrite and final_dir.exists():
                raise FileExistsError(f"checkpoint {final_dir} already exists")
            if final_dir.exists() and not final_dir.is_symlink():
                raise NotImplementedError(
                    f"cannot overwrite a legacy checkpoint directory {final_dir}; save under a new name"
                )
            # a crashed attempt may leave shards or an unpublished version link
            if tmp_dir.is_symlink():
                tmp_dir.unlink()
            elif tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

    def write_metadata():
        if _rank() == 0 and metadata is not None:
            (tmp_dir / "META.json").write_text(json.dumps(metadata, indent=2))

    make_tmp_dir()
    _barrier()
    write_shards(tmp_dir)
    _barrier()
    write_metadata()
    _barrier()
    (publish or publish_checkpoint_dir)(tmp_dir, final_dir)
    _barrier()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
