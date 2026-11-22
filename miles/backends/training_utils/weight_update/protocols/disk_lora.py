"""LoRA weight transfer through versioned adapter directories that engines read while serving."""

from __future__ import annotations

import json
import os
import shutil
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.arguments import sglang_attention_dp_enabled
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.utils import async_utils
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.function_registry import load_function
from miles.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config


class UpdateWeightFromDiskLora(WeightTransferProtocol):
    """``--update-weight-transfer-mode disk-lora``: each version is the whole adapter, one PEFT directory."""

    required_placement = WeightUpdatePlacement(gather_pp=False)
    supports_lora = True
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-disk-lora"
        self._lora_config = build_lora_config(args, target_modules=args.lora_adapter_targets)
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)
        self._pinned = sglang_attention_dp_enabled(
            attn_dp_size=args.sglang.base_args["attn_dp_size"],
            enable_dp_attention=args.sglang.base_args["enable_dp_attention"],
            dp_size=args.sglang.base_args["dp_size"],
        )
        self._run_tag: str | None = None
        self._turn = 0
        self._stage_ranks = 1
        self._position = 0
        self._tensors: dict[str, torch.Tensor] = {}

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
        assert not placement.gather_pp, "disk-lora writes each pipeline stage's adapter tensors where they are"
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = True
        if self._run_tag is None:
            pp_ranks = [None] * dist.get_world_size()
            dist.all_gather_object(pp_ranks, parallel_state.pp.rank, group=get_gloo_group())
            stage = [rank for rank, pp_rank in enumerate(pp_ranks) if pp_rank == parallel_state.pp.rank]
            self._turn, self._stage_ranks = stage.index(dist.get_rank()), len(stage)
            tag = [time.strftime("%Y%m%d-%H%M%S")]
            dist.broadcast_object_list(tag, src=0, group=get_gloo_group())
            self._run_tag = tag[0]

    def begin_sync(
        self,
        weight_version: int,
        iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]],
    ) -> bool:
        self._position = 0
        self._tensors.clear()
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in bucket:
            if self._position % self._stage_ranks == self._turn:
                _, hf_key = name.split(":", 1)
                self._tensors[hf_key] = tensor.detach().to("cpu", copy=True)
            self._position += 1

    def finalize(self, weight_version: int) -> None:
        version_dir = os.path.join(self.args.update_weight_disk_dir, self._version_name(weight_version))
        tensors, self._tensors = self._tensors, {}
        os.makedirs(version_dir, exist_ok=True)
        # safetensors writes 0600; engines may read as another user
        if tensors:
            path = os.path.join(version_dir, f"adapter_model-{dist.get_rank():05d}.safetensors")
            safetensors.torch.save_file(tensors, path)
            os.chmod(path, 0o644)
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            config = Path(version_dir, "adapter_config.json")
            config.write_text(json.dumps(self._lora_config, indent=2))
            os.chmod(config, 0o644)
            os.chmod(version_dir, 0o755)
        # On every rank, as every mode calls it: each node publishes its own writes.
        if self._post_write_hook is not None:
            fn_args = compute_custom_function_config(self.args, self.args.custom_update_weight_post_write_path)
            self._post_write_hook(fn_args, version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if self.args.rollout_external_url is not None:
            # On every rank: the trainer's ranks report one sync output.
            self.publication = {
                "kind": "lora",
                "lora_name": LORA_ADAPTER_NAME,
                "path": self._version_name(weight_version),
            }
        if dist.get_rank() != 0:
            return
        if self.args.rollout_external_url is None:
            self._update_engines(version_dir, weight_version)
        self._prune_old_versions()

    def _version_name(self, weight_version: int) -> str:
        return f"{LORA_ADAPTER_NAME}_{self._run_tag}_v{weight_version}"

    def _update_engines(self, version_dir: str, weight_version: int) -> None:
        async_utils.wait_futures(
            [
                updater.submit_client_call(
                    "update_lora_adapter",
                    lora_name=LORA_ADAPTER_NAME,
                    lora_path=version_dir,
                    weight_version=str(weight_version),
                    pause_mode=self.args.pause_generation_mode,
                    pinned=self._pinned,
                )
                for updater in self.cell_updaters_of_cell_id.values()
            ]
        )

    def _prune_old_versions(self) -> None:
        prefix = f"{LORA_ADAPTER_NAME}_{self._run_tag}_v"
        root = self.args.update_weight_disk_dir
        versions = sorted(int(name.removeprefix(prefix)) for name in os.listdir(root) if name.startswith(prefix))
        for version in versions[: -self.args.disk_lora_keep_versions]:
            shutil.rmtree(os.path.join(root, f"{prefix}{version}"), ignore_errors=True)
