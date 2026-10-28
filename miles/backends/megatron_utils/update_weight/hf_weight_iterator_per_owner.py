import re
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import _iter_mm_tower_units
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
    HfWeightIteratorDirect,
    _check_and_fix_partition,
    _gather_with_stride,
    _get_param_full_size,
    _pack_param_infos_by_size,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.types import ParamInfo


class HfWeightIteratorPerOwner(HfWeightIteratorDirect):
    """Experts are yielded by the rank that converts them. Every other tensor goes, a whole unit
    at a time, to one owner of the stage, which alone receives its TP shards."""

    forced_placement = WeightUpdatePlacement(gather_pp=False, per_owner=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.placement.gather_pp, f"per-owner export plans each pipeline stage alone: {self.placement}"
        assert (
            self._convert_experts_before_gather or not self._expert_batches
        ), "per-owner export requires --expert-tensor-parallel-size 1"
        self._expert_batches = [replace(batch, gathers=()) for batch in self._expert_batches]
        non_expert_infos = [info for infos in self._non_expert_batches for info in infos]
        self._owned_batches = self._plan_owned_batches(non_expert_infos)

    def _plan_owned_batches(self, non_expert_infos: list[ParamInfo]) -> list[tuple[int, list[ParamInfo]]]:
        """(owner, infos) batches of this rank's TP group. Every rank of a stage computes the same
        plan; a param several stages hold (an MTP layer's embedding) belongs to the lowest."""
        parallel = get_parallel_state()
        if parallel.pp.size > 1:
            stage_names: list = [None] * parallel.pp.size
            dist.all_gather_object(stage_names, [info.name for info in non_expert_infos], group=parallel.pp.group)
            lower = {name for names in stage_names[: parallel.pp.rank] for name in names}
            non_expert_infos = [info for info in non_expert_infos if info.name not in lower]
        stage_ranks = dist.get_process_group_ranks(parallel.tp_dp_cp.group)
        owners = _assign_owners(non_expert_infos, stage_ranks)
        tp_ranks = set(dist.get_process_group_ranks(parallel.tp.group))
        return [
            (owner, infos)
            for owner in stage_ranks
            if owner in tp_ranks
            for infos in _pack_param_infos_by_size(
                self.args, [info for info in non_expert_infos if owners[info.name] == owner]
            )
        ]

    def _hf_atomic_update_groups(self):
        # Owners take whole layers, so a group stays on one owner; an owner may hold none.
        return [replace(group, optional=True) for group in super()._hf_atomic_update_groups()]

    def _iter_hf_param_units(self, weights, *, materialize):
        for owner, param_infos in self._owned_batches:
            named_params = _gather_to_owner(self.args, param_infos, weights, owner=owner)
            if materialize:
                yield from self._convert_to_hf_param_units(named_params)
            del named_params
        for batch in self._expert_batches:
            units = self._convert_and_gather_expert_batch(batch, weights)
            if materialize:
                yield from units
            del units
        if dist.get_rank() == 0:
            yield from _iter_mm_tower_units(self.args, materialize=materialize)


def _unit_of(name: str) -> str:
    if match := re.search(r"\.(decoder|mtp)\.layers\.(\d+)\.", name):
        return f"{match[1]}.{match[2]}"
    for unit in ("embedding", "output_layer"):
        if f".{unit}." in name:
            return unit
    return "rest"


def _assign_owners(param_infos: Sequence[ParamInfo], ranks: Sequence[int]) -> dict[str, int]:
    """Owner rank of each param: whole units, largest first, each to the rank with the fewest
    bytes so far."""
    units: dict[str, list[ParamInfo]] = {}
    for info in param_infos:
        units.setdefault(_unit_of(info.name), []).append(info)
    sizes = {unit: sum(_get_param_full_size(info) for info in infos) for unit, infos in units.items()}
    loads = dict.fromkeys(ranks, 0)
    owners = {}
    for unit in sorted(units, key=lambda unit: (-sizes[unit], unit)):
        owner = min(loads, key=lambda rank: (loads[rank], rank))
        loads[owner] += sizes[unit]
        owners.update((info.name, owner) for info in units[unit])
    return owners


def _is_tp_sharded(info: ParamInfo) -> bool:
    """all_gather_params_async's rule, for non-expert params."""
    return (
        get_parallel_state().tp.size > 1
        and info.attrs["tensor_model_parallel"]
        and info.attrs["parallel_mode"] != "duplicated"
        and "expert_bias" not in info.name
    )


def _gather_to_owner(
    args: Namespace, param_infos: Sequence[ParamInfo], megatron_local_weights, *, owner: int
) -> list[tuple[str, torch.Tensor]]:
    """TP shards gathered to ``owner`` only. Every rank of the owner's TP group calls it; only
    the owner gets the full params back."""
    tp = get_parallel_state().tp
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    tasks, handles = [], []
    for info in param_infos:
        sharded = _is_tp_sharded(info)
        if not sharded and rank != owner:
            continue
        param = megatron_local_weights[info.name].detach().to(device=device, non_blocking=True)
        partitions = None
        if sharded:
            partitions = [torch.empty_like(param) for _ in range(tp.size)] if rank == owner else None
            handles.append(dist.gather(param, gather_list=partitions, dst=owner, group=tp.group, async_op=True))
        tasks.append((info, param, partitions))
    for handle in handles:
        handle.wait()
    if rank != owner:
        return []
    named_params = []
    for info, param, partitions in tasks:
        if partitions is not None:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, info.attrs["partition_stride"], info.attrs["partition_dim"]
            )
            param = _gather_with_stride(partitions, partition_dim, partition_stride)
        named_params.append((info.name, param))
    return named_params
