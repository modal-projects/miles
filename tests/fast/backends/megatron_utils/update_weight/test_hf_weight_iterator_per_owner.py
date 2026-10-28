from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["weight-update"])


from argparse import Namespace
from collections import Counter
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct as direct_module
from miles.backends.megatron_utils.update_weight import hf_weight_iterator_per_owner as per_owner_module
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_per_owner import (
    HfWeightIteratorPerOwner,
    _assign_owners,
    _gather_to_owner,
    _is_tp_sharded,
    _unit_of,
)
from miles.backends.training_utils import parallel
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.types import ParamInfo

_UNSHARDED = {"tensor_model_parallel": False, "partition_dim": -1, "partition_stride": 1, "parallel_mode": None}


def _info(name: str, size: int = 1, **attrs) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={**_UNSHARDED, **attrs},
        size=size,
        src_rank=0,
    )


def _set_parallel_state(monkeypatch: pytest.MonkeyPatch, state: SimpleNamespace) -> None:
    monkeypatch.setattr(direct_module, "get_parallel_state", lambda: state)
    monkeypatch.setattr(per_owner_module, "get_parallel_state", lambda: state)


@pytest.fixture
def tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    trivial = SimpleNamespace(size=1, rank=0, group=None)
    _set_parallel_state(monkeypatch, SimpleNamespace(tp=trivial, etp=trivial))


class TestAssignOwners:
    def test_largest_unit_goes_to_the_least_loaded_rank(self, tp1: None) -> None:
        """Ties go to the lowest rank, so every rank computes the same plan."""
        infos = [
            _info("module.module.embedding.word_embeddings.weight", 20),
            _info("module.module.output_layer.weight", 20),
            _info("module.module.decoder.layers.0.self_attention.linear_proj.weight", 6),
            _info("module.module.decoder.layers.0.input_layernorm.weight", 4),
            _info("module.module.decoder.layers.1.self_attention.linear_proj.weight", 6),
            _info("module.module.decoder.layers.1.input_layernorm.weight", 4),
            _info("module.module.decoder.final_layernorm.weight", 1),
        ]

        owners = _assign_owners(infos, [0, 1, 2, 3])

        assert owners == {
            "module.module.embedding.word_embeddings.weight": 0,
            "module.module.output_layer.weight": 1,
            "module.module.decoder.layers.0.self_attention.linear_proj.weight": 2,
            "module.module.decoder.layers.0.input_layernorm.weight": 2,
            "module.module.decoder.layers.1.self_attention.linear_proj.weight": 3,
            "module.module.decoder.layers.1.input_layernorm.weight": 3,
            "module.module.decoder.final_layernorm.weight": 2,
        }

    def test_every_param_has_one_owner_and_units_stay_whole_in_any_order(self, tp1: None) -> None:
        """SGLang fuses tensors within a layer, so a layer must never be split across owners."""
        infos = [
            _info(f"module.module.{prefix}.layers.{layer}.{param}", size)
            for prefix, layers in (("decoder", range(6)), ("mtp", range(1)))
            for layer in layers
            for param, size in (("self_attention.linear_q_down_proj.weight", 7 + layer), ("input_layernorm.weight", 1))
        ] + [
            _info("module.module.embedding.word_embeddings.weight", 30),
            _info("module.module.output_layer.weight", 30),
            _info("module.module.decoder.final_layernorm.weight", 1),
        ]

        owners = _assign_owners(infos, [4, 5, 6])

        assert set(owners) == {info.name for info in infos}
        units: dict[str, set[int]] = {}
        for name, owner in owners.items():
            units.setdefault(_unit_of(name), set()).add(owner)
        assert all(len(unit_owners) == 1 for unit_owners in units.values())
        assert set(owners.values()) == {4, 5, 6}
        assert _assign_owners(list(reversed(infos)), [4, 5, 6]) == owners


class _RecordingDist:
    def __init__(self) -> None:
        self.all_gathers = 0

    def all_gather(self, outputs, tensor, group, async_op):
        self.all_gathers += 1
        for output in outputs:
            output.copy_(tensor)
        return SimpleNamespace(wait=lambda: None)


@pytest.mark.parametrize(
    "name, attrs, tp_size",
    [
        ("decoder.layers.0.self_attention.linear_proj.weight", {"tensor_model_parallel": True, "partition_dim": 1}, 2),
        ("decoder.layers.0.self_attention.linear_proj.weight", {"tensor_model_parallel": True, "partition_dim": 1}, 1),
        (
            "decoder.layers.0.mlp.shared_experts.linear_fc2.weight",
            {"tensor_model_parallel": True, "partition_dim": 1},
            2,
        ),
        (
            "decoder.layers.0.self_attention.linear_q_down_proj.weight",
            {"tensor_model_parallel": True, "partition_dim": 0, "parallel_mode": "duplicated"},
            2,
        ),
        ("decoder.layers.0.input_layernorm.weight", {}, 2),
        ("decoder.layers.0.mlp.router.expert_bias", {"tensor_model_parallel": True, "partition_dim": 0}, 2),
    ],
)
def test_the_owner_gather_shards_exactly_what_all_gather_params_async_gathers(
    monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict, tp_size: int
) -> None:
    """_is_tp_sharded copies Miles' rule for non-expert params; this keeps the copy from drifting."""
    group = SimpleNamespace(size=tp_size, rank=0, group="tp")
    _set_parallel_state(monkeypatch, SimpleNamespace(tp=group, etp=SimpleNamespace(size=1, rank=0, group="etp")))
    recording = _RecordingDist()
    monkeypatch.setattr(direct_module, "dist", recording)
    info = _info(name, 8, **attrs)
    param = torch.zeros(2, 4)
    for key, value in info.attrs.items():
        setattr(param, key, value)

    direct_module.all_gather_params_async(Namespace(backend=Namespace(swiglu=True)), [(info, param)])

    assert _is_tp_sharded(info) == (recording.all_gathers == 1)


def test_load_groups_are_kept_but_optional() -> None:
    """An owner may hold no layer, so may match no group; whole layers keep each group on one owner."""
    iterator = object.__new__(HfWeightIteratorPerOwner)
    iterator.model_name = "glm5"
    iterator.args = Namespace(backend=Namespace(q_lora_rank=64))

    groups = iterator._hf_atomic_update_groups()

    assert [group.suffixes for group in groups] == [
        group.suffixes for group in HfWeightIteratorDirect._hf_atomic_update_groups(iterator)
    ]
    assert groups and all(group.optional for group in groups)


@pytest.fixture
def per_owner_iterator(monkeypatch: pytest.MonkeyPatch):
    def build(edp_rank: int, etp_size: int = 1) -> HfWeightIteratorPerOwner:
        trivial = SimpleNamespace(size=1, rank=0, group="trivial")
        stage = SimpleNamespace(size=2, rank=edp_rank, group="stage")
        etp = SimpleNamespace(size=etp_size, rank=0, group="etp")
        _set_parallel_state(
            monkeypatch, SimpleNamespace(tp=trivial, etp=etp, ep=trivial, pp=trivial, edp=stage, tp_dp_cp=stage)
        )
        model = torch.nn.Linear(1, 1, bias=False)
        model.config = SimpleNamespace(mtp_num_layers=0)
        weights = [(f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}", torch.ones(1)) for i in range(3)]
        weights.append(("decoder.layers.0.input_layernorm.weight", torch.ones(1)))
        monkeypatch.setattr(direct_module, "named_params_and_buffers", lambda _args, _model: iter(weights))
        monkeypatch.setattr(dist, "get_rank", lambda: edp_rank)
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0, 1] if group == "stage" else [edp_rank])
        return HfWeightIteratorPerOwner(
            args=Namespace(
                backend=Namespace(num_experts=3, swiglu=True), update_weight_buffer_size=1024, hf_checkpoint="/unused"
            ),
            model=[model],
            placement=WeightUpdatePlacement(gather_pp=False, per_owner=True),
            model_name="qwen3_moe",
            quantization_config=None,
        )

    return build


class TestPerOwnerIterator:
    @pytest.mark.parametrize("edp_rank,expert_ids", [(0, [0, 1]), (1, [2])])
    def test_experts_stay_with_the_rank_that_converts_them(
        self, per_owner_iterator, edp_rank: int, expert_ids: list[int]
    ) -> None:
        iterator = per_owner_iterator(edp_rank)

        assert [info.name for batch in iterator._expert_batches for info in batch.param_infos] == [
            f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}" for i in expert_ids
        ]
        assert all(batch.gathers == () for batch in iterator._expert_batches)
        assert [(owner, [info.name for info in infos]) for owner, infos in iterator._owned_batches] == (
            [(0, ["decoder.layers.0.input_layernorm.weight"])] if edp_rank == 0 else []
        )

    def test_expert_tensor_parallel_is_refused(self, per_owner_iterator) -> None:
        with pytest.raises(AssertionError, match="--expert-tensor-parallel-size 1"):
            per_owner_iterator(edp_rank=0, etp_size=2)


# name -> (full shape, TP attrs); fc1 is SwiGLU's interleaved [gate; up].
_PARAMS = {
    "module.module.embedding.word_embeddings.weight": ((8, 4), {"tensor_model_parallel": True, "partition_dim": 0}),
    "module.module.decoder.layers.0.input_layernorm.weight": ((4,), {}),
    "module.module.decoder.layers.0.self_attention.linear_proj.weight": (
        (4, 8),
        {"tensor_model_parallel": True, "partition_dim": 1},
    ),
    "module.module.decoder.layers.0.mlp.linear_fc1.weight": (
        (8, 4),
        {"tensor_model_parallel": True, "partition_dim": 0, "partition_stride": 2},
    ),
    "module.module.decoder.layers.1.self_attention.linear_q_down_proj.weight": (
        (4, 4),
        {"tensor_model_parallel": True, "partition_dim": 0, "parallel_mode": "duplicated"},
    ),
    "module.module.decoder.layers.1.mlp.router.expert_bias": ((4,), {}),
    "module.module.decoder.final_layernorm.weight": ((4,), {}),
    "module.module.output_layer.weight": ((8, 4), {"tensor_model_parallel": True, "partition_dim": 0}),
    "module.module.mtp.layers.0.enorm.weight": ((4,), {}),
}
_LAST_STAGE = (
    "module.module.decoder.layers.1.",
    "module.module.decoder.final_layernorm.",
    "module.module.output_layer.",
    "module.module.mtp.",
)


def _full(name: str) -> torch.Tensor:
    shape, _ = _PARAMS[name]
    generator = torch.Generator().manual_seed(list(_PARAMS).index(name))
    return torch.randn(shape, generator=generator)


def _local_shard(name: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    full, attrs = _full(name), {**_UNSHARDED, **_PARAMS[name][1]}
    if not attrs["tensor_model_parallel"] or attrs["parallel_mode"] == "duplicated":
        return full
    dim = attrs["partition_dim"]
    parts = full.chunk(attrs["partition_stride"], dim=dim)
    return torch.cat([part.chunk(tp_size, dim=dim)[tp_rank] for part in parts], dim=dim).contiguous()


def _stage_names(pp_size: int, pp_rank: int) -> list[str]:
    if pp_size == 1:
        return list(_PARAMS)
    last = [name for name in _PARAMS if name.startswith(_LAST_STAGE)]
    # The MTP stage builds the embedding too, as Megatron does.
    embedding = "module.module.embedding.word_embeddings.weight"
    if pp_rank == 1:
        return [embedding, *last]
    return [name for name in _PARAMS if name not in last]


def _export_on_every_rank(rank: int, world_size: int, port: int, *, pp_size: int) -> None:
    init_gloo(rank, world_size, port=port)
    torch.cuda.current_device = lambda: "cpu"
    tp_size, pp_rank = 2, rank // 2
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    pp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp = SimpleNamespace(size=tp_size, rank=rank % 2, group=tp_groups[rank // 2])
    if pp_size == 1:
        pp = SimpleNamespace(size=1, rank=0, group=None)
        stage = SimpleNamespace(size=4, rank=rank, group=dist.group.WORLD)
    else:
        pp = SimpleNamespace(size=2, rank=pp_rank, group=pp_groups[rank % 2])
        stage = tp
    parallel.set_parallel_state(
        SimpleNamespace(tp=tp, pp=pp, tp_dp_cp=stage, etp=SimpleNamespace(size=1, rank=0, group=None))
    )
    names = _stage_names(pp_size, pp_rank)
    weights = {name: _local_shard(name, tp.rank, tp_size) for name in names}
    infos = [
        ParamInfo(
            name=name,
            dtype=torch.float32,
            shape=weights[name].shape,
            attrs={**_UNSHARDED, **_PARAMS[name][1]},
            size=weights[name].numel() * 4,
            src_rank=rank,
        )
        for name in names
    ]
    iterator = object.__new__(HfWeightIteratorPerOwner)
    iterator.args = Namespace(backend=Namespace(swiglu=True), update_weight_buffer_size=1 << 20)

    exported = {
        name: tensor
        for owner, batch in iterator._plan_owned_batches(infos)
        for name, tensor in _gather_to_owner(iterator.args, batch, weights, owner=owner)
    }

    every_export: list = [None] * world_size
    dist.all_gather_object(every_export, exported)
    assert all(every_export), "every rank owns part of its stage"
    counts = Counter(name for export in every_export for name in export)
    assert counts == Counter(dict.fromkeys(_PARAMS, 1))
    for export in every_export:
        for name, tensor in export.items():
            assert torch.equal(tensor, _full(name)), name
    dist.destroy_process_group()


@pytest.mark.parametrize("pp_size", [1, 2], ids=["tp2_dp2", "pp2_tp2"])
def test_each_tensor_reaches_exactly_one_owner_whole(pp_size: int) -> None:
    """Real gloo collectives on 4 ranks: owners in both TP pairs (dst is a global rank), strided
    SwiGLU shards, and with two stages the MTP stage's embedding left to the lowest stage."""
    run_multiprocess(partial(_export_on_every_rank, pp_size=pp_size), world_size=4)
