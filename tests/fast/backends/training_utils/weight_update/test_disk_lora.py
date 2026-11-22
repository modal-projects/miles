import json
import os
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import safetensors.torch
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols.disk_lora import UpdateWeightFromDiskLora
from miles.utils.lora.utils import LORA_ADAPTER_NAME

_MODULE = "miles.backends.training_utils.weight_update.protocols.disk_lora"
# Four trainer ranks in two pipeline stages; each stage exports only its own layers' adapter tensors.
_STAGES = [0, 0, 1, 1]
_STAGE_TENSORS = {
    stage: [f"model.layers.{stage}.mlp.{leaf}.lora_{factor}.weight" for leaf in ("up", "down") for factor in "AB"]
    + [f"model.layers.{stage}.self_attn.q_proj.lora_A.weight"]
    for stage in (0, 1)
}


class _FakeEngine:
    def __init__(self) -> None:
        self.server_url = "http://e:1"
        self.updates: list[dict] = []

    async def update_lora_adapter(self, **request):
        self.updates.append(request)
        return {"success": True}


def _make_args(tmp_path, **overrides) -> Namespace:
    args = Namespace(
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_adapter_targets=["q_proj", "up_proj", "down_proj"],
        update_weight_disk_dir=str(tmp_path),
        custom_update_weight_post_write_path=None,
        rollout_external_url=None,
        pause_generation_mode="in_place",
        disk_lora_keep_versions=3,
        sglang=SimpleNamespace(base_args={"attn_dp_size": 1, "enable_dp_attention": False, "dp_size": 1}),
        ft_components=[],
        update_weight_engine_request_timeout=10.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextmanager
def _as_rank(rank: int, run_tag: str = "run-a"):
    def all_gather_object(out, value, group=None):
        out[:] = _STAGES

    def broadcast_object_list(objects, src=0, group=None):
        objects[0] = run_tag

    with (
        patch(f"{_MODULE}.dist.get_rank", return_value=rank),
        patch(f"{_MODULE}.dist.get_world_size", return_value=len(_STAGES)),
        patch(f"{_MODULE}.dist.all_gather_object", side_effect=all_gather_object),
        patch(f"{_MODULE}.dist.broadcast_object_list", side_effect=broadcast_object_list),
        patch(f"{_MODULE}.dist.barrier"),
        patch(f"{_MODULE}.get_gloo_group"),
    ):
        yield


def _connect(args: Namespace, engines: list, run_tag: str = "run-a") -> list[UpdateWeightFromDiskLora]:
    protocols = []
    for rank, stage in enumerate(_STAGES):
        protocol = UpdateWeightFromDiskLora(args)
        with _as_rank(rank, run_tag):
            protocol.connect(
                engines,
                None,
                None,
                engine_cell_ids=[f"cell-{index}" for index in range(len(engines))],
                parallel_state=SimpleNamespace(pp=SimpleNamespace(rank=stage)),
                placement=WeightUpdatePlacement(gather_pp=False),
                selector="all",
            )
        protocols.append(protocol)
    return protocols


def _sync(protocols: list[UpdateWeightFromDiskLora], weight_version: int, value: float = 1.0) -> None:
    for rank, protocol in enumerate(protocols):
        protocol.begin_sync(weight_version, None)
        names = _STAGE_TENSORS[_STAGES[rank]]
        with _as_rank(rank):
            protocol.send_bucket([(f"{LORA_ADAPTER_NAME}:{name}", torch.full((4, 2), value)) for name in names[:2]])
            protocol.send_bucket([(f"{LORA_ADAPTER_NAME}:{name}", torch.full((4, 2), value)) for name in names[2:]])
    for rank in (*range(1, len(protocols)), 0):
        with _as_rank(rank):
            protocols[rank].finalize(weight_version)


def _version_dir(tmp_path, weight_version: int, run_tag: str = "run-a") -> str:
    return os.path.join(tmp_path, f"{LORA_ADAPTER_NAME}_{run_tag}_v{weight_version}")


def test_each_stage_rank_writes_its_turn_of_the_stage_tensors_and_the_config_marks_the_version(tmp_path):
    engine = _FakeEngine()
    protocols = _connect(_make_args(tmp_path), [engine])

    _sync(protocols, 1, value=2.0)

    version_dir = _version_dir(tmp_path, 1)
    written = {
        rank: safetensors.torch.load_file(os.path.join(version_dir, f"adapter_model-{rank:05d}.safetensors"))
        for rank in range(len(_STAGES))
    }
    for rank, stage in enumerate(_STAGES):
        turn = rank - _STAGES.index(stage)
        assert sorted(written[rank]) == sorted(_STAGE_TENSORS[stage][turn::2])
    names = [name for tensors in written.values() for name in tensors]
    assert sorted(names) == sorted(_STAGE_TENSORS[0] + _STAGE_TENSORS[1])
    assert all(
        torch.equal(tensor, torch.full((4, 2), 2.0)) for tensors in written.values() for tensor in tensors.values()
    )
    config = json.loads(open(os.path.join(version_dir, "adapter_config.json")).read())
    assert config["r"] == 16 and config["lora_alpha"] == 32
    assert engine.updates == [
        {
            "lora_name": LORA_ADAPTER_NAME,
            "lora_path": version_dir,
            "weight_version": "1",
            "pause_mode": "in_place",
            "pinned": False,
        }
    ]


def test_every_rank_publishes_its_writes_through_the_post_write_hook(tmp_path):
    seen = []

    def hook(args, version_dir, engines):
        seen.append((dist.get_rank(), sorted(os.listdir(version_dir))))

    with patch(f"{_MODULE}.load_function", return_value=hook), patch(f"{_MODULE}.compute_custom_function_config"):
        protocols = _connect(_make_args(tmp_path, custom_update_weight_post_write_path="hook"), [_FakeEngine()])
        _sync(protocols, 1)

    assert sorted(rank for rank, _ in seen) == [0, 1, 2, 3]
    for rank, files in seen:
        assert f"adapter_model-{rank:05d}.safetensors" in files
    assert "adapter_config.json" in dict(seen)[0]


def test_every_sync_restarts_the_turns(tmp_path):
    protocols = _connect(_make_args(tmp_path), [_FakeEngine()])

    _sync(protocols, 1)
    _sync(protocols, 2)

    first, second = (sorted(os.listdir(_version_dir(tmp_path, version))) for version in (1, 2))
    assert first == second
    for rank in range(len(_STAGES)):
        name = f"adapter_model-{rank:05d}.safetensors"
        assert sorted(safetensors.torch.load_file(os.path.join(_version_dir(tmp_path, 1), name))) == sorted(
            safetensors.torch.load_file(os.path.join(_version_dir(tmp_path, 2), name))
        )


def test_every_rank_reports_the_version_directory_to_an_external_rollout_and_no_engine_is_called(tmp_path):
    engine = _FakeEngine()
    protocols = _connect(_make_args(tmp_path, rollout_external_url="https://rollout"), [engine])

    _sync(protocols, 4)

    assert engine.updates == []
    assert all(
        protocol.publication
        == {"kind": "lora", "lora_name": LORA_ADAPTER_NAME, "path": f"{LORA_ADAPTER_NAME}_run-a_v4"}
        for protocol in protocols
    )
    assert os.path.exists(os.path.join(tmp_path, protocols[0].publication["path"], "adapter_config.json"))


def test_a_resumed_run_writes_its_versions_apart_and_prunes_only_its_own(tmp_path):
    first = _connect(_make_args(tmp_path, disk_lora_keep_versions=2), [_FakeEngine()], run_tag="run-a")
    for version in (3, 4, 5):
        _sync(first, version, value=1.0)
    # resumed from a step-3 checkpoint: version 4 again, with other content
    resumed = _connect(_make_args(tmp_path, disk_lora_keep_versions=2), [_FakeEngine()], run_tag="run-b")
    for version in (3, 4, 5):
        _sync(resumed, version, value=7.0)

    assert sorted(os.listdir(tmp_path)) == sorted(
        f"{LORA_ADAPTER_NAME}_{tag}_v{version}" for tag in ("run-a", "run-b") for version in (4, 5)
    )
    old = safetensors.torch.load_file(
        os.path.join(_version_dir(tmp_path, 4, "run-a"), "adapter_model-00000.safetensors")
    )
    assert all(torch.equal(tensor, torch.full((4, 2), 1.0)) for tensor in old.values())


def test_dp_attention_engines_get_the_adapter_pinned(tmp_path):
    engine = _FakeEngine()
    sglang = SimpleNamespace(base_args={"attn_dp_size": 4, "enable_dp_attention": True, "dp_size": 4})
    protocols = _connect(_make_args(tmp_path, sglang=sglang, pause_generation_mode="retract"), [engine])

    _sync(protocols, 1)

    assert engine.updates[0]["pinned"] is True and engine.updates[0]["pause_mode"] == "retract"
