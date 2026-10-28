import re

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["weight-update"])

from pathlib import Path

import torch
from safetensors.torch import save_file

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.backends.training_utils.weight_update.protocols.weight_delta import protocol as weight_delta
from miles.backends.training_utils.weight_update.protocols.weight_delta.protocol import (
    UpdateWeightFromDelta,
    _check_exported,
    _check_exported_values,
)
from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import version_dir
from miles.utils.args.custom_view import ImmutableNamespace


class _Engine:
    def __init__(
        self, calls: list, index: int, label: str = "default", failing: str | None = None, stats: dict | None = None
    ):
        self._calls, self._index, self._label, self._failing = calls, index, label, failing
        self._stats = stats or {}

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((self._index, name, kwargs))
            if name == "get_weight_version":
                return self._label
            return {"success": name != self._failing, "message": f"{name} failed", "stats": self._stats.get(name)}

        return method


@pytest.fixture(autouse=True)
def single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    def gather_object(obj, object_gather_list=None, dst=0, group=None):
        object_gather_list[0] = obj

    def all_gather_object(object_list, obj, group=None):
        object_list[0] = obj

    monkeypatch.setattr(weight_delta.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(weight_delta.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(weight_delta.dist, "gather_object", gather_object)
    monkeypatch.setattr(weight_delta.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(weight_delta.dist, "broadcast_object_list", lambda object_list, src=0, group=None: None)
    monkeypatch.setattr(weight_delta, "get_gloo_group", lambda: None)


def _protocol(tmp_path: Path, engines: list[_Engine]) -> UpdateWeightFromDelta:
    args = ImmutableNamespace(
        update_weight_disk_dir=str(tmp_path / "delta"),
        custom_update_weight_post_write_path=None,
        hf_checkpoint=str(tmp_path / "base"),
        weight_delta_codec="gap-zstd",
        sglang=SglangConfig(models=[], base_args={"weight_version": "default"}),
        ft_components=[],
        update_weight_engine_request_timeout=10.0,
        pause_generation_mode="retract",
        fully_async=False,
        colocate=False,
    )
    protocol = UpdateWeightFromDelta(args)
    _connect(protocol, engines)
    return protocol


def _connect(protocol: UpdateWeightFromDelta, engines: list[_Engine]) -> None:
    protocol.connect(
        engines, None, None, [f"cell-{i}" for i in range(len(engines))], None, protocol.required_placement, "all"
    )


def _launch_exporting_nothing(protocol: UpdateWeightFromDelta, version: int) -> None:
    assert protocol.begin_sync(version, iter_buckets=None)
    protocol.finalize(version)


def _publish(delta_dir: Path, *versions: int, sealed: bool = True) -> None:
    for version in versions:
        directory = version_dir(delta_dir, version)
        directory.mkdir(parents=True)
        if sealed:
            (directory / "manifest.json").write_text("{}")


def _published(delta_dir: Path) -> list[str]:
    return sorted(entry.name for entry in delta_dir.iterdir())


def test_a_fresh_launch_publishes_nothing_and_labels_every_engine(tmp_path: Path):
    calls: list = []
    _launch_exporting_nothing(_protocol(tmp_path, [_Engine(calls, 0), _Engine(calls, 1)]), 0)

    for index in (0, 1):
        assert [(name, kwargs) for i, name, kwargs in calls if i == index] == [
            ("get_weight_version", {}),
            ("update_weight_version", {"weight_version": "0"}),
        ]
    assert not (tmp_path / "delta").exists()


def test_a_resume_replays_the_history_in_order_after_deleting_what_came_later(tmp_path: Path):
    delta_dir = tmp_path / "delta"
    _publish(delta_dir, 1, 2, 3, 4)
    calls: list = []
    protocol = _protocol(tmp_path, [_Engine(calls, 0)])
    hooked: list = []
    protocol._post_write_hook = lambda args, path, engines: hooked.append((path, _published(delta_dir)))

    _launch_exporting_nothing(protocol, 2)

    assert _published(delta_dir) == ["weight_v000001", "weight_v000002"]
    assert hooked == [(str(delta_dir), ["weight_v000001", "weight_v000002"])]
    expected = [("get_weight_version", {})]
    for version in (1, 2):
        manifest = str(version_dir(delta_dir, version) / "manifest.json")
        expected += [
            ("prepare_weights_from_delta", {"manifest": manifest}),
            ("pause_generation", {"mode": "retract"}),
            ("flush_cache", {}),
            ("update_weights_from_delta", {"manifest": manifest, "weight_version": str(version)}),
            ("continue_generation", {}),
        ]
    assert [(name, kwargs) for _, name, kwargs in calls] == expected


@pytest.mark.parametrize(
    "labels, sealed, unsealed, version, message",
    [
        (["default", "7"], [1, 3], [], 0, "rollout engines ['cell-1'] hold updated weights"),
        (["default"], [1, 2, 4], [], 3, "holds no published version 3"),
        (["default"], [1, 4], [2], 2, "holds no published version 2"),
    ],
    ids=["an engine past its startup label", "history ends before the trainer", "the trainer's version unsealed"],
)
def test_a_launch_that_cannot_start_from_the_history_is_refused_before_anything_is_deleted(
    tmp_path: Path, labels: list[str], sealed: list[int], unsealed: list[int], version: int, message: str
):
    delta_dir = tmp_path / "delta"
    _publish(delta_dir, *sealed)
    _publish(delta_dir, *unsealed, sealed=False)
    before = _published(delta_dir)
    calls: list = []
    protocol = _protocol(tmp_path, [_Engine(calls, i, label) for i, label in enumerate(labels)])

    with pytest.raises(RuntimeError, match=re.escape(message)):
        protocol.begin_sync(version, iter_buckets=None)

    assert _published(delta_dir) == before
    assert {name for _, name, _ in calls} == {"get_weight_version"}


def test_engines_resume_even_when_an_update_fails(tmp_path: Path):
    _publish(tmp_path / "delta", 1)
    calls: list = []
    protocol = _protocol(tmp_path, [_Engine(calls, 0, failing="update_weights_from_delta")])

    with pytest.raises(RuntimeError, match="update_weights_from_delta failed on a rollout engine"):
        _launch_exporting_nothing(protocol, 1)

    assert [name for _, name, _ in calls][-2:] == ["update_weights_from_delta", "continue_generation"]


def test_each_engine_stat_is_logged_at_its_slowest_gpu(tmp_path: Path):
    _publish(tmp_path / "delta", 1)
    stats = [
        {
            "prepare_weights_from_delta": [{"read_s": 1.0, "codec": "gap-zstd"}, {"read_s": 3.0}],
            "update_weights_from_delta": [{"gpu_s": 0.1}, None],
        },
        {
            "prepare_weights_from_delta": [{"read_s": 2.0, "plan_s": 0.3}],
            "update_weights_from_delta": [{"gpu_s": 0.4, "batches": 6}],
        },
    ]
    protocol = _protocol(tmp_path, [_Engine([], index, stats=engine) for index, engine in enumerate(stats)])

    _launch_exporting_nothing(protocol, 1)

    assert {key: value for key, value in protocol.pop_metrics().items() if "_engine_" in key} == {
        "perf/update_weights_engine_prepare_read_s": 3.0,
        "perf/update_weights_engine_prepare_plan_s": 0.3,
        "perf/update_weights_engine_apply_gpu_s": 0.4,
        "perf/update_weights_engine_apply_batches": 6,
    }


def test_engines_that_change_after_the_launch_are_refused(tmp_path: Path):
    engines = [_Engine([], 0)]
    protocol = _protocol(tmp_path, engines)
    _launch_exporting_nothing(protocol, 0)

    with pytest.raises(RuntimeError, match="changed after the launch"):
        _connect(protocol, engines)


def test_the_export_must_match_the_checkpoint_headers_and_come_from_one_rank_each(tmp_path: Path):
    tmp_path.joinpath("base").mkdir()
    save_file(
        {"w": torch.zeros(2, 3, dtype=torch.bfloat16), "s": torch.zeros(3, dtype=torch.float32)},
        tmp_path / "base" / "model-00001.safetensors",
    )
    matching = {"w": ("BF16", [2, 3]), "s": ("F32", [3])}
    _check_exported([matching], str(tmp_path / "base"))

    with pytest.raises(ValueError, match=r"2 exported tensors differ .*s: exported \('F32', \[4\]\).*x: exported"):
        _check_exported([{"w": ("BF16", [2, 3]), "s": ("F32", [4])}, {"x": ("BF16", [1])}], str(tmp_path / "base"))
    with pytest.raises(RuntimeError, match=r"1 tensors are exported by two ranks, e\.g\. \['w'\]"):
        _check_exported([matching, {"w": ("BF16", [2, 3])}], str(tmp_path / "base"))


def test_the_export_must_have_the_checkpoints_values(tmp_path: Path):
    tmp_path.joinpath("base").mkdir()
    weights = {"w": torch.arange(6, dtype=torch.bfloat16).view(2, 3), "s": torch.ones(3)}
    save_file(weights, tmp_path / "base" / "model-00001.safetensors")
    exported = {name: tensor.view(-1).view(torch.uint8).clone() for name, tensor in weights.items()}
    _check_exported_values(exported, str(tmp_path / "base"))

    exported["s"][0] ^= 1
    with pytest.raises(ValueError, match=r"1 exported tensors differ from --hf-checkpoint's values, e\.g\. \['s'\]"):
        _check_exported_values(exported, str(tmp_path / "base"))
