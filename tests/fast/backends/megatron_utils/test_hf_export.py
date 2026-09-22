import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors.torch
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


@pytest.fixture
def exporter(monkeypatch, tmp_path):
    class WeightUpdatePlacement:
        def __init__(self, *, gather_pp):
            self.gather_pp = gather_pp

    stubs = {
        "megatron.core.distributed": {"DistributedDataParallel": object},
        "miles.backends.megatron_utils.lora.utils": {
            "is_lora_model": lambda model: False,
            "save_lora_checkpoint": lambda *args: None,
        },
        "miles.backends.megatron_utils.named_weights": {
            "named_params_and_buffers": lambda *args, **kwargs: [],
        },
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct": {
            "HfWeightIteratorDirect": object,
        },
        "miles.backends.training_utils.weight_update.hf_weight_iterator": {
            "WeightUpdatePlacement": WeightUpdatePlacement,
        },
        "miles.utils.hf_config": {
            "HF_EXPORT_COMPLETE_MARKER": ".complete",
            "load_hf_config": lambda path: SimpleNamespace(),
        },
        "miles.utils.megatron_bridge_utils": {"patch_megatron_model": lambda model: nullcontext()},
        "miles.utils.distributed_utils": {"get_gloo_group": lambda: None},
        "miles.backends.training_utils.parallel": {
            "get_parallel_state": lambda: SimpleNamespace(
                effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=0)
            ),
        },
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/hf_export.py"
    spec = importlib.util.spec_from_file_location("hf_export_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    args = SimpleNamespace(
        hf_checkpoint=str(base),
        save_hf=str(tmp_path / "export"),
        model_name=None,
        megatron_to_hf_mode="raw",
    )
    chunks = [[("first", torch.arange(4))], [("second", torch.ones(4))]]
    monkeypatch.setattr(
        module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(iter_hf_weights=lambda weights: iter(chunks)),
    )
    rank = threading.local()
    rank.value = 0
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: getattr(rank, "value", 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, **kwargs: output.__setitem__(0, value)
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    return SimpleNamespace(module=module, args=args, path=Path(args.save_hf), chunks=chunks, rank=rank)


def test_direct_export_buffers_an_immutable_snapshot(exporter, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    save = safetensors.torch.save_file

    def blocked_save(tensors, path):
        started.set()
        assert release.wait(5)
        save(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", blocked_save)
    export = exporter.module._start_direct_export(
        exporter.args,
        [],
        exporter.path,
        model_name="model",
        quantization_config=None,
        megatron_local_weights={},
    )
    try:
        assert started.wait(5)
        exporter.chunks[0][0][1].zero_()
    finally:
        release.set()
        export.finish()

    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    tensor = safetensors.torch.load_file(exporter.path / index["weight_map"]["first"])["first"]
    assert torch.equal(tensor, torch.arange(4))


def test_parallel_writers_use_one_tensor_parallel_replica(exporter, monkeypatch):
    world_size = 4
    rendezvous = threading.Barrier(world_size, timeout=10)
    values = [None] * world_size
    expected = {f"weight_{index}": torch.arange(8, dtype=torch.float32) + index for index in range(4)}

    def gather(output, value, **kwargs):
        values[exporter.rank.value] = value
        rendezvous.wait()
        output[:] = values
        rendezvous.wait()

    def chunks():
        for name, tensor in expected.items():
            yield [(name, tensor + (100 if exporter.rank.value % 2 else 0))]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        exporter.module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            effective_dp_cp=SimpleNamespace(rank=0), tp=SimpleNamespace(rank=exporter.rank.value % 2)
        ),
    )
    monkeypatch.setattr(
        exporter.module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(iter_hf_weights=lambda weights: chunks()),
    )

    def run(rank):
        exporter.rank.value = rank
        exporter.module.export_hf_model_direct(
            exporter.args,
            [],
            exporter.path,
            model_name="model",
            quantization_config=None,
            megatron_local_weights={},
        )

    with ThreadPoolExecutor(world_size) as pool:
        futures = [pool.submit(run, rank) for rank in range(world_size)]
        for future in futures:
            future.result(timeout=10)

    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    for name, tensor in expected.items():
        saved = safetensors.torch.load_file(exporter.path / index["weight_map"][name])[name]
        assert torch.equal(saved, tensor), name


def test_duplicate_tensor_drains_the_collective_iterator_before_failing(exporter, monkeypatch):
    visited = []

    def chunks():
        for index in range(3):
            visited.append(index)
            yield [("duplicate", torch.ones(4))]

    monkeypatch.setattr(
        exporter.module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(iter_hf_weights=lambda weights: chunks()),
    )
    with pytest.raises(RuntimeError, match="duplicate HF tensor"):
        exporter.module.export_hf_model_direct(
            exporter.args,
            [],
            exporter.path,
            model_name="model",
            quantization_config=None,
            megatron_local_weights={},
        )

    assert visited == [0, 1, 2]
