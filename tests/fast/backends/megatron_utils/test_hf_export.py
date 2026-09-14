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
    stubs = {
        "megatron.core.distributed": {"DistributedDataParallel": object},
        "miles.backends.megatron_utils.lora_utils": {
            "is_lora_model": lambda model: False,
            "save_lora_checkpoint": lambda *args: None,
        },
        "miles.backends.megatron_utils.update_weight.common": {
            "named_params_and_buffers": lambda *args, **kwargs: [],
        },
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct": {
            "HfWeightIteratorDirect": object,
        },
        "miles.utils.hf_config": {
            "HF_EXPORT_COMPLETE_MARKER": ".complete",
            "load_hf_config": lambda path: SimpleNamespace(),
        },
        "miles.utils.megatron_bridge_utils": {"patch_megatron_model": lambda model: nullcontext()},
        "miles.utils.distributed_utils": {"get_gloo_group": lambda: None},
        "miles.backends.training_utils.parallel": {
            "get_parallel_state": lambda: SimpleNamespace(tp=SimpleNamespace(rank=0)),
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
        update_weight_transfer_mode="disk-delta",
    )
    chunks = [[("first", torch.arange(4))], [("second", torch.ones(4))]]
    monkeypatch.setattr(
        module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(get_hf_weight_chunks=lambda weights: iter(chunks)),
    )
    rank = threading.local()
    rank.value = 0
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: getattr(rank, "value", 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, **kwargs: output.__setitem__(0, value)
    )
    return SimpleNamespace(module=module, args=args, path=Path(args.save_hf), chunks=chunks, rank=rank)


def test_start_buffers_an_immutable_snapshot_and_finish_closes_it(exporter, monkeypatch):
    started, release = threading.Event(), threading.Event()
    save = safetensors.torch.save_file

    def blocked_save(tensors, path):
        started.set()
        assert release.wait(5)
        save(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", blocked_save)
    export = exporter.module.start_hf_export(exporter.args, 7, [])
    try:
        assert started.wait(5)
        assert not (exporter.path / ".complete").exists()
        exporter.chunks[0][0][1].zero_()
    finally:
        release.set()
        export.finish()
    assert (exporter.path / ".complete").is_file()
    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    tensor = safetensors.torch.load_file(exporter.path / index["weight_map"]["first"])["first"]
    assert torch.equal(tensor, torch.arange(4))


@pytest.mark.parametrize("fail", [False, True])
def test_completion_waits_for_all_writers_and_propagates_failure(exporter, monkeypatch, fail):
    rendezvous = threading.Barrier(2, timeout=5)
    values = [None, None]

    def gather(output, value, **kwargs):
        values[exporter.rank.value] = value
        rendezvous.wait()
        output[:] = values
        rendezvous.wait()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    started, release = threading.Event(), threading.Event()
    save = safetensors.torch.save_file

    def blocked_save(tensors, path):
        if "second" in tensors:
            started.set()
            assert release.wait(5)
            if fail:
                raise OSError("disk full on writer 1")
        save(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", blocked_save)

    def run(rank):
        exporter.rank.value = rank
        exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, rank) for rank in (0, 1)]
        try:
            assert started.wait(5)
            assert not (exporter.path / ".complete").exists()
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        for future in futures:
            if fail:
                with pytest.raises(RuntimeError, match="disk full on writer 1"):
                    future.result(timeout=5)
            else:
                future.result(timeout=5)
    assert (exporter.path / ".complete").exists() is not fail


def test_standalone_export_is_complete_on_return(exporter):
    exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"first", "second"}
    assert all((exporter.path / name).is_file() for name in index["weight_map"].values())
    assert (exporter.path / ".complete").is_file()


def test_parallel_writers_preserve_the_checkpoint_tensor_parallel_replica(exporter, monkeypatch):
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
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=exporter.rank.value % 2)),
        raising=False,
    )
    monkeypatch.setattr(
        exporter.module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(get_hf_weight_chunks=lambda weights: chunks()),
    )

    def run(rank):
        exporter.rank.value = rank
        exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)

    with ThreadPoolExecutor(world_size) as pool:
        futures = [pool.submit(run, rank) for rank in range(world_size)]
        for future in futures:
            future.result(timeout=10)
    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    for name, tensor in expected.items():
        saved = safetensors.torch.load_file(exporter.path / index["weight_map"][name])[name]
        assert torch.equal(saved, tensor), name


@pytest.mark.parametrize("lora", [False, True])
def test_bridge_and_lora_keep_their_completion_contract(exporter, monkeypatch, lora):
    exporter.args.megatron_to_hf_mode = "bridge"
    exporter.args.update_weight_transfer_mode = "broadcast"
    monkeypatch.setattr(exporter.module, "is_lora_model", lambda model: lora)
    adapter_saved = []
    monkeypatch.setattr(exporter.module, "save_lora_checkpoint", lambda *args: adapter_saved.append(True))
    monkeypatch.setattr(
        exporter.module,
        "_get_hf_bridge",
        lambda checkpoint: SimpleNamespace(
            save_hf_pretrained=lambda model, path: safetensors.torch.save_file(
                {"weight": torch.ones(4)}, path / "model.safetensors"
            )
        ),
    )
    exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    assert (exporter.path / ".complete").is_file()
    assert bool(adapter_saved) is lora


def test_disk_delta_checkpoint_uses_canonical_export_in_bridge_mode(exporter, monkeypatch):
    exporter.args.megatron_to_hf_mode = "bridge"
    monkeypatch.setattr(
        exporter.module, "_get_hf_bridge", lambda checkpoint: pytest.fail("noncanonical bridge export")
    )
    exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    assert (exporter.path / ".complete").is_file()


def test_fixed_source_buffers_are_preserved_without_overwriting_live_weights(exporter):
    tensors = {
        "first": torch.zeros(4),
        "layer.input_scale": torch.tensor([0.25]),
        "layer.rotary_emb.inv_freq": torch.ones(2),
    }
    base = Path(exporter.args.hf_checkpoint)
    safetensors.torch.save_file(tensors, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    index = json.loads((exporter.path / "model.safetensors.index.json").read_text())
    for name in tensors:
        saved = safetensors.torch.load_file(exporter.path / index["weight_map"][name])[name]
        expected = torch.arange(4) if name == "first" else tensors[name]
        assert torch.equal(saved, expected)


def test_duplicate_tensor_drains_the_collective_iterator_before_failing(exporter, monkeypatch):
    visited = []

    def chunks():
        for index in range(3):
            visited.append(index)
            yield [("duplicate", torch.ones(4))]

    monkeypatch.setattr(
        exporter.module,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(get_hf_weight_chunks=lambda weights: chunks()),
    )
    with pytest.raises(RuntimeError, match="duplicate HF tensor"):
        exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    assert visited == [0, 1, 2]
    assert not (exporter.path / ".complete").exists()


def test_metadata_failure_removes_stale_completion(exporter, monkeypatch):
    exporter.path.mkdir()
    (exporter.path / ".complete").touch()

    def fail(*args):
        raise OSError("metadata write failed")

    monkeypatch.setattr(exporter.module, "_write_metadata", fail)
    with pytest.raises(RuntimeError, match="metadata write failed"):
        exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    assert not (exporter.path / ".complete").exists()


def test_empty_export_cannot_be_completed(exporter):
    exporter.chunks.clear()
    with pytest.raises(RuntimeError, match="produced no weights"):
        exporter.module.save_hf_model(exporter.args, 7, [], raise_on_error=True)
    assert not (exporter.path / ".complete").exists()
