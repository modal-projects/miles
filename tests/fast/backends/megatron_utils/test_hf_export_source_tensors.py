import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors.torch
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


@pytest.fixture
def exporter(monkeypatch):
    class WeightUpdatePlacement:
        def __init__(self, *, gather_pp):
            self.gather_pp = gather_pp

    stubs = {
        "megatron.core.distributed": {"DistributedDataParallel": object},
        "miles.backends.megatron_utils.lora.utils": {
            "is_lora_model": lambda model: False,
            "save_lora_checkpoint": lambda *args: None,
        },
        "miles.backends.megatron_utils.named_weights": {"named_params_and_buffers": lambda *args, **kwargs: []},
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct": {"HfWeightIteratorDirect": object},
        "miles.backends.training_utils.parallel": {"get_parallel_state": lambda: None},
        "miles.backends.training_utils.weight_update.hf_weight_iterator": {
            "WeightUpdatePlacement": WeightUpdatePlacement,
        },
        "miles.utils.hf_config": {
            "HF_EXPORT_COMPLETE_MARKER": ".complete",
            "load_hf_config": lambda path: None,
        },
        "miles.utils.megatron_bridge_utils": {"patch_megatron_model": lambda model: nullcontext()},
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/hf_export.py"
    spec = importlib.util.spec_from_file_location("hf_export_source_tensors_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_indexed_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, path / shard_name)
    index = {
        "metadata": {},
        "weight_map": {name: shard_name for name in tensors},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_copy_source_tensors_fills_only_missing_weights(exporter, tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_indexed_checkpoint(
        source,
        {
            "model.visual.proj.weight": torch.arange(4, dtype=torch.float32),
            "model.visual.norm.weight": torch.ones(2),
            "model.language.weight": torch.zeros(3),
        },
    )
    weight_map = {
        # A live trainer tensor remains authoritative even when it matches the prefix.
        "model.visual.proj.weight": "model-00001.safetensors",
        "model.language.weight": "model-00001.safetensors",
    }

    copied_size = exporter._copy_source_tensors(source, output, weight_map, ["model.visual."])

    assert copied_size == 8
    assert weight_map["model.visual.proj.weight"] == "model-00001.safetensors"
    source_shard = weight_map["model.visual.norm.weight"]
    assert torch.equal(
        safetensors.torch.load_file(output / source_shard)["model.visual.norm.weight"],
        torch.ones(2),
    )
    assert "model.language.weight" not in safetensors.torch.load_file(output / source_shard)


def test_copy_source_tensors_supports_single_file_checkpoint(exporter, tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    safetensors.torch.save_file(
        {"model.visual.weight": torch.arange(3)},
        source / "model.safetensors",
    )
    weight_map = {}

    exporter._copy_source_tensors(source, output, weight_map, ["model.visual."])

    assert set(weight_map) == {"model.visual.weight"}


def test_copy_source_tensors_rejects_unmatched_prefix(exporter, tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_indexed_checkpoint(source, {"model.language.weight": torch.ones(1)})

    with pytest.raises(ValueError, match="no matching weights"):
        exporter._copy_source_tensors(source, output, {}, ["model.visual."])


def test_direct_export_commits_source_tensors_but_not_source_marker(exporter, monkeypatch, tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_indexed_checkpoint(source, {"model.visual.weight": torch.ones(2)})
    (source / ".complete").touch()
    (source / "config.json").write_text("{}")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        exporter,
        "HfWeightIteratorDirect",
        lambda *args, **kwargs: SimpleNamespace(
            iter_hf_weights=lambda weights: iter([[("model.language.weight", torch.zeros(3))]])
        ),
    )
    args = SimpleNamespace(
        hf_checkpoint=str(source),
        hf_export_source_tensor_prefixes=["model.visual."],
    )

    exporter.export_hf_model_direct(
        args,
        [],
        output,
        model_name="model",
        quantization_config=None,
        megatron_local_weights={},
    )

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"model.language.weight", "model.visual.weight"}
    assert index["metadata"]["total_size"] == 20
    assert (output / "config.json").exists()
    assert not (output / ".complete").exists()


def test_source_completion_marker_is_not_metadata(exporter, tmp_path):
    marker = tmp_path / ".complete"
    marker.touch()

    assert not exporter._is_hf_metadata_file(marker)
