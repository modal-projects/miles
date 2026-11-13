import json

import tools.convert_hf_to_raw_export as converter
import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERTS, INTERMEDIATE, HIDDEN = 3, 2, 4
LAYER = "model.language_model.layers.0."


def _release(path):
    torch.manual_seed(0)
    gate_up = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN).bfloat16()
    down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE).bfloat16()
    first = {
        f"{LAYER}mlp.experts.gate_up_proj": gate_up,
        f"{LAYER}input_layernorm.weight": torch.randn(HIDDEN).bfloat16(),
    }
    second = {f"{LAYER}mlp.experts.down_proj": down, f"{LAYER}linear_attn.A_log": torch.randn(HIDDEN).bfloat16()}
    path.mkdir()
    save_file(first, path / "model-00001-of-00002.safetensors")
    save_file(second, path / "model-00002-of-00002.safetensors")
    weight_map = {**dict.fromkeys(first, "model-00001-of-00002.safetensors")}
    weight_map |= dict.fromkeys(second, "model-00002-of-00002.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text("{}")
    (path / ".cache").mkdir()
    return {**first, **second}


def _written(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    tensors = {}
    for file_name in set(index["weight_map"].values()):
        with safe_open(path / file_name, framework="pt") as reader:
            tensors |= {name: reader.get_tensor(name) for name in reader.keys()}
    return index, tensors


def test_experts_split_per_expert_and_fp32_tensors_upcast(tmp_path, monkeypatch):
    release = _release(tmp_path / "release")
    # one shard per few tensors, so the index spans several files
    monkeypatch.setattr(converter, "SHARD_BYTES", 40)

    converter.main(str(tmp_path / "release"), str(tmp_path / "export"), ("linear_attn.A_log",))

    index, tensors = _written(tmp_path / "export")
    gate_up, down = release[f"{LAYER}mlp.experts.gate_up_proj"], release[f"{LAYER}mlp.experts.down_proj"]
    expected = {f"{LAYER}input_layernorm.weight": release[f"{LAYER}input_layernorm.weight"]}
    expected[f"{LAYER}linear_attn.A_log"] = release[f"{LAYER}linear_attn.A_log"].float()
    for expert in range(EXPERTS):
        expected[f"{LAYER}mlp.experts.{expert}.gate_proj.weight"] = gate_up[expert, :INTERMEDIATE]
        expected[f"{LAYER}mlp.experts.{expert}.up_proj.weight"] = gate_up[expert, INTERMEDIATE:]
        expected[f"{LAYER}mlp.experts.{expert}.down_proj.weight"] = down[expert]
    assert set(tensors) == set(expected) == set(index["weight_map"])
    for name, tensor in expected.items():
        assert tensors[name].dtype == tensor.dtype, name
        assert torch.equal(tensors[name], tensor), name
    assert len(set(index["weight_map"].values())) > 1
    assert index["metadata"]["total_size"] == sum(tensor.nbytes for tensor in expected.values())
    assert sorted(path.name for path in (tmp_path / "export").iterdir() if not path.suffix == ".safetensors") == [
        "config.json",
        "model.safetensors.index.json",
    ]
