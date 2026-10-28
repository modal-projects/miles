from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=["weight-update"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)


import json
from pathlib import Path

import pyarrow
import pytest
import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.backends.training_utils.weight_update.protocols.weight_delta import protocol as weight_delta
from miles.backends.training_utils.weight_update.protocols.weight_delta.encoder import GAP_NBYTES
from miles.backends.training_utils.weight_update.protocols.weight_delta.protocol import UpdateWeightFromDelta
from miles.backends.training_utils.weight_update.protocols.weight_delta.publication import version_dir
from miles.utils.args.custom_view import ImmutableNamespace

# name -> (dtype, shape, bytes changed per sync)
TENSORS = {
    "model.layers.0.mlp.down_proj.weight": (torch.bfloat16, (1024, 1536), 3000),
    "model.layers.0.mlp.up_proj.weight": (torch.uint8, (1024, 1024), 40),
    "model.norm.weight": (torch.float32, (4096,), 7),
    "model.embed_tokens.weight": (torch.bfloat16, (256, 512), 0),
}


class _Engine:
    def __getattr__(self, name: str):
        async def method(**kwargs):
            return "default" if name == "get_weight_version" else {"success": True}

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


def _initial() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.randint(0, 256, (torch.Size(shape).numel() * dtype.itemsize,), generator=generator)
        .to(torch.uint8)
        .view(dtype)
        .view(shape)
        for name, (dtype, shape, _) in TENSORS.items()
    }


def _flip_bytes(weights: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    changed = {}
    for name, tensor in weights.items():
        data = tensor.clone()
        flat = data.view(-1).view(torch.uint8)
        positions = torch.randperm(flat.numel(), generator=generator)[: TENSORS[name][2]]
        flat[positions] ^= torch.randint(1, 256, (positions.numel(),), generator=generator, dtype=torch.uint8)
        changed[name] = data
    return changed


def _protocol(tmp_path: Path, codec: str, change_metrics: bool) -> UpdateWeightFromDelta:
    protocol = UpdateWeightFromDelta(
        ImmutableNamespace(
            update_weight_disk_dir=str(tmp_path / "delta"),
            custom_update_weight_post_write_path=None,
            hf_checkpoint=str(tmp_path / "base"),
            weight_delta_codec=codec,
            sglang=SglangConfig(models=[], base_args={"weight_version": "default"}),
            ft_components=[],
            update_weight_engine_request_timeout=10.0,
            pause_generation_mode="retract",
            fully_async=False,
            colocate=False,
            weight_delta_change_metrics=change_metrics,
        )
    )
    protocol.connect([_Engine()], None, None, ["cell-0"], None, protocol.required_placement, "all")
    return protocol


def _sync(protocol: UpdateWeightFromDelta, version: int, weights: dict[str, torch.Tensor]) -> None:
    exported = [(name, tensor.cuda()) for name, tensor in weights.items()]
    assert protocol.begin_sync(version, iter_buckets=None)
    protocol.send_bucket(exported[:2])
    protocol.send_bucket(exported[2:])
    protocol.after_base_weights()
    protocol.finalize(version)


def _decode_frame(codec: str, inner: bytes, nbytes: int) -> torch.Tensor:
    if codec == "snappy-zstd":
        data = pyarrow.Codec("snappy").decompress(inner, decompressed_size=nbytes).to_pybytes()
        return torch.frombuffer(bytearray(data), dtype=torch.uint8)
    payload = torch.frombuffer(bytearray(inner), dtype=torch.uint8)
    count = payload.numel() // (GAP_NBYTES + 1)
    planes = payload[: GAP_NBYTES * count].view(GAP_NBYTES, count).to(torch.int64)
    frame = torch.zeros(nbytes, dtype=torch.uint8)
    frame[torch.cumsum(sum(planes[plane] << 8 * plane for plane in range(GAP_NBYTES)), 0)] = payload[
        GAP_NBYTES * count :
    ]
    return frame


def _apply_like_an_engine(
    directory: Path, previous: dict[str, torch.Tensor]
) -> tuple[dict, dict[str, torch.Tensor], list[str]]:
    """The manifest, every changed tensor's new bytes, and which tensors came raw."""
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {f["name"]: f["nbytes"] for f in manifest["files"]} == {
        f["name"]: (directory / f["name"]).stat().st_size for f in manifest["files"]
    }
    applied, raw_names = {}, []
    for file in manifest["files"]:
        with safe_open(directory / file["name"], framework="numpy") as owner:
            column = {key: owner.get_tensor(key) for key in owner.keys()}
        names, payload = bytes(column["tensors.name"]).decode(), bytes(column["payload"])
        name_start = chunk_start = frame_start = 0
        for index, offset in enumerate(column["tensors.offset"]):
            name_end, chunk_end, frame_end = (
                int(column[f"tensors.{ends}"][index]) for ends in ("name_end", "chunk_end", "frame_end")
            )
            name = names[name_start:name_end]
            region = payload[offset : offset + column["tensors.nbytes"][index]]
            data = previous[name].clone().view(-1).view(torch.uint8)
            if column["tensors.raw"][index]:
                raw_names.append(name)
                data.copy_(torch.frombuffer(bytearray(region), dtype=torch.uint8))
            else:
                inner = b"".join(
                    zstandard.ZstdDecompressor().decompress(
                        region[column["chunks.offset"][chunk] :][: column["chunks.nbytes"][chunk]],
                        max_output_size=int(column["chunks.inner_nbytes"][chunk]),
                    )
                    for chunk in range(chunk_start, chunk_end)
                )
                for frame in range(frame_start, frame_end):
                    frame_offset, nbytes, inner_offset, inner_nbytes = (
                        int(column[f"frames.{field}"][frame])
                        for field in ("offset", "nbytes", "inner_offset", "inner_nbytes")
                    )
                    data[frame_offset:][:nbytes] ^= _decode_frame(
                        manifest["codec"], inner[inner_offset:][:inner_nbytes], nbytes
                    )
            applied[name] = data
            name_start, chunk_start, frame_start = name_end, chunk_end, frame_end
    return manifest, applied, raw_names


@pytest.mark.parametrize("change_metrics", [True, False])
@pytest.mark.parametrize("codec", ["gap-zstd", "snappy-zstd"])
def test_each_version_brings_the_previous_bytes_to_exactly_the_export(
    tmp_path: Path, codec: str, change_metrics: bool
):
    weights = _initial()
    (tmp_path / "base").mkdir()
    save_file(weights, tmp_path / "base" / "model.safetensors")
    protocol = _protocol(tmp_path, codec, change_metrics)
    hooked: list = []
    protocol._post_write_hook = lambda args, path, engines: hooked.append(sorted(p.name for p in Path(path).iterdir()))
    _sync(protocol, 0, weights)
    assert hooked == [] and not (tmp_path / "delta").exists()
    ever_changed = {
        name: torch.zeros(tensor.numel() * tensor.element_size(), dtype=torch.bool) for name, tensor in weights.items()
    }

    for version in (1, 2):
        previous, weights = weights, _flip_bytes(weights, seed=version)
        _sync(protocol, version, weights)

        manifest, applied, raw_names = _apply_like_an_engine(version_dir(tmp_path / "delta", version), previous)
        assert (manifest["codec"], manifest["base_version"], manifest["target_version"]) == (
            codec,
            version - 1,
            version,
        )
        assert sorted(applied) == sorted(name for name, (_, _, changed) in TENSORS.items() if changed)
        for name, data in applied.items():
            assert torch.equal(data, weights[name].view(-1).view(torch.uint8)), name
        assert raw_names == ["model.norm.weight"]
        assert hooked[-2:] == [["owner-00000.safetensors"], ["manifest.json", "owner-00000.safetensors"]]
        exported = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
        changed = sum(changed for _, _, changed in TENSORS.values())
        metrics = protocol.pop_metrics()
        assert metrics["perf/update_weights_density"] == pytest.approx(changed / exported)
        rechanged = 0
        for name, before in previous.items():
            flipped = before.view(-1).view(torch.uint8) != weights[name].view(-1).view(torch.uint8)
            rechanged += int((ever_changed[name] & flipped).sum())
            ever_changed[name] |= flipped
        if change_metrics:
            ever = sum(int(mask.sum()) for mask in ever_changed.values())
            assert metrics["perf/update_weights_ever_changed"] == pytest.approx(ever / exported)
            assert metrics["perf/update_weights_rechanged"] == pytest.approx(rechanged / changed)
        else:
            assert "perf/update_weights_ever_changed" not in metrics and "perf/update_weights_rechanged" not in metrics


def test_a_fresh_launch_refuses_an_export_that_differs_from_the_checkpoint(tmp_path: Path):
    weights = _initial()
    (tmp_path / "base").mkdir()
    save_file(_flip_bytes(weights, seed=9), tmp_path / "base" / "model.safetensors")
    protocol = _protocol(tmp_path, "gap-zstd", change_metrics=False)
    with pytest.raises(RuntimeError, match=r"3 exported tensors differ from --hf-checkpoint's values"):
        _sync(protocol, 0, weights)
