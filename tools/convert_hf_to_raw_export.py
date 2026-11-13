"""
Write an HF checkpoint as Miles' raw exporter (--megatron-to-hf-mode raw) names and stores its tensors in a BF16
run: each fused routed-expert tensor split into the per-expert tensors the exporter yields, and the tensors the
trainer keeps in FP32 upcast (exact from BF16). Weight-delta sync starts the engines from --hf-checkpoint and checks
at launch that it holds every exported tensor under its name, dtype and bytes; this writes that checkpoint from the
release. Every other tensor and file is copied as is.

Fused experts are stored as transformers stores them for Qwen3.5/3.6 MoE, GLM-5.3-Flash and Muse Spark:
`<block>.mlp.experts.gate_up_proj` [experts, 2 * intermediate, in] with the gate rows first, and
`<block>.mlp.experts.down_proj` [experts, out, intermediate]. A family stored otherwise fails the launch check.

python tools/convert_hf_to_raw_export.py --model-dir <src> --save-dir <dst> [--fp32-suffixes linear_attn.A_log ...]
"""

import json
import re
import shutil
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

INDEX = "model.safetensors.index.json"
SHARD_BYTES = 5 * 1024**3
FUSED_EXPERTS = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")


def exported_tensors(name: str, tensor: torch.Tensor, fp32_suffixes: tuple[str, ...]) -> Iterator[tuple]:
    """The exporter's tensors for one checkpoint tensor."""
    if FUSED_EXPERTS.search(name) is None:
        yield name, tensor.float() if name.endswith(fp32_suffixes) else tensor
        return
    experts = name.rsplit(".", 1)[0]
    if name.endswith("down_proj"):
        for index, down in enumerate(tensor):
            yield f"{experts}.{index}.down_proj.weight", down
        return
    gate, up = tensor.chunk(2, dim=-2)
    for index in range(tensor.shape[0]):
        yield f"{experts}.{index}.gate_proj.weight", gate[index]
        yield f"{experts}.{index}.up_proj.weight", up[index]


def main(model_dir: str, save_dir: str, fp32_suffixes: tuple[str, ...] = ()) -> None:
    src, dst = Path(model_dir), Path(save_dir)
    dst.mkdir(parents=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != INDEX:
            shutil.copy(path, dst / path.name)
    weight_map, shard, shard_bytes, total_bytes = {}, {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes
        file_name = f"model-{len(set(weight_map.values())) + 1:05d}.safetensors"
        save_file(shard, dst / file_name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, file_name))
        shard, shard_bytes = {}, 0

    for path in tqdm(sorted(src.glob("*.safetensors")), desc="shards"):
        with safe_open(path, framework="pt") as reader:
            for checkpoint_name in reader.keys():
                for name, tensor in exported_tensors(
                    checkpoint_name, reader.get_tensor(checkpoint_name), fp32_suffixes
                ):
                    shard[name] = tensor.contiguous()
                    shard_bytes += tensor.nbytes
                    total_bytes += tensor.nbytes
                    if shard_bytes >= SHARD_BYTES:
                        flush()
    if shard:
        flush()
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (dst / INDEX).write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--fp32-suffixes", nargs="*", default=[], help="name suffixes of the tensors the trainer keeps in FP32"
    )
    args = parser.parse_args()
    main(args.model_dir, args.save_dir, tuple(args.fp32_suffixes))
