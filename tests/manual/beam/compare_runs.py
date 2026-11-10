"""Require byte-identical logits, logprobs and every gradient from two validation runs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def compare(left, right):
    files = sorted(left.glob("rank*.pt"))
    assert files and {p.name for p in files} == {p.name for p in right.glob("rank*.pt")}
    results = []
    for path in files:
        reference = torch.load(path, weights_only=True)
        actual = torch.load(right / path.name, weights_only=True)
        expected = {"logits": reference["logits"], "logprobs": reference["logprobs"], **reference["gradients"]}
        observed = {"logits": actual["logits"], "logprobs": actual["logprobs"], **actual["gradients"]}
        assert expected.keys() == observed.keys(), path.name
        hashes = reference.get("gradient_hashes", {})
        assert reference["gradients"] or hashes, "The comparison requires backward results"
        assert hashes.keys() == actual.get("gradient_hashes", {}).keys()
        mismatches = [name for name, value in hashes.items() if value != actual["gradient_hashes"][name]]
        digest = hashlib.sha256()
        for name, tensor in expected.items():
            other = observed[name]
            if (
                tensor.dtype != other.dtype
                or tensor.shape != other.shape
                or not torch.equal(_bytes(tensor), _bytes(other))
            ):
                mismatches.append(name)
            digest.update(name.encode())
            digest.update(_bytes(tensor).numpy().tobytes())
        results.append(
            {
                "rank": path.stem,
                "tensors": len(expected) + len(hashes),
                "mismatches": mismatches,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "reference": str(left),
        "actual": str(right),
        "ranks": results,
        "bitwise_equal": all(not result["mismatches"] for result in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.actual)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    assert result["bitwise_equal"], "Model outputs or gradients differ"


if __name__ == "__main__":
    main()
