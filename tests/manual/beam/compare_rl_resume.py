"""Compare uninterrupted RL with a cold resume from the preceding checkpoint."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def _model_tensors(directory):
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "checkpoint.pt"
        dcp_to_torch_save(directory, target)
        state = torch.load(target, weights_only=False)
    return {
        key: value for key, value in state.items() if isinstance(value, torch.Tensor) and value.is_floating_point()
    }


def _optimizer_files(directory):
    result = {}
    for path in sorted((directory / "nvme_opt_state").rglob("*")):
        if path.suffix == ".bin":
            result[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.suffix == ".json":
            result[str(path.relative_to(directory))] = json.loads(path.read_text())
    assert result, "No NVMe Adam state found"
    return result


def _samples(directory, rollout):
    data = torch.load(directory / "rollouts" / f"{rollout}.pt", weights_only=False)
    samples = sorted(data["samples"], key=lambda sample: sample["index"])
    for sample in samples:
        routes = sample["rollout_routed_experts"]
        assert routes is not None and routes.shape == (len(sample["tokens"]) - 1, 4, 8)
        assert np.logical_and(routes[:, 1:] >= 0, routes[:, 1:] < 16).all()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", required=True, type=Path)
    parser.add_argument("--resumed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before = _model_tensors(args.continuous / "iter_0000000")
    continuous = _model_tensors(args.continuous / "iter_0000001")
    resumed = _model_tensors(args.resumed / "iter_0000001")
    assert before.keys() == continuous.keys() == resumed.keys()
    changed = [
        name
        for name in before
        if not torch.equal(
            before[name].contiguous().view(torch.uint8), continuous[name].contiguous().view(torch.uint8)
        )
    ]
    mismatches = [
        name
        for name in continuous
        if not torch.equal(
            continuous[name].contiguous().view(torch.uint8), resumed[name].contiguous().view(torch.uint8)
        )
    ]
    optimizer_equal = _optimizer_files(args.continuous / "iter_0000001") == _optimizer_files(
        args.resumed / "iter_0000001"
    )
    left, right = _samples(args.continuous, 1), _samples(args.resumed, 1)
    assert len(left) == len(right) > 0
    samples_equal = all(
        all(a[key] == b[key] for key in ("prompt", "tokens", "response_length", "reward", "rollout_log_probs"))
        and np.array_equal(a["rollout_routed_experts"], b["rollout_routed_experts"])
        for a, b in zip(left, right, strict=True)
    )
    metrics = []
    for directory in (args.continuous, args.resumed):
        rows = [json.loads(line) for line in (directory / "events/actor_cell0_rank0.jsonl").read_text().splitlines()]
        metrics.append([row["metrics"] for row in rows if "train/grad_norm" in row.get("metrics", {})])
    report = dict(
        model_tensors=len(continuous),
        updated_model_tensors=len(changed),
        model_mismatches=mismatches,
        nvme_adam_state_equal=optimizer_equal,
        rollout_samples=len(left),
        rollout_tokens_routes_rewards_logprobs_equal=samples_equal,
        continuous_train_metrics=metrics[0],
        resumed_train_metrics=metrics[1],
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert changed and not mismatches and optimizer_equal and samples_equal, report


if __name__ == "__main__":
    main()
