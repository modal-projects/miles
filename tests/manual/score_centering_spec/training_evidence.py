"""Audit the saved artifacts of one short score-centering training arm.

This reads trusted Miles torch dumps and structured event JSONL, never console
log patterns. The probe uses CP=1 and one optimizer step per rollout. Three
steps should produce an initial weight sync plus two post-update syncs; the
last optimizer step is not followed by another rollout or published checksum.

Run: python -m tests.manual.score_centering_spec.training_evidence RUN_DIRECTORY
    --output report.json --expected-steps 3 --expected-dp-shards 2 --expected-engines 2
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

_TRAIN_METRICS = (
    "train/loss",
    "train/pg_loss",
    "train/grad_norm",
    "train/sc_correction",
    "train/sc_train_head_mass",
    "train/sc_rollout_head_mass",
    "train/sc_tail_ratio",
    "train/sc_importance_weight",
)


def _stats(values):
    parts = [torch.as_tensor(value).detach().cpu().reshape(-1).double() for value in values]
    tensor = torch.cat(parts) if parts else torch.empty(0)
    finite = bool(torch.isfinite(tensor).all())
    return dict(
        count=tensor.numel(),
        finite=finite,
        nonzero=int(torch.count_nonzero(tensor)),
        minimum=float(tensor.min()) if tensor.numel() and finite else None,
        maximum=float(tensor.max()) if tensor.numel() and finite else None,
    )


def _read_rollout(path, reward_key=None):
    dump = torch.load(path, map_location="cpu", weights_only=False)
    samples = dump["samples"]
    groups = defaultdict(list)
    versions = set()
    missing_versions = 0
    rewards = []
    for sample in samples:
        reward = sample["reward"]
        if isinstance(reward, dict):
            if reward_key is None:
                raise ValueError("Dictionary rewards require --reward-key or a manifest with --reward-key")
            reward = reward[reward_key]
        rewards.append(float(reward))
        groups[sample["group_index"]].append(float(reward))
        sample_versions = {str(span["version"]) for call in sample.get("weight_versions", []) for span in call}
        missing_versions += not sample_versions
        versions.update(sample_versions)
    return dict(
        artifact=str(path),
        rollout_id=dump["rollout_id"],
        sample_indices=[sample["index"] for sample in samples],
        samples=len(samples),
        rewards=_stats(rewards),
        groups=len(groups),
        missing_group_indices=None in groups,
        mixed_reward_groups=sum(len(set(rewards)) > 1 for rewards in groups.values()),
        group_rewards={
            str(group): [value if math.isfinite(value) else None for value in rewards]
            for group, rewards in groups.items()
        },
        weight_versions=sorted(versions),
        samples_missing_versions=missing_versions,
        response_tokens=sum(sample["response_length"] for sample in samples),
        speculative_verifications=sum(sample.get("spec_info", {}).get("spec_verify_ct", 0) for sample in samples),
    )


def _read_train_shard(path):
    dump = torch.load(path, map_location="cpu", weights_only=False)
    if dump["cp_size"] != 1:
        raise ValueError(f"This probe analyzer requires CP=1: {path}")
    data = dump["rollout_data"]
    active_advantages = []
    for advantages, mask in zip(data["advantages"], data["loss_masks"], strict=True):
        advantages, mask = torch.as_tensor(advantages).flatten(), torch.as_tensor(mask).flatten().bool()
        if advantages.shape != mask.shape:
            raise ValueError(f"Advantage/mask shape mismatch in {path}")
        active_advantages.append(advantages[mask])
    return dict(
        artifact=str(path),
        rollout_id=dump["rollout_id"],
        rank=dump["rank"],
        sample_indices=[int(index) for index in data["sample_indices"]],
        active_advantages=_stats(active_advantages),
    )


def _read_events(root):
    wanted = {"metric", "train_group_step_end", "inference_engine_weight_checksum"}
    events = []
    for path in sorted((root / "events").rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                event = json.loads(line)
                if event["type"] in wanted:
                    events.append({**event, "artifact": f"{path}:{line_number}"})
    return sorted(events, key=lambda event: event["timestamp"])


def _training_metrics(events):
    result = []
    for event in events:
        metrics = event.get("metrics", {})
        if event["type"] != "metric" or "train/grad_norm" not in metrics:
            continue
        selected = {key: metrics.get(key) for key in _TRAIN_METRICS}
        finite = all(value is not None and math.isfinite(value) for value in selected.values())
        result.append(
            dict(
                artifact=event["artifact"],
                rollout_id=event.get("rollout_id"),
                step=metrics.get("train/step"),
                finite_score_centering_metrics=finite,
                positive_grad_norm=finite and selected["train/grad_norm"] > 0,
                metrics={
                    key: value if value is not None and math.isfinite(value) else None
                    for key, value in selected.items()
                },
            )
        )
    return result


def _weight_syncs(events):
    result, previous = [], None
    for event in events:
        if event["type"] != "inference_engine_weight_checksum":
            continue
        # The probe's check-weight-update-selector=target normally excludes
        # draft tensors entirely. Absence is not evidence of unchanged drafts.
        # Its text-only trainer also skips visual tensors; offload-related
        # changes in that frozen encoder cannot prove an optimizer update.
        targets = [
            {
                name: value
                for name, value in engine.items()
                if not name.split("/", 1)[-1].startswith(("draft.", "visual."))
            }
            for engine in event["engine_checksums"]
        ]
        same_layout = previous is not None and len(previous) == len(targets)
        changes = []
        for index, target in enumerate(targets):
            before = previous[index] if same_layout else {}
            changes.append(
                sum(before[name] != value for name, value in target.items())
                if before.keys() == target.keys()
                else None
            )
        result.append(
            dict(
                artifact=event["artifact"],
                rollout_id=event["rollout_id"],
                engine_count=len(targets),
                language_target_tensor_counts=[len(target) for target in targets],
                language_targets_equal_across_engines=bool(targets)
                and all(target == targets[0] for target in targets[1:]),
                language_target_changes_per_engine=changes if previous is not None else None,
                draft_tensor_counts=[
                    sum(name.split("/", 1)[-1].startswith("draft.") for name in engine)
                    for engine in event["engine_checksums"]
                ],
                visual_tensor_counts=[
                    sum(name.split("/", 1)[-1].startswith("visual.") for name in engine)
                    for engine in event["engine_checksums"]
                ],
            )
        )
        previous = targets
    return result


def audit_training(root: Path, *, expected_steps=3, expected_dp_shards=2, expected_engines=2, reward_key=None):
    """Return evidence and fail-closed checks for one fresh, unnamespaced arm."""
    if expected_steps < 2:
        raise ValueError("At least two steps are needed to audit a post-update rollout")
    if reward_key is None and (manifest_path := root / "manifest.json").exists():
        argv = json.loads(manifest_path.read_text()).get("train_argv", [])
        for index, value in enumerate(argv):
            if value == "--reward-key":
                reward_key = argv[index + 1]
    rollouts = sorted(
        (_read_rollout(path, reward_key) for path in (root / "rollouts").glob("*.pt")),
        key=lambda item: item["rollout_id"],
    )
    shards = [_read_train_shard(path) for path in sorted((root / "train").glob("*.pt"))]
    events = _read_events(root)
    metrics = _training_metrics(events)
    syncs = _weight_syncs(events)
    ends = [event for event in events if event["type"] == "train_group_step_end"]
    expected_ids = list(range(expected_steps))
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(sorted(item["rollout_id"] for item in rollouts) == expected_ids, "Missing or duplicate rollout dumps")
    require(
        len(metrics) == expected_steps
        and sorted(item["step"] for item in metrics if item["step"] is not None) == expected_ids,
        "Missing or duplicate actual training metrics",
    )
    require(
        sorted({shard["rollout_id"] for shard in shards}) == expected_ids, "Missing or unexpected train dump rollouts"
    )
    require(
        sorted(event["rollout_id"] for event in ends) == expected_ids,
        "Missing or duplicate completed training outcomes",
    )
    for event in ends:
        outcomes = event["cell_outcomes"].values()
        require(
            bool(event["cell_outcomes"])
            and all(
                isinstance(values, list) and values and all(value == "normal" for value in values)
                for values in outcomes
            ),
            f"Training step {event['rollout_id']} did not finish normally",
        )
    for item in metrics:
        require(
            item["finite_score_centering_metrics"] and item["positive_grad_norm"],
            f"Step {item['step']} lacks finite score-centering metrics and a positive gradient norm",
        )
        require(item["rollout_id"] == item["step"], f"Step {item['step']} has an unexpected rollout association")

    versions = []
    for rollout in rollouts:
        rid = rollout["rollout_id"]
        require(
            rollout["samples"] > 0
            and rollout["rewards"]["finite"]
            and not rollout["missing_group_indices"]
            and rollout["mixed_reward_groups"] > 0,
            f"Rollout {rid} lacks a finite nonzero within-group reward signal",
        )
        selected = [shard for shard in shards if shard["rollout_id"] == rid]
        require(
            len(selected) == expected_dp_shards and len({shard["rank"] for shard in selected}) == expected_dp_shards,
            f"Rollout {rid} lacks all expected DP train dumps",
        )
        require(
            Counter(index for shard in selected for index in shard["sample_indices"])
            == Counter(rollout["sample_indices"]),
            f"Rollout {rid} train dumps do not cover each sample once",
        )
        require(
            all(shard["active_advantages"]["finite"] for shard in selected)
            and sum(shard["active_advantages"]["nonzero"] for shard in selected) > 0,
            f"Rollout {rid} lacks finite nonzero active-token advantages",
        )
        recorded = rollout["weight_versions"]
        valid_version = len(recorded) == 1 and recorded[0].isdigit() and rollout["samples_missing_versions"] == 0
        require(valid_version, f"Rollout {rid} lacks a single recorded engine weight version")
        versions.append(int(recorded[0]) if valid_version else None)
    require(
        all(a is not None and b is not None and b > a for a, b in zip(versions, versions[1:], strict=False)),
        "Rollouts did not observe successive updated weight versions",
    )

    require(
        [sync["rollout_id"] for sync in syncs] == [-1, *range(expected_steps - 1)],
        "Expected initial sync and exactly one sync after each non-final step",
    )
    for index, sync in enumerate(syncs):
        require(
            sync["engine_count"] == expected_engines
            and all(sync["language_target_tensor_counts"])
            and sync["language_targets_equal_across_engines"],
            f"Sync {sync['rollout_id']} lacks matching language-target checksums on every expected engine",
        )
        if index:
            require(
                all(count is not None and count > 0 for count in sync["language_target_changes_per_engine"]),
                f"Sync {sync['rollout_id']} does not demonstrate changed language-target weights",
            )
    return dict(
        passed=not failures,
        failures=failures,
        root=str(root.resolve()),
        expected_training_steps=expected_steps,
        expected_post_update_syncs=expected_steps - 1,
        reward_key=reward_key,
        rollouts=rollouts,
        train_shards=shards,
        training_metrics=metrics,
        weight_syncs=syncs,
        completed_outcomes=[
            dict(rollout_id=event["rollout_id"], cell_outcomes=event["cell_outcomes"], artifact=event["artifact"])
            for event in ends
        ],
        interpretation="Short integration evidence only; no statistical or long-run parity claim. Final-step weight changes are not published or checksum-verified by this recipe.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=3)
    parser.add_argument("--expected-dp-shards", type=int, default=2)
    parser.add_argument("--expected-engines", type=int, default=2)
    parser.add_argument("--reward-key", default=None, help="Defaults to the run manifest's --reward-key")
    args = parser.parse_args()
    report = audit_training(
        args.root,
        expected_steps=args.expected_steps,
        expected_dp_shards=args.expected_dp_shards,
        expected_engines=args.expected_engines,
        reward_key=args.reward_key,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("passed", "failures", "expected_training_steps", "expected_post_update_syncs")
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
