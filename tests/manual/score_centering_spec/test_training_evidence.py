"""Reject training reports whose apparent progress lacks useful update evidence."""

import json

import pytest
import torch
from tests.manual.score_centering_spec.training_evidence import audit_training


@pytest.fixture
def saved_run(tmp_path):
    for directory in ("rollouts", "train", "events"):
        (tmp_path / directory).mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"train_argv": ["--reward-key", "score"]}))
    events = []

    def event(kind, **fields):
        events.append(
            dict(type=kind, timestamp=f"2026-10-03T00:00:{len(events):02d}Z", source={"component": "actor"}, **fields)
        )

    def sync(rollout_id):
        engines = [
            {"rank0/model.layers.0.weight": str(rollout_id), "rank0/visual.weight": f"{rollout_id}-{engine}"}
            for engine in range(2)
        ]
        event("inference_engine_weight_checksum", rollout_id=rollout_id, engine_checksums=engines)

    sync(-1)
    for rollout_id in range(3):
        samples = [
            dict(
                index=index,
                group_index=index // 2,
                reward={"score": float(index % 2), "acc": 0},
                response_length=2,
                weight_versions=[[dict(version=str(rollout_id + 1), abs_start=1, abs_end=3)]],
            )
            for index in range(4)
        ]
        torch.save(dict(rollout_id=rollout_id, samples=samples), tmp_path / "rollouts" / f"{rollout_id}.pt")
        for rank, indices in ((0, [0, 1]), (4, [2, 3])):
            data = dict(
                sample_indices=indices,
                advantages=[torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5])],
                loss_masks=[torch.ones(2), torch.ones(2)],
            )
            torch.save(
                dict(rollout_id=rollout_id, rank=rank, cp_size=1, rollout_data=data),
                tmp_path / "train" / f"{rollout_id}_{rank}.pt",
            )
        metrics = {"train/step": rollout_id, "train/loss": 0.0, "train/pg_loss": 0.0, "train/grad_norm": 0.25}
        metrics.update(
            {
                f"train/sc_{key}": 0.5
                for key in ("correction", "train_head_mass", "rollout_head_mass", "tail_ratio", "importance_weight")
            }
        )
        event("metric", rollout_id=rollout_id, metrics=metrics)
        event("train_group_step_end", rollout_id=rollout_id, cell_outcomes={"0": ["normal"]})
        if rollout_id < 2:
            sync(rollout_id)
    (tmp_path / "events" / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
    return tmp_path


def test_three_steps_prove_only_two_post_update_syncs(saved_run):
    report = audit_training(saved_run)
    assert report["passed"], report["failures"]
    assert report["expected_training_steps"] == 3
    assert report["expected_post_update_syncs"] == 2
    assert [sync["rollout_id"] for sync in report["weight_syncs"]] == [-1, 0, 1]
    assert report["weight_syncs"][1]["language_target_changes_per_engine"] == [1, 1]
    assert report["weight_syncs"][1]["draft_tensor_counts"] == [0, 0]
    assert report["reward_key"] == "score"
    json.dumps(report, allow_nan=False)


def test_masked_advantages_do_not_count_as_training_signal(saved_run):
    for path in (saved_run / "train").glob("0_*.pt"):
        dump = torch.load(path, weights_only=False)
        dump["rollout_data"]["advantages"] = [torch.tensor([5.0, 0.0])] * 2
        dump["rollout_data"]["loss_masks"] = [torch.tensor([0, 1])] * 2
        torch.save(dump, path)
    report = audit_training(saved_run)
    assert not report["passed"]
    assert any("nonzero active-token advantages" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    "defect",
    ["nan_gradient", "missing_sc_loss", "failed_step", "unchanged_language", "inconsistent_engine", "missing_sync"],
)
def test_structured_events_do_not_hide_failed_update_evidence(saved_run, defect):
    path = saved_run / "events" / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = next(event for event in events if event["type"] == "metric")
    syncs = [event for event in events if event["type"] == "inference_engine_weight_checksum"]
    if defect == "nan_gradient":
        metrics["metrics"]["train/grad_norm"] = float("nan")
    elif defect == "missing_sc_loss":
        del metrics["metrics"]["train/sc_correction"]
    elif defect == "failed_step":
        next(event for event in events if event["type"] == "train_group_step_end")["cell_outcomes"] = {"0": "error"}
    elif defect == "unchanged_language":
        for sync in syncs:
            for engine in sync["engine_checksums"]:
                engine["rank0/model.layers.0.weight"] = "unchanged"
    elif defect == "inconsistent_engine":
        syncs[1]["engine_checksums"][1]["rank0/model.layers.0.weight"] = "wrong"
    else:
        events.remove(syncs[-1])
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    report = audit_training(saved_run)
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("defect", ["all_equal_rewards", "stale_version", "missing_shard", "duplicate_samples"])
def test_torch_dumps_must_prove_reward_signal_and_post_update_rollouts(saved_run, defect):
    if defect in ("all_equal_rewards", "stale_version"):
        path = saved_run / "rollouts" / "1.pt"
        dump = torch.load(path, weights_only=False)
        for sample in dump["samples"]:
            if defect == "all_equal_rewards":
                sample["reward"]["score"] = -1.0
            else:
                sample["weight_versions"][0][0]["version"] = "1"
        torch.save(dump, path)
    elif defect == "missing_shard":
        (saved_run / "train" / "1_4.pt").unlink()
    else:
        path = saved_run / "train" / "1_4.pt"
        dump = torch.load(path, weights_only=False)
        dump["rollout_data"]["sample_indices"] = [0, 1]
        torch.save(dump, path)
    assert not audit_training(saved_run)["passed"]
