import json

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


def _module():
    return import_launch_script(REPO_ROOT / "scripts/run_beam_lora.py")


def _args(**overrides):
    module = _module()
    tokens = module._train_args(module.ScriptArgs(run_id="test", **overrides))
    flags = {token for token in tokens if token.startswith("--")}
    values = {
        tokens[i][2:]: tokens[i + 1]
        for i in range(len(tokens) - 1)
        if tokens[i].startswith("--") and not tokens[i + 1].startswith("--")
    }
    return flags, values


def test_lora_recipe_shards_experts_by_ep_only_and_drops_the_disk_optimizer():
    flags, values = _args()
    assert (values["lora-rank"], values["lora-alpha"]) == ("16", "32")
    # Expert LoRA is wrong under expert-TP > 1 in Megatron-Bridge.
    assert values["expert-tensor-parallel-size"] == "1"
    assert int(values["actor-num-nodes"]) * int(values["actor-num-gpus-per-node"]) == int(
        values["expert-model-parallel-size"]
    )
    assert not {"--stream-optimizer-state-to-disk", "--offload-train-disk-dir"} & flags
    assert values["megatron-to-hf-mode"] == "bridge"
    # One TP16 engine serves the whole rollout batch.
    assert values["rollout-num-gpus"] == values["rollout-num-gpus-per-engine"] == "16"
    assert values["sglang-max-running-requests"] == "256"


def test_lora_lr_warms_up_then_decays_and_routing_replay_is_on():
    flags, values = _args()
    assert (values["lr"], values["min-lr"], values["lr-decay-style"]) == ("0.0001", "1e-05", "cosine")
    assert (values["lr-warmup-iters"], values["lr-decay-iters"]) == ("10", "400")
    assert {"--use-rollout-routing-replay", "--use-miles-router"} <= flags
    flags, _ = _args(use_rollout_routing_replay=False)
    assert "--use-rollout-routing-replay" not in flags


def test_eval_prompts_carry_the_training_answer_format(tmp_path):
    module = _module()
    (tmp_path / "aime-2025").mkdir()
    problem = {"prompt": [{"role": "user", "content": "Find x."}], "label": "7"}
    (tmp_path / "aime-2025/aime-2025.jsonl").write_text(json.dumps(problem) + "\n")
    module._write_answer_format_eval(tmp_path)
    (row,) = [json.loads(line) for line in (tmp_path / module.EVAL_FILE).read_text().splitlines()]
    assert row["label"] == "7"
    assert row["prompt"][-1]["content"] == module.ANSWER_FORMAT_PREFIX + "Find x." + module.ANSWER_FORMAT_SUFFIX
    assert "\\boxed{$Answer}" in module.ANSWER_FORMAT_PREFIX
    tokens = module._train_args(module.ScriptArgs(run_id="test", data_dir=str(tmp_path)))
    i = tokens.index("--eval-prompt-data")
    assert tokens[i + 1 : i + 3] == ["aime25", f"{tmp_path}/{module.EVAL_FILE}"]


def test_prepare_downloads_the_checkpoint_and_both_datasets(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = _module()
    # The downloads are only recorded, so AIME25 must already be there for the eval rewrite.
    (tmp_path / "aime-2025").mkdir()
    problem = {"prompt": [{"role": "user", "content": "Find x."}], "label": "7"}
    (tmp_path / "aime-2025/aime-2025.jsonl").write_text(json.dumps(problem) + "\n")
    overrides = {"model_dir": str(tmp_path / "Beam"), "data_dir": str(tmp_path)}
    call_entrypoint(module, "prepare", overrides, sandbox=tmp_path)
    assert recording.commands == [
        f"hf download reflection/Beam-1.0-501B-A23B --local-dir {tmp_path}/Beam",
        f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {tmp_path}/dapo-math-17k",
        f"hf download --repo-type dataset zhuzilin/aime-2025 --local-dir {tmp_path}/aime-2025",
    ]
    assert (tmp_path / module.EVAL_FILE).is_file()


def test_lora_resume_loads_the_latest_adapter(tmp_path):
    # LoRA saves iter_*/adapter without Megatron's latest_checkpointed_iteration.txt.
    for it in (19, 39, 100):
        (tmp_path / f"iter_{it:07d}" / "adapter").mkdir(parents=True)
    _, values = _args(resume=True, output_dir=str(tmp_path))
    assert values["lora-adapter-path"] == str(tmp_path / "iter_0000100" / "adapter")
    assert values["load"] == values["hf-checkpoint"]


def test_lora_resume_requires_an_adapter_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="adapter checkpoint"):
        _args(resume=True, output_dir=str(tmp_path))
