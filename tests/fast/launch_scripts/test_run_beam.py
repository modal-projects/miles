import json

from tests.fast.launch_scripts.py_harness import REPO_ROOT, import_launch_script


def test_eval_prompts_carry_the_training_answer_format(tmp_path):
    module = import_launch_script(REPO_ROOT / "scripts/run_beam.py")
    (tmp_path / "aime-2025").mkdir()
    problem = {"prompt": [{"role": "user", "content": "Find x."}], "label": "7"}
    (tmp_path / "aime-2025/aime-2025.jsonl").write_text(json.dumps(problem) + "\n")
    module._write_answer_format_eval(tmp_path)
    (row,) = [json.loads(line) for line in (tmp_path / module.EVAL_FILE).read_text().splitlines()]
    assert row["label"] == "7"
    assert row["prompt"][-1]["content"] == module.ANSWER_FORMAT_PREFIX + "Find x." + module.ANSWER_FORMAT_SUFFIX
    assert "\\boxed{$Answer}" in module.ANSWER_FORMAT_PREFIX
    for colocate in (False, True):
        tokens = module._train_args(module.ScriptArgs(run_id="test", data_dir=str(tmp_path)), colocate=colocate)
        i = tokens.index("--eval-prompt-data")
        assert tokens[i + 1 : i + 3] == ["aime25", f"{tmp_path}/{module.EVAL_FILE}"]
