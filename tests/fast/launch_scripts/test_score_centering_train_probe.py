"""Keep the paired training experiment controlled and independent of live GPUs."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.manual.score_centering_spec import train_probe


def _without_speculation(argv):
    result = []
    index = 0
    while index < len(argv):
        if argv[index].startswith("--sglang-speculative-"):
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


@pytest.mark.parametrize("sampling", ["unfiltered", "filtered"])
def test_pair_differs_only_in_speculation_and_artifact_paths(sampling):
    regular = train_probe.ScriptArgs(sampling=sampling, arm="regular", output_dir="/artifacts with spaces")
    speculative = replace(regular, arm="dflash")
    first, second = train_probe._manifest(regular), train_probe._manifest(speculative)
    regular_argv = [value.replace(str(regular.artifact_dir), "<output>") for value in first["train_argv"]]
    spec_argv = [value.replace(str(speculative.artifact_dir), "<output>") for value in second["train_argv"]]
    assert regular_argv == _without_speculation(spec_argv)
    assert first["model_argv"] == second["model_argv"]
    assert "--use-kl-loss" not in regular_argv
    assert "--ci-disable-weight-update-checker" not in regular_argv
    assert regular_argv[regular_argv.index("--check-weight-update-selector") + 1] == "target"
    assert regular_argv[regular_argv.index("--loss-type") + 1] == "score_centering"
    assert regular_argv[regular_argv.index("--global-batch-size") + 1] == "32"
    assert regular_argv[regular_argv.index("--num-steps-per-rollout") + 1] == "1"
    assert regular_argv[regular_argv.index("--update-weights-interval") + 1] == "1"
    assert regular_argv[regular_argv.index("--sglang-attention-backend") + 1] == "fa3"
    assert regular_argv[regular_argv.index("--sglang-mamba-ssm-dtype") + 1] == "float32"
    assert regular_argv[regular_argv.index("--sglang-sampling-mask-max-tokens") + 1] == "128"
    assert regular_argv[regular_argv.index("--sglang-max-running-requests") + 1] == "16"


def test_training_uses_standard_launcher_and_refuses_to_resume_an_old_arm(tmp_path, monkeypatch):
    args = train_probe.ScriptArgs(
        model_dir=str(tmp_path / "models"),
        checkpoint_dir=str(tmp_path / "converted"),
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        arm="dflash",
    )
    args.hf_checkpoint.mkdir(parents=True)
    args.draft_checkpoint.mkdir(parents=True)
    args.converted_checkpoint.mkdir(parents=True)
    (args.converted_checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
    args.dataset.parent.mkdir(parents=True)
    args.dataset.write_text("{}\n")
    backend = Mock()
    monkeypatch.setattr(train_probe.ScriptArgs, "create_backend", lambda _: backend)

    train_probe._execute(args)

    backend.execute_train.assert_called_once()
    assert backend.execute_train.call_args.kwargs["job_lifetime"] == "launcher"
    assert backend.execute_train.call_args.kwargs["megatron_model_type"] == "qwen3.8-27B"
    assert (args.artifact_dir / "manifest.json").is_file()
    with pytest.raises(FileExistsError):
        train_probe._execute(args)
    backend.execute_train.assert_called_once()


def test_preparation_uses_the_same_model_definition_and_conversion_utility(monkeypatch):
    args = train_probe.ScriptArgs(model_dir="/models", checkpoint_dir="/converted")
    backend = Mock()
    monkeypatch.setattr(train_probe.ScriptArgs, "create_backend", lambda _: backend)
    train_probe._prepare(args)
    backend.convert_checkpoint.assert_called_once_with(
        model_name="Qwen3.8-27B",
        megatron_model_type="qwen3.8-27B",
        num_gpus_per_node=8,
        hf_checkpoint="/models/Qwen3.8-27B",
        dir_dst="/converted",
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1",
    )
