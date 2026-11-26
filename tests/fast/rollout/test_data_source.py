import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.rollout import data_source
from miles.rollout.data_source import (
    DataSource,
    LegacyRolloutDataSourceWithBuffer,
    RolloutDataResumeMismatch,
    RolloutDataSource,
)
from miles.utils.simple_checkpointer import save_simple_checkpoint
from miles.utils.types import Sample


class TestReadOnlyDataSource:
    def test_a_custom_source_only_implements_read_and_checkpoint_operations(self, tmp_path: Path) -> None:
        """A custom source can instantiate without implementing sample recycling."""
        source = _ReadOnlyDataSource()

        assert source.get_samples(num_samples=1)[0][0].prompt == "0"
        source.save(tmp_path)
        source.load(tmp_path)


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(tmp_path)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_load_rejects_a_directory_without_the_data_source_file(tmp_path: Path) -> None:
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True)

    with pytest.raises(AssertionError, match="state.pt"):
        source.load(tmp_path)


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(tmp_path)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    torch.save({"sample_offset": 7, "epoch_id": 1}, tmp_path / "state.pt")
    source = _bare_source(rollout_global_dataset=True)

    source.load(tmp_path)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


class _ReadOnlyDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass


def test_legacy_source_rejects_checkpoint_loading(tmp_path: Path) -> None:
    """Legacy buffered sources explicitly refuse checkpoint loading."""
    source = LegacyRolloutDataSourceWithBuffer.__new__(LegacyRolloutDataSourceWithBuffer)
    with pytest.raises(NotImplementedError, match="does not support checkpoint loading"):
        source.load(tmp_path)


def test_a_mix_checkpoint_starts_the_single_dataset_over_without_the_flag(tmp_path: Path) -> None:
    """A --rollout-data-yaml-config position means nothing to one dataset, so --prompt-data starts it over."""
    save_simple_checkpoint(
        directory=tmp_path,
        data={"offsets": {"math": 3}, "epoch_ids": {"math": 1}, "sample_group_index": 5, "sample_index": 10},
    )
    source = _bare_source(rollout_global_dataset=True)

    with pytest.raises(RolloutDataResumeMismatch, match="--prompt-data dataset starts from its first prompt") as err:
        source.load(tmp_path)
    assert not err.value.needs_flag

    source.sample_group_index = source.sample_index = 0
    source.load_sample_ids(tmp_path)
    assert (source.sample_group_index, source.sample_index) == (5, 10)


class _WordTokenizer:
    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [prompt.split() for prompt in prompts]}


class _LabelSizedDataSource(RolloutDataSource):
    """Deals each prompt as many samples as its label names, as a source choosing each group's size would."""

    def group_size(self, prompt: Sample) -> int:
        return int(prompt.label)


def _dataset_source(tmp_path: Path, monkeypatch, cls: type[RolloutDataSource], sizes: list[int]) -> RolloutDataSource:
    monkeypatch.setattr(data_source, "_load_prompt_tokenizer_and_processor", lambda args: (_WordTokenizer(), None))
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"input": f"p{i}", "label": str(size)}) + "\n" for i, size in enumerate(sizes)))
    args = _make_args(
        rollout_global_dataset=True,
        rollout_shuffle=True,
        rollout_seed=1,
        n_samples_per_prompt=4,
        prompt_data=str(path),
        rollout_max_prompt_len=None,
        input_key="input",
        label_key="label",
        metadata_key="metadata",
        tool_key="tools",
        multimodal_keys=None,
        multimodal_token_space_processing=False,
        shared_media_dir=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
    )
    return cls(args)


def _drawn(groups: list[list[Sample]]) -> list[list[tuple]]:
    return [[(sample.prompt, sample.group_index, sample.index) for sample in group] for group in groups]


class TestGroupSize:
    def test_every_group_holds_n_samples_per_prompt_by_default(self, tmp_path: Path, monkeypatch) -> None:
        """A run whose data source chooses no size keeps the fixed groups every recipe was tuned with."""
        source = _dataset_source(tmp_path, monkeypatch, RolloutDataSource, sizes=[1, 2, 3])

        assert [len(group) for group in source.get_samples(5)] == [4] * 5

    def test_a_source_deals_each_group_the_size_it_chooses(self, tmp_path: Path, monkeypatch) -> None:
        """Each group is one prompt copied as often as the source says, with ids that keep counting across groups."""
        source = _dataset_source(tmp_path, monkeypatch, _LabelSizedDataSource, sizes=[1, 2, 3, 4])

        groups = source.get_samples(4)

        assert sorted(len(group) for group in groups) == [1, 2, 3, 4]
        assert all(len(group) == int(group[0].label) for group in groups)
        assert all(len({sample.prompt for sample in group}) == 1 for group in groups)
        assert [group[0].group_index for group in groups] == [0, 1, 2, 3]
        assert [sample.index for group in groups for sample in group] == list(range(10))

    def test_a_resumed_source_deals_the_same_groups_at_the_same_sizes(self, tmp_path: Path, monkeypatch) -> None:
        """The size comes from the prompt the restored position deals, so a resume repeats the groups it would have."""
        sizes = [1, 2, 3, 4, 2]
        source = _dataset_source(tmp_path, monkeypatch, _LabelSizedDataSource, sizes=sizes)
        source.get_samples(3)
        source.save(tmp_path / "ckpt")
        expected = _drawn(source.get_samples(4))

        resumed = _dataset_source(tmp_path, monkeypatch, _LabelSizedDataSource, sizes=sizes)
        resumed.load(tmp_path / "ckpt")

        assert _drawn(resumed.get_samples(4)) == expected
