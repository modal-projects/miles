"""Checkpoint filesystem errors propagate to the trainer cell."""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint_io import publish_checkpoint_dir, write_checkpoint_dir


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    def make_dir(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


@pytest.mark.parametrize("crash_before_publish", [True, False])
def test_crashed_overwrite_keeps_a_complete_checkpoint(tmp_path, crash_before_publish):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("old"))
    old_version = checkpoint.resolve()

    def overwrite_and_crash():
        replace = os.replace

        def crash_at_publish(source, destination):
            if Path(destination) == checkpoint and crash_before_publish:
                os._exit(73)
            replace(source, destination)
            if Path(source) == checkpoint or Path(destination) == checkpoint:
                os._exit(73)

        os.replace = crash_at_publish
        write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("new"))

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert (checkpoint / "value").read_text() == ("old" if crash_before_publish else "new")
    assert (old_version / "value").read_text() == "old"

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"))
    assert (checkpoint / "value").read_text() == "retry"


def test_custom_publish_replaces_rename(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    published = []

    def recorder(tmp_dir, final_dir):
        assert (tmp_dir / "value").read_text() == "new"
        assert json.loads((tmp_dir / "META.json").read_text()) == {"k": 1}
        published.append((tmp_dir, final_dir))

    write_checkpoint_dir(
        checkpoint,
        lambda directory: (directory / "value").write_text("new"),
        metadata={"k": 1},
        publish=recorder,
    )

    assert len(published) == 1
    assert published[0][1] == checkpoint
    assert not checkpoint.exists()


def test_custom_publish_can_use_default(tmp_path):
    checkpoint = tmp_path / "checkpoint"

    write_checkpoint_dir(
        checkpoint,
        lambda directory: (directory / "value").write_text("new"),
        publish=lambda tmp_dir, final_dir: publish_checkpoint_dir(tmp_dir, final_dir),
    )

    assert (checkpoint / "value").read_text() == "new"
    assert checkpoint.is_symlink()


def test_publish_errors_propagate(tmp_path):
    def fail(_tmp_dir, _final_dir):
        raise RuntimeError("sync failed")

    with pytest.raises(RuntimeError, match="sync failed"):
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None, publish=fail)
