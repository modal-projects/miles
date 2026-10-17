from pathlib import Path

import pytest

from miles.utils.disk_delta import prepare_delta_directory


def _write(root: Path, version: int, content: bytes = b"empty-index") -> Path:
    directory = root / f"weight_v{version:06d}"
    directory.mkdir(parents=True)
    index = directory / "model.safetensors.index.json"
    index.write_bytes(content)
    return index


def test_resume_retains_replay_prefix_and_prunes_incomplete_suffix(tmp_path):
    prefix = [_write(tmp_path, v) for v in range(1, 4)]
    _write(tmp_path, 4, b"abandoned")
    (tmp_path / "weight_v000005").mkdir()
    (tmp_path / "unrelated").write_text("keep")
    for _ in range(2):
        prepare_delta_directory(str(tmp_path), base_version=3)
        assert all(p.read_bytes() == b"empty-index" for p in prefix)
        assert not (tmp_path / "weight_v000004").exists()
        assert not (tmp_path / "weight_v000005").exists()
        assert (tmp_path / "unrelated").read_text() == "keep"
        _write(tmp_path, 4, b"failed-retry")


def test_fresh_run_clears_previous_stream(tmp_path):
    _write(tmp_path, 1)
    prepare_delta_directory(str(tmp_path), base_version=0)
    assert list(tmp_path.iterdir()) == []


def test_negative_version_does_not_delete_history(tmp_path):
    index = _write(tmp_path, 1)
    with pytest.raises(ValueError):
        prepare_delta_directory(str(tmp_path), base_version=-1)
    assert index.exists()
