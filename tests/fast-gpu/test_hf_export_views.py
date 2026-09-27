from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import hf_export


class _Publisher:
    has_weight_views = True

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def write_model(self, *_args, **_kwargs) -> None:
        self.events.append("write")

    def mark_weight_views_complete(self, _path: Path) -> None:
        self.events.append("mark")


def _args(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        save_hf=str(path),
        megatron_to_hf_mode="raw",
        hf_checkpoint="/base",
        hf_export_source_tensor_prefixes=(),
    )


def _patch_export_dependencies(monkeypatch) -> None:
    parallel_state = SimpleNamespace(
        effective_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(hf_export, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(hf_export, "is_lora_model", lambda _model: False)
    monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *_args, **_kwargs: {})


def test_weight_view_markers_follow_the_checkpoint_commit(monkeypatch, tmp_path) -> None:
    events = []
    publisher = _Publisher(events)
    _patch_export_dependencies(monkeypatch)

    def write_checkpoint_dir(path, write_weights, **_kwargs) -> None:
        write_weights(Path(path))
        events.append("commit")

    monkeypatch.setattr(hf_export, "write_checkpoint_dir", write_checkpoint_dir)

    hf_export.save_hf_model(
        _args(tmp_path / "checkpoint"),
        0,
        [],
        publisher=publisher,
        raise_on_error=True,
    )

    assert events == ["write", "commit", "mark"]


def test_failed_checkpoint_write_does_not_publish_view_markers(
    monkeypatch, tmp_path
) -> None:
    events = []
    publisher = _Publisher(events)
    _patch_export_dependencies(monkeypatch)

    def fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(hf_export, "write_checkpoint_dir", fail_write)

    with pytest.raises(RuntimeError, match="write failed"):
        hf_export.save_hf_model(
            _args(tmp_path / "checkpoint"),
            0,
            [],
            publisher=publisher,
            raise_on_error=True,
        )

    assert events == []
