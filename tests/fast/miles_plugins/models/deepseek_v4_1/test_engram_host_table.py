"""DeepSeek-V4.1 Engram host table: read from the checkpoint, or from one node-local copy shared by the node's ranks
(``MILES_DSV41_ENGRAM_TABLE_DIR``) — no GPU."""

import json
import os
import shutil
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from miles_plugins.models.deepseek_v4_1 import engram

ROWS, HEAD_DIM = 64, 64
LAYER = 3


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    weight = torch.randint(0, 256, (ROWS, HEAD_DIM), dtype=torch.uint8)
    scale = torch.randint(0, 256, (ROWS, HEAD_DIM // 32), dtype=torch.uint8)
    save_file({f"layers.{LAYER}.engram.embed.weight": weight}, str(ckpt / "model-00001.safetensors"))
    save_file({f"layers.{LAYER}.engram.embed.scale": scale}, str(ckpt / "model-00002.safetensors"))
    weight_map = {
        f"layers.{LAYER}.engram.embed.weight": "model-00001.safetensors",
        f"layers.{LAYER}.engram.embed.scale": "model-00002.safetensors",
    }
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return ckpt, weight, scale


def _engram(ckpt, row_start=0, rows_avail=ROWS):
    return SimpleNamespace(
        config=SimpleNamespace(v41_hf_checkpoint=str(ckpt)),
        layer_id=LAYER,
        row_start=row_start,
        rows_avail=rows_avail,
        head_dim=HEAD_DIM,
    )


@pytest.mark.parametrize("row_start,rows_avail", [(0, ROWS), (16, 32)])
def test_tables_from_the_local_copy_match_the_checkpoint(checkpoint, tmp_path, monkeypatch, row_start, rows_avail):
    ckpt, weight, scale = checkpoint
    from_checkpoint = _engram(ckpt, row_start, rows_avail)
    engram.DeepSeekV41Engram._load_table(from_checkpoint)

    local_dir = tmp_path / "shm"
    monkeypatch.setattr(engram, "ENGRAM_TABLE_DIR", str(local_dir))
    from_copy = _engram(ckpt, row_start, rows_avail)
    engram.DeepSeekV41Engram._load_table(from_copy)

    rows = slice(row_start, row_start + rows_avail)
    assert torch.equal(from_checkpoint.table, weight[rows]) and torch.equal(from_copy.table, weight[rows])
    assert torch.equal(from_checkpoint.table_scale, scale[rows]) and torch.equal(from_copy.table_scale, scale[rows])
    for name in ("model-00001.safetensors", "model-00002.safetensors"):
        assert (local_dir / str(ckpt / name).lstrip("/")).is_file()


def test_no_table_dir_reads_the_checkpoint(checkpoint, monkeypatch):
    ckpt, _, _ = checkpoint
    monkeypatch.setattr(engram, "ENGRAM_TABLE_DIR", "")
    copies = []
    monkeypatch.setattr(engram, "node_local_copy", lambda path, directory: copies.append(path) or path)
    engram.DeepSeekV41Engram._load_table(_engram(ckpt))
    assert copies == []


def test_the_node_copies_once_and_reuses_the_copy(checkpoint, tmp_path, monkeypatch):
    ckpt, _, _ = checkpoint
    source = str(ckpt / "model-00001.safetensors")
    copied = []
    real_copyfile = shutil.copyfile

    def slow_copyfile(src, dst):
        copied.append(src)
        time.sleep(0.2)
        return real_copyfile(src, dst)

    monkeypatch.setattr(shutil, "copyfile", slow_copyfile)
    local_dir = str(tmp_path / "shm")
    paths = []
    ranks = [threading.Thread(target=lambda: paths.append(engram.node_local_copy(source, local_dir))) for _ in range(8)]
    for rank in ranks:
        rank.start()
    for rank in ranks:
        rank.join()

    assert copied == [source]
    assert len(set(paths)) == 1 and open(paths[0], "rb").read() == open(source, "rb").read()
    assert not os.path.exists(paths[0] + ".partial")
    engram.node_local_copy(source, local_dir)
    assert copied == [source]


def test_a_partial_copy_is_replaced(checkpoint, tmp_path):
    ckpt, _, _ = checkpoint
    source = str(ckpt / "model-00001.safetensors")
    local_dir = str(tmp_path / "shm")
    local = engram.node_local_copy(source, local_dir)
    with open(local, "r+b") as f:
        f.truncate(16)
    assert engram.node_local_copy(source, local_dir) == local
    assert open(local, "rb").read() == open(source, "rb").read()
