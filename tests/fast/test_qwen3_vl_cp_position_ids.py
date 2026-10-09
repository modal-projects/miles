"""CPU unit test: the patched Qwen3-VL forward passes rank-local MRoPE position_ids to the
bridge when CP has pre-sharded the THD row, and leaves other inputs unchanged."""

import sys
import types

import pytest
import torch

from miles_plugins.models import qwen3_vl

_MODEL_MOD = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"


def _install_fake_bridge(monkeypatch):
    calls = []

    class Qwen3VLModel:
        _cp_local_vision_embed_indices = None

        def forward(self, *args, **kwargs):
            calls.append(kwargs)

    model_mod = types.ModuleType(_MODEL_MOD)
    model_mod.Qwen3VLModel = Qwen3VLModel
    model_mod.get_rope_index = lambda *args, **kwargs: (None, None)
    model_mod.preprocess_packed_seqs = lambda input_ids, attention_mask, *args, **kwargs: (input_ids, None)
    monkeypatch.setitem(sys.modules, _MODEL_MOD, model_mod)
    qwen3_vl._patch_model_forward_and_rope_index()
    return Qwen3VLModel(), calls


def _thd_kwargs(local_len, full_len, position_ids=None):
    psp = types.SimpleNamespace(qkv_format="thd", cu_seqlens_q=torch.tensor([0, full_len]))
    return {
        "input_ids": torch.zeros(1, local_len, dtype=torch.long),
        "packed_seq_params": psp,
        "position_ids": position_ids,
    }


@pytest.fixture
def positions(monkeypatch):
    positions = torch.arange(3 * 8).reshape(3, 1, 8)
    monkeypatch.setattr(qwen3_vl, "_build_packed_positions", lambda *args: positions)
    return positions


def test_pre_sharded_cp_row_gets_explicit_position_ids(monkeypatch, positions):
    monkeypatch.setattr(qwen3_vl, "_cp_size_rank", lambda: (2, 1))
    model, calls = _install_fake_bridge(monkeypatch)

    model.forward(**_thd_kwargs(local_len=8, full_len=16))

    assert calls[0]["position_ids"] is positions


def test_explicit_position_ids_are_kept(monkeypatch, positions):
    monkeypatch.setattr(qwen3_vl, "_cp_size_rank", lambda: (2, 1))
    model, calls = _install_fake_bridge(monkeypatch)
    explicit = torch.zeros(3, 1, 8, dtype=torch.long)

    model.forward(**_thd_kwargs(local_len=8, full_len=16, position_ids=explicit))

    assert calls[0]["position_ids"] is explicit


def test_non_cp_row_keeps_rope_index_path(monkeypatch, positions):
    monkeypatch.setattr(qwen3_vl, "_cp_size_rank", lambda: (1, 0))
    model, calls = _install_fake_bridge(monkeypatch)

    model.forward(**_thd_kwargs(local_len=8, full_len=8))

    assert calls[0]["position_ids"] is None
