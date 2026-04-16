from argparse import Namespace
from pathlib import Path
from unittest.mock import sentinel

import torch.nn as nn


def _make_args(**overrides):
    defaults = dict(
        moe_use_upcycling=False,
        load="/tmp/fake",
        pretrained_checkpoint=None,
        use_torch_fsdp2=False,
        use_megatron_fsdp=False,
        enable_gloo_process_groups=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_setup_model_and_optimizer_restores_marked_dtypes_before_ddp(monkeypatch):
    import sys

    megatron_root = str(Path("/root/Megatron-LM"))
    if megatron_root not in sys.path:
        sys.path.insert(0, megatron_root)

    from miles.backends.megatron_utils import model as model_mod

    events = []
    raw_model = [nn.Linear(4, 4)]

    monkeypatch.setattr(model_mod, "is_lora_enabled", lambda args: False)
    monkeypatch.setattr(model_mod, "get_model_provider_func", lambda args, role: sentinel.provider)

    def fake_get_model(provider, model_type, wrap_with_ddp=True, config=None, pg_collection=None):
        events.append(("get_model", wrap_with_ddp))
        assert provider is sentinel.provider
        return raw_model

    def fake_enforce(model_chunks):
        events.append(("enforce", model_chunks))
        assert model_chunks is raw_model
        return ["module.decoder.layers.0.self_attention.linear_attn.A_log"]

    def fake_wrap(args, model_chunks):
        events.append(("wrap", model_chunks))
        assert model_chunks is raw_model
        return ["wrapped-ddp-model"]

    monkeypatch.setattr(model_mod, "get_model", fake_get_model)
    monkeypatch.setattr(model_mod, "enforce_marked_param_dtypes", fake_enforce)
    monkeypatch.setattr(model_mod, "_wrap_model_with_ddp", fake_wrap)
    monkeypatch.setattr(model_mod, "get_megatron_optimizer", lambda **kwargs: sentinel.optimizer)
    monkeypatch.setattr(model_mod, "get_optimizer_param_scheduler", lambda args, optimizer: sentinel.scheduler)

    model, optimizer, scheduler = model_mod.setup_model_and_optimizer(_make_args(), role="actor")

    assert model == ["wrapped-ddp-model"]
    assert optimizer is sentinel.optimizer
    assert scheduler is sentinel.scheduler
    assert events == [
        ("get_model", False),
        ("enforce", raw_model),
        ("wrap", raw_model),
    ]
