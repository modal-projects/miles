"""The multi-LoRA actor's slot RPCs in the order the Tinker gateway issues them -- no GPU."""

from types import SimpleNamespace

import pytest
import torch

from tests.fast.backends.megatron_utils.test_lora_actor import _lora_actor_module

from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.bridge_adapter_slots import BridgeAdapterSlots
from miles.ray.specs.train import MULTI_LORA_TRAINER_CONCURRENCY_GROUPS
from miles.utils.workers.rpc.common.metadata import DEFAULT_CONCURRENCY_GROUP, declared_concurrency_groups


class _RecordedSlots(BridgeAdapterSlots):
    """``AdapterSlots``' own claim bookkeeping over recorded backend draws."""

    def __init__(self, events: list):
        self._multi_lora, self.num_slots, self._claimed_slots, self.targets = True, 3, set(), None
        self._events, self.head_rows = events, None

    def _init_slot(self, slot, lora_rank, lora_alpha, seed):
        self._events.append(("init", slot, lora_alpha))

    def _clear_slot(self, slot):
        self._events.append(("clear", slot))

    def slot_parameters(self, slot):
        return [torch.nn.Parameter(torch.zeros(1))]

    def parametrization(self, **fields):
        return None

    def global_shapes(self, lora_rank):
        return {}


_SOURCE = {"path": "/ckpt", "load_optimizer": True, "parametrization_fields": {}}


@pytest.fixture
def actor_and_events(monkeypatch):
    events = []
    monkeypatch.setattr(
        lora_model,
        "SlotOptimizer",
        lambda args, model, slot, params: SimpleNamespace(slot=slot, zero_grads=lambda: None),
    )
    monkeypatch.setattr(lora_checkpoint, "load_slot", lambda *args, **kwargs: events.append(("load", args[3])))
    # the real actor module imports torch_memory_saver, which CPU CI lacks
    actor_module = _lora_actor_module(monkeypatch, model=lora_model, checkpoint=lora_checkpoint)
    actor = actor_module.MultiLoRATrainRayActor.__new__(actor_module.MultiLoRATrainRayActor)
    actor.args, actor.model, actor.adapter_slots, actor.slot_optimizers = None, [], _RecordedSlots(events), {}
    actor._checkpoint_members = lambda: None
    return actor, events


def test_load_state_into_a_created_model_claims_its_slot_again_at_the_sources_alpha(actor_and_events, monkeypatch):
    """create_model claims the slot; load_state then loads into that same live slot."""
    actor, events = actor_and_events
    monkeypatch.setattr(lora_checkpoint, "check_load_source", lambda *args, **kwargs: (None, None))

    assert actor.load_slot(1, 8, 16.0) is None
    assert actor.load_slot(1, 8, 32.0, source=_SOURCE) is None
    assert events == [("init", 1, 16.0), ("clear", 1), ("init", 1, 32.0), ("load", "/ckpt")]

    actor.unload_slot(1)
    assert actor.load_slot(1, 8, 16.0) is None, "another tenant creates a model on the freed slot"


def test_a_refused_source_leaves_the_live_slot_and_its_optimizer_alone(actor_and_events, monkeypatch):
    actor, events = actor_and_events
    monkeypatch.setattr(
        lora_checkpoint, "check_load_source", lambda *args, **kwargs: ("saved with another layout", None)
    )
    actor.load_slot(1, 8, 16.0)
    optimizer = actor.slot_optimizers[1]

    refused = actor.load_slot(1, 8, 32.0, source=_SOURCE)

    assert refused == {"error": "saved with another layout", "error_category": "user"}
    assert events == [("init", 1, 16.0)] and actor.slot_optimizers[1] is optimizer


def test_creating_a_model_on_a_claimed_slot_fails(actor_and_events):
    actor, _ = actor_and_events
    actor.load_slot(1, 8, 16.0)

    with pytest.raises(AssertionError, match="slot 1 is already claimed"):
        actor.load_slot(1, 8, 16.0)


def test_a_sampler_export_is_made_visible_by_its_own_call(monkeypatch):
    module = _lora_actor_module(monkeypatch)
    monkeypatch.setattr(module, "save_sampler_export", lambda root, name, **kwargs: SimpleNamespace(path=root / name))
    monkeypatch.setattr(module, "compute_custom_function_config", lambda args, path: args)
    published = []
    actor = module.MultiLoRATrainRayActor()
    actor.args = SimpleNamespace(custom_update_weight_post_write_path="hook", lora_dropout=0.0)
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor.adapter_slots = SimpleNamespace(
        parametrization=lambda **fields: SimpleNamespace(lora_rank=8), lora_rank=lambda slot: 8, targets=None
    )
    actor._post_write_hook = lambda args, path, engines: published.append((path, engines))

    written = actor.export_slot(1, "/adapters/m1/sampler_weights", "0", "digest", {})
    assert published == []

    actor.commit_export(str(written.path))

    assert published == [("/adapters/m1/sampler_weights/0", [])]


def test_an_export_commit_does_not_queue_behind_training(monkeypatch):
    groups = declared_concurrency_groups(_lora_actor_module(monkeypatch).MultiLoRATrainRayActor)

    assert groups["commit_export"] != DEFAULT_CONCURRENCY_GROUP
    assert set(groups.values()) <= set(MULTI_LORA_TRAINER_CONCURRENCY_GROUPS)
