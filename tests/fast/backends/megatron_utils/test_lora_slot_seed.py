"""Exercise slot seed forwarding without importing the optional trainer runtimes."""

import ast
import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]


def _function(path, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    scope = (
        next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner) if owner else tree
    )
    function = next(
        node
        for node in scope.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "load_slot"
    )
    function.decorator_list = []
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    return compile(isolated, str(ROOT / path), "exec")


@pytest.mark.parametrize("seed", [0, 42])
def test_slot_seed_reaches_initialization_before_optimizer_and_restore_wins(monkeypatch, seed):
    weights, draws = {}, []

    def initialize(model, slot, rank, alpha, seed=None):
        draws.append(seed)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        weights[slot] = torch.randn(16, generator=generator)

    def optimizer(args, model, slot, *unused):
        return SimpleNamespace(slot=slot, initial_weights=weights[slot].clone(), zero_grads=lambda: None)

    bridge = ModuleType("megatron.bridge.peft.multi_lora_layers")
    bridge.init_adapter_slot = initialize
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    model_globals = {"SlotOptimizer": optimizer}
    exec(_function("miles/backends/megatron_utils/lora/model.py"), model_globals)
    native = "adapter_slots" in model_globals["load_slot"].__annotations__
    slots = SimpleNamespace(
        init=lambda slot, rank, alpha, seed: initialize(None, slot, rank, alpha, seed),
        slot_parameters=lambda slot: [weights[slot]],
        parametrization=lambda **fields: None,
        global_shapes=lambda rank: {},
        targets=None,
        head_rows=None,
    )

    def restore(*args, **kwargs):
        slot_optimizer = args[2] if native else args[1]
        weights[slot_optimizer.slot] = torch.full((16,), 99.0)
        slot_optimizer.initial_weights = weights[slot_optimizer.slot].clone()

    actor_globals = {
        "lora_model": SimpleNamespace(load_slot=model_globals["load_slot"], unload_slot=lambda *args: None),
        "lora_checkpoint": SimpleNamespace(check_load_source=lambda *args, **kwargs: (None, None), load_slot=restore),
    }
    exec(_function("miles/backends/megatron_utils/lora/actor.py", "MultiLoRATrainRayActor"), actor_globals)
    actor = SimpleNamespace(
        args=None, model=[], adapter_slots=slots, slot_optimizers={}, _checkpoint_members=lambda: None
    )

    async def execute(method, **kwargs):
        assert method == "load_slot"
        return [actor_globals["load_slot"](actor, **kwargs)]

    controller_globals = {}
    exec(_function("miles/ray/train/group.py", "TrainerController"), controller_globals)
    controller = SimpleNamespace(_execute_slots=execute)

    async def controller_load(*args, **kwargs):
        return await controller_globals["load_slot"](controller, *args, **kwargs)

    runtime_globals = {"_slot_failure": lambda results: next((item for item in results if item is not None), None)}
    exec(_function("miles/tinker/runtime.py", "MilesBackend"), runtime_globals)
    backend = SimpleNamespace(trainer=SimpleNamespace(load_slot=controller_load))

    async def run():
        async def load(slot, **kwargs):
            await runtime_globals["load_slot"](backend, slot, 8, 16.0, **kwargs)
            return actor.slot_optimizers[slot].initial_weights.clone()

        first = await load(0, seed=seed)
        assert torch.equal(first, await load(1, seed=seed))
        actor.slot_optimizers.pop(0)
        assert not torch.equal(first, await load(0, seed=seed + 1))
        actor.slot_optimizers.pop(0)
        assert torch.equal(first, await load(0, seed=seed))
        source = (
            {"source": {"path": "/saved", "load_optimizer": True, "parametrization_fields": {}}}
            if native
            else {"ckpt_path": "/saved"}
        )
        restored = await load(0, seed=seed, **source)
        assert torch.equal(restored, torch.full((16,), 99.0))
        assert draws == [seed, seed, seed + 1, seed, None]

    asyncio.run(run())
