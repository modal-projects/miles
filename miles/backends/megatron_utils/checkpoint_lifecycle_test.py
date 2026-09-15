import ast
import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _function(path, name, *, class_name=None, namespace):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class _Args(SimpleNamespace):
    def __getattr__(self, name):
        return None


@pytest.mark.parametrize("driver", ["train.py", "train_async.py"])
@pytest.mark.parametrize("trigger", ["periodic", "external", "final"])
def test_driver_saves_sampler_first_and_drains_even_on_debug_exit(driver, trigger):
    final = trigger == "final"
    events = []

    async def call(name, *args, **kwargs):
        events.append((name, args, kwargs))
        return None

    class Model:
        async def update_weights(self, *args, **kwargs):
            await call("update", *args, **kwargs)

        async def train(self, *args, **kwargs):
            await call("train", *args, **kwargs)

        async def save_model(self, *args, **kwargs):
            await call("save", *args, **kwargs)

        async def finish_checkpoints(self, *args, **kwargs):
            await call("drain", *args, **kwargs)

        async def clear_memory(self):
            pass

    model = Model()

    async def models(*args):
        return model, None

    def remote(name):
        return SimpleNamespace(remote=lambda *a, **kw: asyncio.create_task(call(name, *a, **kw)))

    manager = SimpleNamespace(**{name: remote(name) for name in ("save", "generate", "dispose", "eval")})
    manager.save = remote("sampler")
    args = _Args(
        start_rollout_id=0,
        num_rollout=1 if final else 3,
        debug_exit_after_rollout=None if final else 1,
        save_interval=1,
        update_weights_interval=1,
        async_save=True,
        train_backend="megatron",
        save_trigger_sentinel="/trigger" if trigger == "external" else None,
    )
    namespace = {
        "__name__": "driver_under_test",
        "asyncio": asyncio,
        "logger": logging.getLogger("test"),
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda path: True), remove=lambda path: None),
        "object_store": SimpleNamespace(init_instance=lambda *a, **kw: None),
        "MainProcessIdentity": lambda: None,
        "create_placement_groups": lambda args: {"rollout": None},
        "create_rollout_manager": lambda *args: (manager, 1),
        "create_training_models": models,
        "uses_external_disk_deltas": lambda args: False,
        "can_overlap_external_weight_sync": lambda args: True,
        "should_run_periodic_action": lambda rollout_id, interval, *args: interval is not None,
        "EvalDispatcher": lambda *args: SimpleNamespace(drain=lambda: call("eval_drain")),
    }
    for name in (
        "validate_async_off_policy_correction",
        "configure_logger",
        "maybe_start_periodic_pyspy_dump",
        "init_tracking",
        "maybe_start_mini_ft_controller",
        "remove_rollout_data_refs",
    ):
        namespace[name] = lambda *a, **kw: None
    fn = _function(ROOT / driver, "train", namespace=namespace)
    asyncio.run(fn(args))
    names = [event[0] for event in events]
    assert names.index("sampler") < names.index("save"), names
    assert names.index("drain") < names.index("dispose"), names
    save = next(event for event in events if event[0] == "save")
    assert save[2]["force_sync"] is (trigger != "periodic")


@pytest.fixture
def runtime(monkeypatch):
    events = []
    state = SimpleNamespace(pending=False, writer_done=False, rank=0, peers=None)
    args = _Args(
        async_save=True, use_persistent_ckpt_worker=True, rank=0, custom_checkpoint_completed_hook_path="test.complete"
    )
    native = ModuleType("megatron.training.async_utils")

    def init(rank):
        events.append(("init", rank))

    def finalize(blocking=False, terminate=False):
        events.append(("poll", blocking, terminate))
        if blocking or state.writer_done:
            state.pending = False

    native.init_persistent_async_worker = init
    native.maybe_finalize_async_save = finalize
    native.is_empty_async_queue = lambda: not state.pending
    misc = ModuleType("miles.utils.misc")
    misc.load_function = lambda path: lambda *a: events.append(("complete", *a[1:]))
    distributed = ModuleType("torch.distributed")
    distributed.get_rank = lambda: state.rank
    distributed.get_world_size = lambda *a: 2

    def gather(output, value, **kwargs):
        output[:] = state.peers if state.peers is not None else [value, value]

    distributed.all_gather_object = gather
    torch = ModuleType("torch")
    torch.distributed = distributed
    utils = ModuleType("miles.utils.distributed_utils")
    utils.get_gloo_group = lambda: None
    training = ModuleType("megatron.training")
    training.async_utils = native
    for name, module in [
        ("megatron.training", training),
        ("torch", torch),
        ("torch.distributed", distributed),
        ("megatron.training.async_utils", native),
        ("miles.utils.misc", misc),
        ("miles.utils.distributed_utils", utils),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).with_name("checkpoint_lifecycle.py")
    spec = importlib.util.spec_from_file_location("checkpoint_lifecycle_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, args=args, state=state, events=events, native=native, misc=misc)


def test_native_writer_is_initialized_and_pending_save_waits_for_completion(runtime):
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    assert runtime.events == [("init", 0)]
    runtime.state.pending = True
    lifecycle.saved(7, "/model/7", "/hf/7")
    lifecycle.poll()
    assert not any(event[0] == "complete" for event in runtime.events)
    runtime.state.writer_done = True
    lifecycle.poll()
    assert runtime.events[-1] == ("complete", 7, "/model/7", "/hf/7")
    lifecycle.poll()
    assert sum(event[0] == "complete" for event in runtime.events) == 1


def test_starting_another_save_drains_previous_snapshot(runtime):
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    runtime.state.pending = True
    lifecycle.saved(7, "/model/7", "/hf/7")
    with pytest.raises(RuntimeError, match="pending"):
        lifecycle.saved(8, "/model/8", "/hf/8")
    lifecycle.prepare()
    assert runtime.events[-1] == ("complete", 7, "/model/7", "/hf/7")
    lifecycle.saved(8, "/model/8", "/hf/8")
    lifecycle.poll(blocking=True, terminate=True)
    assert runtime.events[-1] == ("complete", 8, "/model/8", "/hf/8")
    assert ("poll", True, True) in runtime.events


def test_sync_completion_uses_same_hook_without_native_initialization(runtime):
    runtime.args.async_save = False
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    lifecycle.saved(7, "/model/7", "/hf/7")
    lifecycle.poll()
    assert runtime.events == [("complete", 7, "/model/7", "/hf/7")]


def test_no_callback_without_an_accepted_snapshot(runtime):
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    lifecycle.poll(blocking=True, terminate=True)
    assert not any(event[0] == "complete" for event in runtime.events)


def test_completion_failure_reaches_peer_ranks(runtime):
    runtime.state.peers = [None, "rank 1: OSError: failed to persist"]
    runtime.args.async_save = False
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    lifecycle.saved(7, "/model/7", "/hf/7")
    with pytest.raises(RuntimeError, match="rank 1: OSError: failed to persist"):
        lifecycle.poll()


def test_old_megatron_worker_initializer_is_supported(runtime):
    runtime.native.init_persistent_async_worker = lambda: runtime.events.append(("init_old",))
    runtime.module.CheckpointLifecycle(runtime.args)
    assert runtime.events == [("init_old",)]


@pytest.mark.parametrize("fail_hf", [False, True])
def test_actor_completion_follows_hf_finish_and_is_suppressed_on_failure(runtime, monkeypatch, fail_hf):
    runtime.args.async_save = False
    runtime.args.debug_rollout_only = False
    runtime.args.save = "/model"
    runtime.args.save_hf = "/hf/{rollout_id}"
    runtime.args.custom_megatron_post_save_hook_path = None
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    exporter = ModuleType("miles.backends.megatron_utils.hf_export")

    def finish():
        runtime.events.append(("hf_finish",))
        if fail_hf:
            raise OSError("HF writer failed")

    def start(*args):
        runtime.events.append(("hf_start",))
        return SimpleNamespace(finish=finish)

    exporter.start_hf_export = start
    native = ModuleType("megatron.training.checkpointing")
    native.get_checkpoint_name = lambda *args, **kwargs: "/model/7"
    monkeypatch.setitem(sys.modules, exporter.__name__, exporter)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    actor = SimpleNamespace(
        args=runtime.args,
        role="actor",
        model=[],
        optimizer=None,
        opt_param_scheduler=None,
        _heartbeat=SimpleNamespace(bump=lambda: None),
        _checkpoint_lifecycle=lifecycle,
    )
    namespace = {
        "__name__": "actor_under_test",
        "dist": sys.modules["torch.distributed"],
        "is_multi_lora_enabled": lambda args: False,
        "save": lambda *args: runtime.events.append(("native_save",)),
    }
    save = _function(
        ROOT / "miles/backends/megatron_utils/actor.py",
        "save_model",
        class_name="MegatronTrainRayActor",
        namespace=namespace,
    )
    if fail_hf:
        with pytest.raises(OSError, match="HF writer failed"):
            save(actor, 7)
        assert lifecycle.pending is None
        assert not any(event[0] == "complete" for event in runtime.events)
    else:
        save(actor, 7)
        assert [event[0] for event in runtime.events] == ["hf_start", "native_save", "hf_finish", "complete"]


def test_nonpersistent_async_mode_does_not_start_a_persistent_worker(runtime):
    runtime.args.use_persistent_ckpt_worker = False
    lifecycle = runtime.module.CheckpointLifecycle(runtime.args)
    assert runtime.events == []
    lifecycle.saved(7, "/model/7", "/hf/7")
    lifecycle.poll(blocking=True, terminate=True)
    assert runtime.events == [("poll", True, True), ("complete", 7, "/model/7", "/hf/7")]
