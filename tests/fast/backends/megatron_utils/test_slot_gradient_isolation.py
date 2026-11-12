"""CPU autograd/Adam regression for shared DDP buckets, without importing GPU-only Megatron.

Execute the actual Miles pass orchestration, accumulation context and slot-step/zeroing bodies.
The bucket fixture models dense DP2 SUM and expert EDP1 SUM with their .5 pre-scaling;
marked dense gradients also undergo a model-wide TP SUM. No CUDA kernels are exercised.
"""

import __future__

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )


class _Bucket:
    def __init__(self, params, *, replicas, overlap):
        self.params = params
        self.replicas = replicas
        self.overlap = overlap
        self.reset()

    def reset(self):
        self.ready = set()
        self.finished = False

    def register_grad_ready(self, param, *_args):
        self.ready.add(param)
        if len(self.ready) == len(self.params):
            self.finish_grad_sync()

    def finish_grad_sync(self):
        if not self.finished:
            for param in self.params:
                param.main_grad.mul_(0.5 * self.replicas)
            self.finished = True


class _Model(torch.nn.Module):
    def __init__(self, namespace, *, overlap):
        super().__init__()
        self.config = SimpleNamespace(cuda_graph_impl="none")
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64)) for _ in range(6)]
        )
        self.params_with_grad = list(self.weights)
        self.bucket_groups = []
        self.expert_parallel_bucket_groups = []
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        self.force_all_reduce = False
        self.param_to_bucket_group = {}
        for kind, replicas in enumerate([2, 2, 1]):
            params = list(self.weights[2 * kind : 2 * kind + 2])
            storage = torch.zeros(2, dtype=torch.float64)
            bucket = _Bucket(params, replicas=replicas, overlap=overlap)
            (self.expert_parallel_bucket_groups if kind == 2 else self.bucket_groups).append(bucket)
            for slot, param in enumerate(params):
                param.slot = slot
                param.kind = kind
                param.main_grad = storage[slot : slot + 1]
                param.grad_added_to_main_grad = False
                self.param_to_bucket_group[param] = bucket
                # Exercise inactive zero gradients and early readiness using the DDP hook contract.
                param.register_post_accumulate_grad_hook(namespace["_make_backward_post_hook"](self, param))


class _Slot:
    def __init__(self, model, slot, namespace):
        self._model = [model]
        self.slot = slot
        self.params = [p for p in model.weights if p.slot == slot]
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3)
        # Nonempty moments make an incorrect gradient visible in updated parameters.
        for param in self.params:
            param.grad = torch.full_like(param, 0.2)
        self.optimizer.step()
        for param in self.params:
            param.data.fill_(0.5)
        self.zero_grads = MethodType(namespace["zero_grads"], self)
        self.zero_grads()

    def apply_adam_params(self, _params):
        pass

    def prepare_grads(self):
        for param in self.params:
            param.grad = param.main_grad.clone()

    def clip_and_step(self, clip_grad):
        norm = torch.nn.utils.clip_grad_norm_(self.params, clip_grad)
        self.optimizer.step()
        return {"grad_norm": float(norm)}

    def allgather_params(self):
        # The LayerWise BF16 gather copies only this optimizer's owned slot parameters.
        pass


def _namespace():
    namespace = dict(
        torch=torch,
        contextmanager=contextmanager,
        is_graph_capturing=lambda: False,
        adapter_slot_parameters=lambda chunks, slot: [p for p in chunks[0].weights if p.slot == slot],
    )
    _functions(
        ROOT / "miles/backends/megatron_utils/lora/optimizer.py",
        {"accumulate_slot_gradients", "reset_grad_metadata_keep_grads", "step_slot_optimizers", "zero_grads"},
        namespace,
    )
    return namespace


def _run(events, *, overlap=False, n_adapters=2):
    namespace = _namespace()

    # The pinned DDP hook is represented locally to keep this test independent of a
    # downloaded Megatron checkout; its arithmetic and readiness behavior are exercised.
    def backward_hook(model, param):
        def hook(*_unused):
            if param.grad is not None and not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad)
            param.grad = None
            if model.ddp_config.overlap_grad_reduce:
                model.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    namespace["_make_backward_post_hook"] = backward_hook
    model = _Model(namespace, overlap=overlap)
    slots = {slot: _Slot(model, slot, namespace) for slot in range(n_adapters)}

    def backward(_args, _dumper, iterator, chunks, *_args2, forward_only=False, **_kwargs):
        slot, contribution = iterator[0].event
        loss = sum(p.sum() * (contribution * (p.kind + 1) if p.slot == slot else 0.0) for p in model.weights)
        if not forward_only:
            loss.backward()
            # Model-wide marked-LoRA SUM happens before explicit DDP finalization.
            for param in model.weights:
                if param.kind == 1:
                    param.main_grad.mul_(8)
            for bucket in model.bucket_groups + model.expert_parallel_bucket_groups:
                bucket.finish_grad_sync()
        return [{"per_datum": []}]

    namespace.update(
        get_data_iterator=lambda _args, _model, data: (
            [SimpleNamespace(reset=lambda: None, event=data["event"])],
            [1],
        ),
        setup_train_iteration_config=lambda *_args, **_kwargs: None,
        DumperMegatronUtil=lambda *_args, **_kwargs: SimpleNamespace(finalize=lambda _model: None),
        DumperPhase=SimpleNamespace(FWD_BWD=None),
        run_forward_backward_pass=backward,
        get_parallel_state=lambda: SimpleNamespace(is_pp_last_stage=True),
        aggregate_train_losses=lambda *_args: {},
    )
    _functions(ROOT / "miles/backends/megatron_utils/lora/model.py", {"run_forward_backward"}, namespace)
    args = SimpleNamespace(multi_lora_n_adapters=n_adapters)
    observations = []
    for operation, slot, value in events:
        if operation == "backward":
            namespace["run_forward_backward"](args, 0, [model], {"event": (slot, value)})
        elif operation == "forward":
            namespace["run_forward_backward"](args, 0, [model], {"event": (slot, value)}, forward_only=True)
        else:
            namespace["step_slot_optimizers"](slots, {slot: {"grad_clip_norm": 1.0}})
        observations.append([p.main_grad.clone() for p in model.weights])
    parameters = [p.detach().clone() for p in model.weights]
    moments = [slots[p.slot].optimizer.state[p] for p in model.weights if p.slot < n_adapters]
    return observations, parameters, moments


class TestSlotGradientIsolation(unittest.TestCase):
    def test_interleaving_preserves_gradients_parameters_and_adam(self):
        for first, second in [(0, 1), (1, 0)]:
            for overlap in [False, True]:
                with self.subTest(first=first, overlap=overlap):
                    b1, b2 = ("backward", first, 2.0), ("backward", second, 3.0)
                    o1, o2 = ("optim", first, None), ("optim", second, None)
                    interleaved = _run([b1, b2, o1, o2], overlap=overlap)
                    serial = _run([b1, o1, b2, o2], overlap=overlap)
                    for index in [first, 2 + first, 4 + first]:
                        torch.testing.assert_close(interleaved[0][0][index], interleaved[0][1][index], atol=0, rtol=0)
                    for index in [second, 2 + second, 4 + second]:
                        torch.testing.assert_close(interleaved[0][1][index], interleaved[0][2][index], atol=0, rtol=0)
                    for actual, expected in zip(interleaved[1], serial[1], strict=True):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for actual, expected in zip(interleaved[2], serial[2], strict=True):
                        for key in ["step", "exp_avg", "exp_avg_sq"]:
                            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_repeated_active_slot_accumulates_finalized_contributions(self):
        for overlap in [False, True]:
            result, _, _ = _run([("backward", 0, 2.0), ("backward", 0, 3.0)], overlap=overlap)
            expected = [5.0, 0.0, 80.0, 0.0, 7.5, 0.0]
            self.assertEqual([float(value) for value in result[-1]], expected)

    def test_forward_only_does_not_modify_pending_gradients(self):
        result, _, _ = _run([("backward", 0, 2.0), ("forward", 1, 3.0)])
        for actual, expected in zip(*result, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_failed_pass_restores_previous_gradients(self):
        namespace = _namespace()
        param = SimpleNamespace(main_grad=torch.tensor([7.0]))
        model = [SimpleNamespace(params_with_grad=[param])]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with namespace["accumulate_slot_gradients"](model, enabled=True):
                param.main_grad.add_(99)
                raise RuntimeError("failed")
        self.assertEqual(float(param.main_grad), 7.0)

    def test_disabled_context_leaves_one_slot_path_unmodified(self):
        namespace = _namespace()
        param = SimpleNamespace(main_grad=torch.tensor([7.0]))
        with namespace["accumulate_slot_gradients"]([SimpleNamespace(params_with_grad=[param])], enabled=False):
            self.assertEqual(float(param.main_grad), 7.0)
            param.main_grad.add_(2)
        self.assertEqual(float(param.main_grad), 9.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
