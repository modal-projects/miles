from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from miles.backends.megatron_utils.lora import batch_invariant as invariant  # noqa: E402


class Head(torch.nn.Module):
    def __init__(self, external_weight: bool = False) -> None:
        super().__init__()
        weight = torch.arange(104, dtype=torch.float32).reshape(13, 8) / 100
        self.register_buffer("external_weight", weight)
        self.weight = None if external_weight else torch.nn.Parameter(weight, requires_grad=False)
        self.shapes = []

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        self.shapes.append(tuple(inputs.shape))
        return torch.nn.functional.linear(inputs, self.weight if weight is None else weight), None


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrapped_reader: bool = False,
    external_weight: bool = False,
) -> tuple[invariant.Partition, SimpleNamespace, SimpleNamespace, Head]:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    layout = invariant.partition([1, 3, 4], [3, 5, 7], [0, 3, 0, 5, 8], 8, 16)

    def get_batch(
        data_iterator: object,
        keys: Sequence[str],
        pad_multiplier: int = 128,
        qkv_format: str = "thd",
        get_position_ids: bool = False,
        allgather_cp: bool = False,
    ) -> dict[str, Any]:
        return {
            "adapter_slots": [1, 3, 4],
            "unconcat_tokens": [torch.zeros(n) for n in (3, 5, 7)],
            "adapter_token_counts": torch.tensor(layout.counts),
            "tokens": torch.zeros(1, 16),
        }

    if wrapped_reader:
        original = cast(Callable[..., dict[str, Any]], get_batch)

        def get_batch(
            data_iterator: object,
            keys: Sequence[str],
            *args: object,
            allgather_cp: bool = False,
            **kwargs: object,
        ) -> dict[str, Any]:
            return original(data_iterator, keys, *args, allgather_cp=allgather_cp, **kwargs)

    def mm(
        x: torch.Tensor,
        stacked_weights: torch.Tensor,
        *,
        token_splits: Sequence[int],
        offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.cat(
            [torch.nn.functional.linear(z, w) for z, w in zip(x.split(token_splits), stacked_weights, strict=True)]
        )

    layers = SimpleNamespace(_dense_multi_lora_mm=mm)
    model = SimpleNamespace(get_batch=get_batch)
    head = Head(external_weight)
    chunk = torch.nn.Module()
    chunk.add_module("output_layer", head)
    invariant.install(model, [chunk], layers=layers)
    invariant.install(model, [chunk], layers=layers)
    return layout, model, layers, head


@pytest.mark.parametrize("wrapped_reader", [False, True])
@pytest.mark.parametrize("external_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_projection_and_frozen_head_match_independent_solo_gradients(
    monkeypatch: pytest.MonkeyPatch, wrapped_reader: bool, external_weight: bool, dtype: torch.dtype
) -> None:
    layout, model, layers, head = _fixture(monkeypatch, wrapped_reader=wrapped_reader, external_weight=external_weight)
    head.to(dtype=dtype)
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(16, 8, generator=generator, dtype=dtype, requires_grad=True)
    a = torch.randn(5, 8, 8, generator=generator, dtype=dtype, requires_grad=True)
    b = torch.randn(5, 8, 8, generator=generator, dtype=dtype, requires_grad=True)
    upstream = torch.randn(16, 1, 13, generator=generator, dtype=dtype)
    upstream[15:] = 0
    kwargs = {"weight": head.external_weight} if external_weight else {}
    with invariant.work_unit(True):
        model.get_batch(None, ["tokens"], pad_multiplier=8)

        # Checkpoint recomputation may execute the same path on an autograd thread.
        def forward() -> torch.Tensor:
            mid = layers._dense_multi_lora_mm(x, a, token_splits=layout.counts, offsets=None)
            out = layers._dense_multi_lora_mm(mid, b, token_splits=layout.counts, offsets=None)
            return head(out.unsqueeze(1), **kwargs)[0]

        with ThreadPoolExecutor(max_workers=1) as pool:
            actual = pool.submit(forward).result()
        actual_grads = torch.autograd.grad((actual * upstream).sum(), (x, a, b))
    assert head.shapes == []
    expected = []
    for slot, span in enumerate(layout.spans):
        if not span.rows:
            continue
        inputs = torch.cat(
            (
                x[span.start : span.start + span.rows],
                x.new_zeros((span.padded_rows - span.rows, 8)),
            )
        )
        solo = invariant._group_offsets([len(inputs)], inputs.device)
        mid = invariant._grouped_mm(inputs, a[slot : slot + 1], solo)
        out = invariant._grouped_mm(mid, b[slot : slot + 1], solo)
        logits = torch.nn.functional.linear(out.unsqueeze(1), head.external_weight)
        expected.append(logits[: span.rows])
    expected = torch.cat(expected)
    expected_grads = torch.autograd.grad((expected * upstream[:15]).sum(), (x, a, b))
    torch.testing.assert_close(actual[:15], expected, rtol=0, atol=0)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("projection", ["head", "lora"])
def test_padding_gradient_is_rejected_and_failed_unit_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    layout, model, layers, head = _fixture(monkeypatch)
    head.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="nonzero upstream"):
        with invariant.work_unit(True):
            model.get_batch(None, ["tokens"], pad_multiplier=8)
            x = torch.ones(16, 8, dtype=torch.bfloat16, requires_grad=True)
            if projection == "head":
                head(x.unsqueeze(1))[0].sum().backward()
            else:
                weights = torch.ones(5, 8, 8, dtype=torch.bfloat16, requires_grad=True)
                layers._dense_multi_lora_mm(x, weights, token_splits=layout.counts, offsets=None).sum().backward()
            assert invariant._active_unit.bad_padding_grad is not None
            assert x.grad is not None
            if projection == "head":
                assert not x.grad[15:].any()
    assert invariant._active_unit is None
    with invariant.work_unit(True):
        model.get_batch(None, ["tokens"], pad_multiplier=8)
        head(torch.zeros(16, 1, 8, dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="serialized"):
            with invariant.work_unit(True):
                pass


@pytest.mark.parametrize("counts,total", [([0, 3, 0, 5, 7], 15), ([0, 4, 0, 5, 7], 16)])
def test_partition_rejects_unproven_padding(counts: list[int], total: int) -> None:
    with pytest.raises(ValueError, match="padding"):
        invariant.partition([1, 3, 4], [3, 5, 7], counts, 8, total)


@pytest.mark.parametrize("offset", [0, 1, 32])
def test_mm_output_alignment_fallback(monkeypatch: pytest.MonkeyPatch, offset: int) -> None:
    original = torch.mm
    calls = []

    def mm(a, b, *, out=None):
        calls.append(out is not None)
        return original(a, b) if out is None else original(a, b, out=out)

    monkeypatch.setattr(torch, "mm", mm)
    a = torch.randn(3, 7, dtype=torch.float64)
    b = torch.randn(7, 5, dtype=torch.float64)
    out = torch.empty(15 + offset, dtype=torch.float64)[offset:].view(3, 5)
    invariant._mm_out(a, b, out)
    assert calls == [offset % 32 == 0]
    assert torch.equal(out, original(a, b))


def test_disabled_path_preserves_calls_and_missing_head_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, layers, head = _fixture(monkeypatch)
    with invariant.work_unit(False):
        x = torch.ones(5, 3)
        weights = torch.ones(2, 2, 3)
        out = layers._dense_multi_lora_mm(x, weights, token_splits=(2, 3), offsets=None)
        assert out.tolist() == [[3.0, 3.0]] * 5
    with pytest.raises(RuntimeError, match="frozen head"):
        with invariant.work_unit(True):
            model.get_batch(None, ["tokens"], pad_multiplier=8)


def test_option_and_topology_are_validated_before_model_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        deterministic_mode=True,
        bf16=True,
        fp16=False,
        qkv_format="thd",
        allgather_cp=False,
        lora_dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        enable_mtp_training=False,
        num_experts=None,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    invariant.validate_args(args)
    for key, bad in (
        ("tensor_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("expert_model_parallel_size", 2),
        ("lora_dropout", 0.1),
        ("deterministic_mode", False),
        ("qkv_format", "bshd"),
    ):
        changed = SimpleNamespace(**{**vars(args), key: bad})
        with pytest.raises(ValueError):
            invariant.validate_args(changed)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="one initialized rank"):
        invariant.install(SimpleNamespace(), [])


def test_launch_parser_exposes_opt_in_without_importing_gpu_runtime() -> None:
    import argparse
    import ast
    from pathlib import Path

    source = Path(__file__).parents[2] / "miles/utils/lora/arguments.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "add_lora_arguments")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    parser = namespace["add_lora_arguments"](argparse.ArgumentParser())
    assert parser.parse_args([]).lora_batch_invariant is False
    assert parser.parse_args(["--lora-batch-invariant"]).lora_batch_invariant is True


@pytest.mark.parametrize("failure", ["head", "signature"])
def test_install_rejects_trainable_head_and_changed_bridge_signature(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _, model, layers, head = _fixture(monkeypatch)
    chunk = torch.nn.Module()
    chunk.add_module("output_layer", head)
    if failure == "head":
        head.weight.requires_grad_(True)
        error, message = ValueError, "frozen output layer"
    else:
        layers._dense_multi_lora_mm = lambda x: x
        error, message = RuntimeError, "signature"
    with pytest.raises(error, match=message):
        invariant.install(model, [chunk], layers=layers)


# Padding-gradient flags are recorded on module state, which AOT tracing cannot capture.
_OPCHECK_UTILS = ("test_schema", "test_autograd_registration", "test_faketensor")


def test_frozen_head_op_passes_opcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(invariant, "_active_unit", invariant._WorkUnit())
    layout = invariant.partition([0, 2], [5, 4], [5, 0, 11], 8, 16)
    args = invariant._layout_args(layout)
    x = torch.randn(16, 7, dtype=torch.float64, requires_grad=True)
    head = torch.randn(13, 7, dtype=torch.float64)
    torch.library.opcheck(invariant._frozen_head_op, (x, head, *args), test_utils=_OPCHECK_UTILS)


def test_grouped_mm_self_check_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(invariant, "_verified_gemms", set())
    original = invariant._grouped_mm

    def unstable(x: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return original(x, weights, offsets) * (1 + len(weights))

    x = torch.randn(16, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(invariant, "_grouped_mm", unstable)
    with pytest.raises(RuntimeError, match="not batch-invariant"):
        invariant._verify_grouped_mm(x, torch.randn(2, 8, 8, dtype=x.dtype))
    monkeypatch.setattr(invariant, "_grouped_mm", original)
    invariant._verify_grouped_mm(x, torch.randn(2, 8, 8, dtype=x.dtype))
    assert invariant._verified_gemms == {(x.device, x.dtype, 8, 8, False)}


@pytest.mark.skipif(
    invariant.miles_grouped_gemm is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="needs the miles_grouped_gemm extension on sm90/sm100",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cutlass_grouped_mm_matches_torch_and_is_invariant(dtype: torch.dtype) -> None:
    x = torch.randn(40, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(4, 32, 64, device="cuda", dtype=dtype, requires_grad=True)
    offs = invariant._group_offsets((24, 0, 8, 8), x.device)
    torch.library.opcheck(torch.ops.miles_grouped_gemm.grouped_mm, (x, w.transpose(-2, -1), offs))
    if dtype == torch.bfloat16:
        expected = torch._grouped_mm(x, w.transpose(-2, -1), offs)
        assert torch.equal(invariant._grouped_mm(x, w, offs), expected)
    invariant._verify_grouped_mm(x, w)
