import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.gated_delta import (
    enable_gated_delta_prefix_tree,
    tree_causal_convolution,
    tree_gated_delta,
)
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.recurrent import RecurrentTreeLayout
from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree


def _convolution(x, weight, bias=None, activation=None, seq_idx=None):
    if seq_idx is not None:
        _, counts = torch.unique_consecutive(seq_idx[0], return_counts=True)
        return torch.cat(
            [_convolution(part, weight, bias, activation) for part in x.split(counts.tolist(), dim=2)], dim=2
        )
    output = F.conv1d(x, weight[:, None], bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    return F.silu(output[:, :, : x.shape[2]])


def _recurrence(q, k, v, *, g, beta, initial_state=None, cu_seqlens=None, output_final_state=False, **kwargs):
    outputs, states = [], []
    for segment, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        state = q.new_zeros((1, 2, 2)) if initial_state is None else initial_state[segment]
        for row in range(int(start), int(end)):
            state = state * g[0, row].exp()[:, None, None]
            delta = v[0, row] - (state * k[0, row, :, :, None]).sum(-2)
            state = state + k[0, row, :, :, None] * (delta * beta[0, row, :, None])[:, None, :]
            outputs.append((state * q[0, row, :, :, None]).sum(-2))
        states.append(state)
    return torch.stack(outputs)[None], torch.stack(states) if output_final_state else None


def _run(x, weight, g, beta, layout, *, shared):
    def compute(value, decay, update, current):
        convolved = tree_causal_convolution(
            value, weight, None, layout=current, native=_convolution, activation="silu"
        )
        q, k, v = [t.transpose(1, 2).unsqueeze(2) for t in convolved.chunk(3, dim=1)]
        return tree_gated_delta(q, k, v, g=decay, beta=update, layout=current, native=_recurrence)[0]

    if shared:
        output = compute(x, g, beta, layout)
        rows = torch.tensor([row for path in layout.plan.sample_nodes for row in path])
        return output.index_select(1, rows)
    outputs = []
    for path in layout.plan.sample_nodes:
        rows = torch.tensor(path)
        convolved = _convolution(x.index_select(2, rows), weight)
        q, k, v = [t.transpose(1, 2).unsqueeze(2) for t in convolved.chunk(3, dim=1)]
        out, _ = _recurrence(
            q,
            k,
            v,
            g=g.index_select(1, rows),
            beta=beta.index_select(1, rows),
            cu_seqlens=torch.tensor([0, len(rows)]),
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("prefix", [2, 63, 64, 65, 129])
def test_gdn_non_aligned_nested_forks_preserve_output_and_every_gradient(prefix):
    common = list(range(prefix))
    paths = [common + [201, 202, 203], common + [201, 202, 204, 205], common + [301, 302]]
    plan = build_prefix_tree(paths)
    layout = RecurrentTreeLayout.from_plan(plan, device="cpu", convolution_kernel_size=4)
    generator = torch.Generator().manual_seed(33)
    n = len(plan.tokens)
    values = [
        torch.randn(1, 6, n, generator=generator, dtype=torch.float64) * 0.1,
        torch.randn(6, 4, generator=generator, dtype=torch.float64) * 0.1,
        torch.randn(1, n, 1, generator=generator, dtype=torch.float64) * 0.1,
        torch.randn(1, n, 1, generator=generator, dtype=torch.float64) * 0.1,
    ]
    actual_inputs = [value.clone().requires_grad_() for value in values]
    reference_inputs = [value.clone().requires_grad_() for value in values]
    actual = _run(*actual_inputs, layout, shared=True)
    reference = _run(*reference_inputs, layout, shared=False)
    coefficients = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    (actual * coefficients).sum().backward()
    (reference * coefficients).sum().backward()
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-10, atol=1e-12)


def _packed_convolution(x, weight, bias=None, activation=None, seq_idx=None):
    if seq_idx is None:
        return _convolution(x, weight, bias, activation)
    _, counts = torch.unique_consecutive(seq_idx[0], return_counts=True)
    return torch.cat([_convolution(part, weight, bias, activation) for part in x.split(counts.tolist(), dim=2)], dim=2)


def test_unshared_convolution_keeps_native_channel_last_layout_and_one_call():
    plan = build_prefix_tree([[1, 2, 3, 4], [1, 2, 5]], share_prefixes=False)
    layout = RecurrentTreeLayout.from_plan(plan, device="cpu", convolution_kernel_size=4)
    x = torch.randn(1, len(plan.tokens), 6, dtype=torch.float64).transpose(1, 2)
    weight = torch.randn(6, 4, dtype=torch.float64)
    calls = []

    def native(*, x, weight, bias, activation, seq_idx):
        calls.append(x.stride())
        assert x.stride(1) == 1
        return _packed_convolution(x, weight, bias, activation, seq_idx)

    output = tree_causal_convolution(x, weight, None, layout=layout, native=native, activation="silu")
    reference = _packed_convolution(x, weight, activation="silu", seq_idx=layout.convolution.sequence_ids)
    assert calls == [x.stride()]
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_hybrid_forward_and_recompute_keep_tree_metadata(checkpoint):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=8,
        intermediate_size=16,
        vocab_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 0],
        },
    )
    torch.manual_seed(31)
    model = Qwen3_5ForCausalLM(config).double().train()
    # CPU oracles stand in for optional CUDA kernels; the real provider's
    # projection, norm, residual, attention and checkpoint forwards are retained.
    recurrence = _recurrence
    old_module = recurrence.__module__
    recurrence.__module__ = "fla.cpu_oracle"
    try:
        for layer in model.model.layers[:3]:
            layer.linear_attn.causal_conv1d_fn = _packed_convolution
            layer.linear_attn.chunk_gated_delta_rule = recurrence
        enable_transformers_prefix_tree(model, recurrent_adapter=enable_gated_delta_prefix_tree)
    finally:
        recurrence.__module__ = old_module
    forward = model.model.layers[0].linear_attn.forward
    enable_transformers_prefix_tree(model, recurrent_adapter=enable_gated_delta_prefix_tree)
    assert model.model.layers[0].linear_attn.forward is forward
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sequences = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8]]
    targets = [[-100] * 3 + [9, 10, 11], [-100] * 3 + [12, 13]]
    reference = None
    for shared in [False, True]:
        model.zero_grad(set_to_none=True)
        plan = build_prefix_tree(sequences, share_prefixes=shared)
        batch = TreeSFTBatch.from_plan(plan, targets, device="cpu", recurrent_chunk_size=64, convolution_kernel_size=4)
        loss = batch.loss(model, shared=shared)
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        if reference is None:
            reference = (loss.detach(), gradients)
        else:
            torch.testing.assert_close(loss, reference[0], rtol=1e-7, atol=1e-8)
            assert gradients.keys() == reference[1].keys()
            for name, value in gradients.items():
                torch.testing.assert_close(value, reference[1][name], rtol=2e-5, atol=1e-8)
