"""Exercise tree state ownership/backward with independent FP64 kernel oracles."""

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.glm53.kda import tree_kda
from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def _convolution(x, weight, *, bias, initial_state, output_final_state, cu_seqlens, **kwargs):
    outputs, states = [], []
    channels, kernel = weight.shape
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        value = x[:, start:end].transpose(1, 2)
        history = initial_state[index : index + 1] if initial_state is not None else x.new_zeros(1, channels, kernel)
        full = torch.cat((history, value), dim=-1)
        out = F.conv1d(full[:, :, -(end - start + kernel - 1) :], weight[:, None], bias=bias, groups=channels)
        outputs.append(F.silu(out).transpose(1, 2))
        states.append(full[:, :, -kernel:])
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def _recurrence(q, k, v, *, g, beta, initial_state, cu_seqlens, **kwargs):
    outputs, states = [], []
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        state = (
            initial_state[index] if initial_state is not None else q.new_zeros(q.shape[-2], q.shape[-1], v.shape[-1])
        )
        for t in range(int(start), int(end)):
            state = (
                g[0, t].exp()[..., None] * state
                + beta[0, t, :, None, None] * k[0, t, :, :, None] * v[0, t, :, None, :]
            )
            outputs.append(torch.einsum("hd,hdv->hv", q[0, t], state))
        states.append(state)
    return torch.stack(outputs)[None], torch.stack(states)


@pytest.mark.parametrize("prefix", [0, 129])
def test_nested_parent_state_gradients_match_expansion(monkeypatch, prefix):
    for name, function_name, function in (
        ("fla.modules.conv.causal_conv1d", "causal_conv1d", _convolution),
        ("fla.ops.kda", "chunk_kda", _recurrence),
    ):
        module = ModuleType(name)
        setattr(module, function_name, function)
        monkeypatch.setitem(sys.modules, name, module)
    torch.manual_seed(4)
    plan = build_prefix_tree(
        [list(range(100, 100 + prefix)) + p for p in [[1, 2, 3, 4, 5], [1, 2, 3, 6], [1, 2, 7, 8], [9, 10]]]
    )
    n = len(plan.tokens)
    values = [
        torch.randn(1, n, 12, dtype=torch.float64),
        -torch.rand(1, n, 2, 2, dtype=torch.float64),
        torch.rand(1, n, 2, dtype=torch.float64),
        torch.randn(12, 4, dtype=torch.float64),
    ]
    rows = torch.tensor([r for path in plan.sample_nodes for r in path])

    def evaluate(shared):
        x, g, beta, weight = [value.clone().requires_grad_() for value in values]
        if shared:
            out = tree_kda(x, g, beta, weight, GlmTreeLayout.from_plan(plan, device="cpu")).index_select(1, rows)
        else:
            cu = torch.tensor([0, *torch.tensor([len(p) for p in plan.sample_nodes]).cumsum(0).tolist()])
            conv, _ = _convolution(
                x.index_select(1, rows), weight, bias=None, initial_state=None, output_final_state=False, cu_seqlens=cu
            )
            q, k, v = [part.unflatten(-1, (2, 2)) for part in conv.chunk(3, dim=-1)]
            out, _ = _recurrence(
                q, k, v, g=g.index_select(1, rows), beta=beta.index_select(1, rows), initial_state=None, cu_seqlens=cu
            )
        out.square().sum().backward()
        return out, [value.grad for value in (x, g, beta, weight)]

    expected, reference_grads = evaluate(False)
    actual, actual_grads = evaluate(True)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for reference, grad in zip(reference_grads, actual_grads, strict=True):
        torch.testing.assert_close(grad, reference, rtol=1e-12, atol=1e-12)
