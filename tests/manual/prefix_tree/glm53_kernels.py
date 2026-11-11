"""Installed FLA tree correctness probe, including forks inside a kernel chunk."""

import json

import torch

from miles.backends.training_utils.prefix_tree.glm53.kda import tree_kda
from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def difference(reference, value):
    reference, value = reference.float().flatten(), value.float().flatten()
    return float((reference - value).norm() / reference.norm().clamp_min(1e-20))


def run_case(prefix):
    torch.manual_seed(42)
    sequences = [list(range(prefix)) + list(range(1000 + b * 100, 1017 + b * 100)) for b in range(3)]
    plan = build_prefix_tree(sequences)
    layout = GlmTreeLayout.from_plan(plan, device="cuda")
    heads, width = 2, 128
    x = (
        torch.randn(1, len(plan.tokens), heads * width * 3, device="cuda", dtype=torch.bfloat16) * 0.2
    ).requires_grad_()
    g = (-torch.rand(1, len(plan.tokens), heads, width, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    beta = torch.rand(1, len(plan.tokens), heads, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = (torch.randn(heads * width * 3, 4, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    values = [x, g, beta, weight]
    rows = torch.tensor([row for path in plan.sample_nodes for row in path], device="cuda")
    expanded = build_prefix_tree(sequences, share_prefixes=False)

    def evaluate(shared):
        leaves = [v.detach().clone().requires_grad_() for v in values]
        if shared:
            out = tree_kda(*leaves, layout)
            out = out.index_select(1, rows)
        else:
            from fla.modules.conv.causal_conv1d import causal_conv1d
            from fla.ops.kda import chunk_kda

            conv, _ = causal_conv1d(
                leaves[0].index_select(1, rows),
                leaves[3],
                activation="silu",
                backend="triton",
                cu_seqlens=torch.tensor(expanded.segment_offsets, device="cuda", dtype=torch.int32),
            )
            q, k, v = [a.unflatten(-1, (heads, width)) for a in conv.chunk(3, -1)]
            out, _ = chunk_kda(
                q,
                k,
                v,
                g=leaves[1].index_select(1, rows),
                beta=leaves[2].index_select(1, rows),
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.tensor(expanded.segment_offsets, device="cuda", dtype=torch.int32),
            )
        # Distinct suffix adjoints plus shared prompt losses.
        scales = torch.linspace(0.5, 1.5, out.shape[1], device="cuda")[None, :, None, None]
        (out.float().square() * scales).sum().backward()
        return out.detach(), [v.grad.detach() for v in leaves]

    expected, expected_grads = evaluate(False)
    actual, actual_grads = evaluate(True)
    report = {
        "prefix": prefix,
        "outputs": difference(expected, actual),
        "gradients": [difference(a, b) for a, b in zip(expected_grads, actual_grads, strict=True)],
    }
    print(json.dumps(report), flush=True)
    assert report["outputs"] < 0.03, report
    assert max(report["gradients"]) < 0.04, report
    return report


def main():
    return [run_case(prefix) for prefix in (3, 127, 128, 129, 257)]


if __name__ == "__main__":
    main()
