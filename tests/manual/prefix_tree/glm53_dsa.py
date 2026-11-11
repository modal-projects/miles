"""Native GLM kpool/SparseMLA equivalence independent of the full model."""

import json
from types import SimpleNamespace

import torch

import miles_plugins.megatron_bridge  # noqa: F401
from miles.backends.training_utils.prefix_tree.glm53.bridge import _select_segment
from miles.backends.training_utils.prefix_tree.glm53.dsa import tree_kpool_select
from miles.backends.training_utils.prefix_tree.glm53.layout import GlmTreeLayout
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def main():
    # Register the isolated provider before resolving its optional modules.
    from megatron.bridge.models.glm5.tilelang import SparseMLA
    from megatron.bridge.models.glm5_next.kpool_indexer import build_pooled_keys, kpool_select_topk, pool_boundaries

    torch.manual_seed(42)
    sequences = [list(range(129)) + list(range(200 + b * 20, 217 + b * 20)) for b in range(3)]
    plan = build_prefix_tree(sequences)
    layout = GlmTreeLayout.from_plan(plan, device="cuda")
    n = len(plan.tokens)
    rows = torch.tensor([r for p in plan.sample_nodes for r in p], device="cuda")
    cu = torch.tensor([0, 146, 292, 438], device="cuda", dtype=torch.int32)
    module = SimpleNamespace(
        index_kpool=4,
        index_topk=64,
        index_kpool_compress_ape=torch.randn(4, 128, device="cuda", dtype=torch.float32) * 0.1,
    )
    iq = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
    hw = torch.randn(n, 4, device="cuda", dtype=torch.float32) * 0.1
    gates = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
    pooled = build_pooled_keys(ik[rows], gates[rows], module.index_kpool_compress_ape, cu, 4)
    expanded_selection = kpool_select_topk(iq[rows], pooled, hw[rows], cu, pool_boundaries(cu, 4), 64, 4)
    selected = tree_kpool_select(
        iq,
        ik,
        hw,
        gates,
        layout,
        select=lambda q, k, w, g, query_start: _select_segment(module, q, k, w, g, query_start=query_start),
    )
    expected = torch.where(expanded_selection >= 0, rows[expanded_selection.clamp_min(0).long()], -1)
    torch.testing.assert_close(
        selected[rows].sort(-1).values, expected.sort(-1).values.to(selected.dtype), rtol=0, atol=0
    )
    query = torch.randn(n, 4, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn(n, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    query[..., 512:] = 0
    key[..., 512:] = 0

    def evaluate(shared):
        q = query.detach().clone().requires_grad_()
        k = key.detach().clone().requires_grad_()
        if shared:
            output, _ = SparseMLA.apply(q, k, selected, 128**-0.5)
            output = output[rows]
        else:
            output, _ = SparseMLA.apply(q[rows], k[rows], expanded_selection, 128**-0.5)
        output.float().square().sum().backward()
        return output.detach(), q.grad, k.grad

    reference = evaluate(False)
    actual = evaluate(True)
    errors = [
        float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-20))
        for a, b in zip(reference, actual, strict=True)
    ]
    print(
        "GLM_TREE_DSA",
        json.dumps({"outputs_qgrad_kgrad_relative_l2": errors, "selection_sets_exact": True}),
        flush=True,
    )
    assert errors[0] == 0.0, errors
    assert max(errors[1:]) < 0.01, errors


if __name__ == "__main__":
    main()
