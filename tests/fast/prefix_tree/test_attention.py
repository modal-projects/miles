import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.attention import TreeAttentionLayout, tree_attention
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


def test_float64_outputs_and_every_qkv_gradient_against_independent_expanded_attention():
    torch.manual_seed(41)
    sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 9], [1, 2], [1, 2, 3, 4, 5], [8, 7]]
    plan = build_prefix_tree(sequences)
    actual = [torch.randn(len(plan.tokens), h, 8, dtype=torch.float64, requires_grad=True) for h in [4, 2, 2]]
    expected = [tensor.detach().clone().requires_grad_() for tensor in actual]
    output = tree_attention(*actual, TreeAttentionLayout.from_plan(plan, "cpu"), scaling=8**-0.5)
    actual_terms, expected_terms = [], []
    for nodes in plan.sample_nodes:
        q, k, v = [tensor[list(nodes)].transpose(0, 1) for tensor in expected]
        expanded = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, 0),
            v.repeat_interleave(2, 0),
            is_causal=True,
        ).transpose(0, 1)
        torch.testing.assert_close(output[list(nodes)], expanded, atol=1e-12, rtol=1e-12)
        coefficients = torch.randn_like(expanded)
        actual_terms.append((output[list(nodes)] * coefficients).sum())
        expected_terms.append((expanded * coefficients).sum())
    sum(actual_terms).backward()
    sum(expected_terms).backward()
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
