"""DeepSeek-V4's indexer top-k on a rank that scores one query row (a balanced context-parallel plan can leave it
one) against a key count off SCORE_ROW_ALIGN: the canonical selection matches torch's."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=30, suite="stage-c-2-gpu-mi350", labels=["precision"], nightly=True)

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.v4_indexer import indexer_topk  # noqa: E402

HEADS, DIM, TOPK, RATIO = 32, 128, 512, 4


@pytest.mark.parametrize("n_kv", [701, 1027])
def test_one_query_row_over_an_unaligned_key_count(n_kv):
    torch.manual_seed(0)
    q = torch.randn(1, 1, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n_kv, 1, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, 1, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    # the row's position sees every compressed key
    positions = torch.tensor([n_kv * RATIO - 1], device="cuda")
    picks = {
        backend: indexer_topk(
            q,
            k,
            weights,
            positions,
            None,
            compress_ratio=RATIO,
            index_topk=TOPK,
            topk_fn=get_dsa_topk_fn(backend),
            clean_logits=backend != "canonical",
        )
        for backend in ("torch", "canonical")
    }

    assert torch.equal(picks["canonical"].sort(dim=-1).values, picks["torch"].sort(dim=-1).values)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
