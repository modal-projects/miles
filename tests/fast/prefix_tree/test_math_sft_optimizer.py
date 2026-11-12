import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).parents[2] / "manual/prefix_tree/math_sft_optimizer.py"
_SPEC = importlib.util.spec_from_file_location("_math_sft_optimizer", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
MasterOptimizer = _MODULE.MasterOptimizer


def test_fp32_accumulation_preserves_a_bf16_rounding_tie_and_adamw_momentum():
    model = torch.nn.Module()
    model.register_parameter("adapter", torch.nn.Parameter(torch.tensor([1.5], dtype=torch.bfloat16)))
    actual = MasterOptimizer(model, lr=0.01)
    shadow = torch.nn.Parameter(model.adapter.detach().float().clone())
    reference = torch.optim.AdamW([shadow], lr=0.01, weight_decay=0.0, fused=False)
    for contributions in ([1.0, 1 / 256], [0.5, -0.25]):
        actual.clear(model)
        for coefficient in contributions:
            (model.adapter.float() * coefficient).sum().backward()
            actual.accumulate(model)
        # A BF16 accumulation would lose 1/256 at the first tie. The independent
        # FP32 optimizer receives the exact sum of these representable adjoints.
        expected = sum(contributions)
        assert actual.buffers[0].item() == expected
        shadow.grad = torch.tensor([expected])
        reference.step()
        counterfactual = actual.reference_step({"adapter": torch.tensor([expected])})
        torch.testing.assert_close(counterfactual[0], shadow, atol=0, rtol=0)
        actual.step()
        torch.testing.assert_close(actual.masters[0], shadow, atol=0, rtol=0)
        torch.testing.assert_close(model.adapter, shadow.detach().bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(
            actual.optimizer.state[actual.masters[0]]["exp_avg"], reference.state[shadow]["exp_avg"]
        )
    assert actual.optimizer.state[actual.masters[0]]["step"].item() == 2
    snapshot = actual.snapshot()
    actual.clear(model)
    assert snapshot["adapter"].item() == 0.25
