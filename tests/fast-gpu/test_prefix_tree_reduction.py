"""Wide vocabulary restoration must not allocate an expanded FP32 adjoint."""

import torch
from tests.ci.ci_register import register_cuda_ci

from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather

register_cuda_ci(
    est_time=120,
    suite="stage-b-2-gpu-h200",
    labels=["lora", "megatron"],
    hardware=["hopper", "blackwell"],
    num_gpus=1,
)


def test_wide_gather_backward_bounds_peak_memory():
    rows, columns, copies = 8192, 4096, 4
    plan = OrderedRowGather.from_indices(list(range(rows)) * copies, rows, device="cuda")
    value = torch.zeros(rows, columns, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = plan.gather(value)
    gradient = torch.randint(-8, 9, output.shape, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (actual,) = torch.autograd.grad(output, value, gradient)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    assert extra < gradient.numel() * gradient.element_size(), extra
    expected = gradient.view(copies, rows, columns).sum(0, dtype=torch.float32).to(value.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print({"peak_extra_bytes": extra, "expanded_adjoint_bytes": gradient.numel() * gradient.element_size()})
