"""Test row-count-dependent GEMM dispatch with identical corresponding inputs.

Run each workspace/BLAS setting in a fresh process. This isolates the operator
from prefix planning, attention, loss, and backward. No weights are updated.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from miles.backends.training_utils.prefix_tree.operators import rowwise_frozen_linear, rowwise_linear


def measured(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        output = fn()
    kernels = sorted({kernel.name for event in profile.events() for kernel in event.kernels})
    kernels = sorted(
        set(kernels)
        | {event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
    )
    return output, kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-rows", type=int, default=15023)
    parser.add_argument("--compact-rows", type=int, default=3854)
    parser.add_argument("--inner", type=int, default=16384)
    parser.add_argument("--outputs", type=int, default=24)
    parser.add_argument("--blas", choices=("default", "cublas"), default="default")
    parser.add_argument("--rowwise", action="store_true")
    parser.add_argument("--trainable", action="store_true")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--compare-policies", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.compare_policies:
        rows = []
        for label, extra, workspace in [
            ("default", [], None),
            ("no_workspace", [], "0"),
            ("legacy_blas", ["--blas", "cublas"], None),
            ("rowwise", ["--rowwise"], None),
        ]:
            env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
            if workspace is not None:
                env["CUBLASLT_WORKSPACE_SIZE"] = workspace
            command = [
                sys.executable,
                __file__,
                "--expanded-rows",
                str(args.expanded_rows),
                "--compact-rows",
                str(args.compact_rows),
                "--inner",
                str(args.inner),
                "--outputs",
                str(args.outputs),
                "--dtype",
                args.dtype,
                *(["--trainable"] if args.trainable else []),
                *extra,
            ]
            child = subprocess.run(command, env=env, text=True, capture_output=True)
            if child.returncode:
                raise RuntimeError(child.stderr[-4000:])
            row = {"policy": label, **json.loads(child.stdout.splitlines()[-1])}
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2))
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.blas != "default":
        torch.backends.cuda.preferred_blas_library(args.blas)
    torch.manual_seed(71)
    compact = torch.randn(args.compact_rows, args.inner, device="cuda", dtype=torch.bfloat16).float()
    weights = torch.randn(args.outputs, args.inner, device="cuda") * 0.01
    if args.dtype == "bf16":
        compact = compact.bfloat16()
        weights = weights.bfloat16()
    weights.requires_grad_(args.trainable)
    rows = torch.arange(args.expanded_rows, device="cuda") % args.compact_rows
    expanded = compact.index_select(0, rows)

    def project(x):
        if args.rowwise:
            return rowwise_linear(x, weights) if args.trainable else rowwise_frozen_linear(x, weights)
        return x @ weights.t()

    # Warm lazy library loading outside the kernel inventory.
    project(expanded)
    project(compact)
    reference, expanded_kernels = measured(lambda: project(expanded))
    unique, compact_kernels = measured(lambda: project(compact))
    actual = unique.index_select(0, rows)
    delta = reference.double() - actual.double()
    gradient_checks = None
    if args.gradient_check:
        x = compact.detach().clone().requires_grad_()
        w = weights.detach().clone().requires_grad_()
        a, b = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        actual_output = rowwise_linear(x, w)
        expected_output = torch.nn.functional.linear(a, b)
        torch.testing.assert_close(actual_output, expected_output, atol=0.02, rtol=0.02)
        upstream = torch.randn_like(actual_output)
        (actual_output.float() * upstream.float()).sum().backward()
        (expected_output.float() * upstream.float()).sum().backward()
        gradient_checks = [
            float((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
            for got, expected in [(x.grad, a.grad), (w.grad, b.grad)]
        ]
    print(
        json.dumps(
            {
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "workspace": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
                "gradient_relative_l2": gradient_checks,
                "bitwise_equal": torch.equal(reference, actual),
                "relative_l2": float(delta.norm() / reference.double().norm()),
                "max_abs": float(delta.abs().max()),
                "expanded_kernels": expanded_kernels,
                "compact_kernels": compact_kernels,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
