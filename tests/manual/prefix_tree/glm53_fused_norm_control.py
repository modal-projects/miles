"""Diagnostic unfused RMSNorm + native TE GEMM, preserving parameter objects.

No upstream source is copied or edited. Reuse TE's Linear.forward implementation
on the installed LayerNormLinear's common linear state; this test-only adapter
is guarded to unquantized, non-overlapped column linears and is not a production
ModuleSpec replacement. Explicit RMS math makes its precision independently
controllable, including the returned normalized input consumed by native LoRA.
"""

import torch


def unfuse_rmsnorm_linears(model, precision="fp32"):
    import transformer_engine.pytorch as te
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    dtype = {"fp32": torch.float32, "fp64": torch.float64}[precision]
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, te.LayerNormLinear):
            continue
        if module.normalization != "RMSNorm" or module.parallel_mode != "column":
            raise ValueError("Control requires RMSNorm column-parallel LayerNormLinear")
        if module.config.fp8 or getattr(module.config, "fp4", False) or module.config.tp_comm_overlap:
            raise ValueError("Control requires unquantized non-overlapped TE execution")
        # Linear.forward uses this option; LayerNormLinear does not expose it.
        module.save_original_input = False
        native = module.forward

        def forward(x, base=module):
            value = x.to(dtype)
            gamma = base.layer_norm_weight.to(dtype)
            if base.zero_centered_gamma:
                gamma = gamma + 1
            normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + base.eps) * gamma).to(x.dtype)
            first = None if base.disable_parameter_transpose_cache else base.is_first_microbatch
            result = te.Linear.forward(base, normalized, is_first_microbatch=first)
            base.is_first_microbatch = False
            if base.return_bias:
                projected, bias = result
            else:
                projected, bias = result, None
            if not base.return_layernorm_output:
                return projected, bias
            returned = normalized
            if base.sequence_parallel and base.return_layernorm_output_gathered:
                returned = gather_from_sequence_parallel_region(returned, group=base.tp_group)
            if base.te_return_bias:
                return projected, bias, returned
            return (projected, returned), None

        module.forward = forward
        replacements.append((name, module, native))
    return replacements
