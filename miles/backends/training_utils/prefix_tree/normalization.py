"""Explicit RMS arithmetic for the opt-in Megatron prefix kernel policy."""

import torch

from miles.backends.training_utils.prefix_tree.operators import rowwise_frozen_linear


def install_row_math(model):
    # Optional Transformer Engine is only needed in a Megatron GPU worker.
    import transformer_engine.pytorch as te

    for module in model.modules():
        if isinstance(module, te.RMSNorm):

            def forward(x, norm=module):
                value = x.float()
                gamma = norm.weight.float()
                if norm.zero_centered_gamma:
                    gamma = gamma + 1
                value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.eps)
                return (value * gamma).to(x.dtype)

            module.forward = forward
        if hasattr(module, "_proj_rms_op"):  # config-access-exempt: only mHC modules expose the projection operator
            if module._proj_rms_compute_h_op is not None:
                raise ValueError("Prefix row math requires an unfused projection/RMS operator")
            module._proj_rms_op = _projection(module._proj_rms_op)
    unfuse_rmsnorm_linears(model)


def _projection(native):
    def forward(x, weight, eps):
        if weight.requires_grad:
            raise ValueError("Prefix row projection requires frozen base weights")
        # Delegate RMS values and derivatives to the provider: its epsilon
        # placement is part of the architecture contract.
        _, reciprocal = native(x, weight, eps)
        return rowwise_frozen_linear(x, weight), reciprocal

    return forward


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
        if (
            module.config.fp8 or getattr(module.config, "fp4", False) or module.config.tp_comm_overlap
        ):  # config-access-exempt: fp4 is absent on older TE configs
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
