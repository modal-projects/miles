"""Change only mHC projection math, preserving the installed provider's RMS.

GLM and generic Megatron place epsilon differently. A diagnostic must delegate
normalization to the original operator rather than assume either equation.
"""

import torch


def projection_control(native, mode):
    if mode not in ("bmm", "fp64"):
        raise ValueError("Projection control must be bmm or fp64")

    def forward(x, weight, eps):
        if mode == "bmm" and weight.requires_grad:
            raise ValueError("The rowwise BMM control requires frozen projection weights")
        # Preserve both the provider's values and its normalization derivative.
        # The unused native projection is intentional in this numerical oracle.
        _, reciprocal = native(x, weight, eps)
        if mode == "bmm":
            projected = torch.bmm(x.unsqueeze(1), weight.t().unsqueeze(0).expand(len(x), -1, -1)).squeeze(1)
        else:
            projected = (x.double() @ weight.double().t()).to(x.dtype)
        return projected, reciprocal

    return forward
