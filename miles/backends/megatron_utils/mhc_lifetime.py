"""CUDA Megatron compatibility for the hyperconnection norm-manager lifecycle."""

from functools import wraps


def release_mhc_norm_managers() -> None:
    # Megatron is an optional GPU dependency on hosts importing Miles utilities.
    from megatron.core.transformer import transformer_layer

    layer = getattr(
        transformer_layer, "HyperConnectionTransformerLayer", None
    )  # config-access-exempt: older Megatron versions do not have hyperconnection layers
    if layer is not None:
        _release_after_mlp(layer)


def _release_after_mlp(layer) -> None:
    original = layer._forward_post_mlp_with_fused_hyper_connection
    if original.__dict__.get("_miles_releases_norm_manager", False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            # Match TransformerLayer: group_offload already consumed this manager.
            self.mlp_norm_manager = None

    forward._miles_releases_norm_manager = True
    layer._forward_post_mlp_with_fused_hyper_connection = forward
