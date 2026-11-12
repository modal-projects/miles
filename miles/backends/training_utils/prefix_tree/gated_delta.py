"""Opt-in native Gated DeltaNet kernel adapters; projections stay untouched.

The provider must expose callable convolution and chunk-recurrence handles on
each module. No model-name dispatch or copied provider forward is needed.
"""

from contextvars import ContextVar

from miles.backends.training_utils.prefix_tree.recurrent import execute_tree_levels

_ACTIVE = ContextVar("prefix_tree_gated_delta", default=None)
_OBSERVER = ContextVar("prefix_tree_gated_delta_observer", default=None)


def tree_causal_convolution(x, weight, bias, *, layout, native, activation):
    """Batch node halos while preserving native channel-last CUDA dispatch."""
    convolution = layout.convolution
    if convolution is None or convolution.kernel_size != weight.shape[-1]:
        raise ValueError("Prepare a convolution layout with the native kernel width")
    # Gather in NTC order, then expose the native NCT view. A direct NCT
    # index_select silently switches causal-conv1d to its channel-first kernel.
    gathered = x.transpose(1, 2).index_select(1, convolution.rows).transpose(1, 2)
    output = native(x=gathered, weight=weight, bias=bias, activation=activation, seq_idx=convolution.sequence_ids)
    return output.transpose(1, 2).index_select(1, convolution.restore_rows).transpose(1, 2)


def tree_gated_delta(query, key, value, *, g, beta, layout, native, **options):
    if layout.chunk_size != 64 or options.get("chunk_size", 64) != 64:
        raise ValueError("GDN execution maps must follow the validated native 64-token grid")
    if options.get("initial_state") is not None or options.get("output_final_state"):
        raise ValueError("Shared-prefix GDN is training-only and does not consume inference caches")
    options = {
        key: val for key, val in options.items() if key not in ("initial_state", "output_final_state", "cu_seqlens")
    }
    options["chunk_size"] = 64

    def step(q, k, v, decay, update, *, initial_state, cu_seqlens):
        return native(
            q,
            k,
            v,
            g=decay,
            beta=update,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            **options
        )

    output = execute_tree_levels(
        (query, key, value, g, beta), layout.recurrent_levels, layout.recurrent_restore_rows, step=step
    )
    return output, None


def _install_module(module, name):
    forward, convolution, recurrence = module.forward, module.causal_conv1d_fn, module.chunk_gated_delta_rule

    def wrapped_forward(*args, **kwargs):
        layout = kwargs.get("prefix_tree_recurrent") if kwargs.get("prefix_tree_shared", True) else None
        if kwargs.get("prefix_tree_layout") is not None and kwargs.get("prefix_tree_shared", True) and layout is None:
            raise ValueError("Prepare a recurrent tree layout before shared GDN execution")
        if kwargs.get("cache_params") is not None:
            raise ValueError("Tree GDN does not support inference caches")
        token = _ACTIVE.set(layout)
        observer = _OBSERVER.set(kwargs.get("prefix_tree_observer"))
        try:
            return forward(*args, **kwargs)
        finally:
            _OBSERVER.reset(observer)
            _ACTIVE.reset(token)

    def wrapped_convolution(x, weight, bias=None, activation=None, **kwargs):
        layout = _ACTIVE.get()
        if layout is None:
            output = convolution(x=x, weight=weight, bias=bias, activation=activation, **kwargs)
        else:
            output = tree_causal_convolution(x, weight, bias, layout=layout, native=convolution, activation=activation)
        if _OBSERVER.get() is not None:
            _OBSERVER.get()(name + ".convolution", output.transpose(1, 2))
        return output

    def wrapped_recurrence(query, key, value, *, g, beta, **kwargs):
        layout = _ACTIVE.get()
        if layout is None:
            output = recurrence(query, key, value, g=g, beta=beta, **kwargs)
        else:
            output = tree_gated_delta(query, key, value, g=g, beta=beta, layout=layout, native=recurrence, **kwargs)
        if _OBSERVER.get() is not None:
            _OBSERVER.get()(name + ".recurrence", output[0])
        return output

    module.forward = wrapped_forward
    module.causal_conv1d_fn = wrapped_convolution
    module.chunk_gated_delta_rule = wrapped_recurrence
    module._prefix_tree_gdn_enabled = True


def enable_gated_delta_prefix_tree(model):
    """Validate native per-module kernel capabilities before installing adapters."""
    kinds = tuple(model.config.layer_types)
    if any(kind not in ("linear_attention", "full_attention") for kind in kinds):
        raise ValueError("GDN adapter requires full-attention / linear-attention layers")
    modules = [(name, module) for name, module in model.named_modules() if hasattr(module, "chunk_gated_delta_rule")]
    if len(modules) != kinds.count("linear_attention"):
        raise ValueError("Every recurrent layer must expose the native chunk recurrence handle")
    enabled = [getattr(module, "_prefix_tree_gdn_enabled", False) for _, module in modules]
    if enabled and all(enabled):
        return
    if any(enabled):
        raise ValueError("GDN tree adapters must be installed consistently across recurrent layers")
    for _, module in modules:
        if not callable(getattr(module, "causal_conv1d_fn", None)) or not callable(module.chunk_gated_delta_rule):
            raise ValueError("GDN requires native convolution and chunk-recurrence kernels")
        if not module.chunk_gated_delta_rule.__module__.startswith("fla."):
            raise ValueError("GDN tree adapter currently validates the native FLA 64-token chunk backend")
    for name, module in modules:
        _install_module(module, name)
