"""Differentiable KDA tree execution with native chunk-aligned state sharing."""

from miles.backends.training_utils.prefix_tree.glm53.layout import KDA_CHUNK_SIZE, GlmTreeLayout
from miles.backends.training_utils.prefix_tree.recurrent import execute_tree_levels


def tree_kda(mixed_qkv, g, beta, conv_weight, layout: GlmTreeLayout, *, conv_bias=None):
    # Optional GPU dependency: importing prefix planning must work on CPU hosts.
    from fla.ops.kda import chunk_kda

    heads, width = g.shape[-2:]
    convolved = _tree_convolution(mixed_qkv, conv_weight, layout, bias=conv_bias)
    query, key, value = (part.unflatten(-1, (heads, width)) for part in convolved.chunk(3, dim=-1))

    def step(q, k, v, decay, update, *, initial_state, cu_seqlens):
        return chunk_kda(
            q,
            k,
            v,
            g=decay,
            beta=update,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            chunk_size=KDA_CHUNK_SIZE,
        )

    return execute_tree_levels(
        (query, key, value, g, beta), layout.recurrent_levels, layout.recurrent_restore_rows, step=step
    )


def _tree_convolution(x, weight, layout, *, bias):
    from fla.modules.conv.causal_conv1d import causal_conv1d

    def step(value, *, initial_state, cu_seqlens):
        return causal_conv1d(
            value,
            weight,
            bias=bias,
            initial_state=initial_state,
            output_final_state=True,
            activation="silu",
            backend="triton",
            cu_seqlens=cu_seqlens,
        )

    return execute_tree_levels((x,), layout.levels, layout.restore_rows, step=step)
