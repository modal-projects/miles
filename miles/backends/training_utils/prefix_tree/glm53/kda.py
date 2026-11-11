"""Differentiable KDA tree execution with native chunk-aligned state sharing."""

import torch

from miles.backends.training_utils.prefix_tree.glm53.layout import KDA_CHUNK_SIZE, GlmTreeLayout


def tree_kda(mixed_qkv, g, beta, conv_weight, layout: GlmTreeLayout, *, conv_bias=None):
    # Optional GPU dependency: importing prefix planning must work on CPU hosts.
    from fla.ops.kda import chunk_kda

    heads, width = g.shape[-2:]
    convolved = _tree_convolution(mixed_qkv, conv_weight, layout, bias=conv_bias)
    query, key, value = (part.unflatten(-1, (heads, width)) for part in convolved.chunk(3, dim=-1))
    states, outputs = {}, []
    for level in layout.recurrent_levels:
        initial = None
        if level.parents[0] >= 0:
            initial = torch.cat([states[p] for p in level.parents], dim=0)
        out, final = chunk_kda(
            query.index_select(1, level.rows),
            key.index_select(1, level.rows),
            value.index_select(1, level.rows),
            g=g.index_select(1, level.rows),
            beta=beta.index_select(1, level.rows),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=level.cu_seqlens,
            chunk_size=KDA_CHUNK_SIZE,
        )
        for index, segment in enumerate(level.segments):
            states[segment] = final[index : index + 1]
        outputs.append(out)
    return torch.cat(outputs, dim=1).index_select(1, layout.recurrent_restore_rows)


def _tree_convolution(x, weight, layout, *, bias):
    from fla.modules.conv.causal_conv1d import causal_conv1d

    states, outputs = {}, []
    for level in layout.levels:
        initial = None
        if level.parents[0] >= 0:
            initial = torch.cat([states[p] for p in level.parents], dim=0)
        output, final = causal_conv1d(
            x.index_select(1, level.rows),
            weight,
            bias=bias,
            initial_state=initial,
            output_final_state=True,
            activation="silu",
            backend="triton",
            cu_seqlens=level.cu_seqlens,
        )
        for index, segment in enumerate(level.segments):
            states[segment] = final[index : index + 1]
        outputs.append(output)
    return torch.cat(outputs, dim=1).index_select(1, layout.restore_rows)
