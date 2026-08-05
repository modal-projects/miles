from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def get_local_moe_layer_indices(models) -> list[int]:
    """Return the global layer indices of this rank's local MoE layers."""
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)
    return layer_indices


def register_replay_list_moe(
    replay_list,
    replay_data,
    *,
    models,
    global_layer_indices=None,
    **_kwargs,
):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    local_layer_indices = get_local_moe_layer_indices(models)

    if global_layer_indices is None:
        stream_indices = local_layer_indices
    else:
        global_layer_indices = [int(index) for index in global_layer_indices]
        if global_layer_indices != local_layer_indices:
            raise ValueError(
                "Routing replay shard does not match the local Megatron layers: "
                f"shard={global_layer_indices}, local={local_layer_indices}"
            )
        if replay_data.shape[1] != len(global_layer_indices):
            raise ValueError(
                "Routing replay shard has an invalid layer dimension: "
                f"shape={tuple(replay_data.shape)}, layers={global_layer_indices}"
            )
        stream_indices = range(len(global_layer_indices))

    if len(replay_list) != len(local_layer_indices):
        raise ValueError(
            f"Registered {len(replay_list)} routing replay streams for " f"{len(local_layer_indices)} local MoE layers"
        )

    for replay_idx, stream_idx in enumerate(stream_indices):
        layer_data = replay_data[:, stream_idx]
        replay_list[replay_idx].record(layer_data)
