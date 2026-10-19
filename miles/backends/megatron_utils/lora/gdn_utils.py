"""Row layout of LoRA B on a Gated DeltaNet ``in_proj``, in plain torch so it runs without Megatron."""


def is_gdn_in_proj(module) -> bool:
    return module.base_linear_name.rsplit(".", 1)[-1] == "in_proj" and hasattr(
        module.to_wrap.config, "linear_num_key_heads"
    )


def gdn_in_proj_split(config, local_rows: int) -> tuple[int, ...]:
    """Per-TP-shard row sizes of the q, k, v, z, b and a blocks of a Gated DeltaNet ``in_proj``.

    Each shard holds its heads' rows as ``[q; k; v; z; b; a]`` (Bridge's GDN layout), so the
    shards gather into one group per shard."""
    qk = config.linear_key_head_dim * config.linear_num_key_heads
    v = config.linear_value_head_dim * config.linear_num_value_heads
    heads = config.linear_num_value_heads
    total = 2 * qk + 2 * v + 2 * heads
    assert total % local_rows == 0, f"GDN in_proj adapter rows {local_rows} do not divide {total}"
    tp = total // local_rows
    return (qk // tp, qk // tp, v // tp, v // tp, heads // tp, heads // tp)


def gdn_gate_rows(module) -> slice | None:
    """Local rows of LoRA B that feed the GDN b and a gate projections, or ``None`` for other modules.

    HF PEFT and Tinker do not adapt ``in_proj_b`` and ``in_proj_a``, so these rows stay frozen at zero."""
    if not is_gdn_in_proj(module):
        return None
    local_rows = module.adapters[0].linear_out.weight.shape[-2]
    *_, b, a = gdn_in_proj_split(module.to_wrap.config, local_rows)
    return slice(local_rows - b - a, local_rows)
