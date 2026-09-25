import itertools
import logging

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config):
    metadata = {}

    validate_compact_rollout_ids(data)

    if args.async_keep_partial_groups_on_abort:
        data = _select_dp_aligned_prompt_groups(data, train_parallel_config["dp_size"])
        metadata["prompt_group_sizes"] = [_nested_sample_count(group) for group in data]
        data = list(itertools.chain.from_iterable(_flatten_group(group) for group in data))
        metadata["dynamic_global_batch_size"] = len(data)
        logger.info(
            "Collected %d samples from %d prompt groups with dynamic global batch size",
            len(data),
            len(metadata["prompt_group_sizes"]),
        )
        return data, metadata

    # flatten the data if it is a list of lists
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

    # Compact rollouts must not be trimmed by sample count; the schedule drops
    # whole trailing rollouts instead.
    is_compact = any(s.rollout_id is not None for s in data)

    if not args.disable_rollout_trim_samples and not is_compact:
        global_batch_size = args.global_batch_size
        if args.use_dynamic_global_batch_size:
            logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
            dynamic_global_batch_size = _compute_dynamic_global_batch_size(
                args, train_parallel_config=train_parallel_config, num_samples=len(data)
            )
            metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
            global_batch_size = dynamic_global_batch_size

        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            if trim_len == 0:
                raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
            origin_data_length = len(data)
            data = data[:trim_len]
            logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata


def validate_compact_rollout_ids(node, depth=0):
    """Require compact leaves (``list[Sample]`` at depth >= 2, >1 sibling) to
    share a non-None ``rollout_id``; default rollout shapes skip validation."""
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        validate_compact_rollout_ids(item, depth + 1)


def _first_sample(group):
    return _first_sample(group[0]) if isinstance(group[0], list) else group[0]


def _nested_sample_count(group) -> int:
    if not isinstance(group, list):
        return 1
    return sum(_nested_sample_count(item) for item in group)


def _flatten_group(group) -> list[Sample]:
    if isinstance(group, Sample):
        return [group]
    return list(itertools.chain.from_iterable(_flatten_group(item) for item in group))


def _select_dp_aligned_prompt_groups(groups, dp_size: int):
    """Keep the largest sample population made of complete prompt groups."""
    best_by_remainder: dict[int, tuple[int, list[int]]] = {0: (0, [])}
    for index, group in enumerate(groups):
        size = _nested_sample_count(group)
        next_best = dict(best_by_remainder)
        for total, selected in best_by_remainder.values():
            candidate_total = total + size
            remainder = candidate_total % dp_size
            current = next_best.get(remainder)
            if current is None or candidate_total > current[0]:
                next_best[remainder] = (candidate_total, [*selected, index])
        best_by_remainder = next_best

    selected_total, selected_indices = best_by_remainder[0]
    if selected_total == 0:
        raise ValueError(f"No complete prompt groups can form a batch aligned to dp_size={dp_size}")
    if len(selected_indices) != len(groups):
        logger.info(
            "Kept %d/%d complete prompt groups (%d/%d samples) for dp_size=%d alignment",
            len(selected_indices),
            len(groups),
            selected_total,
            sum(_nested_sample_count(group) for group in groups),
            dp_size,
        )
    return [groups[index] for index in selected_indices]


def _compute_dynamic_global_batch_size(args, train_parallel_config, num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps=1, wasted={wasted})"
        )

    return dynamic_gbs
