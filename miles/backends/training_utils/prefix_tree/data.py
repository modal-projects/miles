"""Build fixed, token-exact recorded SFT microbatches from recorded model samples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTMicrobatch:
    sequences: tuple
    targets: tuple
    weights: tuple
    slots: tuple
    computation_keys: tuple
    prompt_indices: tuple
    real_input_tokens: int
    supervised_tokens: int
    padding_tokens: int


def global_token_scale(local_supervised_tokens, global_supervised_tokens, dp_size):
    if min(local_supervised_tokens, global_supervised_tokens, dp_size) < 1:
        raise ValueError("SFT token counts and DP size must be positive")
    return dp_size * local_supervised_tokens / global_supervised_tokens


def _make_batch(prompts, groups, index_slots, *, sp_multiple, pad_id):
    sequences, targets, weights, keys, slots = [], [], [], [], []
    for index, slot in sorted(index_slots, key=lambda item: item[1]):
        shared_prompt = list(prompts[index]["prompt_ids"])
        prompt = shared_prompt
        if not prompt:
            raise ValueError("A recorded prompt is empty")
        shared_length = len(prompt) // sp_multiple * sp_multiple
        for sample in sorted(groups[index], key=lambda row: row["member"]):
            prompt = list(sample.get("prompt_ids", shared_prompt))
            if prompt[: len(shared_prompt)] != shared_prompt:
                raise ValueError("Per-sample prompt must start with the recorded shared prompt")
            full = prompt + list(sample["completion_ids"])
            sequences.append(full[:-1])
            targets.append(full[1:])
            weights.append([0.0] * (len(prompt) - 1) + [1.0] * len(sample["completion_ids"]))
            keys.append(
                [(index, None)] * shared_length + [(index, sample["member"])] * (len(full) - 1 - shared_length)
            )
            slots.append(slot)
    real_tokens = sum(map(len, sequences))
    padding = -real_tokens % sp_multiple
    sequences[-1] += [pad_id] * padding
    targets[-1] += [-100] * padding
    weights[-1] += [0.0] * padding
    keys[-1] += [keys[-1][-1]] * padding
    return SFTMicrobatch(
        sequences=tuple(map(tuple, sequences)),
        targets=tuple(map(tuple, targets)),
        weights=tuple(map(tuple, weights)),
        slots=tuple(slots),
        computation_keys=tuple(map(tuple, keys)),
        prompt_indices=tuple(index for index, _ in index_slots),
        real_input_tokens=real_tokens,
        supervised_tokens=int(sum(sum(values) for values in weights)),
        padding_tokens=padding,
    )


def sft_microbatches(
    rollouts,
    *,
    dp_rank,
    dp_size,
    num_slots=2,
    sp_multiple=2,
    pad_id=0,
    max_groups_per_microbatch=2,
    max_microbatch_tokens=32768
):
    if dp_size < 1:
        raise ValueError("Invalid parallel topology")
    prompts = {row["index"]: row for row in rollouts["prompts"]}
    groups = {index: [] for index in prompts}
    for sample in rollouts["samples"]:
        if not sample["completion_ids"]:
            raise ValueError("A recorded completion is empty")
        groups[sample["prompt_index"]].append(sample)
    counts = {len(samples) for samples in groups.values()}
    if len(counts) != 1 or 0 in counts or len(groups) % dp_size:
        raise ValueError("EP peers require equal numbers of prompt groups and samples")
    if not 0 <= dp_rank < dp_size or min(sp_multiple, num_slots, max_groups_per_microbatch, max_microbatch_tokens) < 1:
        raise ValueError("Invalid parallel topology, adapter capacity or token budget")
    costs = {
        i: sum(len(s.get("prompt_ids", prompts[i]["prompt_ids"])) + len(s["completion_ids"]) - 1 for s in samples)
        for i, samples in groups.items()
    }
    order = sorted(groups, key=lambda i: (-costs[i], i))
    cohorts = [
        tuple(order[step * dp_size + (rank if step % 2 == 0 else dp_size - rank - 1)] for rank in range(dp_size))
        for step in range(len(order) // dp_size)
    ]
    # Make the packing decision across all DP ranks, so every EP peer executes
    # exactly the same number of collectives. A long single group stays intact.
    batches, step = [], 0
    while step < len(cohorts):
        width = 1
        while width < max_groups_per_microbatch and step + width < len(cohorts):
            maximum = max(
                sum(costs[cohorts[j][rank]] for j in range(step, step + width + 1)) for rank in range(dp_size)
            )
            if maximum > max_microbatch_tokens:
                break
            width += 1
        indices = [(cohorts[j][dp_rank], j % num_slots) for j in range(step, step + width)]
        batches.append(_make_batch(prompts, groups, indices, sp_multiple=sp_multiple, pad_id=pad_id))
        step += width
    return tuple(batches)
