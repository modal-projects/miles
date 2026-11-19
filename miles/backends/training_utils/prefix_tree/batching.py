"""Bound both model activations and restored loss rows before prefix execution."""

from miles.backends.training_utils.prefix_tree.packed import prepare_packed_tree
from miles.utils.seqlen_balancing import expand_bins_by_splitting


def split_prefix_microbatches(bins, sequences, slots, *, expanded_limit, compact_limit, alignment, pad_size):
    if min(expanded_limit, compact_limit, alignment, pad_size) < 1:
        raise ValueError("Prefix token budgets and alignments must be positive")
    lengths = list(map(len, sequences))
    pending = [list(indices) for indices in reversed(bins)]
    bounded = []
    while pending:
        indices = sorted(pending.pop(), key=lambda index: slots[index])
        expanded = sum(lengths[index] for index in indices)
        padded = (expanded + pad_size - 1) // pad_size * pad_size
        if padded <= expanded_limit:
            transport = prepare_packed_tree(
                [sequences[index] for index in indices],
                [slots[index] for index in indices],
                # Reserve any possible native padding so subsequent DP bin splitting remains safe.
                padded_rows=expanded + pad_size - 1,
                alignment=alignment,
                num_adapters=max(slots[index] for index in indices) + 1,
            )
            if len(transport.plan.tokens) <= compact_limit:
                bounded.append(indices)
                continue
        if len(indices) == 1:
            raise ValueError(
                f"Sample {indices[0]} cannot fit prefix token budgets: "
                f"expanded_limit={expanded_limit}, compact_limit={compact_limit}, tokens={expanded}"
            )
        halves = [indices]
        expand_bins_by_splitting(halves, 2, lengths)
        pending.extend(reversed(halves))
    return bounded
