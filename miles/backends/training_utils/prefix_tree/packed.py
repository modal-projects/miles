"""Transport maps between compact model rows and unchanged packed RL losses."""

from dataclasses import dataclass

from miles.backends.training_utils.prefix_tree.plan import PrefixTreePlan, build_prefix_tree


@dataclass(frozen=True)
class PackedTreePlan:
    plan: PrefixTreePlan
    restore_rows: tuple[int, ...]
    adapter_counts: tuple[int, ...]


def prepare_packed_tree(sequences, slots, *, padded_rows, alignment, num_adapters, share_prefix=True):
    if alignment < 1 or num_adapters < 1 or not sequences or len(sequences) != len(slots):
        raise ValueError("Invalid packed prefix layout")
    if list(slots) != sorted(slots) or any(slot < 0 or slot >= num_adapters for slot in slots):
        raise ValueError("Packed prefix samples must be sorted by valid adapter slots")
    sequences = [tuple(sequence) for sequence in sequences]
    slots = list(slots)
    padding = padded_rows - sum(map(len, sequences))
    if padding < 0:
        raise ValueError("Packed row inventory is smaller than the original inputs")
    keys = [(slot, "sample" if share_prefix else index) for index, slot in enumerate(slots)]
    if padding:
        sequences.append((0,) * padding)
        slots.append(slots[-1])
        keys.append((slots[-1], "native_padding"))
    original_sequences = len(sequences)
    plan = build_prefix_tree(sequences, sharing_keys=keys)
    extra = -len(plan.tokens) % alignment
    if extra:
        sequences.append((0,) * extra)
        slots.append(slots[-1])
        keys.append((slots[-1], "compact_alignment"))
        plan = build_prefix_tree(sequences, sharing_keys=keys)
    owners = [-1] * len(plan.tokens)
    for slot, path in zip(slots, plan.sample_nodes, strict=True):
        for row in path:
            if owners[row] not in (-1, slot):
                raise ValueError("A compact row cannot belong to different adapters")
            owners[row] = slot
    if owners != sorted(owners):
        raise ValueError("Compact adapter rows must remain contiguous")
    restore = tuple(row for path in plan.sample_nodes[:original_sequences] for row in path)
    if len(restore) != padded_rows or len(plan.tokens) % alignment:
        raise ValueError("Packed prefix transport failed its row inventory check")
    return PackedTreePlan(plan, restore, tuple(owners.count(slot) for slot in range(num_adapters)))
