"""Bucketing for weight update: atomic grouping + size-bounded packing.

A unit is the HF-named tensors one training-side parameter converted into
(a plain weight, or weight + quant scales); units are indivisible.
"""

import dataclasses
from collections.abc import Iterable, Iterator

import torch


@dataclasses.dataclass(frozen=True)
class AtomicUpdateGroup:
    """HF-name suffixes whose tensors must land in the same load call."""

    key: str
    suffixes: tuple[str, ...]
    optional: bool = False


def assemble_atomic_update_groups(
    hf_param_units: Iterable[list[tuple[str, torch.Tensor]]],
    atomic_update_groups: list[AtomicUpdateGroup],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Merge units whose members match an AtomicUpdateGroup into one unit,
    keyed by the matching member's name prefix; other units pass through."""
    for group in atomic_update_groups:
        assert group.suffixes, f"Atomic update group {group.key} has no suffixes"
        assert all(group.suffixes), f"Atomic update group {group.key} contains empty suffix"
        assert len(set(group.suffixes)) == len(
            group.suffixes
        ), f"Atomic update group {group.key} contains duplicate suffixes"
    keys = [group.key for group in atomic_update_groups]
    assert len(set(keys)) == len(keys), f"Duplicate atomic update group keys: {keys}"

    pending: dict[tuple[str, str], list] = {}
    matched_group_keys: set[str] = set()
    for unit in hf_param_units:
        if not unit:
            continue
        matches = [
            (group, suffix_idx, suffix, name)
            for name, _tensor in unit
            for group in atomic_update_groups
            for suffix_idx, suffix in enumerate(group.suffixes)
            if name.endswith(suffix)
        ]
        assert len(matches) <= 1, f"Unit {[n for n, _ in unit]} matches multiple atomic group suffixes"
        if not matches:
            yield unit
            continue
        group, suffix_idx, suffix, name = matches[0]
        matched_group_keys.add(group.key)
        prefix = name[: -len(suffix)]
        slots = pending.setdefault((prefix, group.key), [None] * len(group.suffixes))
        assert slots[suffix_idx] is None, f"Duplicate member in atomic update group: {name}"
        slots[suffix_idx] = unit
        if None not in slots:
            yield [named for slot in slots for named in slot]
            del pending[(prefix, group.key)]

    assert not pending, f"Incomplete atomic update groups at end of stream: {sorted(pending)}"
    for group in atomic_update_groups:
        assert (
            group.optional or group.key in matched_group_keys
        ), f"Atomic update group {group.key} matched no params (suffixes {group.suffixes})"


def pack_units_by_size(
    hf_param_units: Iterable[list[tuple[str, torch.Tensor]]],
    max_bytes: int,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Pack units into buckets <= max_bytes, never splitting a unit."""
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    for unit in hf_param_units:
        unit_size = sum(tensor.nbytes for _name, tensor in unit)
        if bucket and bucket_size + unit_size >= max_bytes:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.extend(unit)
        bucket_size += unit_size
    if bucket:
        yield bucket


def assemble_atomic_update_group_views(
    view_units: Iterable[dict[str, list[tuple[str, torch.Tensor]]]],
    atomic_update_groups: list[AtomicUpdateGroup],
) -> Iterator[dict[str, list[tuple[str, torch.Tensor]]]]:
    """Apply :func:`assemble_atomic_update_groups` coherently across weight views."""
    for group in atomic_update_groups:
        assert group.suffixes, f"Atomic update group {group.key} has no suffixes"
        assert all(group.suffixes), f"Atomic update group {group.key} contains empty suffix"
        assert len(set(group.suffixes)) == len(
            group.suffixes
        ), f"Atomic update group {group.key} contains duplicate suffixes"
    keys = [group.key for group in atomic_update_groups]
    assert len(set(keys)) == len(keys), f"Duplicate atomic update group keys: {keys}"

    pending: dict[tuple[str, str], list] = {}
    matched_group_keys: set[str] = set()
    expected_views: tuple[str, ...] | None = None
    for unit_by_view in view_units:
        views = tuple(unit_by_view)
        if expected_views is None:
            expected_views = views
        assert views == expected_views, f"Weight views changed within the stream: {views} != {expected_views}"
        if not any(unit_by_view.values()):
            continue

        view_matches = {}
        for view, unit in unit_by_view.items():
            matches = [
                (group, suffix_idx, suffix, name)
                for name, _tensor in unit
                for group in atomic_update_groups
                for suffix_idx, suffix in enumerate(group.suffixes)
                if name.endswith(suffix)
            ]
            assert len(matches) <= 1, f"Unit {[n for n, _ in unit]} matches multiple atomic group suffixes"
            if matches:
                group, suffix_idx, suffix, name = matches[0]
                view_matches[view] = (group.key, suffix_idx, name[: -len(suffix)])
        if not view_matches:
            yield unit_by_view
            continue
        assert len(view_matches) == len(
            unit_by_view
        ), f"An atomic update group must be present in every weight view: matched {sorted(view_matches)}, views {sorted(unit_by_view)}"
        match_values = set(view_matches.values())
        assert len(match_values) == 1, f"Weight views disagree on atomic grouping: {view_matches}"
        group_key, suffix_idx, prefix = next(iter(match_values))
        group = next(group for group in atomic_update_groups if group.key == group_key)
        matched_group_keys.add(group.key)
        slots = pending.setdefault((prefix, group.key), [None] * len(group.suffixes))
        assert (
            slots[suffix_idx] is None
        ), f"Duplicate member in atomic update group: {prefix}{group.suffixes[suffix_idx]}"
        slots[suffix_idx] = unit_by_view
        if None not in slots:
            yield {view: [named for slot in slots for named in slot[view]] for view in unit_by_view}
            del pending[(prefix, group.key)]

    assert not pending, f"Incomplete atomic update groups at end of stream: {sorted(pending)}"
    for group in atomic_update_groups:
        assert (
            group.optional or group.key in matched_group_keys
        ), f"Atomic update group {group.key} matched no params (suffixes {group.suffixes})"


def pack_view_units_by_size(
    view_units: Iterable[dict[str, list[tuple[str, torch.Tensor]]]],
    max_bytes: int,
) -> Iterator[dict[str, list[tuple[str, torch.Tensor]]]]:
    """Pack multi-view units while bounding their combined live GPU storage."""
    bucket: dict[str, list[tuple[str, torch.Tensor]]] = {}
    bucket_size = 0
    for unit_by_view in view_units:
        unit_size = sum(tensor.nbytes for unit in unit_by_view.values() for _name, tensor in unit)
        if bucket and bucket_size + unit_size >= max_bytes:
            yield bucket
            bucket = {}
            bucket_size = 0
        for view, unit in unit_by_view.items():
            bucket.setdefault(view, []).extend(unit)
        bucket_size += unit_size
    if bucket:
        yield bucket
