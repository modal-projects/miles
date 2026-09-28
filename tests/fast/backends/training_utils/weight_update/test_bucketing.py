"""Unit tests for HF-namespace atomic grouping and size packing."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    assemble_atomic_update_group_views,
    pack_units_by_size,
    pack_view_units_by_size,
)


def _unit(*names, numel=4):
    return [(name, torch.zeros(numel, dtype=torch.uint8)) for name in names]


PAIR_GROUP = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"))


class TestAssembleAtomicUpdateGroups:
    def test_ungrouped_units_stream_through(self):
        optional = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"), optional=True)
        units = [_unit("layers.0.a.weight"), _unit("layers.1.a.weight")]
        out = list(assemble_atomic_update_groups(units, [optional]))
        assert [[n for n, _ in u] for u in out] == [["layers.0.a.weight"], ["layers.1.a.weight"]]

    def test_group_members_merge_into_one_unit(self):
        units = [_unit("layers.0.a.weight"), _unit("layers.0.b.weight"), _unit("layers.0.c.weight")]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            ["layers.0.a.weight"],
            ["layers.0.b.weight", "layers.0.c.weight"],
        ]

    def test_merge_is_per_prefix_instance(self):
        units = [
            _unit("layers.0.b.weight"),
            _unit("layers.1.b.weight"),
            _unit("layers.1.c.weight"),
            _unit("layers.0.c.weight"),
        ]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            ["layers.1.b.weight", "layers.1.c.weight"],
            ["layers.0.b.weight", "layers.0.c.weight"],
        ]

    def test_quant_scales_ride_inside_the_matching_unit(self):
        """A unit is matched by any member; non-matching members (scales) come along."""
        units = [
            _unit("layers.0.b.weight", "layers.0.b.weight_scale_inv"),
            _unit("layers.0.c.weight", "layers.0.c.weight_scale_inv"),
        ]
        out = list(assemble_atomic_update_groups(units, [PAIR_GROUP]))
        assert [[n for n, _ in u] for u in out] == [
            [
                "layers.0.b.weight",
                "layers.0.b.weight_scale_inv",
                "layers.0.c.weight",
                "layers.0.c.weight_scale_inv",
            ]
        ]

    def test_empty_units_are_skipped(self):
        out = list(assemble_atomic_update_groups([[], _unit("layers.0.a.weight")], []))
        assert [[n for n, _ in u] for u in out] == [["layers.0.a.weight"]]

    def test_incomplete_group_raises_at_end_of_stream(self):
        with pytest.raises(AssertionError, match="Incomplete atomic update groups"):
            list(assemble_atomic_update_groups([_unit("layers.0.b.weight")], [PAIR_GROUP]))

    def test_required_group_matching_nothing_raises(self):
        with pytest.raises(AssertionError, match="matched no params"):
            list(assemble_atomic_update_groups([_unit("layers.0.a.weight")], [PAIR_GROUP]))

    def test_optional_group_matching_nothing_passes(self):
        optional = AtomicUpdateGroup("pair", (".b.weight", ".c.weight"), optional=True)
        out = list(assemble_atomic_update_groups([_unit("layers.0.a.weight")], [optional]))
        assert len(out) == 1

    def test_unit_matching_two_suffixes_raises(self):
        with pytest.raises(AssertionError, match="matches multiple"):
            list(assemble_atomic_update_groups([_unit("layers.0.b.weight", "layers.0.c.weight")], [PAIR_GROUP]))

    def test_invalid_group_specs_raise(self):
        for groups, error in [
            ([AtomicUpdateGroup("empty", ())], "has no suffixes"),
            ([AtomicUpdateGroup("dup", (".a", ".a"))], "duplicate suffixes"),
            ([AtomicUpdateGroup("k", (".a",)), AtomicUpdateGroup("k", (".b",))], "Duplicate atomic update group keys"),
        ]:
            with pytest.raises(AssertionError, match=error):
                list(assemble_atomic_update_groups([], groups))


class TestPackUnitsBySize:
    def test_packs_up_to_max_bytes(self):
        units = [_unit("a", numel=4), _unit("b", numel=4), _unit("c", numel=4)]
        out = list(pack_units_by_size(units, max_bytes=9))
        assert [[n for n, _ in u] for u in out] == [["a", "b"], ["c"]]

    def test_never_splits_a_unit(self):
        units = [_unit("a", numel=4), _unit("b1", "b2", numel=4)]
        out = list(pack_units_by_size(units, max_bytes=6))
        assert [[n for n, _ in u] for u in out] == [["a"], ["b1", "b2"]]

    def test_empty_stream_yields_nothing(self):
        assert list(pack_units_by_size([], max_bytes=8)) == []


def test_multiview_atomic_groups_keep_each_view_independent() -> None:
    units = [
        {
            "fp8": _unit("layers.0.b.weight", "layers.0.b.weight_scale_inv"),
            "nvfp4": _unit("layers.0.b.weight", "layers.0.b.weight_scale"),
        },
        {
            "fp8": _unit("layers.0.c.weight", "layers.0.c.weight_scale_inv"),
            "nvfp4": _unit("layers.0.c.weight", "layers.0.c.weight_scale"),
        },
    ]

    [grouped] = list(assemble_atomic_update_group_views(units, [PAIR_GROUP]))

    assert [name for name, _tensor in grouped["fp8"]] == [
        "layers.0.b.weight",
        "layers.0.b.weight_scale_inv",
        "layers.0.c.weight",
        "layers.0.c.weight_scale_inv",
    ]
    assert [name for name, _tensor in grouped["nvfp4"]] == [
        "layers.0.b.weight",
        "layers.0.b.weight_scale",
        "layers.0.c.weight",
        "layers.0.c.weight_scale",
    ]


def test_multiview_packing_bounds_combined_live_bytes() -> None:
    units = [
        {"fp8": _unit("a.fp8", numel=4), "nvfp4": _unit("a.nvfp4", numel=2)},
        {"fp8": _unit("b.fp8", numel=4), "nvfp4": _unit("b.nvfp4", numel=2)},
    ]

    buckets = list(pack_view_units_by_size(units, max_bytes=10))

    assert len(buckets) == 2
    assert [name for name, _tensor in buckets[0]["fp8"]] == ["a.fp8"]
    assert [name for name, _tensor in buckets[1]["nvfp4"]] == ["b.nvfp4"]
