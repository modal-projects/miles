from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.rollout_data_conversion import (
    _compute_dynamic_global_batch_size,
    postprocess_rollout_data,
    validate_compact_rollout_ids,
)
from miles.utils.dp_schedule import build_dp_schedule


class TestComputeDynamicGlobalBatchSize:
    def test_rounds_down_to_multiple_of_dp_size(self):
        args = make_args(global_batch_size=64)
        # 13 samples, dp_size=4 → floor(13/4)*4 = 12
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=13)
        assert gbs == 12

    def test_returns_num_samples_when_already_aligned(self):
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=16)
        assert gbs == 16

    def test_falls_back_to_dp_size_when_below_dp_size(self):
        """When num_samples < dp_size, the rounded result is 0, which would
        produce a divide-by-zero downstream — fallback to dp_size."""
        args = make_args(global_batch_size=64)
        gbs = _compute_dynamic_global_batch_size(args, train_parallel_config={"dp_size": 4}, num_samples=2)
        assert gbs == 4


class TestPostprocessRolloutData:
    def test_aligned_input_passes_through_unchanged(self):
        """Happy path: len(data) % global_batch_size == 0 → no trim, no metadata."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(8)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_unaligned_input_is_trimmed_to_multiple(self):
        """len % gbs != 0 → trim down to multiple. Tail samples are dropped."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 8
        assert meta == {}

    def test_disable_rollout_trim_samples_keeps_unaligned_data(self):
        """With trim disabled, leave length as-is even if not divisible."""
        args = make_args(global_batch_size=4, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        data = [make_sample(index=i) for i in range(11)]
        out, _meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 1})
        assert len(out) == 11

    def test_raises_when_trim_would_produce_zero_samples(self):
        """5 samples, gbs=64 → trim_len=0 → can't proceed."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        with pytest.raises(ValueError, match="Not enough samples"):
            postprocess_rollout_data(args, [make_sample()] * 5, train_parallel_config={"dp_size": 1})

    def test_dynamic_batch_size_computed_and_recorded_in_metadata(self):
        """When use_dynamic_global_batch_size=True, the function recomputes gbs
        from the actual sample count and records it in metadata."""
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True)
        data = [make_sample(index=i) for i in range(10)]
        out, meta = postprocess_rollout_data(args, data, train_parallel_config={"dp_size": 2})
        # Dynamic gbs = floor(10/2)*2 = 10
        assert meta["dynamic_global_batch_size"] == 10
        assert len(out) == 10

    def test_flattens_nested_list_of_lists(self):
        """The function supports list[list[Sample]] input by flattening."""
        args = make_args(global_batch_size=2, disable_rollout_trim_samples=True, use_dynamic_global_batch_size=False)
        nested = [[make_sample(index=0), make_sample(index=1)], [make_sample(index=2)]]
        out, _meta = postprocess_rollout_data(args, nested, train_parallel_config={"dp_size": 1})
        assert len(out) == 3


class TestValidateRolloutIdAnnotated:
    def test_flat_list_skips_validation(self):
        """Depth-1 leaf (flat list[Sample]) is the default rollout shape — no rollout_id required."""
        validate_compact_rollout_ids([make_sample(index=i) for i in range(4)])

    def test_default_nested_shape_skips_validation(self):
        """list[list[Sample]] (prompt x rollout): leaf at depth 1 — no rollout_id required."""
        validate_compact_rollout_ids([[make_sample(index=0), make_sample(index=1)]])

    def test_compact_shape_requires_rollout_id(self):
        """list[list[list[Sample]]]: leaf at depth 2 with >1 sibling requires rollout_id."""
        compact = [[[make_sample(index=0), make_sample(index=1)]]]
        with pytest.raises(AssertionError, match="rollout_id is unset"):
            validate_compact_rollout_ids(compact)

    def test_compact_shape_siblings_must_share_rollout_id(self):
        a, b = make_sample(index=0), make_sample(index=1)
        a.rollout_id, b.rollout_id = 7, 8
        with pytest.raises(AssertionError, match="share rollout_id"):
            validate_compact_rollout_ids([[[a, b]]])

    def test_compact_shape_valid_when_annotated(self):
        a, b = make_sample(index=0), make_sample(index=1)
        a.rollout_id = b.rollout_id = 7
        validate_compact_rollout_ids([[[a, b]]])

    def test_compact_singleton_leaf_needs_no_annotation(self):
        """A single-sample leaf at depth >= 2 is unambiguous — no rollout_id required."""
        validate_compact_rollout_ids([[[make_sample(index=0)]]])


def _compact_samples(rollout_sizes: list[int]):
    """Flat samples of rollouts with these sample counts; each rollout shares one rollout_id."""
    samples = []
    for rollout_id, size in enumerate(rollout_sizes):
        for _ in range(size):
            samples.append(make_sample(index=len(samples), rollout_id=rollout_id))
    return samples


def _parallel_config(dp_size, vpp_size=1, microbatch_group_size_per_vp_stage=None):
    return {
        "dp_size": dp_size,
        "cp_size": 1,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


class TestCompactDynamicGlobalBatchSize:
    def _postprocess(self, rollout_sizes, train_parallel_config):
        args = make_args(global_batch_size=64, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True)
        return postprocess_rollout_data(args, _compact_samples(rollout_sizes), train_parallel_config)

    def test_records_a_rollout_count_without_trimming(self):
        """Without the metadata, the dynamic global batch size check in convert_samples_to_train_data fails."""
        out, meta = self._postprocess([1, 2, 1], _parallel_config(dp_size=2))
        assert meta["dynamic_global_batch_size"] == 3
        assert len(out) == 4

    def test_keeps_the_most_rollouts_with_an_aligned_sample_count(self):
        # Sample counts after each rollout: 1, 3, 4, 5. Only 4 splits over 2 DP ranks.
        _out, meta = self._postprocess([1, 2, 1, 1], _parallel_config(dp_size=2))
        assert meta["dynamic_global_batch_size"] == 3

    def test_aligns_to_the_vpp_microbatch_group(self):
        config = _parallel_config(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)
        _out, meta = self._postprocess([1] * 5 + [2], config)
        assert meta["dynamic_global_batch_size"] == 4

    def test_falls_back_to_all_rollouts_when_no_prefix_aligns(self):
        _out, meta = self._postprocess([1, 2], _parallel_config(dp_size=4))
        assert meta["dynamic_global_batch_size"] == 2

    def test_not_recorded_without_dynamic_global_batch_size(self):
        args = make_args(global_batch_size=2, disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False)
        out, meta = postprocess_rollout_data(args, _compact_samples([1, 2, 1]), _parallel_config(dp_size=2))
        assert meta == {}
        assert len(out) == 4

    def test_dp_schedule_accepts_an_odd_compact_batch(self):
        """Each sample fills a micro-batch alone, so 5 samples cannot make the 6 micro-batches dp_size=2 needs."""
        config = _parallel_config(dp_size=2)
        rollout_sizes = [2, 1, 1, 1]
        out, meta = self._postprocess(rollout_sizes, config)
        schedule_args = SimpleNamespace(
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=100,
            balance_data=False,
            balance_by_flops=False,
        )
        total_lengths = [60] * len(out)
        rollout_ids = [s.rollout_id for s in out]

        with pytest.raises(AssertionError, match="micro-batches"):
            build_dp_schedule(
                schedule_args,
                config,
                total_lengths,
                global_batch_size=len(rollout_sizes),
                rollout_indices=rollout_ids,
            )

        partitions, _indices, num_microbatches, num_rollouts = build_dp_schedule(
            schedule_args,
            config,
            total_lengths,
            global_batch_size=meta["dynamic_global_batch_size"],
            rollout_indices=rollout_ids,
        )
        assert num_rollouts == [3]
        assert num_microbatches == [2]
        assert sorted(i for rank in partitions for i in rank) == [0, 1, 2, 3]
