"""Parallel score-centering validation must keep the serial accept/reject contract."""

import itertools
import math
import threading
from copy import deepcopy

import numpy as np
import pytest

from miles.ray.rollout import train_data_conversion
from miles.rollout.generate_utils.score_centering import validate_score_centering_sample
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample

K = 4
# Row 2 is a tool observation: untrained, forced to token 9, and padded candidates.
SUPPORTS = [[3, 2], [5, 1, 7], [9], [4, 6]]
PROBABILITIES = [[3 / 7, 4 / 7], [0.5, 0.3, 0.2], None, [0.25, 0.75]]
LOSS_MASK = [1, 1, 0, 1]


def _serial_reference(samples: list[Sample], k: int) -> None:
    """The loop that train_data_conversion ran before validation was parallel."""
    for sample in samples:
        validate_score_centering_sample(sample, k)
        if sample.multimodal_train_inputs:
            raise ValueError("Score centering does not yet support multimodal token expansion")


def _filtered_sample(index: int) -> Sample:
    ids = np.full((len(SUPPORTS), K), -1, dtype=np.int32)
    logps = np.full((len(SUPPORTS), K), -np.inf, dtype=np.float32)
    for row, (support, probabilities) in enumerate(zip(SUPPORTS, PROBABILITIES, strict=True)):
        if probabilities is not None:
            ids[row, : len(support)] = support
            logps[row, : len(support)] = np.log(probabilities)
    return Sample(
        tokens=[0, 1, *(support[0] for support in SUPPORTS)],
        response_length=len(SUPPORTS),
        rollout_log_probs=[float(logps[row, 0]) if LOSS_MASK[row] else 0.0 for row in range(len(SUPPORTS))],
        rollout_topk_token_ids=ids,
        rollout_topk_log_probs=logps,
        rollout_sampling_mask=RolloutSamplingMask.from_mask_list(SUPPORTS),
        loss_mask=list(LOSS_MASK),
        status=Sample.Status.COMPLETED,
        index=index,
        group_index=index // 2,
        reward=float(index % 2),
    )


def _set_logps(sample: Sample, row: int, probabilities: list[float]) -> None:
    sample.rollout_topk_log_probs[row, : len(probabilities)] = np.log(probabilities)


def _pad_candidates(sample: Sample) -> None:
    sample.rollout_topk_token_ids = np.pad(sample.rollout_topk_token_ids, ((0, 0), (0, 1)), constant_values=-1)
    sample.rollout_topk_log_probs = np.pad(sample.rollout_topk_log_probs, ((0, 0), (0, 1)), constant_values=-np.inf)


def _drop_candidates(sample: Sample) -> None:
    sample.rollout_topk_token_ids = sample.rollout_topk_log_probs = None


def _empty_trained_row(sample: Sample) -> None:
    sample.rollout_topk_token_ids[1] = -1
    sample.rollout_topk_log_probs[1] = -np.inf


# One mutation per rejection branch, with the error it must produce.
MUTATIONS = {
    "loss_mask_length": (lambda s: setattr(s, "loss_mask", LOSS_MASK[:-1]), AssertionError, "loss_mask length"),
    "missing_candidates": (_drop_candidates, ValueError, "requires candidate and sampled-token"),
    "wrong_candidate_count": (_pad_candidates, ValueError, "must have shape"),
    "wrong_dtype": (
        lambda s: setattr(s, "rollout_topk_token_ids", s.rollout_topk_token_ids.astype(np.int64)),
        ValueError,
        "require int32 IDs",
    ),
    "invalid_padding_id": (
        lambda s: s.rollout_topk_token_ids.__setitem__((0, 3), -2),
        ValueError,
        "Every trained token needs",
    ),
    "empty_trained_row": (_empty_trained_row, ValueError, "Every trained token needs"),
    "no_candidate_mass": (
        lambda s: s.rollout_topk_log_probs.__setitem__((0, slice(0, 2)), -np.inf),
        ValueError,
        "positive candidate probability mass",
    ),
    "nan_logprob": (
        lambda s: s.rollout_topk_log_probs.__setitem__((0, 0), np.nan),
        ValueError,
        "Invalid score-centering logprobs",
    ),
    "finite_padding": (
        lambda s: s.rollout_topk_log_probs.__setitem__((0, 3), -1.0),
        ValueError,
        "Invalid score-centering logprobs",
    ),
    "mass_over_one": (lambda s: _set_logps(s, 1, [0.6, 0.3, 0.2]), ValueError, "mass exceeds one"),
    "duplicate_id": (
        lambda s: s.rollout_topk_token_ids.__setitem__((1, 1), 5),
        ValueError,
        "Duplicate score-centering candidate",
    ),
    "outside_support": (
        lambda s: s.rollout_topk_token_ids.__setitem__((3, 1), 8),
        ValueError,
        "must equal the sampling support",
    ),
    "support_mass": (lambda s: _set_logps(s, 3, [0.2, 0.7]), ValueError, "must sum to one"),
    "nonfinite_sampled": (
        lambda s: s.rollout_log_probs.__setitem__(0, math.nan),
        ValueError,
        "Invalid score-centering sampled-token",
    ),
    "sampled_mismatch": (
        lambda s: s.rollout_log_probs.__setitem__(0, math.log(0.4)),
        ValueError,
        "same sampler distribution",
    ),
    "multimodal": (
        lambda s: setattr(s, "multimodal_train_inputs", {"pixel_values": np.zeros(1)}),
        ValueError,
        "multimodal token expansion",
    ),
}


def _outcome(validate, samples: list[Sample]) -> tuple[type, str] | None:
    try:
        validate(samples, K)
    except (AssertionError, ValueError) as exc:
        return type(exc), str(exc)
    return None


def _batch(corrupt: dict[int, str], size: int = 6) -> list[Sample]:
    samples = [_filtered_sample(index) for index in range(size)]
    for position, name in corrupt.items():
        MUTATIONS[name][0](samples[position])
    return samples


@pytest.fixture
def threaded(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Force the thread pool and record which threads validated samples."""
    threads: set[str] = set()

    def record(sample: Sample, k: int) -> None:
        threads.add(threading.current_thread().name)
        validate_score_centering_sample(sample, k)

    monkeypatch.setattr(train_data_conversion.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(train_data_conversion, "validate_score_centering_sample", record)
    return threads


def test_valid_filtered_batch_is_accepted_by_both(threaded: set[str]) -> None:
    samples = _batch({})
    assert _outcome(_serial_reference, deepcopy(samples)) is None
    assert _outcome(train_data_conversion._validate_score_centering_samples, samples) is None
    assert any(thread.startswith("score-centering-validation") for thread in threaded)


@pytest.mark.parametrize("name", MUTATIONS)
@pytest.mark.parametrize("position", range(6))
def test_each_rejection_matches_serial(threaded: set[str], name: str, position: int) -> None:
    _, error_type, message = MUTATIONS[name]
    expected = _outcome(_serial_reference, _batch({position: name}))
    assert expected is not None and expected[0] is error_type and message in expected[1]
    assert _outcome(train_data_conversion._validate_score_centering_samples, _batch({position: name})) == expected
    assert any(thread.startswith("score-centering-validation") for thread in threaded)


@pytest.mark.parametrize(
    ("first", "second"), list(itertools.permutations(["sampled_mismatch", "duplicate_id", "multimodal"], 2))
)
@pytest.mark.parametrize(("early", "late"), [(0, 5), (2, 3), (4, 5)])
def test_earliest_rejection_wins_like_serial(
    threaded: set[str], first: str, second: str, early: int, late: int
) -> None:
    corrupt = {early: first, late: second}
    expected = _outcome(_serial_reference, _batch(corrupt))
    assert expected is not None and MUTATIONS[first][2] in expected[1]
    assert _outcome(train_data_conversion._validate_score_centering_samples, _batch(corrupt)) == expected


def test_serial_fallback_for_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_data_conversion.os, "cpu_count", lambda: 1)
    corrupt = {3: "sampled_mismatch"}
    assert _outcome(train_data_conversion._validate_score_centering_samples, _batch(corrupt)) == _outcome(
        _serial_reference, _batch(corrupt)
    )
    assert _outcome(train_data_conversion._validate_score_centering_samples, []) is None
