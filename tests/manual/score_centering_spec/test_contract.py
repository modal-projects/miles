"""CPU tests for the experiment's failure checks and independent reference."""

import json

import numpy as np
import pytest

from tests.manual.score_centering_spec.capture import reference_distribution
from tests.manual.score_centering_spec.contract import Case, collect_sample
from tests.manual.score_centering_spec.oracle import compare_gradient


def response(probs, *, selected=1, support=False):
    ids = np.argsort(-probs)
    meta = {
        "output_token_logprobs": [[float(np.log(probs[selected])), selected, None]],
        "output_top_logprobs": [[[float(np.log(probs[i])), int(i), None] for i in ids]],
    }
    if support:
        meta["output_token_sampling_mask"] = [ids.tolist()]
        meta["output_token_sampling_logprobs"] = [np.log(probs[ids]).tolist()]
    return {"meta_info": meta}


@pytest.mark.parametrize("support", [False, True])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_collected_probabilities_produce_dense_oracle_gradient(support, mode):
    probs = np.asarray([0.08, 0.12, 0.25, 0.55])
    case = Case("fixture", 4 if support else 2, top_k=4 if support else -1)
    raw = response(probs, support=support)
    sample = collect_sample(case, [5, 6], raw)
    result = compare_gradient(np.log(probs), sample, 0, temperature=1.0, mode=mode)
    assert result["max_gradient_error"] < 1e-7


def test_unfiltered_short_capture_is_rejected():
    with pytest.raises(ValueError, match="requested K"):
        collect_sample(Case("short", 8), [7], response(np.asarray([0.2, 0.8])))


def test_incomplete_support_is_rejected_even_when_ids_are_consistent():
    raw = response(np.asarray([0.1, 0.5]), support=True)
    with pytest.raises(ValueError, match="normalized support"):
        collect_sample(Case("short", 2, top_k=2), [7], raw)


def test_sampled_logprob_from_wrong_row_is_rejected():
    raw = response(np.asarray([0.2, 0.8]))
    raw["meta_info"]["output_token_logprobs"][0][0] = np.log(0.2)
    with pytest.raises(ValueError, match="same sampler"):
        collect_sample(Case("wrong", 2), [7], raw)


def test_partial_response_does_not_hide_a_later_server_abort():
    raw = response(np.asarray([0.2, 0.8]))
    raw["meta_info"]["finish_reason"] = {"type": "abort", "message": "support overflow"}
    with pytest.raises(ValueError, match="Generation aborted"):
        collect_sample(Case("aborted", 2), [7], raw)


def test_filter_order_changes_distribution():
    logits = np.log([0.4, 0.3, 0.2, 0.1])
    topk_first = reference_distribution(logits, temperature=1, top_k=2, top_p=0.55, order="top_k_first")
    joint = reference_distribution(logits, temperature=1, top_k=2, top_p=0.55, order="joint")
    np.testing.assert_allclose(topk_first, [1, 0, 0, 0])
    np.testing.assert_allclose(joint, [4 / 7, 3 / 7, 0, 0])


def test_topk_cutoff_ties_are_not_silently_dropped():
    actual = reference_distribution(np.log([0.4, 0.2, 0.2, 0.2]), temperature=1, top_k=2, top_p=1, order="top_k_first")
    np.testing.assert_allclose(actual, [0.4, 0.2, 0.2, 0.2])


def test_topp_cutoff_ties_are_not_silently_dropped():
    actual = reference_distribution(np.log([0.5, 0.25, 0.25]), temperature=1, top_k=-1, top_p=0.6, order="joint")
    np.testing.assert_allclose(actual, [0.5, 0.25, 0.25])


def test_topp_stops_at_exact_cumulative_boundary():
    actual = reference_distribution(
        np.log([0.5, 0.25, 0.125, 0.125]), temperature=1, top_k=-1, top_p=0.5, order="joint"
    )
    np.testing.assert_allclose(actual, [1, 0, 0, 0])


def test_snapshots_cover_later_cases_and_distinguish_bonus_roles(tmp_path, monkeypatch):
    from tests.manual.score_centering_spec import capture

    monkeypatch.setenv("MILES_SCORE_CENTERING_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(capture, "_SNAPSHOT_COUNTS", {})
    monkeypatch.setattr(capture, "_SNAPSHOT_ROWS", 0)
    for block, case in enumerate(("unfiltered32_t1", "topk32_t1"), start=1):
        monkeypatch.setattr(capture, "_BLOCKS", block)
        capture._write_block(
            requests=[
                {"rid": f"sc-{case}-run-{i}", "prompt_length": 3, "temperature": 1.0, "top_k": -1, "top_p": 1.0}
                for i in range(2)
            ],
            logits=np.zeros((6, 4)),
            kernel_probs=np.full((2, 3, 4), 0.25),
            prefix_lens=[3, 3],
            accept_lens=[2, 0],
            commit_lens=[3, 1],
            tokens=np.asarray([[0, 1, 2], [3, 0, 0]]),
            order="top_k_first",
        )
    saved = [row for path in tmp_path.glob("*.rows.json") for row in json.loads(path.read_text())]
    assert {row["role"] for row in saved} == {"accepted_draft", "all_accepted_bonus", "rejection_bonus"}
    assert len({row["rid"].rsplit("-", 2)[0] for row in saved}) == 2
    assert min(row["output_index"] for row in saved) == 1


def test_trace_checker_rejects_incorrect_committed_position(tmp_path):
    from tests.manual.score_centering_spec.check_trace import check

    raw = response(np.asarray([0.2, 0.8]))
    for key in ("output_token_logprobs", "output_top_logprobs"):
        raw["meta_info"][key] *= 2
    case = {"name": "wrong", "width": 2}
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({"case": case, "request": {"rid": "x", "input_ids": [7]}, "response": raw}))
    (tmp_path / "verify-1.jsonl").write_text(json.dumps({"rid": "x", "rows": [{"output_index": 1, "token": 0}]}))
    with pytest.raises(AssertionError, match="Token alignment"):
        check(responses, tmp_path, logprob_atol=2e-5, require_complete=True)
