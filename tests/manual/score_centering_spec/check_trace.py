"""Validate response alignment/probabilities against the actual verifier snapshots.

python -m tests.manual.score_centering_spec.check_trace --responses RUN/responses.jsonl \
    --trace-dir TRACES --output RUN/trace-check.json

The first generated token uses ordinary prefill and is intentionally outside the
DFlash verify trace. End-of-generation trimming may leave extra committed trace
rows; all retained generated rows after prefill must be present by default.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.manual.score_centering_spec.contract import Case, collect_sample
from tests.manual.score_centering_spec.oracle import compare_gradient


def check(responses: Path, trace_dir: Path, *, logprob_atol: float, require_complete: bool) -> dict:
    records = [json.loads(line) for line in responses.read_text().splitlines()]
    sources = {record["request"]["rid"]: record for record in records}
    samples = {
        rid: collect_sample(Case(**record["case"]), record["request"]["input_ids"], record["response"])
        for rid, record in sources.items()
    }
    matched, errors, kernel_errors, roles = set(), [], [], {}
    trimmed_rows = 0
    for path in sorted(trace_dir.glob("verify-*.jsonl")):
        for line in path.read_text().splitlines():
            block = json.loads(line)
            rid = block["rid"]
            if rid not in sources:
                continue
            sample = samples[rid]
            case = Case(**sources[rid]["case"])
            for row in block["rows"]:
                position = row["output_index"]
                if position >= sample.response_length:
                    trimmed_rows += 1
                    continue
                if position < 1 or (rid, position) in matched:
                    raise AssertionError(f"Invalid/duplicate committed position {(rid, position)}")
                token = sample.tokens[len(sample.tokens) - sample.response_length + position]
                assert token == row["token"], f"Token alignment differs at {rid}:{position}"
                assert row["kernel_observed"], f"No actual stochastic verification observed at {rid}:{position}"
                ids = sample.rollout_topk_token_ids[position]
                valid = ids >= 0
                expected = dict(zip(row["top_ids"], row["top_logprobs"], strict=True))
                if case.mode == "support":
                    expected = dict(zip(row["support_ids"], row["support_logprobs"], strict=True))
                    assert set(ids[valid]) == set(expected), f"Support mismatch at {rid}:{position}"
                    selected_expected = expected[token]
                else:
                    selected_expected = row["selected_logprob"]
                    cutoff = sorted(expected.values(), reverse=True)[min(case.width, len(expected)) - 1]
                    assert all(
                        expected[int(candidate)] >= cutoff - logprob_atol for candidate in ids[valid]
                    ), f"Returned candidates are not the top {case.width} at {rid}:{position}"
                candidate_expected = np.asarray([expected[int(candidate)] for candidate in ids[valid]])
                delta = np.abs(sample.rollout_topk_log_probs[position][valid] - candidate_expected)
                errors.extend(delta.tolist())
                errors.append(abs(sample.rollout_log_probs[position] - selected_expected))
                kernel_errors.append(row["kernel_l1_error"])
                assert row["kernel_l1_error"] <= 2e-5, f"Kernel distribution mismatch at {rid}:{position}"
                assert np.max(delta) <= logprob_atol, f"Candidate logprob mismatch at {rid}:{position}"
                assert errors[-1] <= logprob_atol, f"Sampled logprob mismatch at {rid}:{position}"
                matched.add((rid, position))
                roles[row["role"]] = roles.get(row["role"], 0) + 1
    expected_positions = {
        (rid, index) for rid, sample in samples.items() for index in range(1, sample.response_length)
    }
    if require_complete:
        assert matched == expected_positions, f"Missing verify rows: {sorted(expected_positions - matched)[:20]}"
    assert matched, "No response tokens matched verifier snapshots"
    gradient_checks = []
    for path in sorted(trace_dir.glob("verify-*.rows.json")):
        rows = json.loads(path.read_text())
        with np.load(path.with_name(path.name.replace(".rows.json", ".npz"))) as snapshot:
            for index, row in enumerate(rows):
                key = (row["rid"], row["output_index"])
                if key not in matched:
                    continue
                sample = samples[row["rid"]]
                for mode in ("none", "tis", "mis"):
                    gradient_checks.append(
                        {
                            "rid": row["rid"],
                            "position": row["output_index"],
                            **compare_gradient(
                                snapshot["logits"][index],
                                sample,
                                row["output_index"],
                                temperature=row["temperature"],
                                mode=mode,
                            ),
                        }
                    )
    assert gradient_checks, "No full-logit snapshots matched response rows for gradient verification"
    expected_cases = {sources[rid]["case"]["name"] for rid, _ in matched}
    checked_cases = {sources[item["rid"]]["case"]["name"] for item in gradient_checks}
    assert expected_cases == checked_cases, f"Missing gradient checks for cases: {expected_cases - checked_cases}"
    assert roles.get("accepted_draft", 0), "No accepted draft tokens were checked"
    assert roles.get("rejection_bonus", 0), "No rejection bonus tokens were checked"
    return {
        "verified_tokens": len(matched),
        "expected_verify_tokens": len(expected_positions),
        "committed_rows_trimmed_at_response_end": trimmed_rows,
        "roles": roles,
        "all_accepted_bonus_observed": bool(roles.get("all_accepted_bonus", 0)),
        "gradient_cases": sorted(checked_cases),
        "logprob_atol": logprob_atol,
        "logprob_max_error": max(errors),
        "logprob_p99_error": float(np.quantile(errors, 0.99)),
        "independent_kernel_probability_max_l1_error": max(kernel_errors),
        "independent_kernel_probability_l1_atol": 2e-5,
        "gradient_checks": gradient_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logprob-atol", type=float, default=2e-5)
    parser.add_argument("--allow-partial-trace", action="store_true")
    args = parser.parse_args()
    result = check(
        args.responses, args.trace_dir, logprob_atol=args.logprob_atol, require_complete=not args.allow_partial_trace
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "gradient_checks"}))


if __name__ == "__main__":
    main()
