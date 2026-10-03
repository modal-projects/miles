"""Cross-check saved prefixes on a non-spec server, retaining numerical controls.

This is not an oracle for the speculative forward: prefill/cache/decode kernels
can differ. Each prefix is evaluated twice on the reference server to measure
its own numerical variation. No generated-string identity is assumed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.manual.score_centering_spec.contract import Case, collect_sample
from tests.manual.score_centering_spec.probe import generate


def _reference_logprobs(case: Case, response: dict) -> dict[int, float]:
    meta = response["meta_info"]
    if case.mode == "support":
        return dict(zip(meta["output_token_sampling_mask"][0], meta["output_token_sampling_logprobs"][0], strict=True))
    return {entry[1]: entry[0] for entry in meta["output_token_ids_logprobs"][0]}


def _compare(source: dict[int, float], reference: dict[int, float]) -> dict:
    common = source.keys() & reference.keys()
    union = source.keys() | reference.keys()
    return {
        "common_tokens": len(common),
        "source_only_tokens": len(source.keys() - reference.keys()),
        "reference_only_tokens": len(reference.keys() - source.keys()),
        "common_logprob_max_error": max((abs(source[i] - reference[i]) for i in common), default=None),
        "recorded_probability_l1_error": sum(
            abs(np.exp(source.get(i, -np.inf)) - np.exp(reference.get(i, -np.inf))) for i in union
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", default="0,1,7,8,-1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    with (args.output / "replays.jsonl").open("w") as out:
        for line in args.responses.read_text().splitlines():
            source = json.loads(line)
            case = Case(**source["case"])
            prompt = source["request"]["input_ids"]
            sample = collect_sample(case, prompt, source["response"])
            positions = {int(value) % sample.response_length for value in args.positions.split(",")}
            for position in sorted(positions):
                ids = sample.rollout_topk_token_ids[position]
                logps = sample.rollout_topk_log_probs[position]
                source_logps = {int(token): float(logp) for token, logp in zip(ids, logps, strict=True) if token >= 0}
                token = sample.tokens[len(prompt) + position]
                source_logps[token] = sample.rollout_log_probs[position]
                payload = case.request(sample.tokens[: len(prompt) + position], length=1)
                if case.mode == "selected":
                    payload["token_ids_logprob"] = sorted(source_logps)
                first = generate(args.endpoint.rstrip("/"), payload)
                second = generate(args.endpoint.rstrip("/"), payload)
                first_logps = _reference_logprobs(case, first["response"])
                second_logps = _reference_logprobs(case, second["response"])
                result = {
                    "rid": source["request"]["rid"],
                    "position": position,
                    "mode": case.mode,
                    "source_vs_reference": _compare(source_logps, second_logps),
                    "reference_vs_reference": _compare(first_logps, second_logps),
                    "source_token_reference_probability": float(np.exp(second_logps.get(token, -np.inf))),
                }
                out.write(json.dumps({**result, "first": first, "second": second}) + "\n")
                out.flush()
                comparisons.append(result)
    (args.output / "summary.json").write_text(json.dumps(comparisons, indent=2))
    print(json.dumps({"prefixes_checked": len(comparisons)}))


if __name__ == "__main__":
    main()
