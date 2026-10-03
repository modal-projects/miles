"""Fixed-length, matched-concurrency metadata timing, with diagnostic tracing off."""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import requests
from tests.manual.score_centering_spec.contract import CASES, collect_sample
from tests.manual.score_centering_spec.probe import PROMPTS, generate
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=256)
    parser.add_argument("--model", default="/models/Qwen3.8-27B")
    parser.add_argument("--server-manifest", type=Path, default=Path("/artifacts/server-process.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.warmup_tokens < 1:
        parser.error("--warmup-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    args.endpoint = args.endpoint.rstrip("/")
    server_manifest = json.loads(args.server_manifest.read_text())
    assert "MILES_SCORE_CENTERING_TRACE_DIR" not in server_manifest["environment_overrides"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for i in range(8)
    ]
    cases = [c for c in CASES if c.name in ("unfiltered32_t1", "unfiltered128_t1", "topk64_topp90_t1")]
    manifest = {
        "arm": args.arm,
        "server_process": server_manifest,
        "server_info": requests.get(args.endpoint + "/server_info", timeout=30).json(),
        "concurrency": 8,
        "requests_per_trial": 8,
        "output_tokens_per_request": 256,
        "ignore_eos": True,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "cases": [asdict(c) for c in cases],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for case in cases:
        for repeat in range(args.repeats):
            for capture in ((True, False) if repeat % 2 == 0 else (False, True)):
                label = f"{case.name}-r{repeat}-capture{int(capture)}"
                run_id = uuid.uuid4().hex[:12]

                def payload(index, length, case=case, capture=capture, label=label, run_id=run_id):
                    request = case.request(prompts[index], length=length, capture=capture)
                    request["sampling_params"]["ignore_eos"] = True
                    request["rid"] = f"bench-{label}-{run_id}-{index}-{length}"
                    return request

                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(lambda i: generate(args.endpoint, payload(i, args.warmup_tokens)), range(8)))
                    started = time.perf_counter()
                    records = list(
                        executor.map(
                            lambda i, case=case: {"case": asdict(case), **generate(args.endpoint, payload(i, 256))},
                            range(8),
                        )
                    )
                    elapsed = time.perf_counter() - started
                # Preserve raw evidence before validation. Artifact writes and
                # production sample validation are both outside the timer.
                (args.output / f"{label}-responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                total = 0
                for record in records:
                    meta = record["response"]["meta_info"]
                    assert meta["finish_reason"]["type"] == "length", meta["finish_reason"]
                    assert meta["completion_tokens"] == 256, meta["completion_tokens"]
                    total += meta["completion_tokens"]
                    if capture:
                        collect_sample(case, record["request"]["input_ids"], record["response"])
                result = {
                    "case": case.name,
                    "repeat": repeat,
                    "capture": capture,
                    "tokens": total,
                    "seconds": elapsed,
                    "tokens_per_second": total / elapsed,
                }
                results.append(result)
                (args.output / "summary.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
