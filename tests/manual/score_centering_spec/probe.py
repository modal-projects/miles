"""Collect raw native SGLang responses and validate the production Miles contract.

Example:
    python -m tests.manual.score_centering_spec.probe --endpoint http://localhost:30000 \
        --model /models/target --output /artifacts/spec --cases unfiltered32_t1

Run on each pinned server configuration. Disable the verifier trace plugin for
timings. --no-capture measures the same sampling settings without metadata cost.
"""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
from tests.manual.score_centering_spec.contract import CASES, Case, collect_sample
from transformers import AutoTokenizer

PROMPTS = (
    "Compute 17 times 23. Show your reasoning and put the answer in a box.",
    "Find the sum of the first 20 positive odd integers. Explain why your answer is correct.",
    "A bag contains 3 red and 5 blue balls. Draw two without replacement. What is the chance both are red?",
    "Write a short Python function that returns whether a positive integer is prime, then explain its complexity.",
)


def generate(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(f"{endpoint}/generate", json=payload, timeout=900)
    response.raise_for_status()
    return {"request": payload, "response": response.json(), "seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(case.name for case in CASES))
    parser.add_argument("--lengths", default="1,7,8,9,17,32,64")
    parser.add_argument("--prompts", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=17)
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    run_id = uuid.uuid4().hex[:12]
    args.output.mkdir(parents=True, exist_ok=True)
    wanted = set(args.cases.split(","))
    cases = [case for case in CASES if case.name in wanted]
    if wanted != {case.name for case in cases}:
        parser.error(f"Unknown cases: {wanted - {case.name for case in cases}}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for index in range(args.prompts)
    ]
    server = requests.get(f"{args.endpoint}/server_info", timeout=60)
    server.raise_for_status()
    manifest = {**vars(args), "output": str(args.output), "run_id": run_id, "server_info": server.json()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Exercise decode/verification and the tested metadata path at the same
    # concurrency outside timing. A one-token prefill does not warm speculation.
    for case in cases:
        payloads = [
            case.request(prompt, length=args.warmup_tokens, capture=not args.no_capture)
            for prompt in prompts[: max(len(PROMPTS), args.concurrency)]
        ]
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            list(executor.map(lambda payload: generate(args.endpoint, payload), payloads))
    start = time.perf_counter()
    lengths = [int(value) for value in args.lengths.split(",")]
    jobs = [(case, [prompt], [lengths[i % len(lengths)]], i) for case in cases for i, prompt in enumerate(prompts)]

    def run(job: tuple) -> dict:
        case, batch, sizes, index = job
        payload = case.request(batch[0], length=sizes[0], capture=not args.no_capture)
        payload["rid"] = f"sc-{case.name}-{run_id}-{index}"
        return {"case": asdict(case), **generate(args.endpoint, payload)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor, (args.output / "responses.jsonl").open(
        "w"
    ) as out:
        total_tokens = 0
        count = 0
        captured = []
        for record in executor.map(run, jobs):
            out.write(json.dumps(record) + "\n")
            out.flush()
            captured.append(record)
            total_tokens += record["response"]["meta_info"]["completion_tokens"]
            count += 1
    elapsed = time.perf_counter() - start
    # Exclude the expensive diagnostic validator from throughput. Timing still
    # includes HTTP transport and artifact serialization, and is end-to-end.
    if not args.no_capture:
        for record in captured:
            collect_sample(Case(**record["case"]), record["request"]["input_ids"], record["response"])
    summary = {
        "requests": count,
        "tokens": total_tokens,
        "seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
