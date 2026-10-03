"""Check stop completion and native/OpenAI metadata using production collectors.

Run against the engine directly, then against the actual router URL. This checks
non-streaming Chat Completions, the interface used by Miles session records.
Raw OpenAI responses and normalized native-shaped records are both preserved.
"""

import argparse
import json
import time
import uuid
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from tests.manual.score_centering_spec.contract import Case, collect_sample
from transformers import AutoTokenizer

from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord


def check_openai(case: Case, request: dict, response: dict, tokenizer, native_sample) -> None:
    args = Namespace(
        rollout_top_logprobs_num=case.width,
        save_debug_trajectory_data=None,
        sglang_speculative_algorithm="DFLASH",
    )
    record = SessionRecord(
        timestamp=time.time(),
        method="POST",
        path="v1/chat/completions",
        status_code=200,
        request=request,
        response=response,
    )
    (sample,) = compute_samples_from_openai_records(args, [record], tokenizer)
    validate_rollout_topk_logprobs_sample(sample, case.width)
    assert sample.tokens == native_sample.tokens
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, native_sample.rollout_topk_token_ids)
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, native_sample.rollout_topk_log_probs)
    np.testing.assert_array_equal(sample.rollout_log_probs, native_sample.rollout_log_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    summary = []
    cases = (
        Case("endpoint_unfiltered32", 32),
        Case("endpoint_unfiltered128", 128),
        Case("endpoint_support128", 128, top_k=64, top_p=0.9),
    )
    with (args.output / "responses.jsonl").open("w") as records:
        for interface in ("generate", "v1/chat/completions"):
            for case in cases:
                for stopping in ("eos", "string"):
                    messages = [{"role": "user", "content": "Write two short sentences about mountains."}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
                    )
                    native = case.request(prompt, length=128)
                    if stopping == "string":
                        native["sampling_params"]["stop"] = ["."]
                    native["rid"] = f"sc-{case.name}_{stopping}-{run_id}-{len(summary)}"
                    payload = native
                    if interface != "generate":
                        payload = {
                            "rid": native["rid"],
                            "model": args.served_model or args.model,
                            "messages": messages,
                            "input_ids": prompt,
                            "temperature": case.temperature,
                            "top_k": case.top_k,
                            "top_p": case.top_p,
                            "min_p": 0.0,
                            "max_completion_tokens": 128,
                            "logprobs": True,
                            "return_meta_info": True,
                            "stream": False,
                        }
                        if stopping == "string":
                            payload["stop"] = ["."]
                        if case.mode == "support":
                            payload.update(return_sampling_mask=True, sampling_logprobs_mode="support")
                        else:
                            payload["top_logprobs"] = case.width
                    response = requests.post(f"{args.endpoint.rstrip('/')}/{interface}", json=payload, timeout=900)
                    response.raise_for_status()
                    raw = response.json()
                    normalized = raw if interface == "generate" else {"meta_info": raw["choices"][0]["meta_info"]}
                    # Keep the actual engine id for joining verifier observations.
                    native["rid"] = normalized["meta_info"]["id"]
                    records.write(json.dumps({"case": asdict(case), "request": native, "response": normalized}) + "\n")
                    records.flush()
                    (args.output / f"raw-{len(summary)}.json").write_text(
                        json.dumps({"interface": interface, "request": payload, "response": raw}, indent=2)
                    )
                    sample = collect_sample(case, prompt, normalized)
                    if interface != "generate":
                        check_openai(case, payload, raw, tokenizer, sample)
                    finish = normalized["meta_info"]["finish_reason"]
                    assert finish["type"] == "stop", f"Expected {stopping} completion, got {finish}"
                    summary.append(
                        dict(
                            interface=interface,
                            case=case.name,
                            stopping=stopping,
                            tokens=sample.response_length,
                            finish=finish,
                        )
                    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
