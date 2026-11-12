"""Record real DAPO prompts and fresh Tinker completions for paired SFT tests.

Run against an already configured sampling service. Proxy credentials, if needed,
come from MODAL_PROXY_TOKEN_ID/SECRET; the service key comes from TINKER_API_KEY.
The output contains exact token IDs, stop reasons and pinned dataset provenance.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


async def sample(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    import tinker
    from tinker import types

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    revision = HfApi().dataset_info(args.dataset).sha
    rows = load_dataset(args.dataset, "all", revision=revision, split="train")
    rows = rows.shuffle(seed=args.seed).select(range(args.prompts))
    prompts = []
    for index, row in enumerate(rows):
        messages = row["source_prompt"]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True
        )
        prompts.append(
            {
                "index": index,
                "question_id": row.get("extra_info", {}).get("index"),
                "messages": messages,
                "answer": row.get("solution"),
                "prompt_ids": ids,
            }
        )
    headers = {}
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers = {
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        }
    service = tinker.ServiceClient(
        base_url=args.base_url,
        api_key=os.environ["TINKER_API_KEY"],
        default_headers=headers,
        timeout=7200,
        _client_config={
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "use_pyqwest_transport": False,
            "sample_max_concurrent_requests": args.concurrency,
            "sample_enable_stuck_detection": False,
        },
    )
    sampler = await service.create_sampling_client_async(base_model=args.model)
    data = {
        "dataset": args.dataset,
        "revision": revision,
        "dataset_shuffle_seed": args.seed,
        "model": args.model,
        "thinking": True,
        "sampling": {"max_tokens": args.max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "prompts": prompts,
        "samples": [],
    }
    _save(args.output, data)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    async def one(index, member):
        seed = 1000 * args.seed + args.completions * index + member
        async with semaphore:
            response = await sampler.sample_async(
                prompt=types.ModelInput.from_ints(prompts[index]["prompt_ids"]),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=args.max_tokens, temperature=1.0, top_p=0.95, top_k=64, seed=seed
                ),
            )
        sequence = response.sequences[0]
        data["samples"].append(
            {
                "prompt_index": index,
                "member": member,
                "seed": seed,
                "completion_ids": list(sequence.tokens),
                "text": tokenizer.decode(sequence.tokens, skip_special_tokens=False),
                "response": sequence.model_dump(mode="json"),
            }
        )
        _save(args.output, data)
        print("DAPO_ROLLOUT_PROGRESS", len(data["samples"]), flush=True)

    await asyncio.gather(*(one(i, j) for i in range(args.prompts) for j in range(args.completions)))
    data["samples"].sort(key=lambda row: (row["prompt_index"], row["member"]))
    data["generation_s"] = time.monotonic() - started
    prompt_lengths = [len(row["prompt_ids"]) for row in prompts]
    lengths = [len(row["completion_ids"]) for row in data["samples"]]
    data["summary"] = {
        "prompts": len(prompts),
        "trajectories": len(lengths),
        "prompt_length_min": min(prompt_lengths),
        "prompt_length_max": max(prompt_lengths),
        "completion_length_min": min(lengths),
        "completion_length_max": max(lengths),
        "completion_length_mean": sum(lengths) / len(lengths),
        "completion_tokens": sum(lengths),
        "at_generation_cap": sum(n == args.max_tokens for n in lengths),
    }
    data["config_sha256"] = hashlib.sha256((Path(args.tokenizer) / "config.json").read_bytes()).hexdigest()
    _save(args.output, data)
    print("DAPO_ROLLOUT_COMPLETE", json.dumps(data["summary"]), flush=True)
    return data["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, help="Local teacher checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="open-r1/DAPO-Math-17k-Processed")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--completions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=128)
    args = parser.parse_args()
    if min(args.prompts, args.completions, args.max_tokens, args.concurrency) < 1:
        parser.error("Counts and token budget must be positive")
    asyncio.run(sample(args))


if __name__ == "__main__":
    main()
