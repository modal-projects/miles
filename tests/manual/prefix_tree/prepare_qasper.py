"""Prepare real long-document SFT with distinct human Q/A branches per paper.

Uses only QASPER's training split, pinned converted-parquet revision and complete
papers. No prompt repetition, synthetic answers, truncation or fresh sampling.
The recorded token format is shared by Transformers and Megatron probes.

Example: python prepare_qasper.py --tokenizer /assets/model --output rollouts.json
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = "allenai/qasper"
REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"


def _records(values):
    """HF Sequence stores nested records as columns in converted Parquet."""
    if isinstance(values, dict):
        return [dict(zip(values, row, strict=True)) for row in zip(*values.values(), strict=True)]
    return values


def _answer(answers):
    for annotation in _records(answers):
        answer = annotation["answer"]
        if answer["unanswerable"]:
            continue
        if answer["free_form_answer"]:
            return answer["free_form_answer"]
        if answer["extractive_spans"]:
            return "; ".join(answer["extractive_spans"])
        if answer["yes_no"] is not None:
            return "Yes" if answer["yes_no"] else "No"
    return None


def paper_group(paper, tokenizer, *, branches, min_prefix, max_length):
    document = (
        paper["title"]
        + "\n\n"
        + paper["abstract"]
        + "\n\n"
        + "\n\n".join(
            (section["section_name"] or "") + "\n" + "\n".join(section["paragraphs"])
            for section in _records(paper["full_text"])
        )
    )
    prompts, samples, questions = [], [], set()
    for qa in _records(paper["qas"]):
        answer = _answer(qa["answers"])
        if answer is None or qa["question"] in questions:
            continue
        messages = [
            {
                "role": "user",
                "content": (
                    "Read the following paper and answer the question using the paper.\n\n"
                    + document
                    + "\n\nQuestion: "
                    + qa["question"]
                ),
            }
        ]
        prompt = list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
            )
        )
        completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt) + len(completion) - 1 > max_length:
            continue
        questions.add(qa["question"])
        prompts.append(prompt)
        samples.append(
            {
                "member": len(samples),
                "prompt_ids": prompt,
                "completion_ids": completion,
                "question_id": qa["question_id"],
                "answer": answer,
            }
        )
        if len(samples) == branches:
            break
    if len(samples) != branches:
        return None
    common = 0
    for values in zip(*prompts, strict=False):
        if len(set(values)) != 1:
            break
        common += 1
    if common < min_prefix:
        return None
    return {"paper_id": paper["id"], "prompt_ids": prompts[0][:common]}, samples


def prepare(papers, tokenizer, *, groups=8, branches=4, min_prefix=4096, max_length=16384, seed=71):
    if min(groups, branches, min_prefix, max_length) < 1 or branches < 2:
        raise ValueError("Require positive budgets and at least two distinct questions per paper")
    papers = list(papers)
    random.Random(seed).shuffle(papers)
    prompts, samples = [], []
    for paper in papers:
        group = paper_group(paper, tokenizer, branches=branches, min_prefix=min_prefix, max_length=max_length)
        if group is None:
            continue
        prompt, members = group
        index = len(prompts)
        prompts.append({"index": index, **prompt})
        samples.extend({"prompt_index": index, **member} for member in members)
        if len(prompts) == groups:
            break
    if len(prompts) != groups:
        raise ValueError(f"Found only {len(prompts)} eligible papers; need {groups}. No padding/truncation applied")
    lengths = [len(s["prompt_ids"]) for s in samples]
    completions = [len(s["completion_ids"]) for s in samples]
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "train",
        "dataset_shuffle_seed": seed,
        "target_source": "human reference answers",
        "prompts": prompts,
        "samples": samples,
        "summary": {
            "prompts": len(prompts),
            "trajectories": len(samples),
            "prompt_length_min": min(lengths),
            "prompt_length_max": max(lengths),
            "shared_prefix_min": min(len(p["prompt_ids"]) for p in prompts),
            "completion_length_min": min(completions),
            "completion_length_max": max(completions),
            "completion_tokens": sum(completions),
        },
    }


def main():
    # Optional preparation dependencies; the pure converter is CPU-testable.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--min-prefix", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    source = hf_hub_download(DATASET, "qasper/train/0000.parquet", repo_type="dataset", revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    data = prepare(
        pq.read_table(source).to_pylist(),
        tokenizer,
        groups=args.groups,
        branches=args.branches,
        min_prefix=args.min_prefix,
        max_length=args.max_length,
        seed=args.seed,
    )
    data["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    data["config_sha256"] = hashlib.sha256(Path(args.tokenizer, "config.json").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    print(json.dumps(data["summary"]), flush=True)


if __name__ == "__main__":
    main()
