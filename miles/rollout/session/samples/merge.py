"""Training-sample assembly: session records -> per-turn `Sample`s, truncated at turn boundaries.

Owned by the session package so the assembly runs on the owning instance (records never have to leave the session server). The wire codec for the assembled reply lives in `codec`.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay decoders (accepted utils-level dependency: the decoders have other consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE `merge_samples` — truncation is a turn-level budget decision (which turns survive; the overflowing turn is cut at a turn boundary, later turns are dropped) and the turn structure only exists pre-merge.
"""

from argparse import Namespace

import numpy as np

from miles.rollout.generate_utils.generate_endpoint_utils import (
    decode_routed_experts,
    get_indexer_topk_from_response,
)
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata
from miles.rollout.session.types import SessionRecord
from miles.utils.lifecycle import attach_lifecycle_metadata
from miles.utils.types import Sample


def compute_samples_from_openai_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Current compact records carry a prompt length and output token logprobs;
    the prompt is reconstructed from the single accumulated token sequence.
    Older records containing ``request.input_ids`` remain supported.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/session/test_samples.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.request.get("input_ids")
        if prompt_ids is None:
            prompt_token_count = (
                record.prompt_token_count
                if record.prompt_token_count is not None
                else record.request.get("prompt_token_count")
            )
            if prompt_token_count is None:
                raise ValueError("session record has neither request.input_ids nor prompt_token_count")
            if accumulated_token_ids is None:
                raise ValueError("compact session records require accumulated_token_ids")
            if not 0 <= prompt_token_count <= len(accumulated_token_ids):
                raise ValueError(
                    f"invalid prompt_token_count={prompt_token_count} for accumulated "
                    f"sequence of {len(accumulated_token_ids)} tokens"
                )
            prompt_ids = accumulated_token_ids[:prompt_token_count]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            # Step 1: position cursor right after this turn's prompt
            cursor = len(prompt_ids)

            # Step 2: greedily match output_ids against accumulated[cursor:]
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break

            # Step 3: unmatched trailing tokens were consumed by the next
            # turn's template rendering (e.g. stop tokens that double as
            # the next message delimiter) — strip them from the sample.
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={accumulated_token_ids[cursor:cursor+3]}"
            )

            # Step 4: advance cursor past matched output to the next turn
            cursor += matched

        sample = _compute_sample_from_openai_record(
            args,
            record,
            tokenizer,
            trim_count,
            prompt_token_ids=prompt_ids,
        )
        attach_lifecycle_metadata(sample, record, records[i - 1] if i else None, turn=i + 1)
        if is_last and args.save_debug_trajectory_data is not None and record.request.get("messages"):
            sample.metadata["messages"] = record.request["messages"] + [record.response["choices"][0]["message"]]
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace,
    record: SessionRecord,
    tokenizer,
    trim_count: int = 0,
    *,
    prompt_token_ids: list[int] | None = None,
) -> Sample:
    choice = record.response["choices"][0]

    if prompt_token_ids is None:
        prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("prompt token IDs unavailable")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = Sample()
    # Production session records intentionally compact the request down to an
    # empty dict unless trajectory debugging is enabled.  The configured
    # rollout contract is therefore authoritative; the request flag remains a
    # compatibility path for older/debug records and focused tests.
    if getattr(args, "rollout_top_p", 1.0) < 1.0 or record.request.get("return_sampling_mask", False):
        output_log_probs = append_sampling_metadata(sample, output_token_ids, choice["meta_info"])
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    # Session routed-expert payloads may be incremental. They are reconstructed
    # once, after turn-level truncation, instead of materializing a cumulative
    # tensor for every intermediate Sample.
    sample.rollout_routed_experts = None
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, choice, sample)

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    if args.sglang_speculative_algorithm:
        sample.spec_info.add(choice.get("meta_info", {}))
    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


def reconstruct_routed_experts(
    args: Namespace,
    records: list[SessionRecord],
    *,
    final_num_tokens: int,
) -> np.ndarray | None:
    """Stitch incremental per-turn routed-expert suffixes into one tensor."""
    if not any(record.response.get("choices", [{}])[0].get("meta_info", {}).get("routed_experts") is not None for record in records):
        return None

    segments: list[np.ndarray] = []
    accumulated_rows = 0
    for turn, record in enumerate(records, start=1):
        choice = record.response["choices"][0]
        meta_info = choice["meta_info"]
        payload = meta_info.get("routed_experts")
        if payload is None:
            raise ValueError(f"turn {turn} is missing routed_experts")

        output_rows = len(meta_info["output_token_logprobs"])
        prompt_rows = record.prompt_token_count
        if prompt_rows is None:
            prompt_ids = record.request.get("input_ids")
            if prompt_ids is None:
                raise ValueError(f"turn {turn} has no prompt length for routed_experts")
            prompt_rows = len(prompt_ids)
        end_row = prompt_rows + output_rows - 1
        start_row = int(meta_info.get("routed_experts_start_len", 0))
        if not 0 <= start_row <= end_row:
            raise ValueError(f"turn {turn} has invalid routed_experts interval [{start_row}, {end_row})")
        if start_row > accumulated_rows:
            raise ValueError(
                f"turn {turn} routed_experts starts at {start_row}, "
                f"after the {accumulated_rows} reconstructed rows"
            )

        accumulated_rows = _truncate_segments(segments, accumulated_rows, start_row)
        segment = decode_routed_experts(args, payload, end_row - start_row)
        segments.append(segment)
        accumulated_rows += len(segment)
        if accumulated_rows != end_row:
            raise ValueError(
                f"turn {turn} reconstructed {accumulated_rows} routed_experts rows, "
                f"expected {end_row}"
            )

    if final_num_tokens > accumulated_rows:
        raise ValueError(
            f"training sample needs {final_num_tokens} routed_experts rows, "
            f"but only {accumulated_rows} were reconstructed"
        )
    _truncate_segments(segments, accumulated_rows, final_num_tokens)
    if not segments:
        return np.empty(
            (0, args.num_layers, args.moe_router_topk),
            dtype=np.int32,
        )
    return np.concatenate(segments, axis=0)


def _truncate_segments(
    segments: list[np.ndarray],
    current_rows: int,
    target_rows: int,
) -> int:
    """Truncate replay suffixes in place without copying their retained prefix."""
    while current_rows > target_rows:
        segment = segments.pop()
        segment_start = current_rows - len(segment)
        if segment_start < target_rows:
            segments.append(segment[: target_rows - segment_start])
        current_rows = max(segment_start, target_rows)
    return current_rows


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        sample.strip_last_output_tokens(overshoot, tokenizer)
        sample.status = Sample.Status.TRUNCATED
        result.append(sample)
        break

    return result
