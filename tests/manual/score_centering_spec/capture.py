"""Opt-in SGLang hooks that observe actual DFlash verification without changing it.

Install the adjacent plugin and set MILES_SCORE_CENTERING_TRACE_DIR. This copies
GPU tensors to the CPU and deliberately synchronizes: NEVER use it for timings.
All verification blocks are recorded up to the explicit block limit. A bounded
subset also retains full logits/probabilities for independent gradient checks.
"""

import json
import os
from contextvars import ContextVar
from pathlib import Path

import numpy as np

_BATCH = ContextVar("score_centering_batch", default=None)
_VERIFY = ContextVar("score_centering_verify", default=None)
_BLOCKS = 0
_SNAPSHOT_ROWS = 0
_FAILURE_SNAPSHOTS = 0
_SNAPSHOT_COUNTS: dict[tuple[str, str], int] = {}


def reference_distribution(
    logits: np.ndarray, *, temperature: float, top_k: int, top_p: float, order: str
) -> np.ndarray:
    """Independent float64 reference; do not call any SGLang filtering helpers."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    keep = np.ones(len(probs), dtype=bool)
    if 0 < top_k < len(probs):
        cutoff = np.partition(probs, -top_k)[-top_k]
        keep &= probs >= cutoff
    nucleus_probs = probs * keep if order == "top_k_first" else probs.copy()
    nucleus_probs /= nucleus_probs.sum()
    if top_p < 1:
        ids = np.argsort(-nucleus_probs, kind="stable")
        cumulative_before = np.cumsum(nucleus_probs[ids]) - nucleus_probs[ids]
        cutoff = nucleus_probs[ids[cumulative_before < top_p][-1]]
        keep &= nucleus_probs >= cutoff
    probs *= keep
    return probs / probs.sum()


def _with_batch(original, worker, batch, *args, **kwargs):
    if worker.model_runner.tp_rank != 0:
        return original(worker, batch, *args, **kwargs)
    requests = [
        {
            "rid": req.rid,
            "prompt_length": len(req.origin_input_ids),
            "temperature": req.sampling_params.temperature,
            "top_k": req.sampling_params.top_k,
            "top_p": req.sampling_params.top_p,
        }
        for req in batch.reqs
    ]
    token = _BATCH.set(requests)
    try:
        return original(worker, batch, *args, **kwargs)
    finally:
        _BATCH.reset(token)


def _observe_kernel(original, *args, **kwargs):
    state = _VERIFY.get()
    if state is not None:
        # Snapshot the exact distribution handed to the kernel, before temporary
        # graph-pool storage is released or reused. Sampling itself is untouched.
        state["kernel_probs"] = kwargs["target_probs"].detach().float().cpu().numpy().copy()
    return original(*args, **kwargs)


def _observe_accept(original, worker, **kwargs):
    global _BLOCKS
    requests = _BATCH.get()
    if requests is None or _BLOCKS >= int(os.environ.get("MILES_SCORE_CENTERING_TRACE_BLOCKS", "4096")):
        return original(worker, **kwargs)
    info = kwargs["sampling_info"]
    state = {
        "effective_sampling": {
            "temperature": info.temperatures.reshape(-1).cpu().tolist(),
            "top_k": info.top_ks.reshape(-1).cpu().tolist(),
            "top_p": info.top_ps.reshape(-1).cpu().tolist(),
        }
    }
    token = _VERIFY.set(state)
    try:
        result = original(worker, **kwargs)
    finally:
        _VERIFY.reset(token)
    # Optional dependency: this module also supports CPU-only reference tests.
    from sglang.srt.runtime_context import get_exec

    _BLOCKS += 1
    _write_block(
        requests=requests,
        logits=kwargs["next_token_logits"].detach().float().cpu().numpy(),
        kernel_probs=state.get("kernel_probs"),
        prefix_lens=kwargs["prefix_lens"].cpu().tolist(),
        accept_lens=result[0].cpu().tolist(),
        commit_lens=result[1].cpu().tolist(),
        tokens=result[3].cpu().numpy(),
        order=get_exec().kernel.sampling_filter_order,
        effective_sampling=state["effective_sampling"],
    )
    return result


def _write_block(
    *, requests, logits, kernel_probs, prefix_lens, accept_lens, commit_lens, tokens, order, effective_sampling=None
):
    global _SNAPSHOT_ROWS, _FAILURE_SNAPSHOTS
    directory = Path(os.environ["MILES_SCORE_CENTERING_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    block_size = tokens.shape[1]
    logits = logits.reshape(len(requests), block_size, -1)
    if kernel_probs is not None:
        kernel_probs = kernel_probs.reshape(logits.shape)
    records, saved_logits, saved_probs, saved_rows = [], [], [], []
    for batch_index, request in enumerate(requests):
        rows = []
        for position in range(commit_lens[batch_index]):
            row_logits = logits[batch_index, position]
            selected = int(tokens[batch_index, position])
            actual = None if kernel_probs is None else kernel_probs[batch_index, position]
            row = _describe_row(row_logits, actual, selected, request, order)
            bonus_role = "all_accepted_bonus" if accept_lens[batch_index] == block_size - 1 else "rejection_bonus"
            role = "accepted_draft" if position < accept_lens[batch_index] else bonus_role
            row.update(
                output_index=prefix_lens[batch_index] - request["prompt_length"] + 1 + position,
                role=role,
            )
            # The probe embeds case/run/index in its ID. Retain two rows per
            # case and role, so later filtered/temperature cases are covered.
            case_key = request["rid"][3:].rsplit("-", 1)[0]
            key = (case_key, role)
            failed = row["kernel_l1_error"] is not None and row["kernel_l1_error"] > 2e-5
            if (
                request["rid"].startswith("sc-")
                and (_SNAPSHOT_COUNTS.get(key, 0) < 2 or (failed and _FAILURE_SNAPSHOTS < 8))
                and _SNAPSHOT_ROWS < int(os.environ.get("MILES_SCORE_CENTERING_FULL_LOGIT_ROWS", "256"))
            ):
                _SNAPSHOT_COUNTS[key] = _SNAPSHOT_COUNTS.get(key, 0) + 1
                _SNAPSHOT_ROWS += 1
                _FAILURE_SNAPSHOTS += int(failed)
                row["snapshot_row"] = len(saved_rows)
                saved_logits.append(row_logits.copy())
                saved_probs.append(actual.copy() if actual is not None else np.full_like(row_logits, np.nan))
                saved_rows.append({**request, **row, "order": order})
            rows.append(row)
        effective = {key: values[batch_index] for key, values in (effective_sampling or {}).items()}
        records.append(
            {**request, "effective_sampling": effective, "order": order, "block_size": block_size, "rows": rows}
        )
    stem = f"verify-{os.getpid()}-{_BLOCKS:06d}"
    if saved_rows:
        np.savez_compressed(
            directory / f"{stem}.npz", logits=np.stack(saved_logits), kernel_probs=np.stack(saved_probs)
        )
        (directory / f"{stem}.rows.json").write_text(json.dumps(saved_rows))
    with (directory / f"verify-{os.getpid()}.jsonl").open("a") as out:
        for record in records:
            out.write(json.dumps({**record, "snapshot": f"{stem}.npz"}) + "\n")


def _describe_row(logits, actual, selected, request, order):
    unfiltered = reference_distribution(logits, temperature=request["temperature"], top_k=-1, top_p=1.0, order=order)
    filtered = reference_distribution(
        logits, temperature=request["temperature"], top_k=request["top_k"], top_p=request["top_p"], order=order
    )
    # The API may choose any tied IDs at the head boundary. Keep all boundary
    # ties in the diagnostic reference, so a valid alternative is comparable.
    cutoff = np.partition(unfiltered, -min(128, len(unfiltered)))[-min(128, len(unfiltered))]
    top_ids = np.flatnonzero(unfiltered >= cutoff)
    bounded = 0 < request["top_k"] < len(logits) or request["top_p"] < 1
    support_ids = np.flatnonzero(actual > 0) if bounded and actual is not None else np.empty(0, dtype=int)
    return {
        "token": selected,
        "selected_logprob": float(np.log(unfiltered[selected])),
        "top_ids": top_ids.tolist(),
        "top_logprobs": np.log(unfiltered[top_ids]).tolist(),
        "support_ids": support_ids.tolist(),
        "support_logprobs": np.log(actual[support_ids]).tolist() if len(support_ids) else [],
        "kernel_observed": actual is not None,
        "kernel_l1_error": float(np.abs(actual - filtered).sum()) if actual is not None else None,
        "kernel_max_error": float(np.abs(actual - filtered).max()) if actual is not None else None,
    }


def install() -> None:
    if not os.environ.get("MILES_SCORE_CENTERING_TRACE_DIR"):
        return
    # Import hooks only inside the installed SGLang plugin entry point.
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    worker = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2"
    HookRegistry.register(f"{worker}.forward_batch_generation", _with_batch, HookType.AROUND)
    HookRegistry.register(f"{worker}._accept_block", _observe_accept, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.speculative.dflash_utils.tree_speculative_sampling_target_only", _observe_kernel, HookType.AROUND
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.dspark.dspark_accept.chain_speculative_sampling_triton",
        _observe_kernel,
        HookType.AROUND,
    )
