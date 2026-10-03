"""Read-only DFlash2 checksums after each rollout, before the normal offload.

The Qwen DFlash2 worker registers the target's shared ``lm_head`` on its draft
model. Its input embeddings are supplied externally by the target. Only the
registered shared head is excluded from private-draft equality; all returned
hashes and names remain in the artifact. Missing private hashes fail the probe.
"""

import json
from pathlib import Path

import httpx


def _private_checksums(payload):
    if not payload.get("success") or not payload.get("ranks"):
        raise ValueError("Draft checksum request returned no successful rank payloads")
    private, shared = {}, {}
    for rank in payload["ranks"]:
        placements = rank["parallelism_info"]
        if len(placements) != 1 or placements[0]["role"] != "draft":
            raise ValueError("Expected one DFlash draft placement per TP rank")
        index = placements[0]["tp_rank"]
        for name, checksum in rank["checksums"].items():
            if not name.startswith("draft."):
                raise ValueError(f"Unexpected non-draft checksum: {name}")
            destination = shared if name.startswith("draft.lm_head.") else private
            key = f"rank{index}/{name}"
            if key in destination:
                raise ValueError(f"Duplicate draft checksum: {key}")
            destination[key] = checksum
    if not private:
        raise ValueError("No private draft tensors were captured")
    return private, shared


def log_draft_checksums(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Existing rollout-log hook: preserve default metrics and record live hashes."""
    if getattr(args, "sglang_speculative_algorithm", None) != "DFLASH":
        return False
    root = Path(args.save_debug_rollout_data).parent.parent / "draft_checksums"
    root.mkdir(exist_ok=True)
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    captures = {}
    with httpx.Client(timeout=300.0) as client:
        response = client.get(f"{router}/workers")
        response.raise_for_status()
        urls = sorted(worker["url"] for worker in response.json()["workers"])
        expected = args.actor_num_gpus_per_node // args.rollout_num_gpus_per_engine
        if len(urls) != expected or len(set(urls)) != expected:
            raise ValueError(f"Expected {expected} distinct rollout engines, got {urls}")
        for url in urls:
            response = client.post(f"{url}/weights_checker", json={"action": "checksum", "selector": "draft"})
            response.raise_for_status()
            payload = response.json()
            private, shared = _private_checksums(payload)
            ranks = [rank["parallelism_info"][0]["tp_rank"] for rank in payload["ranks"]]
            if sorted(ranks) != list(range(args.rollout_num_gpus_per_engine)):
                raise ValueError("Incomplete draft TP rank coverage")
            captures[url] = dict(raw=payload, private=private, shared_target_head=shared)
    record = dict(rollout_id=rollout_id, engines=captures)
    # Preserve a mismatching capture before failing, including every shared hash.
    (root / f"{rollout_id}.json").write_text(json.dumps(record, indent=2))
    if rollout_id:
        baseline = json.loads((root / "0.json").read_text())["engines"]
        if captures.keys() != baseline.keys():
            raise ValueError("Draft engines changed during the paired training probe")
        for url, capture in captures.items():
            if capture["private"] != baseline[url]["private"]:
                raise ValueError(f"Private draft tensors changed after target update: {url}")
    return False
