from collections.abc import Sequence

from miles.backends.training_utils.weight_update.checksum_utils import hash_tensor_sha256


def record_lora_checksums(bucket, checksums) -> None:
    """Accumulate the sha256 checksums the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        checksums[lora_name][hf_key] = hash_tensor_sha256(tensor)


def engine_stat_metrics(replies: Sequence, phase: str) -> dict[str, float]:
    """Each numeric stat the engines report per GPU, at its largest: the slowest GPU sets the pace."""
    gpus = [gpu for reply in replies if isinstance(reply, dict) for gpu in reply.get("stats") or () if gpu]
    keys = {key for gpu in gpus for key, value in gpu.items() if isinstance(value, int | float)}
    return {
        f"perf/update_weights_engine_{phase}_{key}": max(gpu[key] for gpu in gpus if key in gpu)
        for key in sorted(keys)
    }
