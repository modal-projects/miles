import logging
from typing import Any

import numpy as np
import torch

from miles.rollout.failures import is_loss_masked_failure
from miles.utils import object_store
from miles.utils.object_store import ValueSpec
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.timer import Timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": "int32",
    "loss_masks": "int32",
    "rollout_log_probs": "float32",
    "rollout_sampling_mask_ids": "int32",
    "rollout_sampling_mask_offsets": "int32",
    "teacher_log_probs": "float32",
    "opd_reverse_kl": "float32",
    "rollout_routed_experts": "int32",
    "rollout_indexer_topk": "int32",
}

ROLLOUT_DATA_VALUE_SPEC: dict[str, ValueSpec] = {
    **{field: ValueSpec(codec="typed_ragged") for field in ROLLOUT_DATA_TENSOR_DTYPES},
    "partition": ValueSpec(codec="ndarray", dtype="int64"),
    "seq_witness_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
    "rewards": ValueSpec(codec="ndarray", dtype="float32"),
    "truncated": ValueSpec(codec="ndarray", dtype="int64"),
    "round_number": ValueSpec(codec="ndarray", dtype="int64"),
    "sample_indices": ValueSpec(codec="ndarray", dtype="int64"),
    "loss_denominator_mask": ValueSpec(codec="ndarray", dtype="bool"),
    "loss_global_batch_sizes": ValueSpec(codec="ndarray", dtype="int64"),
    "multimodal_train_inputs": ValueSpec(codec="ragged_tensor_dict"),
    "prompt": ValueSpec(codec="msgpack_ragged"),
    "metadata": ValueSpec(codec="msgpack_ragged"),
    "weight_versions": ValueSpec(codec="msgpack_ragged"),
    "raw_reward": ValueSpec(codec="auto"),
    "total_lengths": ValueSpec(codec="auto"),
    "dynamic_global_batch_size": ValueSpec(codec="auto"),
    "prompt_group_sizes": ValueSpec(codec="auto"),
}

ROUTING_REPLAY_LAYER_INDICES_KEY = "rollout_routed_experts_layer_indices"
ROUTING_REPLAY_VALUE_SPEC = {
    "rollout_routed_experts": ValueSpec(codec="typed_ragged"),
    ROUTING_REPLAY_LAYER_INDICES_KEY: ValueSpec(codec="ndarray", dtype="int64"),
}


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        return f(args, samples)

    for sample in samples:
        sample.validate()

    has_rollout_log_probs = [sample.rollout_log_probs is not None for sample in samples]
    if any(has_rollout_log_probs) and not all(has_rollout_log_probs):
        missing_indices = [
            sample.index
            for sample, present in zip(
                samples,
                has_rollout_log_probs,
                strict=True,
            )
            if not present
        ]
        raise ValueError(
            "rollout_log_probs must be present for every sample or none; "
            f"missing sample indices={missing_indices[:20]}"
        )
    if (getattr(args, "use_rollout_logprobs", False) or getattr(args, "use_tis", False)) and not all(
        has_rollout_log_probs
    ):
        raise ValueError("Async behavior-policy correction requires rollout_log_probs for " "every sample")

    raw_rewards, rewards = _post_process_rewards(
        args,
        samples,
        custom_reward_post_process_func=custom_reward_post_process_func,
        prompt_group_sizes=metadata.get("prompt_group_sizes"),
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks
    loss_denominator_mask = [not is_loss_masked_failure(sample) for sample in samples]
    if not all(loss_denominator_mask):
        train_data["loss_denominator_mask"] = loss_denominator_mask

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if all(has_rollout_log_probs):
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if args.rollout_top_p < 1.0:
        sampling_mask_ids = []
        sampling_mask_offsets = []
        synthesized_indices = []
        for position, (sample, loss_mask) in enumerate(zip(samples, loss_masks, strict=True)):
            sample.validate()
            ids = sample.rollout_sampling_mask_ids
            offsets = sample.rollout_sampling_mask_offsets
            if ids is None:
                is_zero_loss_placeholder = (
                    is_loss_masked_failure(sample) and sample.remove_sample and not any(loss_mask)
                )
                if not is_zero_loss_placeholder:
                    raise ValueError(
                        "--rollout-top-p < 1 requires sampling-mask data for every active training sample; "
                        f"position={position}, sample_index={sample.index}, group_index={sample.group_index}, "
                        f"response_length={sample.response_length}, remove_sample={sample.remove_sample}, "
                        f"loss_mask_sum={sum(loss_mask)}, status={sample.status}, "
                        f"exit_status={sample.metadata.get('exit_status')!r}, "
                        f"sampling_mask_ids_present={sample.rollout_sampling_mask_ids is not None}, "
                        f"sampling_mask_offsets_present={sample.rollout_sampling_mask_offsets is not None}"
                    )

                # This row is excluded from reward normalization, the loss
                # denominator, and token loss. Singleton support keeps the
                # top-p replay tensors shape-safe without inventing a policy
                # distribution for a trainable token.
                response_tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
                ids = [int(token_id) for token_id in response_tokens]
                offsets = list(range(sample.response_length + 1))
                if "rollout_log_probs" in train_data:
                    train_data["rollout_log_probs"][position] = [0.0] * sample.response_length
                synthesized_indices.append(sample.index)

            sampling_mask_ids.append(ids)
            sampling_mask_offsets.append(offsets)

        if synthesized_indices:
            logger.warning(
                "Synthesized singleton top-p masks for %d zero-loss failure placeholders; " "sample_indices=%s",
                len(synthesized_indices),
                synthesized_indices[:20],
            )
        train_data["rollout_sampling_mask_ids"] = sampling_mask_ids
        train_data["rollout_sampling_mask_offsets"] = sampling_mask_offsets

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [sample.weight_versions for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if any(sample.adapter is not None for sample in samples):
        assert all(sample.adapter is not None for sample in samples), "Cannot mix adapter and adapter-less samples"
        train_data["adapter_slots"] = [sample.adapter.slot for sample in samples]
        # Slots whose adapter batch completes with this batch: the trainer scales their
        # accumulated gradients by 1/adapter-batch-size and advances the LR schedule.
        step_slots = sorted(metadata.get("step_slots", []))
        train_data["step_slots"] = step_slots
        train_data["step_adapter_names"] = sorted(metadata.get("step_adapter_names", []))
        step_slot_set = set(step_slots)
        train_data["step_adapter_batch_sizes"] = {
            sample.adapter.slot: sample.metadata["adapter_global_batch_size"]
            for sample in samples
            if sample.adapter.slot in step_slot_set
        }

    if (prompt_group_sizes := metadata.get("prompt_group_sizes")) is not None:
        train_data["prompt_group_sizes"] = prompt_group_sizes

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _post_process_rewards(
    args,
    samples: list[Sample] | list[list[Sample]],
    custom_reward_post_process_func,
    prompt_group_sizes: list[int] | None = None,
):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    valid_mask = torch.tensor(
        [not is_loss_masked_failure(sample) for sample in samples],
        dtype=torch.bool,
    )

    # Keep ordinary homogeneous rollouts on Miles' established normalization
    # path. Explicit group boundaries below exist only for Multi-LoRA or
    # batches carrying masked infrastructure placeholders.
    if (
        valid_mask.all()
        and prompt_group_sizes is None
        and args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        rewards = rewards - rewards.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
        return raw_rewards, rewards.flatten().tolist()

    if prompt_group_sizes is None:
        group_size = args.n_samples_per_prompt
        if group_size > 0 and len(samples) % group_size == 0:
            prompt_group_sizes = [group_size] * (len(samples) // group_size)
        else:
            prompt_group_sizes = [len(samples)]
    assert sum(prompt_group_sizes) == len(
        raw_rewards
    ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(raw_rewards)} rewards"

    if (
        args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        normalized = torch.zeros_like(rewards)
        offset = 0
        for size in prompt_group_sizes:
            group_rewards = rewards[offset : offset + size]
            group_valid = valid_mask[offset : offset + size]
            valid_rewards = group_rewards[group_valid]
            if valid_rewards.numel():
                centered = valid_rewards - valid_rewards.mean()
                if (
                    args.advantage_estimator in ["grpo", "gspo", "cispo"]
                    and args.grpo_std_normalization
                    and valid_rewards.numel() > 1
                ):
                    centered = centered / (valid_rewards.std() + 1e-6)
                normalized[offset : offset + size][group_valid] = centered
            offset += size
        return raw_rewards, normalized.tolist()

    processed_rewards = [
        reward if is_valid else 0.0 for reward, is_valid in zip(raw_rewards, valid_mask.tolist(), strict=True)
    ]
    return raw_rewards, processed_rewards


def split_train_data_by_dp(args, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data_list = split_train_data_by_dp_raw(args, data, dp_size=dp_size)
    store = object_store.get_instance()
    return [store.put(value=rollout_data, value_spec=ROLLOUT_DATA_VALUE_SPEC) for rollout_data in rollout_data_list]


def put_train_data(args, data, train_parallel_config: dict[str, Any]) -> dict[str, Any]:
    """Store rollout data in the layout consumed by the trainer ranks."""
    store = object_store.get_instance()
    if args.delay_split_train_data_by_dp:
        return {"data_ref": store.put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)}

    dp_size = train_parallel_config["dp_size"]
    dp_shards = split_train_data_by_dp_raw(args, data, dp_size=dp_size)
    routing_specs = train_parallel_config.get("routing_replay_specs")
    if "rollout_routed_experts" not in data:
        return {"data_ref": [store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC) for shard in dp_shards]}
    if routing_specs is None:
        raise RuntimeError("Trainer did not provide its routing replay topology")

    routing_by_dp = [shard.pop("rollout_routed_experts") for shard in dp_shards]
    data_refs = [store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC) for shard in dp_shards]

    wire_dtype = _routing_replay_wire_dtype(args.num_experts)
    refs_by_spec = {}
    routing_replay_refs = []
    source_bytes = 0
    stored_bytes = 0
    for rank, spec in enumerate(routing_specs):
        if spec["rank"] != rank:
            raise ValueError(
                "Routing replay specs must be ordered by trainer rank; " f"position={rank}, spec_rank={spec['rank']}"
            )
        dp_rank = int(spec["dp_rank"])
        if not 0 <= dp_rank < dp_size:
            raise ValueError(
                f"Trainer rank {rank} has invalid routing replay DP rank " f"{dp_rank} for dp_size={dp_size}"
            )
        layer_indices = tuple(int(index) for index in spec["layer_indices"])
        if len(set(layer_indices)) != len(layer_indices) or any(index < 0 for index in layer_indices):
            raise ValueError(f"Trainer rank {rank} has invalid routing replay layers: " f"{layer_indices}")

        key = (dp_rank, layer_indices)
        if key not in refs_by_spec:
            routed_experts = []
            for sample in routing_by_dp[dp_rank]:
                sample = np.asarray(sample)
                if sample.dtype != np.int32 or sample.ndim != 3:
                    raise ValueError(
                        "rollout_routed_experts must contain int32 arrays with "
                        f"shape [tokens, layers, topk], got dtype={sample.dtype}, "
                        f"shape={sample.shape}"
                    )
                if sample.shape[2] != args.moe_router_topk:
                    raise ValueError(
                        "rollout_routed_experts has an invalid topk dimension: "
                        f"shape={sample.shape}, expected_topk={args.moe_router_topk}"
                    )
                if layer_indices and max(layer_indices) >= sample.shape[1]:
                    raise ValueError(
                        f"Routing replay layer {max(layer_indices)} is outside "
                        f"the rollout tensor's {sample.shape[1]} layers"
                    )
                local = sample[:, layer_indices, :]
                _validate_routing_replay_ids(local, args.num_experts)
                compact = np.ascontiguousarray(local, dtype=wire_dtype)
                routed_experts.append(compact)
                source_bytes += local.nbytes
                stored_bytes += compact.nbytes

            refs_by_spec[key] = store.put(
                value={
                    "rollout_routed_experts": routed_experts,
                    ROUTING_REPLAY_LAYER_INDICES_KEY: np.asarray(
                        layer_indices,
                        dtype=np.int64,
                    ),
                },
                value_spec=ROUTING_REPLAY_VALUE_SPEC,
            )
        routing_replay_refs.append(refs_by_spec[key])

    logger.info(
        "Stored routing replay for %d trainer ranks as %d unique shards " "(%s -> %s)",
        len(routing_specs),
        len(refs_by_spec),
        _format_bytes(source_bytes),
        _format_bytes(stored_bytes),
    )
    return {"data_ref": data_refs, "routing_replay_refs": routing_replay_refs}


def _routing_replay_wire_dtype(num_experts: int) -> np.dtype:
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if num_experts <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if num_experts <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


def _validate_routing_replay_ids(data: np.ndarray, num_experts: int) -> None:
    if data.size == 0:
        return
    minimum = int(data.min())
    maximum = int(data.max())
    if minimum < 0 or maximum >= num_experts:
        raise ValueError(
            "Routing replay contains an invalid expert ID: " f"range=[{minimum}, {maximum}], num_experts={num_experts}"
        )


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    # Multi-LoRA: sort partitions by adapter slot so each microbatch is
    # contiguous-by-slot (required by the per-adapter token-count math).
    adapter_slots = data.get("adapter_slots")
    if adapter_slots is not None:
        partitions = [sorted(p, key=lambda i: adapter_slots[i]) for p in partitions]

    loss_global_batch_sizes = _compute_loss_global_batch_sizes(args, data, partitions, dp_size)
    shards = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_log_probs",
            "rollout_sampling_mask_ids",
            "rollout_sampling_mask_offsets",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "adapter_slots",
            "loss_denominator_mask",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        if loss_global_batch_sizes is not None:
            rollout_data["loss_global_batch_sizes"] = loss_global_batch_sizes[i]
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
            "step_slots",
            "step_adapter_names",
            "step_adapter_batch_sizes",
            "prompt_group_sizes",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        if "adapter_slots" in rollout_data:
            rollout_data["n_adapters"] = args.multi_lora_n_adapters
        shards.append(rollout_data)
    return shards


def _compute_loss_global_batch_sizes(
    args,
    data,
    partitions,
    dp_size: int,
) -> list[list[int]] | None:
    """Return the effective sample denominator for each local training row.

    Infrastructure placeholders preserve the fixed tensor shape but have a
    zero loss mask. Normalizing by the configured batch size would therefore
    shrink every gradient in proportion to the infrastructure failure rate.
    Compute the denominator after DP partitioning so every microbatch in a
    training step uses the number of real trajectories in that exact step.
    """
    denominator_mask = data.get("loss_denominator_mask")
    if denominator_mask is None:
        return None

    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    if global_batch_size % dp_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} must be divisible by dp_size={dp_size}")
    local_batch_size = global_batch_size // dp_size
    if any(len(partition) % local_batch_size != 0 for partition in partitions):
        raise ValueError("Each DP partition must contain a whole number of local training batches")

    num_steps = len(partitions[0]) // local_batch_size
    result = [[] for _ in range(dp_size)]
    for step in range(num_steps):
        start = step * local_batch_size
        end = start + local_batch_size
        step_indices = [index for partition in partitions for index in partition[start:end]]
        denominator = sum(bool(denominator_mask[index]) for index in step_indices)
        if denominator == 0:
            raise ValueError(f"Training step {step} contains no trainable samples")
        for rank in range(dp_size):
            result[rank].extend([denominator] * local_batch_size)
    return result


def process_rollout_data_shard(args, rollout_data):
    """Train-side completion of the DP split: drop the ``partition`` key and
    reorder the batch-global ``total_lengths`` into this shard's row order."""
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
