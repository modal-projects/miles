from argparse import Namespace
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import all_gather_with_cp, allgather_cp_redistribute
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.sampling_mask import get_rollout_sampling_masks
from miles.utils.sampling_mask import sampling_support_replay_enabled
from miles.utils.score_centering import RolloutScoreCenteringHead


def get_rollout_score_centering_heads(batch: Mapping[str, object]) -> list[RolloutScoreCenteringHead]:
    ids_batch = batch.get("rollout_score_centering_head_ids")
    offsets_batch = batch.get("rollout_score_centering_head_offsets")
    logprobs_batch = batch.get("rollout_score_centering_head_logprobs")
    if ids_batch is None or offsets_batch is None or logprobs_batch is None:
        raise ValueError("modeled-tail score centering requires all sampler-head wire fields")
    if not all(isinstance(values, Sequence) for values in (ids_batch, offsets_batch, logprobs_batch)):
        raise TypeError("score-centering head wire fields must be sequences with one entry per sample")
    if len(ids_batch) != len(offsets_batch) or len(ids_batch) != len(logprobs_batch):
        raise ValueError("score-centering head wire fields must have the same batch size")
    return [
        RolloutScoreCenteringHead(
            ids=torch.as_tensor(ids),
            offsets=torch.as_tensor(offsets),
            logprobs=torch.as_tensor(logprobs),
        )
        for ids, offsets, logprobs in zip(ids_batch, offsets_batch, logprobs_batch, strict=True)
    ]


def get_score_centered_log_probs(
    logits: torch.Tensor,
    *,
    args: Namespace,
    batch: Mapping[str, object],
) -> dict[str, list[torch.Tensor]]:
    """Score response tokens and construct the detached centering coefficients."""
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens")
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("score centering requires the real tokenizer vocab size")

    exact_support = sampling_support_replay_enabled(args)
    sampling_masks = get_rollout_sampling_masks(batch) if exact_support else None
    sampler_heads = None if exact_support else get_rollout_score_centering_heads(batch)
    if exact_support:
        if len(sampling_masks) != len(response_lengths):
            raise ValueError("sampling-mask batch size must match response-length batch size")
        for sample_index, (sampling_mask, response_length) in enumerate(
            zip(sampling_masks, response_lengths, strict=True)
        ):
            if len(sampling_mask) != response_length:
                raise ValueError(
                    f"sampling-mask length {len(sampling_mask)} != response length {response_length} for sample {sample_index}"
                )
            if sampling_mask._as_distribution_tensors()[2] is None:
                raise ValueError(
                    f"sampling-mask support logprobs are missing for score-centering sample {sample_index}"
                )
    else:
        if len(sampler_heads) != len(response_lengths):
            raise ValueError("sampler-head batch size must match response-length batch size")
        for sample_index, (sampler_head, response_length) in enumerate(
            zip(sampler_heads, response_lengths, strict=True)
        ):
            if len(sampler_head) != response_length:
                raise ValueError(
                    f"sampler-head length {len(sampler_head)} != response length {response_length} for sample {sample_index}"
                )

    result = {
        "log_probs": [],
        "sampler_log_probs": [],
        "corrections": [],
        "importance_weights": [],
        "sampler_mass": [],
        "train_mass": [],
    }
    response_chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=True,
    )
    for sample_index, (logits_chunk, tokens_chunk, response_indices) in enumerate(response_chunks):
        scaled_logits = logits_chunk.float()
        if not args.true_on_policy_mode and args.rollout_temperature != 1.0:
            scaled_logits = scaled_logits / args.rollout_temperature

        if exact_support:
            chunk_result = _score_exact_support(
                scaled_logits,
                tokens_chunk,
                response_indices,
                sampling_masks[sample_index],
                args=args,
            )
        else:
            sampler_token_logprobs = _select_sampler_token_logprobs(
                batch["rollout_log_probs"][sample_index],
                response_indices,
                total_length=total_lengths[sample_index],
                response_length=response_lengths[sample_index],
                max_seq_len=max_seq_lens[sample_index] if max_seq_lens is not None else None,
                args=args,
            )
            chunk_result = _score_modeled_tail(
                scaled_logits,
                tokens_chunk,
                sampler_token_logprobs,
                response_indices,
                sampler_heads[sample_index],
                args=args,
                vocab_size=vocab_size,
            )
        for key, value in chunk_result.items():
            result[key].append(value)

    if args.allgather_cp:
        allgather_cp_redistribute(
            result,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    return result


def sparse_vocab_parallel_log_probs(
    logits: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None,
    vocab_size: int,
    chunk_size: int,
    normalize_over_support: bool,
) -> torch.Tensor:
    """Return selected log-probs without materializing full-vocabulary log-softmax."""
    ids = ids.to(device=logits.device, dtype=torch.long)
    offsets = offsets.to(device=logits.device, dtype=torch.long)
    _validate_sparse_queries(logits, ids, offsets, vocab_size)
    if logits.size(0) == 0:
        return logits.sum(dim=-1)
    return _SparseVocabParallelLogProbs.apply(
        logits,
        ids,
        offsets,
        process_group,
        vocab_size,
        chunk_size,
        normalize_over_support,
    )


def _score_exact_support(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    response_indices: Sequence[int],
    sampling_mask,
    *,
    args: Namespace,
) -> dict[str, torch.Tensor]:
    support_ids, sampler_support_logprobs, lengths = sampling_mask._select_distribution(response_indices)
    offsets = _lengths_to_offsets(lengths, logits.device)
    support_ids = support_ids.to(logits.device)
    sampler_support_logprobs = sampler_support_logprobs.to(logits.device)
    actor_support_logprobs = sparse_vocab_parallel_log_probs(
        logits,
        support_ids,
        offsets,
        process_group=get_parallel_state().tp.group,
        vocab_size=args.vocab_size,
        chunk_size=_chunk_size(args),
        normalize_over_support=True,
    )

    row_indices = torch.repeat_interleave(
        torch.arange(logits.size(0), device=logits.device),
        lengths.to(device=logits.device, dtype=torch.long),
    )
    sampled = support_ids == tokens[row_indices]
    sampled_flat_indices = torch.nonzero(sampled, as_tuple=False).flatten()
    if sampled_flat_indices.numel() != logits.size(0) or not torch.equal(
        row_indices[sampled_flat_indices], torch.arange(logits.size(0), device=logits.device)
    ):
        raise ValueError("every sampled token must occur exactly once in its replay support")

    ratios = (actor_support_logprobs - sampler_support_logprobs).exp()
    weights = _importance_weights(ratios, args)
    corrections = _segment_sum(
        sampler_support_logprobs.exp() * weights.detach() * actor_support_logprobs,
        row_indices,
        logits.size(0),
    )
    return {
        "log_probs": actor_support_logprobs[sampled_flat_indices],
        "sampler_log_probs": sampler_support_logprobs[sampled_flat_indices],
        "corrections": corrections,
        "importance_weights": weights[sampled_flat_indices].detach(),
        "sampler_mass": logits.new_ones(logits.size(0)),
        "train_mass": logits.new_ones(logits.size(0)),
    }


def _score_modeled_tail(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    sampler_token_logprobs: torch.Tensor,
    response_indices: Sequence[int],
    sampler_head: RolloutScoreCenteringHead,
    *,
    args: Namespace,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    head_ids, sampler_head_logprobs, head_lengths = sampler_head._select_rows(response_indices)
    head_ids = head_ids.to(device=logits.device, dtype=torch.long)
    sampler_head_logprobs = sampler_head_logprobs.to(logits.device)
    head_lengths = head_lengths.to(device=logits.device, dtype=torch.long)

    query_lengths = head_lengths + 1
    query_offsets = _lengths_to_offsets(query_lengths, logits.device)
    token_positions = query_offsets[1:] - 1
    query_ids = torch.empty(query_offsets[-1], dtype=torch.long, device=logits.device)
    is_token = torch.zeros_like(query_ids, dtype=torch.bool)
    is_token[token_positions] = True
    query_ids[is_token] = tokens
    query_ids[~is_token] = head_ids
    query_logprobs = sparse_vocab_parallel_log_probs(
        logits,
        query_ids,
        query_offsets,
        process_group=get_parallel_state().tp.group,
        vocab_size=vocab_size,
        chunk_size=_chunk_size(args),
        normalize_over_support=False,
    )
    actor_token_logprobs = query_logprobs[is_token]
    actor_head_logprobs = query_logprobs[~is_token]

    head_rows = torch.repeat_interleave(
        torch.arange(logits.size(0), device=logits.device),
        head_lengths,
    )
    p_head = actor_head_logprobs.exp()
    q_head = sampler_head_logprobs.exp()
    p_mass = _segment_sum(p_head, head_rows, logits.size(0))
    q_mass = _segment_sum(q_head, head_rows, logits.size(0))
    p_tail = (1.0 - p_mass).clamp_min(1e-6)
    q_tail = (1.0 - q_mass).clamp_min(1e-6)
    rho = q_tail / p_tail
    alpha = rho * _importance_weights(rho.reciprocal(), args)

    head_weights = _importance_weights((actor_head_logprobs - sampler_head_logprobs).exp(), args)
    residual = q_head * head_weights - alpha[head_rows] * p_head
    corrections = _segment_sum(residual.detach() * actor_head_logprobs, head_rows, logits.size(0))
    token_weights = _importance_weights((actor_token_logprobs - sampler_token_logprobs).exp(), args)
    return {
        "log_probs": actor_token_logprobs,
        "sampler_log_probs": sampler_token_logprobs,
        "corrections": corrections,
        "importance_weights": token_weights.detach(),
        "sampler_mass": q_mass.detach(),
        "train_mass": p_mass.detach(),
    }


def _select_sampler_token_logprobs(
    local_logprobs: torch.Tensor,
    response_indices: Sequence[int],
    *,
    total_length: int,
    response_length: int,
    max_seq_len: int | None,
    args: Namespace,
) -> torch.Tensor:
    if args.allgather_cp:
        full_logprobs = all_gather_with_cp(
            local_logprobs,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_len,
        )
        indices = torch.as_tensor(response_indices, dtype=torch.long, device=full_logprobs.device)
        return full_logprobs[indices].detach()
    if local_logprobs.numel() != len(response_indices):
        raise ValueError(
            f"sampler token logprobs have {local_logprobs.numel()} rows, expected {len(response_indices)}"
        )
    return local_logprobs.detach()


def _importance_weights(ratios: torch.Tensor, args: Namespace) -> torch.Tensor:
    if not args.use_tis:
        return torch.ones_like(ratios)
    if args.custom_tis_function_path is not None:
        return torch.where(
            (ratios >= args.tis_clip_low) & (ratios <= args.tis_clip),
            ratios,
            torch.zeros_like(ratios),
        )
    return ratios.clamp(min=args.tis_clip_low, max=args.tis_clip)


def _lengths_to_offsets(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    lengths = lengths.to(device=device, dtype=torch.long)
    return torch.cat((torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)))


def _segment_sum(values: torch.Tensor, row_indices: torch.Tensor, row_count: int) -> torch.Tensor:
    result = values.new_zeros(row_count)
    result.scatter_add_(0, row_indices, values)
    return result


def _chunk_size(args: Namespace) -> int:
    configured = int(args.log_probs_chunk_size)
    return configured if configured > 0 else 1024


def _validate_sparse_queries(
    logits: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    vocab_size: int,
) -> None:
    if logits.ndim != 2:
        raise ValueError(f"sparse logprob logits must have shape [rows, local_vocab], got {logits.shape}")
    if ids.ndim != 1 or offsets.ndim != 1:
        raise ValueError("sparse logprob ids and offsets must be one-dimensional")
    if offsets.numel() != logits.size(0) + 1 or offsets[0] != 0 or offsets[-1] != ids.numel():
        raise ValueError("sparse logprob offsets must delimit exactly one query row per logit row")
    if torch.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("every sparse logprob row must contain at least one query")
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError(f"sparse logprob token ids must lie in [0, {vocab_size})")


class _SparseVocabParallelLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        ids: torch.Tensor,
        offsets: torch.Tensor,
        process_group: dist.ProcessGroup | None,
        vocab_size: int,
        chunk_size: int,
        normalize_over_support: bool,
    ) -> torch.Tensor:
        world_size = 1 if process_group is None else dist.get_world_size(process_group)
        rank = 0 if process_group is None else dist.get_rank(process_group)
        local_vocab_size = logits.size(1)
        if vocab_size > local_vocab_size * world_size:
            raise ValueError("real vocab size exceeds the tensor-parallel padded vocabulary")

        row_lengths = offsets[1:] - offsets[:-1]
        query_rows = torch.repeat_interleave(torch.arange(logits.size(0), device=logits.device), row_lengths)
        selected_logits = _gather_selected_logits(logits, ids, query_rows, process_group, rank)
        if normalize_over_support:
            row_max = logits.new_full((logits.size(0),), float("-inf"))
            row_max.scatter_reduce_(0, query_rows, selected_logits, reduce="amax", include_self=True)
            exp_selected = (selected_logits - row_max[query_rows]).exp()
            denominator = _segment_sum(exp_selected, query_rows, logits.size(0))
            probabilities = exp_selected / denominator[query_rows]
            output = selected_logits - row_max[query_rows] - denominator[query_rows].log()
            ctx.save_for_backward(ids, query_rows, probabilities)
        else:
            row_max, denominator = _full_vocab_normalizers(
                logits,
                process_group=process_group,
                rank=rank,
                world_size=world_size,
                vocab_size=vocab_size,
                chunk_size=chunk_size,
            )
            output = selected_logits - row_max[query_rows] - denominator[query_rows].log()
            ctx.save_for_backward(logits, ids, query_rows, row_max, denominator)

        ctx.process_group = process_group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.local_vocab_size = local_vocab_size
        ctx.vocab_size = vocab_size
        ctx.chunk_size = chunk_size
        ctx.normalize_over_support = normalize_over_support
        ctx.row_count = logits.size(0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.normalize_over_support:
            ids, query_rows, probabilities = ctx.saved_tensors
            row_grad = _segment_sum(grad_output, query_rows, ctx.row_count)
            selected_grad = grad_output - probabilities * row_grad[query_rows]
            grad_logits = grad_output.new_zeros((row_grad.numel(), ctx.local_vocab_size))
        else:
            logits, ids, query_rows, row_max, denominator = ctx.saved_tensors
            row_count = logits.size(0)
            row_grad = _segment_sum(grad_output, query_rows, row_count)
            grad_logits = torch.empty_like(logits)
            for start in range(0, row_count, ctx.chunk_size):
                end = min(start + ctx.chunk_size, row_count)
                chunk = logits[start:end]
                probabilities = (chunk - row_max[start:end, None]).exp() / denominator[start:end, None]
                _zero_padded_probabilities_(
                    probabilities,
                    rank=ctx.rank,
                    local_vocab_size=ctx.local_vocab_size,
                    vocab_size=ctx.vocab_size,
                )
                grad_logits[start:end] = -probabilities * row_grad[start:end, None]
            selected_grad = grad_output

        vocab_start = ctx.rank * ctx.local_vocab_size
        is_local = (ids >= vocab_start) & (ids < vocab_start + ctx.local_vocab_size)
        flat_indices = query_rows[is_local] * ctx.local_vocab_size + ids[is_local] - vocab_start
        grad_logits.view(-1).index_add_(0, flat_indices, selected_grad[is_local])
        return grad_logits, None, None, None, None, None, None


def _gather_selected_logits(
    logits: torch.Tensor,
    ids: torch.Tensor,
    query_rows: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    rank: int,
) -> torch.Tensor:
    local_vocab_size = logits.size(1)
    vocab_start = rank * local_vocab_size
    is_local = (ids >= vocab_start) & (ids < vocab_start + local_vocab_size)
    selected = logits.new_zeros(ids.numel())
    selected[is_local] = logits[query_rows[is_local], ids[is_local] - vocab_start]
    if process_group is not None and dist.get_world_size(process_group) > 1:
        dist.all_reduce(selected, group=process_group)
    return selected


def _full_vocab_normalizers(
    logits: torch.Tensor,
    *,
    process_group: dist.ProcessGroup | None,
    rank: int,
    world_size: int,
    vocab_size: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_max_parts = []
    denominator_parts = []
    for start in range(0, logits.size(0), chunk_size):
        chunk = logits[start : start + chunk_size]
        valid_local_size = min(logits.size(1), max(0, vocab_size - rank * logits.size(1)))
        if valid_local_size == 0:
            local_max = chunk.new_full((chunk.size(0),), float("-inf"))
        else:
            local_max = chunk[:, :valid_local_size].max(dim=-1).values
        if world_size > 1:
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=process_group)
        exp_logits = (chunk - local_max[:, None]).exp()
        _zero_padded_probabilities_(
            exp_logits,
            rank=rank,
            local_vocab_size=logits.size(1),
            vocab_size=vocab_size,
        )
        denominator = exp_logits.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(denominator, group=process_group)
        row_max_parts.append(local_max)
        denominator_parts.append(denominator)
    return torch.cat(row_max_parts), torch.cat(denominator_parts)


def _zero_padded_probabilities_(
    values: torch.Tensor,
    *,
    rank: int,
    local_vocab_size: int,
    vocab_size: int,
) -> None:
    valid_local_size = min(local_vocab_size, max(0, vocab_size - rank * local_vocab_size))
    if valid_local_size < local_vocab_size:
        values[:, valid_local_size:] = 0
