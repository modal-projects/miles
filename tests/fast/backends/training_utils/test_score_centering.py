from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import miles.backends.training_utils.loss_hub.losses as losses
import miles.backends.training_utils.score_centering as score_centering
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.score_centering import RolloutScoreCenteringHead


def _args(*, correction: str = "none") -> Namespace:
    return Namespace(
        custom_tis_function_path=(
            ("miles.backends.training_utils.loss_hub.corrections:icepop_function") if correction == "mis" else None
        ),
        log_probs_chunk_size=1,
        tis_clip=1.4,
        tis_clip_low=0.6,
        use_tis=correction != "none",
        vocab_size=5,
    )


def _weights(ratio: torch.Tensor, correction: str) -> torch.Tensor:
    if correction == "none":
        return torch.ones_like(ratio)
    if correction == "tis":
        return ratio.clamp(0.6, 1.4)
    return torch.where((ratio >= 0.6) & (ratio <= 1.4), ratio, torch.zeros_like(ratio))


def _run_tensor_parallel_sparse_logprob_check(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        padded_logits = torch.tensor(
            [[0.2, -0.3, 1.1, 0.7, -0.8, 50.0], [1.2, -0.4, 0.3, -0.2, 0.9, 60.0]],
            dtype=torch.float32,
        )
        ids = torch.tensor([2, 2, 4, 0, 3])
        offsets = torch.tensor([0, 3, 5])
        coefficients = torch.tensor([0.7, -0.2, 1.1, -0.4, 0.6])
        shard_start = rank * 3
        shard_end = shard_start + 3

        for normalize_over_support in (False, True):
            local_logits = padded_logits[:, shard_start:shard_end].clone().requires_grad_()
            reference_logits = padded_logits[:, :5].clone().requires_grad_()
            actual = score_centering.sparse_vocab_parallel_log_probs(
                local_logits,
                ids,
                offsets,
                process_group=dist.group.WORLD,
                vocab_size=5,
                chunk_size=1,
                normalize_over_support=normalize_over_support,
            )
            if normalize_over_support:
                expected = torch.cat(
                    (
                        reference_logits[0, ids[:3]].log_softmax(-1),
                        reference_logits[1, ids[3:]].log_softmax(-1),
                    )
                )
            else:
                reference_logprobs = reference_logits.log_softmax(-1)
                expected = torch.cat((reference_logprobs[0, ids[:3]], reference_logprobs[1, ids[3:]]))
            torch.testing.assert_close(actual, expected)

            (actual * coefficients).sum().backward()
            (expected * coefficients).sum().backward()
            expected_local_grad = torch.zeros_like(local_logits)
            real_shard_end = min(shard_end, reference_logits.size(1))
            if shard_start < real_shard_end:
                expected_local_grad[:, : real_shard_end - shard_start] = reference_logits.grad[
                    :, shard_start:real_shard_end
                ]
            torch.testing.assert_close(local_logits.grad, expected_local_grad)
    finally:
        dist.destroy_process_group()


def test_sparse_full_vocab_logprobs_match_dense_reference_with_duplicate_queries():
    logits = torch.tensor(
        [[0.2, -0.3, 1.1, 0.7, -0.8], [1.2, -0.4, 0.3, -0.2, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_()
    ids = torch.tensor([2, 2, 4, 0, 3])
    offsets = torch.tensor([0, 3, 5])
    coefficients = torch.tensor([0.7, -0.2, 1.1, -0.4, 0.6])

    actual = score_centering.sparse_vocab_parallel_log_probs(
        logits,
        ids,
        offsets,
        process_group=None,
        vocab_size=5,
        chunk_size=1,
        normalize_over_support=False,
    )
    reference_logprobs = reference_logits.log_softmax(-1)
    expected = torch.cat((reference_logprobs[0, ids[:3]], reference_logprobs[1, ids[3:]]))
    torch.testing.assert_close(actual, expected)

    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_sparse_support_logprobs_match_dense_reference():
    logits = torch.tensor(
        [[0.2, -0.3, 1.1, 0.7, -0.8], [1.2, -0.4, 0.3, -0.2, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_()
    ids = torch.tensor([0, 2, 4, 1, 3])
    offsets = torch.tensor([0, 3, 5])
    coefficients = torch.tensor([0.7, -0.2, 1.1, -0.4, 0.6])

    actual = score_centering.sparse_vocab_parallel_log_probs(
        logits,
        ids,
        offsets,
        process_group=None,
        vocab_size=5,
        chunk_size=1,
        normalize_over_support=True,
    )
    expected = torch.cat(
        (
            reference_logits[0, ids[:3]].log_softmax(-1),
            reference_logits[1, ids[3:]].log_softmax(-1),
        )
    )
    torch.testing.assert_close(actual, expected)

    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_sparse_logprobs_match_dense_reference_with_two_tensor_parallel_ranks(tmp_path):
    mp.spawn(
        _run_tensor_parallel_sparse_logprob_check,
        args=(2, str(tmp_path / "score-centering-tp-init")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("correction", ["none", "tis", "mis"])
def test_modeled_tail_matches_paper_reference(monkeypatch, correction: str):
    monkeypatch.setattr(
        score_centering,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    logits = torch.tensor(
        [[0.2, -0.3, 1.1, 0.7, -0.8], [1.2, -0.4, 0.3, -0.2, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_()
    sampler_logits = torch.tensor(
        [[-0.4, 0.8, 0.1, 1.0, -0.7], [0.6, -0.1, 1.2, 0.2, -0.8]],
        dtype=torch.float32,
    )
    sampler_logprobs = sampler_logits.log_softmax(-1)
    head_ids = torch.tensor([[3, 1], [2, 0]])
    sampled_tokens = torch.tensor([2, 2])
    head = RolloutScoreCenteringHead.from_rows(
        head_ids.tolist(),
        sampler_logprobs.gather(-1, head_ids).tolist(),
    )
    args = _args(correction=correction)

    actual = score_centering._score_modeled_tail(
        logits,
        sampled_tokens,
        sampler_logprobs.gather(-1, sampled_tokens[:, None]).squeeze(-1),
        range(2),
        head,
        args=args,
        vocab_size=5,
    )
    train_logprobs = reference_logits.log_softmax(-1)
    train_head_logprobs = train_logprobs.gather(-1, head_ids)
    sampler_head_logprobs = sampler_logprobs.gather(-1, head_ids)
    p_head = train_head_logprobs.exp()
    q_head = sampler_head_logprobs.exp()
    p_tail = (1 - p_head.sum(-1)).clamp_min(1e-6)
    q_tail = (1 - q_head.sum(-1)).clamp_min(1e-6)
    rho = q_tail / p_tail
    alpha = rho * _weights(rho.reciprocal(), correction)
    residual = q_head * _weights((train_head_logprobs - sampler_head_logprobs).exp(), correction)
    residual = residual - alpha[:, None] * p_head
    expected_correction = (residual.detach() * train_head_logprobs).sum(-1)
    expected_token_logprobs = train_logprobs.gather(-1, sampled_tokens[:, None]).squeeze(-1)
    expected_weights = _weights(
        (expected_token_logprobs - sampler_logprobs.gather(-1, sampled_tokens[:, None]).squeeze(-1)).exp(),
        correction,
    )

    torch.testing.assert_close(actual["log_probs"], expected_token_logprobs)
    torch.testing.assert_close(actual["corrections"], expected_correction)
    torch.testing.assert_close(actual["importance_weights"], expected_weights)
    advantages = torch.tensor([0.7, -1.1])
    actual_loss = (-advantages * (actual["importance_weights"] * actual["log_probs"] - actual["corrections"])).sum()
    expected_loss = (-advantages * (expected_weights.detach() * expected_token_logprobs - expected_correction)).sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("correction", ["none", "tis", "mis"])
def test_exact_support_matches_full_support_reference(monkeypatch, correction: str):
    monkeypatch.setattr(
        score_centering,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    logits = torch.tensor(
        [[0.2, -0.3, 1.1, 0.7, -0.8], [1.2, -0.4, 0.3, -0.2, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_()
    support_rows = [[0, 2, 4], [1, 3], [2]]
    q_rows = [
        torch.tensor([0.2, 0.5, 0.3]).log().tolist(),
        torch.tensor([0.65, 0.35]).log().tolist(),
        [0.0],
    ]
    sampling_mask = RolloutSamplingMask.from_mask_list(support_rows, q_rows)
    sampled_tokens = torch.tensor([4, 1, 2])
    args = _args(correction=correction)

    actual = score_centering._score_exact_support(
        logits,
        sampled_tokens,
        range(3),
        sampling_mask,
        args=args,
    )
    expected_logprobs = []
    expected_q = []
    expected_corrections = []
    expected_weights = []
    for row, (support, q_values, sampled) in enumerate(zip(support_rows, q_rows, sampled_tokens, strict=True)):
        support_tensor = torch.tensor(support)
        p_logprobs = reference_logits[row, support_tensor].log_softmax(-1)
        q_logprobs = torch.tensor(q_values)
        weights = _weights((p_logprobs - q_logprobs).exp(), correction)
        sampled_index = support.index(int(sampled))
        expected_logprobs.append(p_logprobs[sampled_index])
        expected_q.append(q_logprobs[sampled_index])
        expected_corrections.append((q_logprobs.exp() * weights.detach() * p_logprobs).sum())
        expected_weights.append(weights[sampled_index])
    expected_logprobs = torch.stack(expected_logprobs)
    expected_q = torch.stack(expected_q)
    expected_corrections = torch.stack(expected_corrections)
    expected_weights = torch.stack(expected_weights)

    torch.testing.assert_close(actual["log_probs"], expected_logprobs)
    torch.testing.assert_close(actual["sampler_log_probs"], expected_q)
    torch.testing.assert_close(actual["corrections"], expected_corrections)
    torch.testing.assert_close(actual["importance_weights"], expected_weights)
    advantages = torch.tensor([0.7, -1.1, 2.0])
    actual_loss = (-advantages * (actual["importance_weights"] * actual["log_probs"] - actual["corrections"])).sum()
    expected_loss = (-advantages * (expected_weights.detach() * expected_logprobs - expected_corrections)).sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits.grad[2], torch.zeros(5))


def test_policy_loss_uses_centered_reinforce_expression_and_loss_mask(monkeypatch):
    log_probs = torch.tensor([-0.5, -0.2], requires_grad=True)
    score = {
        "log_probs": [log_probs],
        "sampler_log_probs": [torch.tensor([-0.6, -0.3])],
        "corrections": [0.25 * log_probs],
        "importance_weights": [torch.tensor([1.5, 0.7])],
        "sampler_mass": [torch.tensor([0.8, 0.9])],
        "train_mass": [torch.tensor([0.7, 0.85])],
    }
    monkeypatch.setattr(losses, "get_score_centered_log_probs", lambda *_args, **_kwargs: score)
    monkeypatch.setattr(
        losses,
        "get_local_response_loss_masks",
        lambda *_args, **_kwargs: [torch.tensor([1, 0])],
    )
    monkeypatch.setattr(
        losses,
        "compute_ess_ratio_contribution",
        lambda **_kwargs: torch.tensor(0.75),
    )
    args = SimpleNamespace(
        use_score_centering=True,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_tis=False,
        use_kl_loss=False,
        qkv_format="thd",
        calculate_per_token_loss=False,
    )
    batch = {
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 0])],
        "advantages": [torch.tensor([2.0, float("nan")])],
    }

    def reducer(values):
        return values[0]

    loss, metrics = losses.policy_loss_function(
        args,
        batch,
        torch.zeros(1, 2, 5, requires_grad=True),
        reducer,
    )

    # -A * (w * log p - correction) = -2 * (1.5 * -0.5 - 0.25 * -0.5)
    torch.testing.assert_close(loss, torch.tensor(1.25))
    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(1.25))
    torch.testing.assert_close(metrics["ess_ratio"], torch.tensor(0.75))
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([-2.5, 0.0]))
