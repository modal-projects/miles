"""Use the production request builder and collectors in live experiments."""

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import numpy as np

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata
from miles.utils.types import Sample


@dataclass(frozen=True)
class Case:
    name: str
    width: int
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0

    @property
    def mode(self) -> str:
        return "support" if self.top_k > 0 or self.top_p < 1 else "selected"

    def request(self, prompt: list[int], *, length: int, capture: bool = True) -> dict[str, Any]:
        payload = {
            "input_ids": prompt,
            "sampling_params": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "min_p": 0.0,
                "max_new_tokens": length,
            },
        }
        if capture:
            payload["return_logprob"] = True
            if self.mode == "support":
                payload["return_sampling_mask"] = True
            configure_rollout_topk_logprobs_request(
                Namespace(
                    rollout_top_logprobs_num=self.width,
                    rollout_sampling_logprobs_mode=self.mode,
                    rollout_temperature=self.temperature,
                ),
                payload,
            )
        return payload


CASES = tuple(
    Case(f"{name}_t{temperature:g}", width, temperature, top_k, top_p)
    for name, width, top_k, top_p in (
        ("unfiltered32", 32, -1, 1.0),
        ("unfiltered128", 128, -1, 1.0),
        ("topk32_capture128", 128, 32, 1.0),
        ("topk64_topp90", 128, 64, 0.9),
    )
    for temperature in (0.7, 1.0, 1.3)
)


def collect_sample(case: Case, prompt: list[int], response: dict[str, Any]) -> Sample:
    meta = response["meta_info"]
    if (meta.get("finish_reason") or {}).get("type") == "abort":
        raise ValueError(f"Generation aborted: {meta['finish_reason']}")
    generated = meta["output_token_logprobs"]
    tokens = [int(item[1]) for item in generated]
    if not tokens:
        raise ValueError(f"No generated tokens: {meta.get('finish_reason')}")
    sample = Sample(tokens=list(prompt))
    if case.mode == "support":
        selected = append_sampling_metadata(sample, tokens, meta, sampling_logprobs_mode="support")
    else:
        selected = [float(item[0]) for item in generated]
    sample.tokens.extend(tokens)
    sample.response_length = len(tokens)
    sample.rollout_log_probs = selected
    append_rollout_topk_logprobs(sample, meta, case.width, sampling_logprobs_mode=case.mode)
    validate_rollout_topk_logprobs_sample(sample, case.width)
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    valid = ids >= 0
    if not np.isfinite(logps[valid]).all() or not np.isfinite(selected).all():
        raise ValueError("Non-finite behavior log probability")
    if (logps[valid] > 1e-6).any() or (np.asarray(selected) > 1e-6).any():
        raise ValueError("Positive behavior log probability")
    mass = np.exp(logps.astype(np.float64)).sum(-1)
    if case.mode == "support":
        if not np.allclose(mass, 1.0, atol=1e-5, rtol=0):
            raise ValueError(f"Incomplete normalized support: mass={mass}")
    elif not valid.all() or (mass > 1 + 1e-5).any():
        raise ValueError("Unfiltered capture must contain the requested K candidates with mass <= 1")
    return sample
