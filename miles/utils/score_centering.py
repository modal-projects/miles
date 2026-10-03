"""Configuration contracts for score centering (arXiv:2609.20807)."""

import math
import os
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling


def validate_score_centering_args(args: Namespace) -> None:
    if getattr(args, "loss_type", None) != "score_centering":
        return
    if args.rollout_top_logprobs_num <= 0:
        raise ValueError("Score centering requires a positive --rollout-top-logprobs-num")
    if not math.isfinite(args.score_centering_tis_clip) or args.score_centering_tis_clip <= 0:
        raise ValueError("--score-centering-tis-clip must be finite and positive")
    low, high = args.score_centering_mis_low, args.score_centering_mis_high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    validate_rollout_topk_logprobs_sampling(
        {"top_k": args.rollout_top_k},
        temperature=args.rollout_temperature,
        candidate_count=args.rollout_top_logprobs_num,
    )
    if args.advantage_estimator != "grpo":
        raise ValueError("Score centering currently supports --advantage-estimator grpo (group-centered rewards)")
    incompatible = {
        "use_tis": "use --score-centering-is instead",
        "custom_tis_function_path": "only the built-in score-centering TIS/MIS weights are supported",
        "use_opsm": "sequence masking changes the score-centering estimator",
        "true_on_policy_mode": "score centering uses float32 probability arithmetic",
        "recompute_logprobs_via_prefill": "sampler probabilities must be recorded at generation time",
        "custom_pg_loss_reducer_function_path": "use the standard token/sample reducer",
        "multi_lora": "per-sample Tinker losses bypass the score-centering loss",
        "use_opd": "distillation composition is not supported",
        "custom_convert_samples_to_train_data_path": "custom converters skip rollout top-k logprobs validation",
    }
    for option, reason in incompatible.items():
        if getattr(args, option, None):
            raise ValueError(f"Score centering is incompatible with --{option.replace('_', '-')}: {reason}")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("Score centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
    # YAML groups can override these defaults; validate their effective values at
    # engine launch. External engines are checked from their own /server_info.
    if not getattr(args, "sglang_config", None) and not getattr(args, "rollout_external", False):
        validate_score_centering_speculative_config(
            {name.removeprefix("sglang_"): value for name, value in vars(args).items() if name.startswith("sglang_")},
            environ=os.environ,
        )


def validate_score_centering_speculative_config(server_args: Mapping[str, Any], *, environ: Mapping[str, str]) -> None:
    """Require committed speculative tokens to follow the captured target policy.

    DFlash and DFlash2 checkpoints both use SGLang's DFLASH algorithm. Filtering
    and candidate-width constraints are enforced by the existing request contract.
    """
    algorithm = server_args.get("speculative_algorithm")
    if algorithm is None:
        return
    if algorithm != "DFLASH":
        raise ValueError(
            "Score centering supports speculative decoding only with --sglang-speculative-algorithm DFLASH"
        )
    for field in ("speculative_accept_threshold_single", "speculative_accept_threshold_acc"):
        if server_args.get(field, 1.0) != 1.0:
            raise ValueError(
                f"Score centering requires --sglang-{field.replace('_', '-')}=1.0 for exact DFlash sampling"
            )
    try:
        simulate_acceptance = float(environ.get("SGLANG_SIMULATE_ACC_LEN", "-1"))
    except (TypeError, ValueError) as error:
        raise ValueError("Score centering requires a finite SGLANG_SIMULATE_ACC_LEN<=0") from error
    if not math.isfinite(simulate_acceptance) or simulate_acceptance > 0:
        raise ValueError("Score centering requires SGLANG_SIMULATE_ACC_LEN<=0 (no simulated speculative acceptance)")


def validate_score_centering_server_info(server_info: Mapping[str, Any]) -> None:
    """Check the serving workers, including external engines and per-DP overrides.

    /server_info's top-level fields describe startup; internal_states reflects
    the scheduler's effective settings. Environment introspection is mandatory
    for speculative engines so a hidden simulated-acceptance setting cannot pass.
    """
    states = server_info.get("internal_states") or []
    if "speculative_algorithm" not in server_info and (
        not states or any("speculative_algorithm" not in state for state in states)
    ):
        raise ValueError("Score centering requires rollout servers to report speculative_algorithm in /server_info")
    if server_info.get("speculative_algorithm") is not None and not states:
        raise ValueError("Score centering requires speculative engines to report internal_states in /server_info")
    for state in states:
        effective = {**server_info, **state}
        if effective.get("speculative_algorithm") is None:
            continue
        if state.get("dflash_sampling_verify_available") is not True:
            raise ValueError(
                "Score centering requires SGLang to report dflash_sampling_verify_available=true in internal_states; "
                "upgrade SGLang and use a DFlash worker with exact sampling verification"
            )
        for field in ("speculative_accept_threshold_single", "speculative_accept_threshold_acc"):
            if field not in state:
                raise ValueError(f"Score centering requires speculative engines to report {field} in internal_states")
        environ = state.get("env_vars")
        if not isinstance(environ, Mapping):
            raise ValueError("Score centering requires SGLANG_EXPOSE_OWN_ENV_VARS=1 on speculative rollout servers")
        if str(environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0")).lower() in ("1", "true"):
            raise ValueError("Score centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
        validate_score_centering_speculative_config(effective, environ=environ)
