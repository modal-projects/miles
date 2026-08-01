"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
  Returning a dict with ``_miles_abort=True`` marks the episode ABORTED. This
  lets environment adapters exclude infrastructure failures from training even
  when policy calls were already recorded before the failure occurred.
"""

import argparse
import asyncio
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.failures import mark_infrastructure_failure, mark_non_retryable_failure
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _aborted_sample(
    input_sample: Sample,
    *,
    agent_metadata: dict | None,
    session_metadata: dict | None,
    exit_status: str,
    error: Exception | None = None,
    error_key: str = "agent_function_error",
    infrastructure_failure: bool = True,
) -> Sample:
    sample = deepcopy(input_sample)
    sample.status = Sample.Status.ABORTED
    if infrastructure_failure:
        mark_infrastructure_failure(sample)
    else:
        mark_non_retryable_failure(sample)
    if isinstance(agent_metadata, dict):
        sample.metadata.update({key: value for key, value in agent_metadata.items() if key != "_miles_abort"})
    sample.metadata.update(session_metadata or {})
    sample.metadata["exit_status"] = exit_status
    if error is not None:
        sample.metadata[error_key] = (f"{type(error).__name__}: {error}")[:1000]
    return sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert getattr(input.args, "session_server_ip", None) and getattr(
        input.args,
        "session_server_ports",
        None,
    ), "agentic_tool_call.generate requires session_server_ip/session_server_ports. Pass --use-session-server to start the session server."
    session_create_started = time.monotonic()
    try:
        tracer = await OpenAIEndpointTracer.create(input.args)
    except Exception as error:
        # Session servers are control-plane infrastructure. A transient
        # connection failure must invalidate only this trajectory, not the
        # long-lived fully-async producer (and therefore the whole run).
        logger.warning(
            "Failed to create agent session: %s: %s",
            type(error).__name__,
            error,
        )
        sample = _aborted_sample(
            input.sample,
            agent_metadata=None,
            session_metadata=None,
            exit_status="session_create_error",
            error=error,
            error_key="session_create_error",
        )
        sample.metadata["session_create_error_type"] = type(error).__name__
        sample.metadata["session_create/error_seconds"] = time.monotonic() - session_create_started
        return GenerateFnOutput(samples=sample)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert custom_agent_function is not None, f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    metadata = {**metadata, "session_server_id": tracer.session_server_id}

    agent_metadata = None
    agent_error: Exception | None = None
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            base_url=tracer.base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic() - t_start:.1f}s")
    except asyncio.CancelledError:
        await tracer.discard_session()
        raise
    except Exception as e:
        agent_error = e
        logger.warning("%s Agent function failed: %s: %s", log_prefix, type(e).__name__, e)

    logger.debug(f"{log_prefix} Calling collect_samples...")
    try:
        result = await tracer.collect_samples(
            input.sample,
            max_seq_len=max_seq_len,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning(
            "%s Sample collection failed: %s: %s",
            log_prefix,
            type(error).__name__,
            error,
        )
        sample = _aborted_sample(
            input.sample,
            agent_metadata=agent_metadata,
            session_metadata=None,
            exit_status="session_sample_collection_error",
            error=error,
            error_key="session_sample_collection_error",
        )
        return GenerateFnOutput(samples=sample)

    session_metadata = result.session_metadata
    if isinstance(agent_metadata, dict) and agent_metadata.get("_miles_abort"):
        logger.debug(
            "Agent aborted the sample; the rollout scheduler will apply its failure policy: %s",
            agent_metadata.get("exit_status", ""),
        )
        sample = _aborted_sample(
            input.sample,
            agent_metadata=agent_metadata,
            session_metadata=session_metadata,
            exit_status=str(agent_metadata.get("exit_status") or "agent_abort"),
        )
        return GenerateFnOutput(samples=sample)

    if agent_error is not None:
        sample = _aborted_sample(
            input.sample,
            agent_metadata=agent_metadata,
            session_metadata=session_metadata,
            exit_status="agent_function_exception",
            error=agent_error,
        )
        return GenerateFnOutput(samples=sample)

    if not result.samples:
        all_truncated = result.empty_reason == "all_truncated"
        sample = _aborted_sample(
            input.sample,
            agent_metadata=agent_metadata,
            session_metadata=session_metadata,
            exit_status=("prompt_exceeds_max_seq_len" if all_truncated else "no_model_calls"),
            infrastructure_failure=not all_truncated,
        )
        return GenerateFnOutput(samples=sample)

    (sample,) = result.samples
    sample.metadata.update(agent_metadata or {})
    sample.metadata.update(result.session_metadata)
    non_generation_time = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if non_generation_time is not None:
        sample.non_generation_time = non_generation_time
    logger.debug(
        "%s Sample collection finished in %.1fs",
        log_prefix,
        time.monotonic() - t_start,
    )
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) per session. Truncation happens inside the session server during sample assembly; also forwarded to the Harbor agent server (as max_seq_len) to abort the trial early.",
    )


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
