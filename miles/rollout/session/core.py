"""Logic layer of the session server: ``SessionCore``.

HTTP-agnostic: the FastAPI adapter (``sessions.py`` + ``server.py``) turns each request into primitives and calls these methods. Owns one ``SessionRegistry`` (per-session TITO/trajectory state) and one proxy ``backend``.

- ``chat_completions`` strips the R3 replay payloads (``routed_experts`` /
  ``indexer_topk``) from the client reply copy-on-write; the ``SessionRecord``
  keeps only the token/logprob metadata needed for server-side sample assembly.
- ``chat_completions`` serializes backend calls per session with a dedicated
  generation lock. The shorter state lock still gates prep/update and lets
  DELETE mark an in-flight session as closing without waiting for inference.
- ``stream: true`` is served as fake streaming: the backend call stays
  non-streaming so TITO can retain the complete response and token metadata.
- ``collect_samples`` computes, truncates, and merges the training sample on
  the owning server; raw session records never cross the wire.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from starlette.responses import Response

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.errors import (
    MessageValidationError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples.codec import encode_samples
from miles.rollout.session.samples.merge import compute_samples_from_openai_records, truncate_samples_by_total_tokens
from miles.rollout.session.types import GetSessionResponse, SessionRecord

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"

# Hop-by-hop / length-framing headers dropped from the upstream response so the
# transport layer recomputes them from the body we actually send. "server" and
# "date" are dropped because our own ASGI server always emits them, so echoing
# upstream's copy puts two of each on the wire; aiohttp's parser rejects that
# outright with "Duplicate 'Server' header found" instead of reading the body.
_DROP_RESPONSE_HEADERS = ("content-length", "transfer-encoding", "content-encoding", "server", "date")


@dataclass
class ProxyRequest:
    """Primitive carrier for the proxy backend (replaces fastapi.Request)."""

    method: str
    query: str = ""


def _render_json(payload) -> bytes:
    """Encode like Starlette's JSONResponse (compact, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _samples_response(payload: bytes) -> Response:
    """The samples-op reply: one safetensors binary payload."""
    return Response(content=payload, status_code=200, media_type="application/octet-stream")


_CLIENT_STRIPPED_META_KEYS = ("routed_experts", "indexer_topk")
_RECORD_STRIPPED_META_KEYS = (
    # The growing prompt is represented once by accumulated_token_ids.
    "input_token_logprobs",
    "input_top_logprobs",
)


def _strip_replay_payloads(response: dict) -> dict:
    stripped_choices = []
    for choice in response.get("choices", []):
        meta = choice.get("meta_info")
        if isinstance(meta, dict) and any(k in meta for k in _CLIENT_STRIPPED_META_KEYS):
            meta = {k: v for k, v in meta.items() if k not in _CLIENT_STRIPPED_META_KEYS}
            choice = {**choice, "meta_info": meta}
        stripped_choices.append(choice)
    return {**response, "choices": stripped_choices}


def _compact_record_response(response: dict, *, include_message: bool = False) -> dict:
    """Keep only the response fields required to reconstruct a training sample."""
    choices = response.get("choices") or []
    if not choices:
        return {"choices": []}
    choice = choices[0]
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    compact_choice = {
        "finish_reason": choice.get("finish_reason"),
        "meta_info": {key: value for key, value in meta_info.items() if key not in _RECORD_STRIPPED_META_KEYS},
    }
    if include_message:
        compact_choice["message"] = choice.get("message")
    return {"choices": [compact_choice]}


def _response_to_stream_chunk(response: dict) -> dict:
    """Synthesize the single ``chat.completion.chunk`` for a fake stream.

    Adapted from NVIDIA-NeMo/ProRL-Agent-Server (``gateway/server.py::_response_to_stream_chunk``)
    and THUDM/slime (``agent/adapters/openai.py::_render_stream``).

    One big delta is protocol-legal (streaming deltas concatenate). All
    tool_calls ride in this one chunk with their ``index`` set: some clients
    mis-assemble arguments fragmented across chunks. The chunk carries no
    ``meta_info``; server-side sample assembly retains the complete response.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    delta = {"role": message.get("role", "assistant"), "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**tool_call, "index": i} for i, tool_call in enumerate(message["tool_calls"])]
    chunk = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}],
    }
    if response.get("usage") is not None:
        chunk["usage"] = response["usage"]
    return chunk


def _chat_client_response(result: dict, response: dict, client_stream: bool) -> Response:
    if client_stream:
        sse = b"data: " + _render_json(_response_to_stream_chunk(response)) + b"\n\ndata: [DONE]\n\n"
        # Fresh headers: upstream's headers describe its JSON body, not this SSE body.
        # X-Accel-Buffering keeps reverse proxies from buffering the stream.
        return Response(
            content=sse,
            status_code=result["status_code"],
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return Response(
        content=_render_json(_strip_replay_payloads(response)),
        status_code=result["status_code"],
        headers=headers,
        media_type=JSON_MEDIA_TYPE,
    )


def proxy_result_to_response(result: dict) -> Response:
    """Build the client response from a proxy result.

    Mirrors the previous ``SessionServer.build_proxy_response``: re-emit JSON
    bodies as compact JSON (application/json), pass non-JSON bodies through
    unchanged, and drop wire-level framing headers from upstream.
    """
    content = result["response_body"]
    status_code = result["status_code"]
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    content_type = headers.get("content-type", "")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Match the old Response(media_type=content_type): pass it through verbatim
        # (incl. "" when upstream sent no content-type) so the wire bytes are identical.
        return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)
    return Response(content=_render_json(data), status_code=status_code, headers=headers, media_type=JSON_MEDIA_TYPE)


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(self, backend, registry: SessionRegistry, args, session_server_instance_id=None):
        self.backend = backend
        self.registry = registry
        self.args = args
        self.instance_id = session_server_instance_id

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self) -> Response:
        session_id = self.registry.create_session()
        return Response(content=_render_json({"session_id": session_id}), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def get_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        async with session.generation_lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            async with session.lock:
                metadata = await self._session_metadata(session_id, session)
                payload = GetSessionResponse(
                    session_id=session_id,
                    records=session.records,
                    metadata=metadata,
                )
                content = await asyncio.to_thread(
                    _render_json,
                    payload.model_dump(mode="json"),
                )
                return Response(
                    content=content,
                    status_code=200,
                    media_type=JSON_MEDIA_TYPE,
                )

    async def _session_metadata(self, session_id: str, session) -> dict:
        """Build metadata shared by debug inspection and sample collection."""
        metadata: dict = {}
        mismatch = None
        sample_rate = float(getattr(self.args, "tito_session_mismatch_sample_rate", 1.0))
        mismatch_sampled = sample_rate >= 1.0 or (
            sample_rate > 0.0
            and int(session_id, 16) / float(16 ** len(session_id)) < sample_rate
        )
        metadata["tito_session_mismatch_sampled"] = mismatch_sampled
        mismatch_started = time.monotonic()
        if mismatch_sampled:
            try:
                # Canonical rendering can be expensive for a 64k-token
                # trajectory. Keep it out of the event loop; production
                # configs may sample it while CI retains the default 100%.
                mismatch = await asyncio.to_thread(self.registry.compute_session_mismatch, session)
            except TokenizationError:
                metadata["tito_session_mismatch_error"] = True
                logger.warning("Failed to compute TITO mismatch audit for session %s", session_id)
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["session_collect/mismatch_seconds"] = time.monotonic() - mismatch_started
        model_requests = list(session.model_requests)
        model_request_durations = [float(item["duration_seconds"]) for item in model_requests]
        if model_request_durations:
            completion_tokens = sum(int(item["completion_tokens"]) for item in model_requests)
            non_200_count = sum(int(item["status_code"]) != 200 for item in model_requests)
            # Exact vectors let the rollout log hook compute batch-wide
            # distributions without redundant per-session summaries.
            metadata["model_request/durations_seconds"] = model_request_durations
            metadata["model_request/prompt_tokens"] = sum(int(item["prompt_tokens"]) for item in model_requests)
            metadata["model_request/completion_tokens"] = completion_tokens
            metadata["model_request/non_200_count"] = non_200_count
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        return metadata

    async def collect_samples(self, session_id: str, *, max_seq_len: int | None) -> Response:
        """Assemble a consistent training sample after all generation updates."""
        session = self.registry.get_session(session_id)
        async with session.generation_lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            async with session.lock:
                metadata = await self._session_metadata(session_id, session)
                records = list(session.records)
                accumulated_token_ids = list(session.token_ids)
                max_trim_tokens = int(metadata.pop("max_trim_tokens", 0))
                metadata.pop("accumulated_token_ids", None)

        try:
            payload = await asyncio.to_thread(
                self._assemble_samples,
                records,
                accumulated_token_ids,
                metadata,
                max_trim_tokens,
                max_seq_len,
            )
        except (AssertionError, ValueError) as exc:
            return Response(
                content=str(exc).encode(),
                status_code=422,
                media_type="text/plain",
            )
        return _samples_response(payload)

    def _assemble_samples(
        self,
        records: list[SessionRecord],
        accumulated_token_ids: list[int],
        metadata: dict,
        max_trim_tokens: int,
        max_seq_len: int | None,
    ) -> bytes:
        assembly_started = time.monotonic()
        if not records:
            metadata["session_collect/assembly_seconds"] = (
                time.monotonic() - assembly_started
            )
            return encode_samples([], metadata, empty_reason="no_records")

        tokenizer = self.registry.tokenizer
        samples = compute_samples_from_openai_records(
            self.args,
            records,
            tokenizer,
            accumulated_token_ids=accumulated_token_ids,
            max_trim_tokens=max_trim_tokens,
        )
        if max_seq_len is not None:
            samples = truncate_samples_by_total_tokens(
                samples,
                max_seq_len,
                tokenizer,
            )
        if not samples:
            metadata["session_collect/assembly_seconds"] = (
                time.monotonic() - assembly_started
            )
            return encode_samples([], metadata, empty_reason="all_truncated")
        merged_sample = merge_samples(samples, tokenizer)
        metadata["session_collect/assembly_seconds"] = (
            time.monotonic() - assembly_started
        )
        return encode_samples([merged_sample], metadata)

    async def delete_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        # Do not wait for the generation lock: cancellation should retire the
        # session immediately. The short state lock prevents removal during a
        # prepare/update section; an in-flight proxy sees ``closing`` and skips
        # its eventual state update.
        await session.lock.acquire()
        try:
            self.registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    async def chat_completions(self, session_id: str, *, method: str, query: str, headers: dict, body: bytes) -> Response:
        """Serialize one session's generations without limiting other sessions."""
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        async with session.generation_lock:
            return await self._chat_completions_serialized(
                session_id,
                method=method,
                query=query,
                headers=headers,
                body=body,
            )

    async def _chat_completions_serialized(self, session_id: str, *, method: str, query: str, headers: dict, body: bytes) -> Response:
        """Proxy a chat completion through the backend with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (generation lock only) → validate response → update trajectory
        checkpoint and append record (state lock held briefly). DELETE can mark
        a session closing during inference; GET waits for the complete update.
        """
        request_timestamp = time.time()
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            try:
                request_body = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                raise MessageValidationError(f"invalid JSON body: {e}") from e

            # Fake streaming: the backend must stay non-streaming (TITO needs the
            # complete message + meta_info, and sglang rejects return_meta_info
            # with stream=true), so pop the client's intent here and honor it
            # when rendering the client response.
            client_stream = bool(request_body.pop("stream", False))
            request_body.pop("stream_options", None)

            # TITO token tracking needs Miles-owned input_ids plus SGLang output
            # metadata: logprobs=True populates meta_info.output_token_logprobs and
            # return_meta_info wraps it in choice.meta_info. Hardcoded (not
            # setdefault) so agent-side overrides cannot break token accumulation.
            request_body["logprobs"] = True
            request_body["return_meta_info"] = True
            if getattr(self.args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            if getattr(self.args, "use_rollout_indexer_replay", False):
                request_body["return_indexer_topk"] = True
            # Must be False so stop-token text is trimmed from assistant content;
            # token IDs still come from logprobs below.
            request_body["no_stop_trim"] = False
            # FIXME(session): Only nested `chat_template_kwargs` reach the local renderer;
            # top-level `reasoning` and `reasoning_effort` are not mapped to template kwargs.
            request_ctk = request_body.get("chat_template_kwargs")
            if request_ctk is not None and not isinstance(request_ctk, dict):
                raise MessageValidationError("chat_template_kwargs must be an object")
            tito_tokenizer = self.registry.tito_tokenizer
            if request_ctk:
                try:
                    tito_tokenizer = tito_tokenizer.clone_with_chat_template_kwargs(request_ctk)
                except ValueError as e:
                    raise MessageValidationError(str(e)) from e
            if tito_tokenizer.chat_template_kwargs:
                request_body["chat_template_kwargs"] = dict(tito_tokenizer.chat_template_kwargs)
            else:
                request_body.pop("chat_template_kwargs", None)
            effective_chat_template_kwargs = dict(
                tito_tokenizer.chat_template_kwargs
            )
            if (
                session.token_ids
                and session.chat_template_kwargs is not None
                and session.chat_template_kwargs
                != effective_chat_template_kwargs
            ):
                raise MessageValidationError(
                    "chat_template_kwargs cannot change within a session"
                )

            request_messages = request_body.get("messages", [])
            # Tokenizer/chat-template work grows with the full trajectory.
            # Running it directly here can monopolize this server's event loop
            # for every other session sharing the process, including the final
            # GET/DELETE control requests.  Keep the per-session state lock
            # while moving the synchronous CPU work to the thread pool.
            prompt_token_ids = await asyncio.to_thread(
                session.prepare_pretokenized,
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=tito_tokenizer,
            )
            # `message_matches` intentionally ignores wire-only differences
            # such as tool-call indices and a known client-added GLM tool
            # boundary newline. Keep the already-tokenized stored prefix as
            # canonical session state instead of replacing it with that
            # round-tripped representation, which would make the state and
            # retained token prefix disagree during the final TITO audit.
            canonical_request_messages = [
                *session.messages,
                *request_messages[len(session.messages) :],
            ]
            max_seq_len = getattr(self.args, "max_seq_len", None)
            if max_seq_len is not None:
                max_seq_len = int(max_seq_len)
                # SGLang rejects equality at its physical context boundary:
                # prompt + completion must remain strictly below max_seq_len.
                remaining_tokens = max_seq_len - len(prompt_token_ids) - 1
                if remaining_tokens <= 0:
                    raise MessageValidationError(
                        "TITO context limit reached: prompt has "
                        f"{len(prompt_token_ids)} tokens, configured max_seq_len "
                        f"is {max_seq_len}, and one boundary token is reserved"
                    )
                requested_max_tokens = request_body.get("max_tokens")
                if requested_max_tokens is not None:
                    request_body["max_tokens"] = min(
                        int(requested_max_tokens),
                        remaining_tokens,
                    )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            proxy_body = await asyncio.to_thread(_render_json, request_body)
        # --- lock released ---

        # --- Phase 2: proxy to backend (generation lock held; state lock released) ---
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        proxy_round_trip_started = time.monotonic()
        result = await self.backend.do_proxy(ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers)
        proxy_round_trip_seconds = time.monotonic() - proxy_round_trip_started
        model_request_seconds = float(result.get("backend_request_seconds", proxy_round_trip_seconds))
        async with session.lock:
            model_request_index = session.record_model_request(
                duration_seconds=model_request_seconds,
                status_code=result["status_code"],
                prompt_tokens=len(prompt_token_ids),
            )

        # Non-200 (e.g. an upstream 400) contributes to request timing/error
        # metrics but does not become a training SessionRecord.
        if result["status_code"] != 200:
            return proxy_result_to_response(result)

        # Long generations carry thousands of token/logprob entries.  Parsing
        # and rendering those payloads is CPU work and must not block unrelated
        # sessions or their collection requests on this event loop.
        response = await asyncio.to_thread(json.loads, result["response_body"])
        choice = response.get("choices", [{}])[0]

        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError("meta_info and output_token_logprobs must be in choice (requires logprobs=True)")
        assistant_message = choice.get("message") or {}
        if assistant_message.get("content") is None:
            raise UpstreamResponseError(
                "assistant message content is None, when tool call parser failed "
                "SGLang should still return an empty content rather than None. "
                "Please check your modified SGLang version."
            )

        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info["completion_tokens"]

        actual_output_logprobs_len = len(output_token_logprobs)
        if actual_output_logprobs_len != completion_tokens:
            raise UpstreamResponseError(
                "invalid chat completion response: "
                "len(output_token_logprobs)="
                f"{actual_output_logprobs_len} != "
                f"completion_tokens={completion_tokens}. Please check whether "
                "you use the correct SGLang branch which has the tokenizer "
                "batch decode fix."
            )

        completion_token_ids = [t[1] for t in output_token_logprobs]

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            session.set_model_request_completion_tokens(
                model_request_index,
                completion_tokens,
            )
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return await asyncio.to_thread(
                    _chat_client_response,
                    result,
                    response,
                    client_stream,
                )

            await asyncio.to_thread(
                session.update_pretokenized_state,
                canonical_request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=self.registry.tito_tokenizer.max_trim_tokens,
                tools=request_body.get("tools"),
                chat_template_kwargs=effective_chat_template_kwargs,
            )

            record_debug_trajectory = getattr(self.args, "save_debug_trajectory_data", None) is not None
            record = SessionRecord(
                timestamp=time.time(),
                request_timestamp=request_timestamp,
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                prompt_token_count=len(prompt_token_ids),
                request=(
                    {"messages": canonical_request_messages}
                    if record_debug_trajectory
                    else {}
                ),
                response=await asyncio.to_thread(
                    _compact_record_response,
                    response,
                    include_message=record_debug_trajectory,
                ),
            )
            session.append_record(record)
        # --- lock released ---

        return await asyncio.to_thread(_chat_client_response, result, response, client_stream)

    async def proxy(self, session_id: str, path: str, *, method: str, query: str, headers: dict, body: bytes) -> Response:
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(ProxyRequest(method=method, query=query), path, body=body, headers=headers)
        return proxy_result_to_response(result)
