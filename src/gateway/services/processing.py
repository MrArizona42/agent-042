from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal, Sequence

from app_config.catalog import get_catalog, get_kb_config
from gateway.schemas.openai_chat import ChatCompletionRequest, RAGSource
from gateway.services.budget import build_budget_meta
from gateway.services.celery_client import CeleryClient
from gateway.services.prompt_builder import PromptBuilder
from gateway.services.rag_service import RAGService
from gateway.services.redis_stream import RedisStreamService
from gateway.services.task_router import RuleBasedTaskRouter
from gateway.services.vllm_client import VllmOpenAIClient
from shared.config import get_settings, secret_value
from shared.events import InferenceEventProducer, InferenceEventType
from shared.logging import bind_log_context, reset_log_context
from shared.telemetry import get_tracer
from shared.vllm_payloads import (
    canonicalize_assistant_content,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

PROMPT_PREVIEW_TTL_SECONDS = 900
SERVICE_USER_ID = "__service__"


@dataclass(frozen=True)
class PreparedChatRequest:
    generation_payload: dict[str, Any]
    budget_meta: dict[str, int]
    rag_context_chunks: list[dict[str, Any]]
    prompt_messages: list[dict[str, Any]]


@dataclass(frozen=True)
class ResolvedRAGRequest:
    mode: Literal["auto", "off", "explicit"]
    sources: tuple[RAGSource, ...]
    rag_requested: bool
    task_has_knowledge_bases: bool


class _ProcessChat:
    def __init__(self) -> None:
        self._router = RuleBasedTaskRouter()
        self._prompt_builder = PromptBuilder()
        self._rag_service: RAGService | None = None

        # Services injected via init_services() during lifespan startup
        self._celery_client: CeleryClient | None = None
        self._redis_stream: RedisStreamService | None = None
        self._event_producer: InferenceEventProducer | None = None

    def init_services(
        self,
        *,
        redis_stream: RedisStreamService | None = None,
        celery_client: CeleryClient | None = None,
        event_producer: InferenceEventProducer | None = None,
    ) -> None:
        """Inject managed service instances (called from lifespan startup)."""
        self._redis_stream = redis_stream
        self._celery_client = celery_client
        self._event_producer = event_producer

    def publish_inference_event(
        self,
        event_type: InferenceEventType,
        *,
        request_id: str | None = None,
        user_id: str | None = None,
        chat_session_id: str | None = None,
        celery_task_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._event_producer is None:
            return
        self._event_producer.publish(
            event_type,
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
            celery_task_id=celery_task_id,
            conversation_id=conversation_id,
            model=model,
            payload=payload,
        )

    def ensure_rag_service(
        self,
        *,
        settings=None,
        validate: bool = False,
    ) -> RAGService | None:
        """Create or refresh the shared RAG service using current settings."""
        if settings is None:
            settings = get_settings()
        rag_settings = settings.rag

        if not rag_settings.enabled:
            self._rag_service = None
            return None

        if self._rag_service is None or self._rag_service.settings is not settings:
            self._rag_service = RAGService(settings)

        if validate and self._rag_service is not None:
            self._rag_service.validate_knowledge_bases()

        return self._rag_service

    def reload_config_caches(self, *, settings=None) -> None:
        """Invalidate and best-effort rebuild config-derived caches.

        Reload is intentionally fail-open: cache rebuild issues are logged and
        runtime request handling falls back to lazy rebuild or safe defaults.
        """
        if settings is None:
            settings = get_settings()

        self._router.invalidate_cache()
        try:
            self._router.warm_cache()
        except Exception:
            logger.warning("Task router cache warmup failed after config reload", exc_info=True)

        if self._rag_service is not None:
            self._rag_service.invalidate_caches()

        try:
            rag_service = self.ensure_rag_service(settings=settings, validate=True)
        except Exception:
            logger.warning("RAG service validation failed after config reload", exc_info=True)
            return

        if rag_service is None:
            return

        try:
            rag_service.warm_caches()
        except Exception:
            logger.warning("RAG cache warmup failed after config reload", exc_info=True)

    def _client(self) -> VllmOpenAIClient:
        settings = get_settings()
        platform_settings = settings.platform
        gateway_settings = settings.gateway
        return VllmOpenAIClient(
            base_url=platform_settings.vllm_base_url,
            api_key=secret_value(gateway_settings.api_key),
        )

    def _get_celery_client(self) -> CeleryClient:
        """Return the Celery client injected during lifespan startup."""
        if self._celery_client is None:
            raise RuntimeError(
                "CeleryClient is not available. "
                "Ensure gateway startup initialized the Celery client."
            )
        return self._celery_client

    def _get_redis_stream(self) -> RedisStreamService:
        """Return the Redis stream service injected during lifespan startup."""
        if self._redis_stream is None:
            raise RuntimeError(
                "RedisStreamService is not available. "
                "Ensure gateway startup initialized Redis streaming."
            )
        return self._redis_stream

    async def list_models(self) -> Any:
        return await self._client().list_models()

    def _passthrough_fields(self, req: ChatCompletionRequest) -> dict[str, Any]:
        disallowed = {
            "messages",
            "model",
            "temperature",
            "top_p",
            "stream",
            "chat_session_id",
            "rag_sources",
            "max_tokens",
            "max_completion_tokens",
        }
        extra_fields = dict(req.model_extra or {})
        return {
            key: value
            for key, value in extra_fields.items()
            if key not in disallowed and value is not None
        }

    @staticmethod
    def _task_has_knowledge_bases(task: str) -> bool:
        task_cfg = get_catalog().get(task)
        return bool(task_cfg and task_cfg.knowledge_bases)

    def _resolve_rag_request(
        self,
        req: ChatCompletionRequest,
        *,
        task: str,
    ) -> ResolvedRAGRequest:
        task_has_knowledge_bases = self._task_has_knowledge_bases(task)

        if req.rag_sources is None:
            return ResolvedRAGRequest(
                mode="auto",
                sources=(),
                rag_requested=False,
                task_has_knowledge_bases=task_has_knowledge_bases,
            )

        if not req.rag_sources:
            return ResolvedRAGRequest(
                mode="off",
                sources=(),
                rag_requested=False,
                task_has_knowledge_bases=task_has_knowledge_bases,
            )

        return ResolvedRAGRequest(
            mode="explicit",
            sources=tuple(req.rag_sources),
            rag_requested=True,
            task_has_knowledge_bases=task_has_knowledge_bases,
        )

    def _auto_select_rag_sources(
        self,
        *,
        query: str,
        task: str,
    ) -> tuple[RAGSource, ...]:
        with tracer.start_as_current_span("rag.auto_select_sources") as span:
            span.set_attribute("rag.task", task)
            rag_service = self.ensure_rag_service()
            if rag_service is None or not rag_service.enabled:
                span.set_attribute("rag.enabled", False)
                return ()
            span.set_attribute("rag.enabled", True)

            try:
                sources = tuple(rag_service.select_knowledge_bases(query, task))
                span.set_attribute("rag.sources_count", len(sources))
                return sources
            except Exception:
                logger.warning(
                    "Automatic KB selection failed for task=%s",
                    task,
                    exc_info=True,
                )
                return ()

    @staticmethod
    def _resolve_task_model(task: str, *, settings: Any) -> str:
        task_cfg = get_catalog().get(task)
        if task_cfg is not None and task_cfg.adapter.enabled:
            return f"{task_cfg.adapter.name}-{task_cfg.adapter.alias}"
        return settings.vllm.model

    def _retrieve_rag_chunks(
        self,
        rag_sources: Sequence[RAGSource],
        *,
        last_user: str,
    ) -> dict[str, list[dict[str, Any]]]:
        rag_chunks_by_source: dict[str, list[dict[str, Any]]] = {}

        if not rag_sources:
            return rag_chunks_by_source

        rag_service = self.ensure_rag_service()

        if rag_service is None or not rag_service.enabled:
            raise RuntimeError(
                "RAG sources were requested, but the RAG service is disabled or unavailable"
            )

        with tracer.start_as_current_span("rag.retrieve_context") as span:
            span.set_attribute("rag.sources_count", len(rag_sources))
            for src in rag_sources:
                kb_cfg = get_kb_config(src.knowledge_base)
                effective_alias = src.alias or (kb_cfg.default_alias if kb_cfg else "champion")
                logger.info(
                    "RAG — retrieving from kb=%s alias=%s",
                    src.knowledge_base,
                    effective_alias,
                )
                try:
                    docs = rag_service.retrieve_documents(
                        query=last_user,
                        knowledge_base=src.knowledge_base,
                        alias=effective_alias,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to retrieve RAG context for "
                        f"knowledge base '{src.knowledge_base}' alias '{effective_alias}'"
                    ) from exc

                if not docs:
                    continue

                source_key = f"{src.knowledge_base}:{effective_alias}"
                rag_chunks_by_source[source_key] = [
                    {
                        "content": doc.content,
                        "score": doc.score if doc.score is not None else 0.0,
                        "source": f"{src.knowledge_base}_{effective_alias}",
                        "knowledge_base": src.knowledge_base,
                        "alias": effective_alias,
                        "metadata": dict(doc.metadata),
                    }
                    for doc in docs
                ]
            context_sources_count = len(rag_chunks_by_source)
            context_chunks_count = sum(len(chunks) for chunks in rag_chunks_by_source.values())
            span.set_attribute("rag.context_sources_count", context_sources_count)
            span.set_attribute(
                "rag.context_chunks_count",
                context_chunks_count,
            )

        return rag_chunks_by_source

    def _prepare_request(self, req: ChatCompletionRequest) -> PreparedChatRequest:
        settings = get_settings()
        gateway_settings = settings.gateway
        budget_settings = settings.gateway.budget
        rag_settings = settings.rag
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        with tracer.start_as_current_span("gateway.task_routing") as span:
            decision = self._router.decide(last_user)
            span.set_attribute("gateway.task", decision.task)
        rag_request = self._resolve_rag_request(req, task=decision.task)

        rag_sources = rag_request.sources
        rag_requested = rag_request.rag_requested
        if (
            rag_request.mode == "auto"
            and rag_request.task_has_knowledge_bases
            and rag_settings.enabled
        ):
            rag_requested = True
            rag_sources = self._auto_select_rag_sources(query=last_user, task=decision.task)

        rag_chunks_by_source = self._retrieve_rag_chunks(
            rag_sources,
            last_user=last_user,
        )
        with tracer.start_as_current_span("gateway.prompt_build") as span:
            prompt = self._prompt_builder.build_budgeted_messages(
                task=decision.task,
                request_messages=[m.model_dump(exclude_none=True) for m in req.messages],
                rag_chunks_by_source=rag_chunks_by_source,
                rag_requested=rag_requested,
                budget_settings=budget_settings,
            )
            span.set_attribute("gateway.task", decision.task)
            span.set_attribute("rag.requested", rag_requested)
            span.set_attribute("rag.context_chunks_count", len(prompt.rag_context_chunks))
            span.set_attribute("llm.prompt_messages_count", len(prompt.messages))
        logger.info("Built budgeted prompt with %s message(s)", len(prompt.messages))

        generation_payload: dict[str, Any] = {
            "model": req.model
            if req.model
            else self._resolve_task_model(decision.task, settings=settings),
            "messages": prompt.messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": gateway_settings.repetition_penalty,
        }
        requested_max_tokens = req.max_completion_tokens
        if requested_max_tokens is None:
            requested_max_tokens = req.max_tokens
        if requested_max_tokens is not None:
            generation_payload["max_completion_tokens"] = requested_max_tokens
        generation_payload = {
            key: value for key, value in generation_payload.items() if value is not None
        }
        generation_payload.update(self._passthrough_fields(req))

        return PreparedChatRequest(
            generation_payload=generation_payload,
            budget_meta=build_budget_meta(budget_settings),
            rag_context_chunks=prompt.rag_context_chunks,
            prompt_messages=prompt.messages,
        )

    @staticmethod
    def _usage_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    async def _store_prompt_preview(
        self,
        *,
        request_id: str,
        prepared: PreparedChatRequest,
        user_id: str | None,
        chat_session_id: str | None,
    ) -> None:
        redis_stream = self._get_redis_stream()
        preview = {
            "request_id": request_id,
            "owner_user_id": user_id,
            "chat_session_id": chat_session_id,
            "model": prepared.generation_payload.get("model"),
            "prompt_messages": prepared.prompt_messages,
            "rag_context": prepared.rag_context_chunks,
        }
        with tracer.start_as_current_span("gateway.prompt_preview.store") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("prompt_preview.ttl_seconds", PROMPT_PREVIEW_TTL_SECONDS)
            await redis_stream.store_prompt_preview(
                request_id,
                preview,
                ttl_seconds=PROMPT_PREVIEW_TTL_SECONDS,
            )

    async def get_prompt_preview(
        self,
        request_id: str,
        *,
        requester_user_id: str | None,
    ) -> dict[str, Any] | None:
        preview = await self._get_redis_stream().get_prompt_preview(request_id)
        if not isinstance(preview, dict):
            return None

        owner_user_id = preview.get("owner_user_id")
        if owner_user_id and requester_user_id not in {owner_user_id, SERVICE_USER_ID}:
            return None

        return {key: value for key, value in preview.items() if key not in {"owner_user_id"}}

    @staticmethod
    def _chat_completion_result(
        *,
        request_id: str,
        assistant_content: str,
        finish_reason: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @staticmethod
    def _sse_bytes(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    @staticmethod
    def _sse_event_bytes(event: str, payload: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()

    def _answer_delta_chunk(self, *, request_id: str, content: str) -> bytes:
        return self._sse_bytes(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
        )

    def _finish_chunk(self, *, request_id: str, finish_reason: str) -> bytes:
        return self._sse_bytes(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )

    def _usage_chunk(self, *, request_id: str, usage: dict[str, Any]) -> bytes:
        return self._sse_bytes(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            }
        )

    def _error_chunk(self, *, error: str, error_type: str = "server_error") -> bytes:
        return self._sse_bytes(
            {
                "error": {
                    "message": error,
                    "type": error_type,
                }
            }
        )

    def _rich_stream_chunk(
        self,
        *,
        event: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> bytes:
        return self._sse_event_bytes(
            event,
            {
                "request_id": request_id,
                **payload,
            },
        )

    @staticmethod
    def _rag_context_event_payload(prepared: PreparedChatRequest) -> dict[str, Any]:
        sources: set[tuple[str | None, str | None]] = set()
        for chunk in prepared.rag_context_chunks:
            sources.add((chunk.get("knowledge_base"), chunk.get("alias")))
        return {
            "context_chunks_count": len(prepared.rag_context_chunks),
            "context_sources_count": len(sources),
            "sources": [
                {"knowledge_base": knowledge_base, "alias": alias}
                for knowledge_base, alias in sorted(
                    sources,
                    key=lambda item: (item[0] or "", item[1] or ""),
                )
            ],
        }

    async def stream_chat(
        self,
        req: ChatCompletionRequest,
        *,
        user_id: str | None = None,
        chat_session_id: str | None = None,
        request_id: str | None = None,
        rich_stream: bool = False,
    ) -> AsyncIterator[bytes]:
        request_id = request_id or str(_uuid.uuid4())
        log_token = bind_log_context(
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
        )
        try:
            with tracer.start_as_current_span("gateway.prepare_chat_request"):
                prepared = self._prepare_request(req)
            if prepared.rag_context_chunks:
                self.publish_inference_event(
                    "rag.context.selected",
                    request_id=request_id,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    model=prepared.generation_payload.get("model"),
                    payload=self._rag_context_event_payload(prepared),
                )
            try:
                await self._store_prompt_preview(
                    request_id=request_id,
                    prepared=prepared,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                )
            except Exception:
                logger.warning("Failed to store prompt preview", exc_info=True)
            return await self._stream_chat_async(
                prepared,
                req=req,
                user_id=user_id,
                chat_session_id=chat_session_id,
                request_id=request_id,
                rich_stream=rich_stream,
            )
        finally:
            reset_log_context(log_token)

    async def _stream_chat_async(
        self,
        prepared: PreparedChatRequest,
        *,
        req: ChatCompletionRequest,
        user_id: str | None = None,
        chat_session_id: str | None = None,
        request_id: str | None = None,
        rich_stream: bool = False,
    ) -> AsyncIterator[bytes]:
        settings = get_settings()
        gateway_settings = settings.gateway
        conversation_id = str(_uuid.uuid4())
        request_id = request_id or str(_uuid.uuid4())
        celery_client = self._get_celery_client()
        redis_stream = self._get_redis_stream()

        with tracer.start_as_current_span("celery.enqueue_generate_response") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("conversation_id", conversation_id)
            span.set_attribute("llm.model", prepared.generation_payload.get("model") or "")
            task_id = celery_client.enqueue_generate_response(
                conversation_id=conversation_id,
                request_id=request_id,
                generation_payload=prepared.generation_payload,
                budget_meta=prepared.budget_meta,
            )
            span.set_attribute("celery.task_id", task_id)
        log_token = bind_log_context(
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
            celery_task_id=task_id,
            conversation_id=conversation_id,
            model=prepared.generation_payload.get("model"),
        )
        logger.info(
            "Enqueued async stream task",
            extra={"event": "celery.task.enqueued"},
        )
        self.publish_inference_event(
            "celery.task.enqueued",
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
            celery_task_id=task_id,
            conversation_id=conversation_id,
            model=prepared.generation_payload.get("model"),
        )
        reset_log_context(log_token)

        def _revoke_generation(*, reason: str) -> None:
            try:
                celery_client.revoke_task(task_id, terminate=True, signal="SIGTERM")
            except Exception:
                logger.warning(
                    "Failed to revoke async stream task %s for request %s: %s",
                    task_id,
                    request_id,
                    reason,
                    exc_info=True,
                )
                return

            logger.warning(
                "Revoked async stream task %s for request %s conversation %s: %s",
                task_id,
                request_id,
                conversation_id,
                reason,
            )

        async def _event_stream() -> AsyncIterator[bytes]:
            stream_log_token = bind_log_context(
                request_id=request_id,
                user_id=user_id,
                chat_session_id=chat_session_id,
                celery_task_id=task_id,
                conversation_id=conversation_id,
                model=prepared.generation_payload.get("model"),
            )
            thinking_content = ""
            answer_content = ""
            finish_reason = "stop"
            usage: dict[str, Any] = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            terminal_event_seen = False
            revocation_requested = False

            try:
                logger.info(
                    "Async stream started",
                    extra={"event": "chat.stream.started"},
                )
                with tracer.start_as_current_span("gateway.stream_response") as span:
                    span.set_attribute("conversation_id", conversation_id)
                    span.set_attribute("celery.task_id", task_id)
                    span.set_attribute("llm.model", prepared.generation_payload.get("model") or "")
                async for event in redis_stream.subscribe(
                    conversation_id,
                    timeout=gateway_settings.streaming_timeout,
                ):
                    event_type = event.get("type")

                    if event_type == "thinking_token":
                        content = event.get("content", "")
                        thinking_content += content
                        if rich_stream and content:
                            yield self._rich_stream_chunk(
                                event="thinking_token",
                                request_id=request_id,
                                payload={"content": content},
                            )
                        continue

                    if event_type in {"answer_token", "token"}:
                        content = event.get("content", "")
                        if content:
                            answer_content += content
                            if rich_stream:
                                yield self._rich_stream_chunk(
                                    event="answer_token",
                                    request_id=request_id,
                                    payload={"content": content},
                                )
                            else:
                                yield self._answer_delta_chunk(
                                    request_id=request_id,
                                    content=content,
                                )
                        continue

                    if event_type == "done":
                        terminal_event_seen = True
                        thinking_content = event.get("thinking_content", thinking_content)
                        answer_content = event.get("answer_content", answer_content)
                        finish_reason = event.get("finish_reason", "stop")
                        event_usage = event.get("usage")
                        if isinstance(event_usage, dict):
                            usage = event_usage

                        assistant_content = event.get(
                            "content",
                            canonicalize_assistant_content(thinking_content, answer_content),
                        )
                        result = self._chat_completion_result(
                            request_id=request_id,
                            assistant_content=assistant_content,
                            finish_reason=finish_reason,
                            usage=usage,
                        )

                        if rich_stream:
                            yield self._rich_stream_chunk(
                                event="usage",
                                request_id=request_id,
                                payload={"usage": usage},
                            )
                            yield self._rich_stream_chunk(
                                event="done",
                                request_id=request_id,
                                payload={
                                    "thinking_content": thinking_content,
                                    "answer_content": answer_content,
                                    "content": assistant_content,
                                    "finish_reason": finish_reason,
                                },
                            )
                        else:
                            yield self._finish_chunk(
                                request_id=request_id, finish_reason=finish_reason
                            )
                            yield self._usage_chunk(request_id=request_id, usage=usage)

                        if chat_session_id and user_id:
                            await self._persist_exchange(
                                req,
                                result,
                                chat_session_id,
                                request_id=request_id,
                                user_id=user_id,
                                celery_task_id=task_id,
                                conversation_id=conversation_id,
                                model=prepared.generation_payload.get("model"),
                            )

                        logger.info(
                            "Async stream completed",
                            extra={
                                "event": "chat.stream.completed",
                                "finish_reason": finish_reason,
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "total_tokens": usage.get("total_tokens"),
                            },
                        )
                        self.publish_inference_event(
                            "chat.response.completed",
                            request_id=request_id,
                            user_id=user_id,
                            chat_session_id=chat_session_id,
                            celery_task_id=task_id,
                            conversation_id=conversation_id,
                            model=prepared.generation_payload.get("model"),
                            payload={
                                "finish_reason": finish_reason,
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "total_tokens": usage.get("total_tokens"),
                            },
                        )
                        if not rich_stream:
                            yield b"data: [DONE]\n\n"
                        return

                    if event_type == "error":
                        error_type = event.get("error_type") or "server_error"
                        logger.warning(
                            "Async stream received error event",
                            extra={"event": "chat.stream.error", "error_type": error_type},
                        )
                        if error_type != "timeout":
                            terminal_event_seen = True
                        elif not revocation_requested:
                            _revoke_generation(
                                reason=("idle timeout while waiting for worker stream events")
                            )
                            revocation_requested = True

                        if rich_stream:
                            yield self._rich_stream_chunk(
                                event="error",
                                request_id=request_id,
                                payload={
                                    "error": event.get("error", "Unknown error"),
                                    "error_type": error_type,
                                },
                            )
                        else:
                            yield self._error_chunk(
                                error=event.get("error", "Unknown error"),
                                error_type=error_type,
                            )
                        return
            finally:
                if not terminal_event_seen and not revocation_requested:
                    _revoke_generation(
                        reason="stream closed before worker reported completion",
                    )
                reset_log_context(stream_log_token)

        return _event_stream()

    async def _persist_exchange(
        self,
        req: ChatCompletionRequest,
        result: dict,
        chat_session_id: str,
        *,
        request_id: str,
        user_id: str | None,
        celery_task_id: str,
        conversation_id: str,
        model: str | None,
    ) -> None:
        """Persist the user message and assistant response to PostgreSQL."""
        try:
            from sqlalchemy import select

            from shared.db.engine import get_session_factory
            from shared.db.models import ChatMessage, ChatSession

            with tracer.start_as_current_span("gateway.persist_exchange") as span:
                span.set_attribute("chat_session_id", chat_session_id)
                last_user_msg = next(
                    (m.content for m in reversed(req.messages) if m.role == "user"),
                    None,
                )
                assistant_content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}

                session_uuid = _uuid.UUID(chat_session_id)

                async with get_session_factory()() as db:
                    sess_result = await db.execute(
                        select(ChatSession).where(ChatSession.id == session_uuid)
                    )
                    session = sess_result.scalar_one_or_none()
                    if session and not session.title and last_user_msg:
                        session.title = last_user_msg[:100]
                        session.updated_at = datetime.now(timezone.utc)

                    messages_inserted = 0
                    if last_user_msg:
                        db.add(
                            ChatMessage(
                                session_id=session_uuid,
                                role="user",
                                content=last_user_msg,
                            )
                        )
                        messages_inserted += 1

                    if assistant_content:
                        db.add(
                            ChatMessage(
                                session_id=session_uuid,
                                role="assistant",
                                content=assistant_content,
                                prompt_tokens=self._usage_int(usage.get("prompt_tokens")),
                                completion_tokens=self._usage_int(usage.get("completion_tokens")),
                            )
                        )
                        messages_inserted += 1

                    if session:
                        session.updated_at = datetime.now(timezone.utc)
                    await db.commit()
                span.set_attribute("db.messages_inserted", messages_inserted)
                self.publish_inference_event(
                    "chat.persistence.completed",
                    request_id=request_id,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    celery_task_id=celery_task_id,
                    conversation_id=conversation_id,
                    model=model,
                    payload={"messages_inserted": messages_inserted},
                )
        except Exception:
            logger.warning("Failed to persist chat exchange", exc_info=True)


process_chat = _ProcessChat()
