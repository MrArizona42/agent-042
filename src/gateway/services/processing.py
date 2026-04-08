from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from gateway.config import get_settings
from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.budget import build_budget_meta
from gateway.services.celery_client import CeleryClient
from gateway.services.prompt_builder import PromptBuilder
from gateway.services.rag_service import RAGService
from gateway.services.redis_stream import RedisStreamService
from gateway.services.task_router import RuleBasedTaskRouter
from gateway.services.vllm_client import VllmOpenAIClient
from shared.config import get_kb_config
from shared.vllm_payloads import (
    ResponseBudgetExceededError,
    apply_response_token_budget,
    extract_tokenize_payload,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedChatRequest:
    generation_payload: dict[str, Any]
    budget_meta: dict[str, int]
    rag_context_chunks: list[dict[str, Any]]
    prompt_messages: list[dict[str, Any]]


class _ProcessChat:
    def __init__(self) -> None:
        self._router = RuleBasedTaskRouter()
        self._prompt_builder = PromptBuilder()
        self._rag_service: RAGService | None = None

        # Services injected via init_services() during lifespan startup
        self._celery_client: CeleryClient | None = None
        self._redis_stream: RedisStreamService | None = None

    def init_services(
        self,
        *,
        redis_stream: RedisStreamService | None = None,
        celery_client: CeleryClient | None = None,
    ) -> None:
        """Inject managed service instances (called from lifespan startup)."""
        self._redis_stream = redis_stream
        self._celery_client = celery_client

    def ensure_rag_service(
        self,
        *,
        settings=None,
        validate: bool = False,
    ) -> RAGService | None:
        """Create or refresh the shared RAG service using current settings."""
        if settings is None:
            settings = get_settings()

        if not settings.rag_enabled:
            self._rag_service = None
            return None

        if self._rag_service is None or self._rag_service.settings is not settings:
            self._rag_service = RAGService(settings)

        if validate and self._rag_service is not None:
            self._rag_service.validate_knowledge_bases()

        return self._rag_service

    def _client(self) -> VllmOpenAIClient:
        settings = get_settings()
        return VllmOpenAIClient(base_url=settings.vllm_base_url, api_key=settings.api_key)

    def _get_celery_client(self) -> CeleryClient:
        """Return the Celery client injected during lifespan startup."""
        if self._celery_client is None:
            raise RuntimeError(
                "CeleryClient is not available. "
                "Ensure async_enabled=true and CELERY_BROKER_URL is set."
            )
        return self._celery_client

    def _get_redis_stream(self) -> RedisStreamService:
        """Return the Redis stream service injected during lifespan startup."""
        if self._redis_stream is None:
            raise RuntimeError(
                "RedisStreamService is not available. "
                "Ensure async_enabled=true and REDIS_URL is set."
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

    def _retrieve_rag_chunks(
        self,
        req: ChatCompletionRequest,
        *,
        last_user: str,
    ) -> dict[str, list[dict[str, Any]]]:
        rag_chunks_by_source: dict[str, list[dict[str, Any]]] = {}

        if not req.rag_sources:
            return rag_chunks_by_source

        try:
            rag_service = self.ensure_rag_service()
        except Exception as exc:
            logger.error("Failed to initialize RAG service: %s", exc)
            return rag_chunks_by_source

        if rag_service is None or not rag_service.enabled:
            return rag_chunks_by_source

        try:
            for src in req.rag_sources:
                kb_cfg = get_kb_config(src.knowledge_base)
                effective_alias = src.alias or (kb_cfg.default_alias if kb_cfg else "champion")
                logger.info(
                    "RAG — retrieving from kb=%s alias=%s",
                    src.knowledge_base,
                    effective_alias,
                )
                docs = rag_service.retrieve_documents(
                    query=last_user,
                    knowledge_base=src.knowledge_base,
                    alias=effective_alias,
                )
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
        except Exception as exc:
            logger.error("Error retrieving RAG context: %s", exc)

        return rag_chunks_by_source

    def _prepare_request(self, req: ChatCompletionRequest) -> PreparedChatRequest:
        settings = get_settings()
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        decision = self._router.decide(last_user)

        rag_chunks_by_source = self._retrieve_rag_chunks(req, last_user=last_user)
        prompt = self._prompt_builder.build_budgeted_messages(
            task=decision.task,
            request_messages=[m.model_dump(exclude_none=True) for m in req.messages],
            rag_chunks_by_source=rag_chunks_by_source,
            rag_requested=bool(req.rag_sources),
            settings=settings,
        )

        logger.info("Built budgeted prompt with %s message(s)", len(prompt.messages))

        generation_payload: dict[str, Any] = {
            "model": req.model if req.model else settings.default_model,
            "messages": prompt.messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        generation_payload = {
            key: value for key, value in generation_payload.items() if value is not None
        }
        generation_payload.update(self._passthrough_fields(req))

        return PreparedChatRequest(
            generation_payload=generation_payload,
            budget_meta=build_budget_meta(settings),
            rag_context_chunks=prompt.rag_context_chunks,
            prompt_messages=prompt.messages,
        )

    async def _build_exact_generation_payload(
        self,
        prepared: PreparedChatRequest,
        *,
        stream: bool,
    ) -> tuple[dict[str, Any], int]:
        client = self._client()
        tokenize_payload = extract_tokenize_payload(prepared.generation_payload)
        tokenize_response = await client.tokenize(tokenize_payload)
        prompt_tokens = int(tokenize_response["count"])
        final_payload, _ = apply_response_token_budget(
            prepared.generation_payload,
            prompt_tokens=prompt_tokens,
            budget_meta=prepared.budget_meta,
            stream=stream,
        )
        return final_payload, prompt_tokens

    @staticmethod
    def _apply_usage(result: dict[str, Any], *, prompt_tokens: int | None) -> dict[str, Any]:
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        if prompt_tokens is not None and usage.get("prompt_tokens") is None:
            usage["prompt_tokens"] = prompt_tokens

        completion_tokens = usage.get("completion_tokens")
        if (
            usage.get("total_tokens") is None
            and isinstance(prompt_tokens, int)
            and isinstance(completion_tokens, int)
        ):
            usage["total_tokens"] = prompt_tokens + completion_tokens

        result["usage"] = usage
        return result

    @staticmethod
    def _usage_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    async def chat(
        self,
        req: ChatCompletionRequest,
        *,
        user_id: str | None = None,
        chat_session_id: str | None = None,
    ) -> Any:
        """Process chat request (sync or async based on configuration)."""
        settings = get_settings()
        prepared = self._prepare_request(req)

        if settings.async_enabled:
            result = await self._chat_async(prepared)
        else:
            result = await self._chat_sync(prepared)

        result["_prompt_messages"] = prepared.prompt_messages
        if prepared.rag_context_chunks:
            result["rag_context"] = prepared.rag_context_chunks

        if chat_session_id and user_id:
            await self._persist_exchange(req, result, chat_session_id)

        return result

    async def _chat_sync(self, prepared: PreparedChatRequest) -> Any:
        payload, prompt_tokens = await self._build_exact_generation_payload(
            prepared,
            stream=False,
        )
        result = await self._client().chat_completions(payload)
        return self._apply_usage(result, prompt_tokens=prompt_tokens)

    async def _chat_async(self, prepared: PreparedChatRequest) -> Any:
        settings = get_settings()
        conversation_id = str(_uuid.uuid4())
        celery_client = self._get_celery_client()
        redis_stream = self._get_redis_stream()

        task_id = celery_client.enqueue_generate_response(
            conversation_id=conversation_id,
            generation_payload=prepared.generation_payload,
            budget_meta=prepared.budget_meta,
        )
        logger.info("Enqueued async chat task %s for conversation %s", task_id, conversation_id)

        full_content = ""
        finish_reason = "stop"
        usage: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

        async for event in redis_stream.subscribe(
            conversation_id,
            timeout=settings.streaming_timeout,
        ):
            event_type = event.get("type")

            if event_type == "token":
                full_content += event.get("content", "")
                continue

            if event_type == "done":
                full_content = event.get("content", full_content)
                finish_reason = event.get("finish_reason", "stop")
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
                break

            if event_type == "error":
                if event.get("error_type") == "budget_exceeded":
                    raise ResponseBudgetExceededError(event.get("error", "Budget exceeded"))
                raise RuntimeError(f"Async inference error: {event.get('error')}")

        return {
            "id": f"chatcmpl-{conversation_id}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    async def stream_chat(
        self,
        req: ChatCompletionRequest,
        *,
        user_id: str | None = None,
        chat_session_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        del user_id, chat_session_id

        settings = get_settings()
        prepared = self._prepare_request(req)
        if settings.async_enabled:
            return await self._stream_chat_async(prepared)
        return await self._stream_chat_sync(prepared)

    async def _stream_chat_sync(self, prepared: PreparedChatRequest) -> AsyncIterator[bytes]:
        payload, _ = await self._build_exact_generation_payload(prepared, stream=True)
        return self._client().chat_completions_stream(payload)

    async def _stream_chat_async(self, prepared: PreparedChatRequest) -> AsyncIterator[bytes]:
        settings = get_settings()
        conversation_id = str(_uuid.uuid4())
        celery_client = self._get_celery_client()
        redis_stream = self._get_redis_stream()

        task_id = celery_client.enqueue_generate_response(
            conversation_id=conversation_id,
            generation_payload=prepared.generation_payload,
            budget_meta=prepared.budget_meta,
        )
        logger.info(
            "Enqueued async stream task %s for conversation %s",
            task_id,
            conversation_id,
        )
        return redis_stream.subscribe_sse(
            conversation_id,
            timeout=settings.streaming_timeout,
        )

    async def _persist_exchange(
        self,
        req: ChatCompletionRequest,
        result: dict,
        chat_session_id: str,
    ) -> None:
        """Persist the user message and assistant response to PostgreSQL."""
        try:
            from sqlalchemy import select

            from shared.db.engine import get_session_factory
            from shared.db.models import ChatMessage, ChatSession

            last_user_msg = next(
                (m.content for m in reversed(req.messages) if m.role == "user"),
                None,
            )
            assistant_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
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

                if last_user_msg:
                    db.add(
                        ChatMessage(
                            session_id=session_uuid,
                            role="user",
                            content=last_user_msg,
                        )
                    )

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

                if session:
                    session.updated_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception:
            logger.warning("Failed to persist chat exchange", exc_info=True)


process_chat = _ProcessChat()
