from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

from gateway.config import get_settings
from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.celery_client import CeleryClient
from gateway.services.prompt_builder import PromptBuilder
from gateway.services.rag_service import RAGService
from gateway.services.redis_stream import RedisStreamService
from gateway.services.task_router import RuleBasedTaskRouter
from gateway.services.vllm_client import VllmOpenAIClient
from shared.config import get_kb_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _ProcessChat:
    def __init__(self) -> None:
        self._router = RuleBasedTaskRouter()
        self._prompt_builder = PromptBuilder()

        # Initialize RAG service
        try:
            settings = get_settings()
            self._rag_service = RAGService(settings)
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            self._rag_service = None

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

    def _client(self) -> VllmOpenAIClient:
        s = get_settings()
        return VllmOpenAIClient(base_url=s.vllm_base_url, api_key=None)

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

    def _build_payload(self, req: ChatCompletionRequest) -> Dict[str, Any]:
        # Decide task based on last user message (or fallback).
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        decision = self._router.decide(last_user)

        # Try to retrieve RAG context using rag_sources (multi-KB + alias)
        retrieved_context = None
        rag_mode = "off"
        rag_context_chunks: list[Dict[str, Any]] = []

        if self._rag_service and self._rag_service.enabled and req.rag_sources:
            try:
                context_parts: list[str] = []
                for src in req.rag_sources:
                    kb_cfg = get_kb_config(src.knowledge_base)
                    effective_alias = src.alias or (kb_cfg.default_alias if kb_cfg else "champion")
                    logger.info(
                        f"RAG — retrieving from kb={src.knowledge_base} alias={effective_alias}"
                    )
                    docs = self._rag_service.retrieve_documents(
                        query=last_user,
                        knowledge_base=src.knowledge_base,
                        alias=effective_alias,
                    )
                    if docs:
                        source_label = f"{src.knowledge_base}_{effective_alias}"
                        for doc in docs:
                            rag_context_chunks.append(
                                {
                                    "content": doc.content,
                                    "score": doc.score if doc.score is not None else 0.0,
                                    "source": source_label,
                                }
                            )
                        ctx = self._rag_service.format_documents(docs)
                        if ctx:
                            context_parts.append(ctx)
                if context_parts:
                    retrieved_context = "\n\n".join(context_parts)
                    rag_mode = "on"
                    logger.info(f"RAG context retrieved from {len(context_parts)} source(s)")
                else:
                    logger.info("No RAG context found from any source")
            except Exception as e:
                logger.error(f"Error retrieving RAG context: {e}")

        prompt = self._prompt_builder.build_system_prompt(
            task=decision.task,
            rag_mode=rag_mode,
            retrieved_context=retrieved_context,
        )
        logger.info(f"RAG built system prompt: {prompt.system_prompt}")

        messages = list(req.messages)
        if not any(m.role == "system" for m in messages):
            messages = [
                {"role": "system", "content": prompt.system_prompt},
                *[m.model_dump(exclude_none=True) for m in messages],
            ]
        else:
            messages = [m.model_dump(exclude_none=True) for m in messages]

        # Use the default model from settings if none is provided.
        model = req.model if req.model else get_settings().default_model

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }

        # Drop None fields (vLLM is generally tolerant, but keep it clean).
        payload = {k: v for k, v in payload.items() if v is not None}

        # Include any extra openai-like fields.
        payload.update(req.extra)

        # Attach RAG context chunks for eval groundedness scoring
        payload["_rag_context_chunks"] = rag_context_chunks

        return payload

    async def chat(
        self,
        req: ChatCompletionRequest,
        *,
        user_id: str | None = None,
        chat_session_id: str | None = None,
    ) -> Any:
        """Process chat request (sync or async based on configuration)."""
        settings = get_settings()

        payload = self._build_payload(req)

        # Extract RAG context chunks before forwarding to inference
        rag_context_chunks = payload.pop("_rag_context_chunks", [])

        if settings.async_enabled:
            result = await self._chat_async(req, payload)
        else:
            result = await self._chat_sync(req, payload)

        # Attach the full prompt messages for UI debugging display
        result["_prompt_messages"] = payload.get("messages", [])

        # Include RAG context in response when RAG was used
        if rag_context_chunks:
            result["rag_context"] = rag_context_chunks

        # Persist messages if a chat session is associated
        if chat_session_id and user_id:
            await self._persist_exchange(req, result, chat_session_id)

        return result

    async def _chat_sync(
        self, req: ChatCompletionRequest, payload: Dict[str, Any] | None = None
    ) -> Any:
        """Synchronous chat: direct call to vLLM."""
        if payload is None:
            payload = self._build_payload(req)
        return await self._client().chat_completions(payload)

    async def _chat_async(
        self, req: ChatCompletionRequest, payload: Dict[str, Any] | None = None
    ) -> Any:
        """Asynchronous chat: enqueue task and wait for result via Redis."""
        settings = get_settings()
        if payload is None:
            payload = self._build_payload(req)

        conversation_id = str(_uuid.uuid4())
        celery_client = self._get_celery_client()
        redis_stream = self._get_redis_stream()

        # Enqueue the task
        task_id = celery_client.enqueue_generate_response(
            conversation_id=conversation_id,
            messages=payload.get("messages", []),
            model=payload.get("model"),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            max_tokens=payload.get("max_tokens"),
        )

        logger.info(f"Enqueued async chat task {task_id} for conversation {conversation_id}")

        # Wait for completion via Redis
        full_content = ""
        finish_reason = "stop"

        async for event in redis_stream.subscribe(
            conversation_id, timeout=settings.streaming_timeout
        ):
            event_type = event.get("type")

            if event_type == "token":
                full_content += event.get("content", "")
            elif event_type == "done":
                full_content = event.get("content", full_content)
                finish_reason = event.get("finish_reason", "stop")
                break
            elif event_type == "error":
                raise RuntimeError(f"Async inference error: {event.get('error')}")

        # Return OpenAI-compatible response
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
            "usage": {
                "prompt_tokens": 0,  # Not tracked in async mode
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def stream_chat(
        self,
        req: ChatCompletionRequest,
        *,
        user_id: str | None = None,
        chat_session_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream chat response (sync or async based on configuration)."""
        settings = get_settings()

        if settings.async_enabled:
            async for chunk in self._stream_chat_async(req):
                yield chunk
        else:
            async for chunk in self._stream_chat_sync(req):
                yield chunk

    async def _stream_chat_sync(self, req: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Synchronous streaming: direct call to vLLM."""
        payload = self._build_payload(req)
        payload.pop("_rag_context_chunks", None)
        # Ensure stream true.
        payload["stream"] = True
        async for chunk in self._client().chat_completions_stream(payload):
            yield chunk

    async def _stream_chat_async(self, req: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Asynchronous streaming: enqueue task and stream via Redis."""
        settings = get_settings()
        payload = self._build_payload(req)
        payload.pop("_rag_context_chunks", None)

        conversation_id = str(_uuid.uuid4())
        celery_client = self._get_celery_client()
        redis_stream = self._get_redis_stream()

        # Enqueue the task
        task_id = celery_client.enqueue_generate_response(
            conversation_id=conversation_id,
            messages=payload.get("messages", []),
            model=payload.get("model"),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            max_tokens=payload.get("max_tokens"),
        )

        logger.info(f"Enqueued async stream task {task_id} for conversation {conversation_id}")

        # Stream from Redis
        async for chunk in redis_stream.subscribe_sse(
            conversation_id, timeout=settings.streaming_timeout
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _persist_exchange(
        self,
        req: ChatCompletionRequest,
        result: dict,
        chat_session_id: str,
    ) -> None:
        """Persist the user message and assistant response to PostgreSQL."""
        try:
            from shared.db.engine import get_session_factory
            from shared.db.models import ChatMessage, ChatSession

            last_user_msg = next(
                (m.content for m in reversed(req.messages) if m.role == "user"), None
            )
            assistant_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            session_uuid = _uuid.UUID(chat_session_id)

            async with get_session_factory()() as db:
                # Update session title if empty (first message)
                from sqlalchemy import select

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
                        )
                    )

                if session:
                    session.updated_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception:
            logger.warning("Failed to persist chat exchange", exc_info=True)


process_chat = _ProcessChat()
