from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Dict

from gateway.config import get_settings
from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.celery_client import CeleryClient
from gateway.services.prompt_builder import PromptBuilder
from gateway.services.rag_service import RAGService
from gateway.services.redis_stream import RedisStreamService
from gateway.services.task_router import RuleBasedTaskRouter
from gateway.services.vllm_client import VllmOpenAIClient

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

        # Try to retrieve RAG context using explicitly selected knowledge base
        retrieved_context = None
        rag_mode = "off"

        if self._rag_service and self._rag_service.enabled and req.knowledge_base:
            try:
                logger.info(f"RAG — retrieving from knowledge base: {req.knowledge_base}")
                retrieved_context = self._rag_service.retrieve_context(
                    query=last_user,
                    knowledge_base=req.knowledge_base,
                    top_k=5,
                )
                if retrieved_context:
                    rag_mode = "on"
                    logger.info(f"RAG context retrieved (kb={req.knowledge_base})")
                else:
                    logger.info(f"No RAG context found (kb={req.knowledge_base})")
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
        return payload

    async def chat(self, req: ChatCompletionRequest) -> Any:
        """Process chat request (sync or async based on configuration)."""
        settings = get_settings()

        if settings.async_enabled:
            return await self._chat_async(req)
        else:
            return await self._chat_sync(req)

    async def _chat_sync(self, req: ChatCompletionRequest) -> Any:
        """Synchronous chat: direct call to vLLM."""
        payload = self._build_payload(req)
        return await self._client().chat_completions(payload)

    async def _chat_async(self, req: ChatCompletionRequest) -> Any:
        """Asynchronous chat: enqueue task and wait for result via Redis."""
        payload = self._build_payload(req)

        conversation_id = str(uuid.uuid4())
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

        async for event in redis_stream.subscribe(conversation_id):
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

    async def stream_chat(self, req: ChatCompletionRequest) -> AsyncIterator[bytes]:
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
        # Ensure stream true.
        payload["stream"] = True
        async for chunk in self._client().chat_completions_stream(payload):
            yield chunk

    async def _stream_chat_async(self, req: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Asynchronous streaming: enqueue task and stream via Redis."""
        payload = self._build_payload(req)

        conversation_id = str(uuid.uuid4())
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
        async for chunk in redis_stream.subscribe_sse(conversation_id):
            yield chunk


process_chat = _ProcessChat()
