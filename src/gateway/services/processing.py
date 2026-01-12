from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from gateway.config import get_settings
from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.prompt_builder import PromptBuilder
from gateway.services.task_router import RuleBasedTaskRouter
from gateway.services.vllm_client import VllmOpenAIClient


class _ProcessChat:
    def __init__(self) -> None:
        self._router = RuleBasedTaskRouter()
        self._prompt_builder = PromptBuilder()

    def _client(self) -> VllmOpenAIClient:
        s = get_settings()
        return VllmOpenAIClient(base_url=s.vllm_base_url, api_key=None)

    async def list_models(self) -> Any:
        return await self._client().list_models()

    def _build_payload(self, req: ChatCompletionRequest) -> Dict[str, Any]:
        # Decide task based on last user message (or fallback).
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        decision = self._router.decide(last_user)
        prompt = self._prompt_builder.build_system_prompt(task=decision.task, rag_mode="off")

        messages = list(req.messages)
        if not any(m.role == "system" for m in messages):
            messages = [{"role": "system", "content": prompt.system_prompt}, *[m.model_dump(exclude_none=True) for m in messages]]
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
        payload = self._build_payload(req)
        return await self._client().chat_completions(payload)

    async def stream_chat(self, req: ChatCompletionRequest) -> AsyncIterator[bytes]:
        payload = self._build_payload(req)
        # Ensure stream true.
        payload["stream"] = True
        async for chunk in self._client().chat_completions_stream(payload):
            yield chunk


process_chat = _ProcessChat()

