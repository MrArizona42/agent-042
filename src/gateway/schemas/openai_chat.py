from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    # Keep request compatible with OpenAI-like clients.
    model: str | None = None
    messages: list[ChatMessage]

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")

    stream: bool = False

    # Knowledge base selection for RAG retrieval.
    # Valid values: None (disabled), or a key from KNOWLEDGE_BASES (e.g. "arxiv", "pytorch_docs").
    knowledge_base: str | None = Field(
        default=None,
        description="Knowledge base to use for RAG retrieval (None = disabled)",
    )

    # passthrough for additional openai-ish fields (frequency_penalty, etc.)
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True
        extra = "allow"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
