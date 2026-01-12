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

    # passthrough for additional openai-ish fields (frequency_penalty, etc.)
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True
        extra = "allow"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]

