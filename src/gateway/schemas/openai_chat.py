from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class RAGSource(BaseModel):
    """A single knowledge-base source for RAG retrieval."""

    knowledge_base: str
    alias: str | None = Field(
        default=None,
        description="Alias role to use (e.g. 'champion', 'challenger'). "
        "None uses the KB's default_alias from the catalog.",
    )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Keep request compatible with OpenAI-like clients.
    model: str | None = None
    messages: list[ChatMessage]

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Legacy OpenAI-style completion cap used as an upper bound only.",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Requested completion cap used as an upper bound only.",
    )

    stream: bool = False

    # Chat session ID for persisting history in PostgreSQL.
    chat_session_id: str | None = Field(
        default=None,
        description="Chat session UUID for server-side history persistence",
    )

    # RAG sources: multiple knowledge bases with explicit alias selection.
    rag_sources: list[RAGSource] | None = Field(
        default=None,
        description=(
            "Knowledge bases for RAG retrieval. null = auto-select knowledge bases for the "
            "routed task, [] = disable RAG, non-empty list = explicit override."
        ),
    )


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
