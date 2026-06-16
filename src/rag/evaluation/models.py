"""RAG evaluation data contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NormalizedEvalRow(BaseModel):
    """A single evaluation query with expected answer and relevant document ids."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str
    benchmark_scope: str
    query_id: str
    query: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_name", "dataset_version", "benchmark_scope", "query_id", "query")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class Qrel(BaseModel):
    """Relevance judgment linking a query to a relevant document."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    document_id: str
    relevance_grade: int = Field(ge=0)
    evidence_ref: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "document_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class RetrievalEvalObservation(BaseModel):
    """Runtime retrieval result for one eval row."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    knowledge_base: str
    alias: str
    resolved_collection: str
    manifest_id: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)

    @field_validator("query_id", "knowledge_base", "alias", "resolved_collection")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class AnswerEvalObservation(BaseModel):
    """Generated answer and citation result for one eval row."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    answer: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)

    @field_validator("query_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class EvalResult(BaseModel):
    """Persisted result record for one eval row in one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    query_id: str
    rag_manifest_id: str | None = None
    knowledge_base: str
    alias: str
    metrics: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    retrieval: RetrievalEvalObservation | None = None
    answer: AnswerEvalObservation | None = None

    @field_validator("run_id", "query_id", "knowledge_base", "alias")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class PromotionDecision(BaseModel):
    """Outcome of evaluating a candidate collection against promotion gates."""

    model_config = ConfigDict(extra="forbid")

    candidate: str
    promote: bool
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    gate_details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()
