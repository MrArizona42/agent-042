"""RAG benchmark and evaluation data contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EntityType = Literal["document", "chunk"]


class BenchmarkCase(BaseModel):
    """Case input only: query/messages and provenance, no expected outputs.

    Expected outputs (reference answers, rubrics, relevance judgments) live
    entirely on `BenchmarkLabel`, so labels can be revised without touching
    the cases artifact.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    benchmark_source_instance_id: str
    split: str | None = None
    query: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "benchmark_source_instance_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _query_or_messages_required(self) -> "BenchmarkCase":
        if not self.query and not self.messages:
            raise ValueError("case requires at least one of 'query' or 'messages'")
        return self


class Qrel(BaseModel):
    """Graded relevance judgment for one case against one document or chunk."""

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType = "document"
    entity_id: str
    relevance_grade: int = Field(ge=0)
    evidence_ref: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class BenchmarkLabel(BaseModel):
    """Expected outputs for one benchmark case: qrels, answers, scores, rubrics.

    Labels are optional on a benchmark source instance: retrieval benchmarks,
    answer benchmarks, scored datasets, and unlabeled smoke/regression sets
    all share this same shape, populating only the fields relevant to them.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    qrels: list[Qrel] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    reference_answers: list[str] = Field(default_factory=list)
    reference_answer_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    rubrics: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    def relevant_doc_ids(self, *, min_grade: int = 1) -> list[str]:
        """Derive document ids with relevance_grade >= min_grade from qrels.

        Not a stored field: filtering `qrels` keeps exactly one stored
        relevance judgment per case-entity pair instead of a flat list that
        can drift from `qrels`.
        """
        return [
            qrel.entity_id
            for qrel in self.qrels
            if qrel.entity_type == "document" and qrel.relevance_grade >= min_grade
        ]

    def relevant_chunk_ids(self, *, min_grade: int = 1) -> list[str]:
        """Derive chunk ids with relevance_grade >= min_grade from qrels. Not a stored field."""
        return [
            qrel.entity_id
            for qrel in self.qrels
            if qrel.entity_type == "chunk" and qrel.relevance_grade >= min_grade
        ]


class BenchmarkPreparedArtifacts(BaseModel):
    """Normalized cases and labels emitted by a benchmark adapter's `prepare_benchmark()`."""

    model_config = ConfigDict(extra="forbid")

    cases: list[BenchmarkCase] = Field(default_factory=list)
    labels: list[BenchmarkLabel] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """One retrieved chunk's provenance and rank within a retrieval observation."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=0)
    chunk_id: str
    document_id: str
    source_instance_id: str
    score: float
    title: str | None = None
    uri: str | None = None
    in_prompt: bool = False
    prompt_rank: int | None = None
    text_digest: str | None = None

    @field_validator("chunk_id", "document_id", "source_instance_id")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class RetrievalEvalObservation(BaseModel):
    """Structured per-case retrieval observation for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    knowledge_base: str
    alias: str
    qdrant_alias: str | None = None
    collection_name: str
    manifest_id: str | None = None
    retrieval_strategy: str | None = None
    retrieval_capability: str | None = None
    top_k: int | None = Field(default=None, ge=0)
    score_threshold: float | None = None
    reranker: str | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @field_validator("case_id", "knowledge_base", "alias", "collection_name")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class GenerationObservation(BaseModel):
    """Per-case generation facts for an answer benchmark observation."""

    model_config = ConfigDict(extra="forbid")

    prompt_digest: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    error: str | None = None


class AnswerEvalObservation(BaseModel):
    """Generated answer and citation result for one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    answer: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    generation: GenerationObservation = Field(default_factory=GenerationObservation)

    @field_validator("case_id")
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
