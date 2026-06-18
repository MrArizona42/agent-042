"""RAG evaluation contracts, metrics, runners, and promotion gates."""

from rag.evaluation.models import (
    AnswerEvalObservation,
    BenchmarkCase,
    BenchmarkLabel,
    BenchmarkPreparedArtifacts,
    GenerationObservation,
    PromotionDecision,
    Qrel,
    RetrievalEvalObservation,
    RetrievedChunk,
)

__all__ = [
    "AnswerEvalObservation",
    "BenchmarkCase",
    "BenchmarkLabel",
    "BenchmarkPreparedArtifacts",
    "GenerationObservation",
    "PromotionDecision",
    "Qrel",
    "RetrievalEvalObservation",
    "RetrievedChunk",
]
