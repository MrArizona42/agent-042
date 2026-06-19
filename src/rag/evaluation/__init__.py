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
from rag.evaluation.retrieval import (
    ProjectRetrievalEvalResult,
    ProjectRetrieverEvaluator,
    graded_ndcg,
)

__all__ = [
    "AnswerEvalObservation",
    "BenchmarkCase",
    "BenchmarkLabel",
    "BenchmarkPreparedArtifacts",
    "GenerationObservation",
    "PromotionDecision",
    "ProjectRetrievalEvalResult",
    "ProjectRetrieverEvaluator",
    "Qrel",
    "RetrievalEvalObservation",
    "RetrievedChunk",
    "graded_ndcg",
]
