"""RAG evaluation contracts, metrics, and runners."""

from rag.evaluation.models import (
    AnswerEvalObservation,
    BenchmarkCase,
    BenchmarkLabel,
    BenchmarkPreparedArtifacts,
    GenerationObservation,
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
    "ProjectRetrievalEvalResult",
    "ProjectRetrieverEvaluator",
    "Qrel",
    "RetrievalEvalObservation",
    "RetrievedChunk",
    "graded_ndcg",
]
