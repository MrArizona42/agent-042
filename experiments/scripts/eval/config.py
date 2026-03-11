"""EvalConfig — Pydantic model capturing the full identity of an eval run.

Every eval run records a frozen snapshot of all moving parts so that
results are reproducible and comparable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdapterConfig(BaseModel):
    """Adapter (LoRA) identity — nullable for base-model-only evaluation."""

    name: str | None = None
    version: int | None = None
    mlflow_run_id: str | None = None


class RAGConfig(BaseModel):
    """RAG pipeline parameters frozen at eval time."""

    enabled: bool = True
    knowledge_base: str | None = "arxiv"
    qdrant_collection: str | None = None
    qdrant_snapshot_id: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunking_strategy: str = "fixed_token"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    score_threshold: float = 0.35
    reranking_strategy: str | None = "none"


class JudgeConfig(BaseModel):
    """LLM-as-judge configuration."""

    enabled: bool = True
    model: str | None = "gemini-2.0-flash"
    max_rpm: int = 14
    timeout: float = 30.0
    structured_output: bool = True


class GenerationConfig(BaseModel):
    """vLLM generation parameters."""

    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512


class DatasetConfig(BaseModel):
    """Dataset identity for the eval run."""

    name: str = "hotpotqa"
    split: str = "validation"
    max_examples: int | None = 200
    seed: int = 42
    dvc_hash: str | None = None


class MetricsConfig(BaseModel):
    """Automatic metric settings."""

    bert_score_model: str = "roberta-large"


class EvalConfig(BaseModel):
    """Immutable identity of an evaluation run.

    Serialized as JSONB in the ``eval_runs.config`` column so every
    field is queryable.
    """

    # Model
    base_model: str = "/models/Qwen/Qwen3-0.6B"
    vllm_base_url: str = "http://localhost:8000"
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)

    # RAG
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Eval
    task: str = "chat"
    tier: str = "regression"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    task_metrics: list[str] = Field(
        default_factory=lambda: ["relevance", "correctness", "rouge_l", "bert_score"]
    )

    # Judge
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # Generation
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # Metrics
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # DB
    db_url: str = ""
