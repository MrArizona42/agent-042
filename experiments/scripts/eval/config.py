"""EvalConfig — the immutable identity of an evaluation run.

This Pydantic model captures every moving part of an eval: model, adapter, RAG
settings, dataset details, generation parameters, judge config, and metric
settings.  It is serialized to JSONB in the ``eval_runs.config`` column so that
every run is fully reproducible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Frozen snapshot of all parameters for one evaluation run."""

    # ── Model ──
    base_model: str = Field(description="HuggingFace model path served by vLLM")
    vllm_base_url: str = Field(default="http://localhost:8000")

    # ── Adapter (nullable for base model eval) ──
    adapter_name: str | None = Field(default=None, description="MLflow registered model name")
    adapter_version: int | None = Field(
        default=None, description="MLflow model version number"
    )
    adapter_mlflow_run_id: str | None = Field(
        default=None, description="Training run ID for traceability"
    )

    # ── RAG ──
    rag_enabled: bool = Field(default=True)
    knowledge_base: str | None = Field(default=None, description="arxiv | pytorch_docs | None")
    qdrant_collection: str | None = Field(default=None)
    qdrant_snapshot_id: str | None = Field(default=None)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunking_strategy: str = Field(default="fixed_token")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    retrieval_top_k: int = Field(default=5)
    score_threshold: float = Field(default=0.35)
    reranking_strategy: str | None = Field(default=None)

    # ── Eval ──
    dataset_name: str = Field(description="hotpotqa | arxiv-summarization | humaneval | ...")
    dataset_split: str = Field(default="validation")
    dataset_dvc_hash: str | None = Field(default=None)
    task: str = Field(description="chat | summarize | code")
    tier: str = Field(default="regression", description="regression | full")
    max_examples: int | None = Field(default=None, description="None = use all")
    seed: int | None = Field(default=42, description="For reproducible subsampling")

    # ── Judge ──
    judge_enabled: bool = Field(default=False)
    judge_model: str | None = Field(default=None, description="e.g. gemini-2.0-flash")
    bert_score_model: str = Field(default="roberta-large")

    # ── Generation params ──
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    max_tokens: int = Field(default=512)

    model_config = {"frozen": True}
