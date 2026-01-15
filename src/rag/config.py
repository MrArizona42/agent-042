"""RAG system configuration."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """Configuration for RAG system components."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", extra="ignore")

    # Qdrant connection
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    # Embedding model
    # Using lightweight all-MiniLM-L6-v2 (~80MB, fast on CPU)
    # For better quality (but slower): sentence-transformers/all-mpnet-base-v2 (~420MB)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Device for embeddings: cpu, cuda, mps
    embedding_device: str = Field(default="cpu")

    # Retrieval parameters
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0, description="Minimum similarity score")

    # RAG mode
    rag_enabled: bool = Field(default=True)


def get_rag_settings() -> RAGSettings:
    """Get RAG settings from environment."""
    return RAGSettings()
