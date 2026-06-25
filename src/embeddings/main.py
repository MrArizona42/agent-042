"""Embeddings microservice.

A lightweight FastAPI service that wraps sentence-transformers and exposes
an HTTP API for generating text embeddings.  Heavy dependencies (PyTorch,
sentence-transformers) are isolated in this container, keeping the gateway
and Airflow images small.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException
from fastembed.sparse import SparseTextEmbedding
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app_config.runtime import get_settings
from clients.observability.logging import configure_logging
from clients.observability.telemetry import instrument_fastapi_app

configure_logging(service="embeddings")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class EmbeddingsRequest(BaseModel):
    """Request body for the /v1/embeddings endpoint."""

    input: List[str] = Field(..., description="List of texts to embed")


class EmbeddingItem(BaseModel):
    """Single embedding entry in the response."""

    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    """Response body for the /v1/embeddings endpoint."""

    data: List[EmbeddingItem]
    model: str
    dimension: int


class DimensionResponse(BaseModel):
    """Response body for the /v1/dimension endpoint."""

    dimension: int
    model: str


class InfoResponse(BaseModel):
    """Response body for the /v1/info endpoint: provider identity, no inference."""

    dense_model: str
    dense_dimension: int
    sparse_model: str


class SparseEmbeddingItem(BaseModel):
    """Single sparse embedding entry in the response."""

    indices: List[int]
    values: List[float]
    index: int


class SparseEmbeddingsResponse(BaseModel):
    """Response body for the /v1/sparse-embeddings endpoint."""

    data: List[SparseEmbeddingItem]
    model: str


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_sparse_model: SparseTextEmbedding | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model on startup."""
    global _model, _sparse_model
    settings = get_settings()
    rag = settings.rag
    logger.info(f"Loading embedding model: {rag.embedding_model} on device: {rag.embedding_device}")
    _model = SentenceTransformer(rag.embedding_model, device=rag.embedding_device)
    dimension = _model.get_sentence_embedding_dimension()
    logger.info(f"Embedding model loaded — dimension: {dimension}")
    logger.info(f"Loading sparse encoder model: {rag.sparse_encoder_model}")
    _sparse_model = SparseTextEmbedding(rag.sparse_encoder_model)
    logger.info("Sparse encoder model loaded")
    yield
    _model = None
    _sparse_model = None


app = FastAPI(title="Embeddings Service", lifespan=lifespan)
instrument_fastapi_app(app, service="embeddings")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    if _model is None:
        return {"status": "unavailable"}
    return {"status": "ok"}


@app.get("/v1/dimension", response_model=DimensionResponse)
def dimension() -> DimensionResponse:
    """Return the embedding dimension of the loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    settings = get_settings()
    dim = _model.get_sentence_embedding_dimension()
    return DimensionResponse(dimension=dim, model=settings.rag.embedding_model)


@app.get("/v1/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Report the identity of the dense and sparse models this instance has loaded.

    Used to validate catalog-declared encoder identity before build or query,
    without performing any embedding work.
    """
    if _model is None or _sparse_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    settings = get_settings()
    rag = settings.rag
    return InfoResponse(
        dense_model=rag.embedding_model,
        dense_dimension=_model.get_sentence_embedding_dimension(),
        sparse_model=rag.sparse_encoder_model,
    )


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def embed(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Generate embeddings for a list of texts."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    settings = get_settings()
    rag = settings.rag

    if not request.input:
        return EmbeddingsResponse(
            data=[],
            model=rag.embedding_model,
            dimension=_model.get_sentence_embedding_dimension(),
        )

    vectors = _model.encode(
        request.input,
        batch_size=rag.build.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    data = [EmbeddingItem(embedding=vec.tolist(), index=i) for i, vec in enumerate(vectors)]

    return EmbeddingsResponse(
        data=data,
        model=rag.embedding_model,
        dimension=_model.get_sentence_embedding_dimension(),
    )


@app.post("/v1/sparse-embeddings", response_model=SparseEmbeddingsResponse)
def sparse_embed(request: EmbeddingsRequest) -> SparseEmbeddingsResponse:
    """Generate sparse (BM25) embeddings for a list of texts."""
    if _sparse_model is None:
        raise HTTPException(status_code=503, detail="Sparse model not loaded")
    settings = get_settings()
    rag = settings.rag

    if not request.input:
        return SparseEmbeddingsResponse(data=[], model=rag.sparse_encoder_model)

    sparse_vecs = list(_sparse_model.embed(request.input))
    data = [
        SparseEmbeddingItem(
            indices=vec.indices.tolist(),
            values=vec.values.tolist(),
            index=i,
        )
        for i, vec in enumerate(sparse_vecs)
    ]
    return SparseEmbeddingsResponse(data=data, model=rag.sparse_encoder_model)
