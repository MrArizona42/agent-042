"""Reranker microservice.

A lightweight FastAPI service that wraps a cross-encoder model (sentence-transformers)
and exposes an HTTP API for reranking retrieved passages.  Heavy dependencies
are isolated in this container, keeping the gateway image small.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app_config.runtime import get_settings
from shared.logging import configure_logging
from shared.telemetry import instrument_fastapi_app

configure_logging(service="reranker")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    """Request body for the /v1/rerank endpoint."""

    query: str = Field(..., description="Query string")
    passages: List[str] = Field(..., description="Candidate passage texts to score")


class RerankResponse(BaseModel):
    """Response body for the /v1/rerank endpoint."""

    scores: List[float]
    model: str


class InfoResponse(BaseModel):
    """Response body for the /v1/info endpoint: provider identity, no inference."""

    model: str


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_model: CrossEncoder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the cross-encoder model on startup."""
    global _model
    settings = get_settings()
    logger.info(f"Loading reranker model: {settings.rag.reranker_model}")
    _model = CrossEncoder(settings.rag.reranker_model)
    logger.info("Reranker model loaded")
    yield
    _model = None


app = FastAPI(title="Reranker Service", lifespan=lifespan)
instrument_fastapi_app(app, service="reranker")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    if _model is None:
        return {"status": "unavailable"}
    return {"status": "ok"}


@app.get("/v1/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Report the identity of the model this instance has loaded, with no inference."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    settings = get_settings()
    return InfoResponse(model=settings.rag.reranker_model)


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    """Score passages against a query using the cross-encoder model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    settings = get_settings()
    rag = settings.rag

    if not request.passages:
        return RerankResponse(scores=[], model=rag.reranker_model)

    pairs = [[request.query, passage] for passage in request.passages]
    scores: list[float] = _model.predict(pairs).tolist()

    return RerankResponse(scores=scores, model=rag.reranker_model)
