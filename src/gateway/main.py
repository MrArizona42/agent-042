from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.api.routes import router as api_router
from gateway.config import get_settings, validate_settings_on_startup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validate settings at module load time (fail fast)
validate_settings_on_startup()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.service_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Log configuration on startup."""
        logger.info(f"Starting {settings.service_name}")
        logger.info(f"vLLM endpoint: {settings.vllm_base_url}")
        logger.info(f"Default model: {settings.default_model}")
        logger.info(f"RAG enabled: {settings.rag_enabled}")
        if settings.rag_enabled:
            logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
            logger.info(f"Embedding model: {settings.embedding_model}")

    app.include_router(api_router)
    return app


app = create_app()
