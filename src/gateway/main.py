from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.api.routes import router as api_router
from gateway.config import get_settings, validate_settings_on_startup
from gateway.services.celery_client import CeleryClient
from gateway.services.processing import process_chat
from gateway.services.redis_stream import RedisStreamService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validate settings at module load time (fail fast)
validate_settings_on_startup()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application-wide resource lifecycle.

    Creates Redis and Celery connections on startup and ensures
    they are properly closed on shutdown.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.service_name}")
    logger.info(f"vLLM endpoint: {settings.vllm_base_url}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"RAG enabled: {settings.rag_enabled}")
    if settings.rag_enabled:
        logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        logger.info(f"Embedding model: {settings.embedding_model}")

    # --- Create managed connections ---
    redis_stream = RedisStreamService(settings.redis_url)
    logger.info(f"Redis stream service initialised (url={settings.redis_url})")

    celery_client: CeleryClient | None = None
    if settings.async_enabled:
        if not settings.celery_broker_url:
            raise RuntimeError(
                "CELERY_BROKER_URL must be set when async_enabled=true. "
                "Example: amqp://user:password@rabbitmq:5672//"
            )
        celery_client = CeleryClient(settings.celery_broker_url)
        logger.info("Celery client initialised")

    # Inject services into the shared process_chat instance
    process_chat.init_services(
        redis_stream=redis_stream,
        celery_client=celery_client,
    )

    yield

    # --- Cleanup on shutdown ---
    logger.info("Shutting down — closing managed connections")
    await redis_stream.close()
    if celery_client is not None:
        celery_client.close()
    logger.info("All managed connections closed")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.service_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()
