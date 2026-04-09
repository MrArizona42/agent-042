from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.api.routes import router as api_router
from gateway.auth.middleware import AuthMiddleware
from gateway.auth.oidc import OIDCClient
from gateway.auth.router import router as auth_router
from gateway.auth.session import SessionManager
from gateway.config import (
    bootstrap_local_settings_env,
    get_settings,
    validate_settings_on_startup,
)
from gateway.services.celery_client import CeleryClient
from gateway.services.processing import process_chat
from gateway.services.redis_stream import RedisStreamService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

bootstrap_local_settings_env(repo_root=Path(__file__).resolve().parents[2])

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
        # Validate knowledge base aliases at startup
        try:
            process_chat.ensure_rag_service(settings=settings, validate=True)
            logger.info("Knowledge base startup validation complete")
        except Exception:
            if settings.rag_strict_startup:
                raise
            logger.warning("Knowledge base startup validation failed", exc_info=True)

    if not settings.async_enabled:
        raise RuntimeError("GATEWAY_ASYNC_ENABLED=false is no longer supported.")
    if not settings.celery_broker_url:
        raise RuntimeError(
            "CELERY_BROKER_URL must be set because async inference is required. "
            "Example: amqp://user:password@rabbitmq:5672//"
        )

    # --- Create managed connections ---
    redis_stream = RedisStreamService(settings.redis_url)
    logger.info(f"Redis stream service initialized (url={settings.redis_url})")

    celery_client = CeleryClient(settings.celery_broker_url)
    logger.info("Celery client initialized")

    # Inject services into the shared process_chat instance
    process_chat.init_services(
        redis_stream=redis_stream,
        celery_client=celery_client,
    )

    # --- Auth services ---
    auth_redis: aioredis.Redis | None = None
    if settings.google_client_id:
        auth_redis = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
        app.state.oidc_client = OIDCClient(settings)
        app.state.session_manager = SessionManager(auth_redis)
        logger.info("OAuth2 / OIDC services initialized")
    else:
        # Auth disabled — install stubs so middleware skips gracefully
        app.state.oidc_client = None
        app.state.session_manager = None
        logger.warning("OAuth2 disabled (GATEWAY_GOOGLE_CLIENT_ID not set)")

    # --- Database engine ---
    if settings.agent042_db_url:
        from shared.db.engine import get_engine
        from shared.db.models import Base

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("agent042 database tables ensured")

    yield

    # --- Cleanup on shutdown ---
    logger.info("Shutting down — closing managed connections")
    await redis_stream.close()
    celery_client.close()
    if auth_redis is not None:
        await auth_redis.close()
    if settings.agent042_db_url:
        from shared.db.engine import close_engine

        await close_engine()
    logger.info("All managed connections closed")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.service_name, lifespan=lifespan)

    # Auth middleware (only enforced when OAuth is configured)
    if settings.google_client_id:
        app.add_middleware(AuthMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(api_router)
    return app


app = create_app()
