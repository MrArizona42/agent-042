from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

if "celery" not in sys.modules:
    celery_module = ModuleType("celery")

    class _CeleryStub:
        def __init__(self, *args, **kwargs):
            self.conf = SimpleNamespace(update=lambda **kw: None)

        def close(self):
            return None

    celery_module.Celery = _CeleryStub
    sys.modules["celery"] = celery_module

if "redis.asyncio" not in sys.modules:
    redis_module = ModuleType("redis")
    redis_asyncio_module = ModuleType("redis.asyncio")
    redis_exceptions_module = ModuleType("redis.exceptions")

    class _RedisError(Exception):
        pass

    redis_asyncio_module.from_url = Mock()
    redis_exceptions_module.ConnectionError = _RedisError
    redis_exceptions_module.TimeoutError = _RedisError
    redis_module.asyncio = redis_asyncio_module
    sys.modules["redis"] = redis_module
    sys.modules["redis.asyncio"] = redis_asyncio_module
    sys.modules["redis.exceptions"] = redis_exceptions_module

from gateway.services import celery_client as celery_client_module
from gateway.services import redis_stream as redis_stream_module
from gateway.services.celery_client import CeleryClient
from gateway.services.redis_stream import RedisStreamService


class TestCeleryClientLifecycle(TestCase):
    def test_configures_publish_retry_policy(self) -> None:
        fake_app = Mock()
        fake_app.conf = Mock()

        with patch.object(celery_client_module, "Celery", return_value=fake_app):
            client = CeleryClient("amqp://localhost:5672//")
            client._get_app()

        fake_app.conf.update.assert_called_once()
        conf_kwargs = fake_app.conf.update.call_args.kwargs
        self.assertTrue(conf_kwargs["task_publish_retry"])
        self.assertEqual(conf_kwargs["task_publish_retry_policy"]["max_retries"], 3)

    def test_close_global_client_resets_singleton(self) -> None:
        fake_client = Mock()
        celery_client_module._celery_client = fake_client

        celery_client_module.close_celery_client()

        fake_client.close.assert_called_once()
        self.assertIsNone(celery_client_module._celery_client)


class TestRedisStreamLifecycle(IsolatedAsyncioTestCase):
    async def test_get_redis_reconnects_stale_connection(self) -> None:
        service = RedisStreamService("redis://localhost:6379/0")
        stale_client = AsyncMock()
        stale_client.ping.side_effect = redis_stream_module.RedisConnectionError("stale")
        healthy_client = AsyncMock()

        with (
            patch.object(redis_stream_module.aioredis, "from_url", side_effect=[stale_client, healthy_client]),
            patch.object(service, "close", AsyncMock()),
        ):
            redis_client = await service._get_redis()

        self.assertIs(redis_client, healthy_client)

    async def test_close_global_service_resets_singleton(self) -> None:
        service = RedisStreamService("redis://localhost:6379/0")
        service.close = AsyncMock()
        redis_stream_module._redis_stream_service = service

        await redis_stream_module.close_redis_stream_service()

        service.close.assert_awaited_once()
        self.assertIsNone(redis_stream_module._redis_stream_service)
