from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

if "celery" not in sys.modules:
    celery_module = types.ModuleType("celery")

    class _CeleryStub:
        def __init__(self, *args, **kwargs):
            self.conf = SimpleNamespace(update=lambda **kw: None)

        def close(self):
            return None

    celery_module.Celery = _CeleryStub
    sys.modules["celery"] = celery_module

if "kombu.exceptions" not in sys.modules:
    kombu_module = types.ModuleType("kombu")
    kombu_exceptions_module = types.ModuleType("kombu.exceptions")

    class _OperationalError(Exception):
        pass

    kombu_exceptions_module.OperationalError = _OperationalError
    sys.modules["kombu"] = kombu_module
    sys.modules["kombu.exceptions"] = kombu_exceptions_module

if "redis.asyncio" not in sys.modules:
    redis_module = types.ModuleType("redis")
    redis_asyncio_module = types.ModuleType("redis.asyncio")
    redis_exceptions_module = types.ModuleType("redis.exceptions")

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
    def test_enqueue_retries_once_after_connection_error(self) -> None:
        client = CeleryClient("amqp://localhost:5672//")

        failed_app = Mock()
        failed_app.send_task.side_effect = celery_client_module.OperationalError("connection lost")
        successful_app = Mock()
        successful_app.send_task.return_value = SimpleNamespace(id="task-1")

        with (
            patch.object(client, "_get_app", side_effect=[failed_app, successful_app]),
            patch.object(client, "close") as close_mock,
        ):
            task_id = client.enqueue_generate_response(
                conversation_id="conv-1",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(task_id, "task-1")
        close_mock.assert_called_once()

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
