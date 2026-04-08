from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from gateway.config import get_settings

logger = logging.getLogger(__name__)


class VllmOpenAIClient:
    """Minimal async client for vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: Optional[float] = None,
    ):
        """Initialize vLLM client.

        Args:
            base_url: vLLM server URL
            api_key: Optional API key for authentication
            timeout_s: Request timeout in seconds (uses config default if None)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        # Use config timeout if not explicitly provided
        self._timeout = timeout_s if timeout_s is not None else get_settings().vllm_timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def list_models(self) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}/v1/models", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def chat_completions(self, payload: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(f"vLLM error response: {resp.text}")
            resp.raise_for_status()
            return resp.json()

    async def tokenize(self, payload: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/tokenize",
                headers=self._headers(),
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(f"vLLM tokenize error response: {resp.text}")
            resp.raise_for_status()
            return resp.json()

    async def chat_completions_stream(self, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
