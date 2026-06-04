from __future__ import annotations

from typing import Any, Optional

import httpx

from shared.config import get_settings


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
        self._timeout = timeout_s if timeout_s is not None else get_settings().gateway.vllm_timeout

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
