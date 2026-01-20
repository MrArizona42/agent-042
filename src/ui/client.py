from __future__ import annotations

from typing import Any, Dict, Iterator

import requests

from ui.config import get_ui_settings


class GatewayClient:
    """Client for communicating with the Gateway API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._ui_settings = get_ui_settings()

    def health(self) -> dict:
        r = requests.get(
            f"{self.base_url}/health",
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_models(self) -> Any:
        r = requests.get(
            f"{self.base_url}/v1/models",
            timeout=self._ui_settings.models_timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self._ui_settings.chat_timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        payload = {**payload, "stream": True}
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=self._ui_settings.chat_timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                yield line
