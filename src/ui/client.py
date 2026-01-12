from __future__ import annotations

from typing import Any, Dict, Iterator

import requests


class GatewayClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> Any:
        r = requests.get(f"{self.base_url}/v1/models", timeout=30)
        r.raise_for_status()
        return r.json()

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    def chat_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        payload = {**payload, "stream": True}
        with requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, stream=True, timeout=300
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                yield line

