from __future__ import annotations

from typing import Any, Dict, Iterator

import requests

from ui.config import get_ui_settings


class GatewayClient:
    """Client for communicating with the Gateway API."""

    def __init__(self, base_url: str, session_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._ui_settings = get_ui_settings()
        self._session = requests.Session()
        if session_id:
            self._session.headers["Authorization"] = f"Bearer {session_id}"

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def me(self) -> Dict[str, Any] | None:
        """Return the current user profile or ``None`` if not authenticated."""
        try:
            r = self._session.get(
                f"{self.base_url}/auth/me",
                timeout=self._ui_settings.health_timeout,
            )
            if r.status_code == 401:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    def logout(self) -> None:
        """Log out the current session."""
        try:
            self._session.post(
                f"{self.base_url}/auth/logout",
                timeout=self._ui_settings.health_timeout,
            )
        except requests.RequestException:
            pass

    # ------------------------------------------------------------------
    # Chat session management
    # ------------------------------------------------------------------

    def create_chat_session(self, title: str | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if title:
            params["title"] = title
        r = self._session.post(
            f"{self.base_url}/v1/chat/sessions",
            params=params,
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_chat_sessions(self) -> list[Dict[str, Any]]:
        r = self._session.get(
            f"{self.base_url}/v1/chat/sessions",
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_session_messages(self, session_id: str) -> list[Dict[str, Any]]:
        r = self._session.get(
            f"{self.base_url}/v1/chat/sessions/{session_id}/messages",
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()
        return r.json()

    def delete_chat_session(self, session_id: str) -> None:
        r = self._session.delete(
            f"{self.base_url}/v1/chat/sessions/{session_id}",
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()

    # ------------------------------------------------------------------
    # Existing API methods
    # ------------------------------------------------------------------

    def health(self) -> dict:
        r = self._session.get(
            f"{self.base_url}/health",
            timeout=self._ui_settings.health_timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_models(self) -> Any:
        r = self._session.get(
            f"{self.base_url}/v1/models",
            timeout=self._ui_settings.models_timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self._ui_settings.chat_timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        payload = {**payload, "stream": True}
        with self._session.post(
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
