from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Dict

import requests

from app_config.runtime import get_settings


@dataclass(frozen=True)
class GatewayStreamEvent:
    event: str
    data: Any


@dataclass(frozen=True)
class GatewayStreamResponse:
    request_id: str | None
    events: Iterator[GatewayStreamEvent]


class GatewayClient:
    """Client for communicating with the Gateway API."""

    def __init__(self, base_url: str, session_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._ui_settings = get_settings().ui
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

    def get_prompt_preview(self, request_id: str) -> Dict[str, Any]:
        r = self._session.get(
            f"{self.base_url}/v1/chat/prompt-preview/{request_id}",
            timeout=self._ui_settings.chat_timeout,
        )
        r.raise_for_status()
        return r.json()

    def _iter_sse_events(self, response: requests.Response) -> Iterator[GatewayStreamEvent]:
        event_name = "message"
        data_lines: list[str] = []

        try:
            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue

                if line == "":
                    if not data_lines:
                        event_name = "message"
                        continue

                    payload = "\n".join(data_lines)
                    if payload == "[DONE]":
                        yield GatewayStreamEvent(event="done_marker", data=payload)
                    else:
                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            parsed = payload
                        yield GatewayStreamEvent(event=event_name, data=parsed)

                    event_name = "message"
                    data_lines = []
                    continue

                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip() or "message"
                    continue

                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())

            if data_lines:
                payload = "\n".join(data_lines)
                if payload == "[DONE]":
                    yield GatewayStreamEvent(event="done_marker", data=payload)
                else:
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        parsed = payload
                    yield GatewayStreamEvent(event=event_name, data=parsed)
        finally:
            response.close()

    def chat_stream(
        self,
        payload: Dict[str, Any],
        *,
        rich_stream: bool = False,
    ) -> GatewayStreamResponse:
        payload = {**payload, "stream": True}
        headers: Dict[str, str] = {}
        if rich_stream:
            headers["X-UI-Rich-Stream"] = "1"

        response = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            headers=headers or None,
            timeout=self._ui_settings.chat_timeout,
        )
        response.raise_for_status()
        return GatewayStreamResponse(
            request_id=response.headers.get("X-Request-Id"),
            events=self._iter_sse_events(response),
        )
