from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from shared.config import BudgetSettings


class BudgetValidationError(ValueError):
    """Raised when a request cannot satisfy configured prompt budget rules."""


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Approximate token count from character length.

    Uses ceiling rounding so the gateway errs on the conservative side.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / chars_per_token))


def compute_effective_history_budget(
    *,
    min_budget_history: int,
    budget_turn: int,
    current_turn_tokens: int,
) -> int:
    if current_turn_tokens > budget_turn:
        raise BudgetValidationError(
            "Current user turn exceeds the configured request budget. "
            f"Estimated={current_turn_tokens}, budget_turn={budget_turn}."
        )
    return min_budget_history + (budget_turn - current_turn_tokens)


def build_budget_meta(budget_settings: BudgetSettings) -> dict[str, int]:
    return {
        "model_max_tokens": int(budget_settings.model_max_tokens),
        "budget_guard": int(budget_settings.budget_guard),
        "min_response_budget": int(budget_settings.min_response_budget),
    }


def trim_history_pairs(
    messages: Sequence[Mapping[str, Any]],
    budget_history_effective: int,
    chars_per_token: float,
) -> list[dict[str, Any]]:
    """Trim history from the oldest side while preserving completed pairs.

    For non user/assistant messages, the function falls back to keeping or
    dropping them as standalone units so irregular histories still behave
    predictably.
    """
    if budget_history_effective <= 0 or not messages:
        return []

    units = _history_units(messages)
    kept_units: list[list[dict[str, Any]]] = []
    used_tokens = 0

    for unit in reversed(units):
        unit_tokens = sum(
            estimate_tokens(str(message.get("content", "")), chars_per_token) for message in unit
        )
        if used_tokens + unit_tokens > budget_history_effective:
            break
        kept_units.append(unit)
        used_tokens += unit_tokens

    trimmed_history: list[dict[str, Any]] = []
    for unit in reversed(kept_units):
        trimmed_history.extend(unit)
    return trimmed_history


def trim_rag_chunks(
    chunks_per_source: Mapping[str, Sequence[Mapping[str, Any]]],
    budget_rag: int,
    chars_per_token: float,
) -> dict[str, list[dict[str, Any]]]:
    """Keep RAG chunks by fixed per-source shares without redistribution."""
    if not chunks_per_source or budget_rag <= 0:
        return {}

    section_budget = budget_rag // len(chunks_per_source)
    if section_budget <= 0:
        return {}

    kept: dict[str, list[dict[str, Any]]] = {}
    for source_key, chunks in chunks_per_source.items():
        if not chunks:
            continue

        first_chunk = chunks[0]
        available_tokens = section_budget - estimate_tokens(
            _format_rag_section_header(first_chunk), chars_per_token
        )
        if available_tokens <= 0:
            continue

        source_kept: list[dict[str, Any]] = []
        used_tokens = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_text = _format_rag_chunk(index, chunk)
            chunk_tokens = estimate_tokens(chunk_text, chars_per_token)
            if used_tokens + chunk_tokens > available_tokens:
                break
            source_kept.append(dict(chunk))
            used_tokens += chunk_tokens

        if source_kept:
            kept[source_key] = source_kept

    return kept


def render_rag_sections(chunks_per_source: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Render kept RAG chunks into a structured prompt section."""
    if not chunks_per_source:
        return ""

    parts = [
        "--- RETRIEVED CONTEXT ---",
        "Below is relevant information retrieved from the knowledge base. "
        "Use it to provide accurate, well-informed answers. Cite sources when appropriate.",
    ]

    for chunks in chunks_per_source.values():
        if not chunks:
            continue
        parts.append(_format_rag_section_header(chunks[0]))
        for index, chunk in enumerate(chunks, start=1):
            parts.append(_format_rag_chunk(index, chunk))

    parts.append("--- END CONTEXT ---")
    return "\n\n".join(parts)


def _history_units(messages: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    units: list[list[dict[str, Any]]] = []
    pending_user: dict[str, Any] | None = None

    for raw_message in messages:
        message = dict(raw_message)
        role = message.get("role")

        if role == "user":
            if pending_user is not None:
                units.append([pending_user])
            pending_user = message
            continue

        if role == "assistant" and pending_user is not None:
            units.append([pending_user, message])
            pending_user = None
            continue

        if pending_user is not None:
            units.append([pending_user])
            pending_user = None
        units.append([message])

    if pending_user is not None:
        units.append([pending_user])

    return units


def _format_rag_section_header(chunk: Mapping[str, Any]) -> str:
    knowledge_base = chunk.get("knowledge_base", "unknown")
    alias = chunk.get("alias", "unknown")
    return f"### Knowledge Base: {knowledge_base} (alias: {alias})"


def _format_rag_chunk(index: int, chunk: Mapping[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    source = metadata.get("source") or chunk.get("source") or "unknown"
    score = chunk.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "n/a"
    content = str(chunk.get("content", ""))
    return f"[Document {index}] (Source: {source}, Score: {score_text})\n{content}"
