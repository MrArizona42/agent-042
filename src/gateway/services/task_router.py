from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    task: str  # chat | summarize | code


class RuleBasedTaskRouter:
    """Very small baseline router.

    Later you can swap this with an LLM-based decision layer.
    """

    def decide(self, user_text: str) -> RouteDecision:
        t = user_text.lower()
        if any(k in t for k in ["summarize", "summary", "tl;dr", "tldr"]):
            return RouteDecision(task="summarize")
        if any(k in t for k in ["code", "python", "bug", "traceback", "refactor"]):
            return RouteDecision(task="code")
        return RouteDecision(task="chat")

