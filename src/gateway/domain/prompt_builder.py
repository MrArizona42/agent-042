from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app_config.runtime import BudgetSettings
from gateway.domain.budget import (
    BudgetValidationError,
    compute_effective_history_budget,
    estimate_tokens,
    render_rag_sections,
    trim_history_pairs,
    trim_rag_chunks,
)


@dataclass(frozen=True)
class PromptBuildResult:
    system_prompt: str
    messages: list[dict[str, Any]]
    rag_context_chunks: list[dict[str, Any]]
    prompt_debug: dict[str, Any]


class PromptBuilder:
    def build_base_system_prompt(self, task: str) -> str:
        base = "You are an AI assistant for ML/DL/AI/LLM researchers."

        if task == "summarize":
            return (
                base
                + " Summarize the provided content clearly and accurately. "
                + "If the user asks for TL;DR, provide a short summary first, then details."
            )
        if task == "code":
            return (
                base
                + " Help with writing, debugging, and refactoring code. "
                + "Prefer correct, runnable solutions and explain key steps. "
                + "Do not fabricate large binary or bytecode dumps, and stop "
                + "instead of emitting repetitive filler."
            )
        return base + " Answer concisely, but include important caveats."

    def build_budgeted_messages(
        self,
        *,
        task: str,
        request_messages: Sequence[Mapping[str, Any]],
        rag_chunks_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
        rag_requested: bool,
        budget_settings: BudgetSettings,
    ) -> PromptBuildResult:
        base_system_prompt = self.build_base_system_prompt(task)
        system_messages = [
            str(message.get("content", ""))
            for message in request_messages
            if message.get("role") == "system"
        ]
        conversation_messages = [
            dict(message) for message in request_messages if message.get("role") != "system"
        ]

        if not conversation_messages:
            raise BudgetValidationError(
                "Chat request must include at least one non-system message."
            )

        current_turn = dict(conversation_messages[-1])
        history_messages = [dict(message) for message in conversation_messages[:-1]]

        current_turn_tokens = estimate_tokens(
            str(current_turn.get("content", "")),
            budget_settings.chars_per_token,
        )
        history_budget = compute_effective_history_budget(
            min_budget_history=budget_settings.min_budget_history,
            budget_turn=budget_settings.budget_turn,
            current_turn_tokens=current_turn_tokens,
        )
        trimmed_history = trim_history_pairs(
            history_messages,
            history_budget,
            budget_settings.chars_per_token,
        )

        system_prompt = base_system_prompt
        if system_messages:
            system_prompt += "\n\nAdditional caller instructions:\n" + "\n\n".join(system_messages)

        system_prompt_tokens = estimate_tokens(system_prompt, budget_settings.chars_per_token)
        if system_prompt_tokens > budget_settings.budget_system:
            raise BudgetValidationError(
                "System prompt exceeds the configured request budget. "
                f"Estimated={system_prompt_tokens}, budget_system={budget_settings.budget_system}."
            )

        trimmed_rag = trim_rag_chunks(
            rag_chunks_by_source,
            budget_settings.budget_rag,
            budget_settings.chars_per_token,
        )
        rag_context = render_rag_sections(trimmed_rag)
        if rag_context:
            final_system_prompt = system_prompt + "\n\n" + rag_context
        elif rag_requested:
            final_system_prompt = (
                system_prompt
                + "\n\n(No relevant context was found in the knowledge base for this query.)"
            )
        else:
            final_system_prompt = system_prompt

        trimmed_rag_chunks = [dict(chunk) for chunks in trimmed_rag.values() for chunk in chunks]
        messages = [
            {"role": "system", "content": final_system_prompt},
            *trimmed_history,
            current_turn,
        ]

        history_tokens = sum(
            estimate_tokens(str(message.get("content", "")), budget_settings.chars_per_token)
            for message in trimmed_history
        )
        rag_tokens = sum(
            estimate_tokens(str(chunk.get("content", "")), budget_settings.chars_per_token)
            for chunk in trimmed_rag_chunks
        )

        return PromptBuildResult(
            system_prompt=final_system_prompt,
            messages=messages,
            rag_context_chunks=trimmed_rag_chunks,
            prompt_debug={
                "system_tokens_est": system_prompt_tokens,
                "current_turn_tokens_est": current_turn_tokens,
                "history_tokens_est": history_tokens,
                "history_budget_est": history_budget,
                "rag_tokens_est": rag_tokens,
                "rag_sources_kept": list(trimmed_rag.keys()),
            },
        )
