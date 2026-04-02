from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PromptBuildResult:
    system_prompt: str


class PromptBuilder:
    def build_system_prompt(
        self,
        task: str,
        rag_mode: str,
        retrieved_context: Optional[str] = None,
    ) -> PromptBuildResult:
        base = "You are an AI assistant for ML/DL/AI/LLM researchers."

        if task == "summarize":
            sys = (
                base
                + " Summarize the provided content clearly and accurately. "
                + "If the user asks for TL;DR, provide a short summary first, then details."
            )
        elif task == "code":
            sys = (
                base
                + " Help with writing, debugging, and refactoring code. "
                + "Prefer correct, runnable solutions and explain key steps."
            )
        else:
            sys = base + " Answer concisely, but include important caveats."

        # Add RAG context if available
        if retrieved_context:
            sys += (
                "\n\n--- RETRIEVED CONTEXT ---\n"
                "Below is relevant information retrieved from the knowledge base. "
                "Use it to provide accurate, well-informed answers. "
                "Cite sources when appropriate.\n\n" + retrieved_context + "\n--- END CONTEXT ---"
            )
        elif rag_mode != "off":
            # RAG enabled but no context retrieved
            sys += "\n\n(No relevant context was found in the knowledge base for this query.)"

        return PromptBuildResult(system_prompt=sys)
