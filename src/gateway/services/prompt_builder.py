from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBuildResult:
    system_prompt: str


class PromptBuilder:
    def build_system_prompt(self, task: str, rag_mode: str = "off") -> PromptBuildResult:
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

        if rag_mode != "off":
            # Stub for future RAG augmentation.
            sys += "\n\n(You may receive additional retrieved context snippets; use them when relevant.)"

        return PromptBuildResult(system_prompt=sys)

