from __future__ import annotations

from typing import Any, Mapping

CHAT_AFFECTING_FIELDS = (
    "model",
    "messages",
    "tools",
    "chat_template",
    "chat_template_kwargs",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "mm_processor_kwargs",
)


class ResponseBudgetExceededError(ValueError):
    """Raised when the exact prompt leaves too little room for generation."""


def canonicalize_assistant_content(
    thinking_content: str | None,
    answer_content: str | None,
) -> str:
    thinking = thinking_content or ""
    answer = answer_content or ""
    if thinking:
        return f"<think>{thinking}</think>\n\n{answer}"
    return answer


def extract_tokenize_payload(generation_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only chat-affecting fields for vLLM /tokenize parity."""
    return {
        field: generation_payload[field]
        for field in CHAT_AFFECTING_FIELDS
        if field in generation_payload and generation_payload[field] is not None
    }


def compute_response_token_budget(
    prompt_tokens: int,
    budget_meta: Mapping[str, Any],
) -> int:
    final_max_tokens = (
        int(budget_meta["model_max_tokens"]) - int(prompt_tokens) - int(budget_meta["budget_guard"])
    )
    if final_max_tokens < int(budget_meta["min_response_budget"]):
        raise ResponseBudgetExceededError(
            "Exact prompt leaves too little room for generation. "
            f"prompt_tokens={prompt_tokens}, remaining={final_max_tokens}, "
            f"minimum={budget_meta['min_response_budget']}."
        )
    return final_max_tokens


def apply_response_token_budget(
    generation_payload: Mapping[str, Any],
    *,
    prompt_tokens: int,
    budget_meta: Mapping[str, Any],
    stream: bool | None = None,
    include_usage: bool = False,
) -> tuple[dict[str, Any], int]:
    final_max_tokens = compute_response_token_budget(prompt_tokens, budget_meta)
    payload = dict(generation_payload)
    payload["max_tokens"] = final_max_tokens
    if stream is not None:
        payload["stream"] = stream
    if include_usage:
        stream_options = dict(payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    return payload, final_max_tokens
