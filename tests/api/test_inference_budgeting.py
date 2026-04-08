from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.services.budget import (
    compute_effective_history_budget,
    trim_history_pairs,
    trim_rag_chunks,
)
from gateway.services.prompt_builder import PromptBuilder
from shared.vllm_payloads import (
    ResponseBudgetExceededError,
    apply_response_token_budget,
    canonicalize_assistant_content,
    extract_tokenize_payload,
)


def _settings(**overrides):
    return SimpleNamespace(
        chars_per_token=4.0,
        budget_guard=8,
        budget_system=200,
        budget_turn=20,
        min_budget_history=6,
        budget_rag=80,
        min_response_budget=4,
        model_max_tokens=64,
        **overrides,
    )


def test_short_turn_expands_available_history_budget() -> None:
    history_budget = compute_effective_history_budget(
        min_budget_history=6,
        budget_turn=20,
        current_turn_tokens=3,
    )

    assert history_budget == 23


def test_long_turn_shrinks_history_to_minimum_budget() -> None:
    history_budget = compute_effective_history_budget(
        min_budget_history=6,
        budget_turn=20,
        current_turn_tokens=20,
    )

    assert history_budget == 6


def test_history_trimming_never_keeps_half_pair() -> None:
    messages = [
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "old-assistant"},
        {"role": "user", "content": "new-user"},
        {"role": "assistant", "content": "new-assistant"},
    ]

    trimmed = trim_history_pairs(messages, budget_history_effective=25, chars_per_token=1.0)

    assert trimmed == [
        {"role": "user", "content": "new-user"},
        {"role": "assistant", "content": "new-assistant"},
    ]


def test_multi_kb_rag_trimming_respects_fixed_shares() -> None:
    trimmed = trim_rag_chunks(
        {
            "arxiv:champion": [
                {
                    "knowledge_base": "arxiv",
                    "alias": "champion",
                    "content": "AAAA",
                    "score": 0.9,
                    "source": "arxiv_champion",
                },
                {
                    "knowledge_base": "arxiv",
                    "alias": "champion",
                    "content": "B" * 200,
                    "score": 0.8,
                    "source": "arxiv_champion",
                },
            ],
            "pytorch_docs:champion": [
                {
                    "knowledge_base": "pytorch_docs",
                    "alias": "champion",
                    "content": "CC",
                    "score": 0.7,
                    "source": "pytorch_docs_champion",
                },
            ],
        },
        budget_rag=80,
        chars_per_token=4.0,
    )

    assert list(trimmed) == ["arxiv:champion", "pytorch_docs:champion"]
    assert len(trimmed["arxiv:champion"]) <= 1
    assert len(trimmed["pytorch_docs:champion"]) == 1


def test_prompt_builder_trims_history_and_rag_into_messages() -> None:
    builder = PromptBuilder()
    result = builder.build_budgeted_messages(
        task="chat",
        request_messages=[
            {"role": "user", "content": "old-user"},
            {"role": "assistant", "content": "old-assistant"},
            {"role": "user", "content": "current"},
        ],
        rag_chunks_by_source={
            "arxiv:champion": [
                {
                    "knowledge_base": "arxiv",
                    "alias": "champion",
                    "content": "doc",
                    "score": 0.9,
                    "source": "arxiv_champion",
                }
            ]
        },
        rag_requested=True,
        settings=_settings(),
    )

    assert result.messages[0]["role"] == "system"
    assert result.messages[-1] == {"role": "user", "content": "current"}
    assert result.rag_context_chunks[0]["knowledge_base"] == "arxiv"


def test_worker_and_sync_share_same_final_budget_helper() -> None:
    payload, final_max_tokens = apply_response_token_budget(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"other_flag": True},
        },
        prompt_tokens=20,
        budget_meta={
            "model_max_tokens": 64,
            "budget_guard": 8,
            "min_response_budget": 4,
        },
        stream=True,
        include_usage=True,
    )

    assert final_max_tokens == 36
    assert payload["max_tokens"] == 36
    assert payload["stream"] is True
    assert payload["stream_options"] == {"other_flag": True, "include_usage": True}


def test_budget_helper_rejects_when_response_budget_too_small() -> None:
    with pytest.raises(ResponseBudgetExceededError):
        apply_response_token_budget(
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            prompt_tokens=60,
            budget_meta={
                "model_max_tokens": 64,
                "budget_guard": 8,
                "min_response_budget": 4,
            },
            stream=False,
        )


def test_tokenize_payload_keeps_only_chat_affecting_fields() -> None:
    tokenize_payload = extract_tokenize_payload(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
            "tools": [{"type": "function"}],
        }
    )

    assert tokenize_payload == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function"}],
    }


def test_canonicalize_assistant_content_wraps_thinking_block() -> None:
    assert canonicalize_assistant_content("plan", "answer") == "<think>plan</think>\n\nanswer"
    assert canonicalize_assistant_content("", "answer") == "answer"
