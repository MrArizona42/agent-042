from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import shared.config as cfg
from gateway.schemas.openai_chat import ChatCompletionRequest, RAGSource
from gateway.services.processing import _ProcessChat
from gateway.services.task_router import RouteDecision
from shared.catalog import AdapterConfig, KBConfig, TaskConfig, catalog_override
from shared.config import Settings, load_settings


def _alias_config() -> dict[str, object]:
    return {
        "top_k": 5,
        "score_threshold": 0.35,
        "reranker": None,
        "retrieval_strategy": "dense",
        "reranker_multiplier": 4,
    }


@pytest.fixture(autouse=True)
def _loaded_kb_catalog() -> None:
    cfg.clear_knowledge_base_caches()

    arxiv = KBConfig(
        name="arxiv",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        label="ArXiv",
        description="Research papers",
        selection_description="Research papers and literature-grounded answers.",
    )
    pytorch_docs = KBConfig(
        name="pytorch_docs",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        label="PyTorch docs",
        description="API docs",
        selection_description="PyTorch API reference and implementation guidance.",
    )

    catalog = {
        "chat": TaskConfig(
            task="chat",
            label="General knowledge",
            routing_description="General ML research discussion.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[arxiv],
        ),
        "code": TaskConfig(
            task="code",
            label="Coding assistance",
            routing_description="Programming help for ML systems.",
            adapter=AdapterConfig(name="lora-code", alias="champion", enabled=True),
            knowledge_bases=[pytorch_docs],
        ),
        "summarize": TaskConfig(
            task="summarize",
            label="Summarization",
            routing_description="Summarize user-provided content.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[],
        ),
    }
    with catalog_override(catalog):
        yield

    cfg.clear_knowledge_base_caches()


def _settings(
    *,
    behavior: dict[str, object] | None = None,
    budget: dict[str, object] | None = None,
    rag: dict[str, object] | None = None,
) -> Settings:
    gateway_values = {
        "repetition_penalty": 1.1,
    }
    budget_values = {
        "model_max_tokens": 64,
        "budget_guard": 8,
        "min_response_budget": 4,
    }
    rag_values = {
        "enabled": True,
    }
    if behavior is not None:
        gateway_values.update(behavior)
    if budget is not None:
        budget_values.update(budget)
    if rag is not None:
        rag_values.update(rag)
    return load_settings(
        overrides={
            "vllm": {"model": "base-model"},
            "gateway": {**gateway_values, "budget": budget_values},
            "rag": rag_values,
        }
    )


def _prompt_result() -> SimpleNamespace:
    return SimpleNamespace(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        rag_context_chunks=[],
    )


def test_resolve_rag_request_distinguishes_auto_off_and_explicit() -> None:
    process = _ProcessChat()

    auto_request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
    off_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        rag_sources=[],
    )
    explicit_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        rag_sources=[{"knowledge_base": "arxiv"}],
    )

    auto = process._resolve_rag_request(auto_request, task="chat")
    force_off = process._resolve_rag_request(off_request, task="chat")
    explicit = process._resolve_rag_request(explicit_request, task="chat")

    assert auto.mode == "auto"
    assert auto.sources == ()
    assert auto.rag_requested is False
    assert auto.task_has_knowledge_bases is True

    assert force_off.mode == "off"
    assert force_off.sources == ()
    assert force_off.rag_requested is False
    assert force_off.task_has_knowledge_bases is True

    assert explicit.mode == "explicit"
    assert len(explicit.sources) == 1
    assert explicit.sources[0].knowledge_base == "arxiv"
    assert explicit.rag_requested is True
    assert explicit.task_has_knowledge_bases is True


def test_resolve_rag_request_marks_summarize_as_having_no_kbs() -> None:
    process = _ProcessChat()
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    resolved = process._resolve_rag_request(request, task="summarize")

    assert resolved.mode == "auto"
    assert resolved.task_has_knowledge_bases is False


def test_prepare_request_uses_task_adapter_model_when_no_explicit_model() -> None:
    process = _ProcessChat()
    process._prompt_builder = MagicMock()
    process._prompt_builder.build_budgeted_messages.return_value = _prompt_result()

    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    with (
        patch(
            "gateway.services.processing.get_settings",
            return_value=_settings(rag={"enabled": False}),
        ),
        patch.object(process._router, "decide", return_value=RouteDecision(task="code")),
        patch.object(process, "_retrieve_rag_chunks", return_value={}) as retrieve_rag,
    ):
        prepared = process._prepare_request(request)

    assert prepared.generation_payload["model"] == "lora-code-champion"
    retrieve_rag.assert_called_once_with((), last_user="hello")


def test_prepare_request_prefers_explicit_model_over_task_adapter() -> None:
    process = _ProcessChat()
    process._prompt_builder = MagicMock()
    process._prompt_builder.build_budgeted_messages.return_value = _prompt_result()

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        model="manual-model",
    )

    with (
        patch(
            "gateway.services.processing.get_settings",
            return_value=_settings(rag={"enabled": False}),
        ),
        patch.object(process._router, "decide", return_value=RouteDecision(task="code")),
        patch.object(process, "_retrieve_rag_chunks", return_value={}),
    ):
        prepared = process._prepare_request(request)

    assert prepared.generation_payload["model"] == "manual-model"


def test_prepare_request_marks_explicit_rag_sources_as_requested() -> None:
    process = _ProcessChat()
    process._prompt_builder = MagicMock()
    process._prompt_builder.build_budgeted_messages.return_value = _prompt_result()

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        rag_sources=[{"knowledge_base": "arxiv"}],
    )

    with (
        patch("gateway.services.processing.get_settings", return_value=_settings()),
        patch.object(process._router, "decide", return_value=RouteDecision(task="chat")),
        patch.object(process, "_retrieve_rag_chunks", return_value={}) as retrieve_rag,
    ):
        process._prepare_request(request)

    retrieve_rag.assert_called_once()
    _, kwargs = process._prompt_builder.build_budgeted_messages.call_args
    assert kwargs["rag_requested"] is True


def test_prepare_request_auto_selects_task_scoped_rag_sources() -> None:
    process = _ProcessChat()
    process._prompt_builder = MagicMock()
    process._prompt_builder.build_budgeted_messages.return_value = _prompt_result()

    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
    rag_service = MagicMock()
    rag_service.enabled = True
    rag_service.select_knowledge_bases.return_value = [RAGSource(knowledge_base="arxiv")]

    with (
        patch("gateway.services.processing.get_settings", return_value=_settings()),
        patch.object(process._router, "decide", return_value=RouteDecision(task="chat")),
        patch.object(process, "ensure_rag_service", return_value=rag_service),
        patch.object(process, "_retrieve_rag_chunks", return_value={}) as retrieve_rag,
    ):
        process._prepare_request(request)

    rag_service.select_knowledge_bases.assert_called_once_with("hello", "chat")
    retrieve_rag.assert_called_once()
    assert [src.knowledge_base for src in retrieve_rag.call_args.args[0]] == ["arxiv"]
    _, kwargs = process._prompt_builder.build_budgeted_messages.call_args
    assert kwargs["rag_requested"] is True


def test_prepare_request_skips_auto_selection_for_tasks_without_kbs() -> None:
    process = _ProcessChat()
    process._prompt_builder = MagicMock()
    process._prompt_builder.build_budgeted_messages.return_value = _prompt_result()

    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    with (
        patch("gateway.services.processing.get_settings", return_value=_settings()),
        patch.object(process._router, "decide", return_value=RouteDecision(task="summarize")),
        patch.object(process, "ensure_rag_service") as ensure_rag_service,
        patch.object(process, "_retrieve_rag_chunks", return_value={}) as retrieve_rag,
    ):
        process._prepare_request(request)

    ensure_rag_service.assert_not_called()
    retrieve_rag.assert_called_once_with((), last_user="hello")
    _, kwargs = process._prompt_builder.build_budgeted_messages.call_args
    assert kwargs["rag_requested"] is False


def test_reload_config_caches_invalidates_and_warms_router_and_rag_service() -> None:
    process = _ProcessChat()
    process._router = MagicMock()
    existing_rag_service = MagicMock()
    warmed_rag_service = MagicMock()
    process._rag_service = existing_rag_service

    settings = _settings()

    with patch.object(
        process,
        "ensure_rag_service",
        return_value=warmed_rag_service,
    ) as ensure_rag_service:
        process.reload_config_caches(settings=settings)

    process._router.invalidate_cache.assert_called_once_with()
    process._router.warm_cache.assert_called_once_with()
    existing_rag_service.invalidate_caches.assert_called_once_with()
    ensure_rag_service.assert_called_once_with(settings=settings, validate=True)
    warmed_rag_service.warm_caches.assert_called_once_with()


def test_reload_config_caches_is_fail_open_on_warmup_errors() -> None:
    process = _ProcessChat()
    process._router = MagicMock()
    process._router.warm_cache.side_effect = RuntimeError("router down")

    with patch.object(process, "ensure_rag_service", side_effect=RuntimeError("rag down")):
        process.reload_config_caches(settings=_settings())

    process._router.invalidate_cache.assert_called_once_with()
    process._router.warm_cache.assert_called_once_with()
