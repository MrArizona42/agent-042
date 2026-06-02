"""Tests for config contract validation.

Covers AliasConfig completeness, AdapterConfig validation, KB / task registry
requirements, registry reference validation, and validate_kb_alias() error
messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.operator_registry_samples import (
    write_chat_only_operator_registry,
    write_code_only_operator_registry,
)


@pytest.fixture(autouse=True)
def _reset_kb_registry():
    import shared.config as cfg

    cfg.clear_knowledge_base_caches()
    yield
    cfg.clear_knowledge_base_caches()


# ---------------------------------------------------------------------------
# AliasConfig validation
# ---------------------------------------------------------------------------


class TestAliasConfigValidation:
    """AliasConfig rejects incomplete JSON entries."""

    def test_missing_top_k_raises(self):
        from pydantic import ValidationError

        from shared.operator_registry import AliasConfig

        with pytest.raises(ValidationError, match="top_k"):
            AliasConfig(
                score_threshold=0.35,
                reranker=None,
            )

    def test_missing_score_threshold_raises(self):
        from pydantic import ValidationError

        from shared.operator_registry import AliasConfig

        with pytest.raises(ValidationError, match="score_threshold"):
            AliasConfig(top_k=5, reranker=None)

    def test_missing_reranker_raises(self):
        from pydantic import ValidationError

        from shared.operator_registry import AliasConfig

        with pytest.raises(ValidationError, match="reranker"):
            AliasConfig(top_k=5, score_threshold=0.35)

    def test_complete_alias_config_ok(self):
        from shared.operator_registry import AliasConfig

        cfg = AliasConfig(
            top_k=5,
            score_threshold=0.35,
            reranker=None,
            retrieval_strategy="dense",
            reranker_multiplier=4,
        )
        assert cfg.top_k == 5
        assert cfg.reranker is None

    def test_sparse_alias_config_ok(self):
        from shared.operator_registry import AliasConfig

        cfg = AliasConfig(
            top_k=5,
            score_threshold=0.35,
            reranker=None,
            retrieval_strategy="sparse",
            reranker_multiplier=4,
        )

        assert cfg.retrieval_strategy == "sparse"


class TestAdapterConfigValidation:
    def test_disabled_adapter_allows_empty_strings(self):
        from shared.operator_registry import AdapterConfig

        cfg = AdapterConfig(name="", alias="", enabled=False)

        assert cfg.enabled is False
        assert cfg.name == ""
        assert cfg.alias == ""

    def test_enabled_adapter_requires_name(self):
        from pydantic import ValidationError

        from shared.operator_registry import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="", alias="champion", enabled=True)

    def test_enabled_adapter_requires_alias(self):
        from pydantic import ValidationError

        from shared.operator_registry import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="lora-chat", alias="", enabled=True)


# ---------------------------------------------------------------------------
# KBConfig.default_alias must point to declared alias
# ---------------------------------------------------------------------------


class TestKBConfigDefaultAlias:
    def test_default_alias_must_be_declared(self):
        from pydantic import ValidationError

        from shared.operator_registry import KBConfig

        with pytest.raises(ValidationError, match="default_alias"):
            KBConfig(
                name="test_kb",
                default_alias="missing",
                aliases={
                    "champion": {
                        "top_k": 5,
                        "score_threshold": 0.35,
                        "reranker": None,
                        "retrieval_strategy": "dense",
                        "reranker_multiplier": 4,
                    },
                },
                update_strategy="replace",
                selection_description="Selection text",
            )

    def test_valid_default_alias_ok(self):
        from shared.operator_registry import KBConfig

        cfg = KBConfig(
            name="test_kb",
            default_alias="champion",
            aliases={
                "champion": {
                    "top_k": 5,
                    "score_threshold": 0.35,
                    "reranker": None,
                    "retrieval_strategy": "dense",
                    "reranker_multiplier": 4,
                },
            },
            update_strategy="replace",
            selection_description="Selection text",
        )
        assert cfg.default_alias == "champion"

    def test_selection_description_is_required(self):
        from pydantic import ValidationError

        from shared.operator_registry import KBConfig

        with pytest.raises(ValidationError, match="selection_description"):
            KBConfig(
                name="test_kb",
                default_alias="champion",
                aliases={
                    "champion": {
                        "top_k": 5,
                        "score_threshold": 0.35,
                        "reranker": None,
                        "retrieval_strategy": "dense",
                        "reranker_multiplier": 4,
                    },
                },
                update_strategy="replace",
            )


class TestTaskConfigValidation:
    def test_task_config_allows_empty_knowledge_bases(self):
        from shared.operator_registry import TaskConfig

        cfg = TaskConfig(
            task="summarize",
            label="Summarization",
            routing_description="Summarize user-provided content.",
            adapter={"name": "", "alias": "", "enabled": False},
            knowledge_bases=[],
        )

        assert cfg.task == "summarize"
        assert cfg.knowledge_bases == []

    def test_task_config_requires_routing_description(self):
        from pydantic import ValidationError

        from shared.operator_registry import TaskConfig

        with pytest.raises(ValidationError, match="routing_description"):
            TaskConfig(
                task="chat",
                label="General knowledge",
                knowledge_bases=[],
            )


# ---------------------------------------------------------------------------
# Duplicate KB names across tasks rejected
# ---------------------------------------------------------------------------


class TestRegistryReferenceValidation:
    def test_unknown_kb_ref_is_rejected(self, tmp_path: Path):
        from shared.operator_registry import load_knowledge_bases

        path = tmp_path / "invalid.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.chat]",
                    'routing_description = "General chat about ML research."',
                    'kb_refs = ["missing_kb"]',
                    "",
                    "[tasks.chat.adapter]",
                    "enabled = false",
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="references unknown KB 'missing_kb'"):
            load_knowledge_bases(path)


class TestKnowledgeBaseRegistryResolution:
    def test_gateway_settings_expose_grouped_sections(self, monkeypatch):
        from shared.config import load_settings

        monkeypatch.setenv("PLATFORM__VLLM_BASE_URL", "http://platform-vllm:8000")

        settings = load_settings({"rag": {"rag_enabled": False}})

        assert settings.platform.vllm_base_url == "http://platform-vllm:8000"
        assert settings.rag.rag_enabled is False
        assert settings.gateway.service_name == "agent-042-gateway"
        assert settings.auth.session_ttl_seconds == 86400
        with pytest.raises(AttributeError):
            _ = settings.service_name
        with pytest.raises(AttributeError):
            _ = settings.rag.knowledge_bases_path

    def test_load_settings_merges_env_backed_nested_defaults_with_overrides(self, monkeypatch):
        from shared.config import load_settings

        monkeypatch.setenv("GATEWAY__URL", "http://gateway-from-env:9001")
        monkeypatch.setenv("GATEWAY__BUDGET__MODEL_MAX_TOKENS", "4096")

        settings = load_settings(
            {
                "gateway": {
                    "default_model": "override-model",
                    "budget": {"min_response_budget": 1024},
                }
            }
        )

        assert settings.gateway.default_model == "override-model"
        assert settings.gateway.url == "http://gateway-from-env:9001"
        assert settings.gateway.budget.model_max_tokens == 4096
        assert settings.gateway.budget.min_response_budget == 1024

    def test_gateway_cors_and_registry_aliases_use_canonical_nested_names(self, monkeypatch):
        from shared.config import load_settings

        monkeypatch.setenv("GATEWAY__CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
        monkeypatch.setenv("REGISTRY__SYNC_ALIASES", "champion,shadow")

        settings = load_settings()

        assert settings.gateway.cors_allow_origins == (
            "https://a.example",
            "https://b.example",
        )
        assert settings.registry.sync_aliases == ("champion", "shadow")

    def test_legacy_flat_env_names_are_ignored(self, monkeypatch):
        from shared.config import load_settings

        monkeypatch.delenv("PLATFORM__VLLM_BASE_URL", raising=False)
        monkeypatch.setenv("VLLM_BASE_URL", "http://legacy-vllm:8000")

        settings = load_settings()

        assert settings.platform.vllm_base_url == "http://localhost:8000"

    def test_registry_settings_own_operator_registry_path(self):
        from shared.config import RegistryConfig
        from shared.operator_registry import resolve_knowledge_bases_path

        settings = RegistryConfig(operator_registry_path="configs/operator_registry.toml")

        assert settings.operator_registry_path == Path("configs/operator_registry.toml")
        assert resolve_knowledge_bases_path(settings).as_posix().endswith(
            "configs/operator_registry.toml"
        )

    def test_get_knowledge_bases_prefers_registry_settings_path(self, tmp_path: Path, monkeypatch):
        import shared.config as cfg
        from shared.operator_registry import get_kb_names, get_knowledge_bases

        path = tmp_path / "operator-registry.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.code]",
                    'routing_description = "Programming help for ML systems."',
                    'kb_refs = ["pytorch_docs"]',
                    "",
                    "[tasks.code.adapter]",
                    "enabled = false",
                    "",
                    "[knowledge_bases.pytorch_docs]",
                    'default_alias = "champion"',
                    'selection_description = "PyTorch API reference."',
                    "",
                    "[knowledge_bases.pytorch_docs.aliases.champion]",
                    'profile = "champion"',
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("REGISTRY__OPERATOR_REGISTRY_PATH", str(path))
        cfg.clear_knowledge_base_caches()

        registry = get_knowledge_bases()

        assert list(registry) == ["code"]
        assert get_kb_names() == ["pytorch_docs"]

    def test_clear_knowledge_base_caches_refreshes_registry_settings_path(
        self, tmp_path: Path, monkeypatch
    ):
        import shared.config as cfg
        from shared.operator_registry import get_kb_names

        first = tmp_path / "registry-first.toml"
        first.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.chat]",
                    'routing_description = "General chat about ML research."',
                    'kb_refs = ["arxiv"]',
                    "",
                    "[tasks.chat.adapter]",
                    "enabled = false",
                    "",
                    "[knowledge_bases.arxiv]",
                    'default_alias = "champion"',
                    'selection_description = "Research papers and theory."',
                    "",
                    "[knowledge_bases.arxiv.aliases.champion]",
                    'profile = "champion"',
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        second = tmp_path / "registry-second.toml"
        second.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.code]",
                    'routing_description = "Programming help for ML systems."',
                    'kb_refs = ["pytorch_docs"]',
                    "",
                    "[tasks.code.adapter]",
                    "enabled = false",
                    "",
                    "[knowledge_bases.pytorch_docs]",
                    'default_alias = "champion"',
                    'selection_description = "PyTorch API reference."',
                    "",
                    "[knowledge_bases.pytorch_docs.aliases.champion]",
                    'profile = "champion"',
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("REGISTRY__OPERATOR_REGISTRY_PATH", str(first))
        cfg.clear_knowledge_base_caches()
        assert get_kb_names() == ["arxiv"]

        monkeypatch.setenv("REGISTRY__OPERATOR_REGISTRY_PATH", str(second))
        cfg.clear_knowledge_base_caches()
        assert get_kb_names() == ["pytorch_docs"]

    def test_legacy_registry_env_name_is_ignored(self, tmp_path: Path, monkeypatch):
        from shared.operator_registry import resolve_knowledge_bases_path

        path = tmp_path / "operator-registry.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.chat]",
                    'routing_description = "General chat about ML research."',
                    'kb_refs = ["arxiv"]',
                    "",
                    "[tasks.chat.adapter]",
                    "enabled = false",
                    "",
                    "[knowledge_bases.arxiv]",
                    'default_alias = "champion"',
                    'selection_description = "Research papers and theory."',
                    "",
                    "[knowledge_bases.arxiv.aliases.champion]",
                    'profile = "champion"',
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("REGISTRY_OPERATOR_REGISTRY_PATH", str(path))
        resolved = resolve_knowledge_bases_path()

        assert resolved != path

    def test_in_memory_registry_override_bypasses_disk_loading(self):
        from shared.operator_registry import (
            AdapterConfig,
            KBConfig,
            TaskConfig,
            get_kb_config,
            get_kb_names,
            registry_override,
        )

        arxiv = KBConfig(
            name="arxiv",
            default_alias="champion",
            aliases={
                "champion": {
                    "top_k": 5,
                    "score_threshold": 0.35,
                    "reranker": None,
                    "retrieval_strategy": "dense",
                    "reranker_multiplier": 4,
                }
            },
            selection_description="Research papers and theory.",
        )
        registry = {
            "chat": TaskConfig(
                task="chat",
                routing_description="General chat about ML research.",
                adapter=AdapterConfig(name="", alias="", enabled=False),
                knowledge_bases=[arxiv],
            )
        }

        with registry_override(registry):
            assert get_kb_names() == ["arxiv"]
            assert get_kb_config("arxiv") is arxiv

        assert get_kb_names() != ["arxiv"]

    def test_get_knowledge_bases_reloads_when_settings_path_changes(self, tmp_path: Path):
        from shared.config import RegistryConfig
        from shared.operator_registry import get_kb_names, get_knowledge_bases

        first = write_chat_only_operator_registry(tmp_path / "kb-first.toml")

        second = write_code_only_operator_registry(tmp_path / "kb-second.toml")

        first_registry = get_knowledge_bases(
            settings=RegistryConfig(operator_registry_path=str(first))
        )
        second_registry = get_knowledge_bases(
            settings=RegistryConfig(operator_registry_path=str(second))
        )

        assert list(first_registry) == ["chat"]
        assert list(second_registry) == ["code"]
        assert get_kb_names(
            settings=RegistryConfig(operator_registry_path=str(second))
        ) == [
            "pytorch_docs"
        ]

    def test_load_knowledge_bases_from_normalized_toml(self, tmp_path: Path):
        from shared.operator_registry import load_knowledge_bases

        path = tmp_path / "registry.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 2",
                    "",
                    "[tasks.chat]",
                    'label = "General knowledge"',
                    'routing_description = "General chat about ML research."',
                    'kb_refs = ["arxiv"]',
                    "",
                    "[tasks.chat.adapter]",
                    "enabled = false",
                    "",
                    "[tasks.code]",
                    'routing_description = "Programming help for ML systems."',
                    'kb_refs = ["arxiv"]',
                    "",
                    "[tasks.code.adapter]",
                    "enabled = false",
                    "",
                    "[knowledge_bases.arxiv]",
                    'default_alias = "champion"',
                    'selection_description = "Research papers and theory."',
                    "",
                    "[knowledge_bases.arxiv.aliases.champion]",
                    'profile = "champion"',
                    "",
                    "[knowledge_bases.arxiv.aliases.challenger]",
                    'profile = "challenger"',
                    "",
                    "[alias_profiles.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 4",
                    "",
                    "[alias_profiles.challenger]",
                    "top_k = 5",
                    "score_threshold = 0.01",
                    'retrieval_strategy = "hybrid"',
                    'reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"',
                    "reranker_multiplier = 4",
                ]
            ),
            encoding="utf-8",
        )

        registry, index = load_knowledge_bases(path)

        assert list(registry) == ["chat", "code"]
        assert registry["chat"].knowledge_bases[0].name == "arxiv"
        assert registry["code"].knowledge_bases[0].name == "arxiv"
        assert (
            index["arxiv"].aliases["challenger"].reranker
            == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


# ---------------------------------------------------------------------------
# validate_kb_alias() error messages
# ---------------------------------------------------------------------------


class TestValidateKbAlias:
    @pytest.fixture()
    def _loaded_registry(self, tmp_path: Path):
        from shared.operator_registry import load_knowledge_bases, registry_override

        path = write_chat_only_operator_registry(tmp_path / "kb.toml")
        registry, index = load_knowledge_bases(path)
        with registry_override(registry, index=index):
            yield

    def test_unknown_kb_raises_valueerror(self, _loaded_registry):
        from shared.operator_registry import validate_kb_alias

        with pytest.raises(ValueError, match="not found"):
            validate_kb_alias("nonexistent", "champion")

    def test_unknown_alias_raises_valueerror(self, _loaded_registry):
        from shared.operator_registry import validate_kb_alias

        with pytest.raises(ValueError, match="not valid"):
            validate_kb_alias("arxiv", "nonexistent")

    def test_valid_kb_and_alias_passes(self, _loaded_registry):
        from shared.operator_registry import validate_kb_alias

        validate_kb_alias("arxiv", "champion")  # no exception

    def test_kb_only_validation(self, _loaded_registry):
        from shared.operator_registry import validate_kb_alias

        validate_kb_alias("arxiv")  # alias=None is fine


# ---------------------------------------------------------------------------
# Query/build compatibility validation
# ---------------------------------------------------------------------------


class TestQueryBuildCompatibility:
    def _build_config(self, *, retrieval_capability: str, sparse_encoder: str | None):
        from rag.ops.meta import BuildConfig

        return BuildConfig(
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            sparse_encoder=sparse_encoder,
            retrieval_capability=retrieval_capability,
        )

    def test_dense_query_accepts_hybrid_build(self):
        from rag.ops.meta import validate_query_compatibility

        validate_query_compatibility(
            query_strategy="dense",
            build_config=self._build_config(
                retrieval_capability="hybrid",
                sparse_encoder="Qdrant/bm25",
            ),
            context="arxiv_champion",
        )

    def test_dense_query_rejects_sparse_only_build(self):
        from rag.ops.meta import validate_query_compatibility

        with pytest.raises(ValueError, match="query strategy 'dense' requires a dense leg"):
            validate_query_compatibility(
                query_strategy="dense",
                build_config=self._build_config(
                    retrieval_capability="sparse",
                    sparse_encoder="Qdrant/bm25",
                ),
                context="arxiv_champion",
            )

    def test_hybrid_query_rejects_dense_build(self):
        from rag.ops.meta import validate_query_compatibility

        with pytest.raises(ValueError, match="requires build capability 'hybrid'"):
            validate_query_compatibility(
                query_strategy="hybrid",
                build_config=self._build_config(
                    retrieval_capability="dense",
                    sparse_encoder=None,
                ),
                runtime_sparse_encoder="Qdrant/bm25",
                context="arxiv_champion",
            )

    def test_sparse_query_requires_matching_sparse_encoder(self):
        from rag.ops.meta import validate_query_compatibility

        with pytest.raises(ValueError, match="does not match build sparse encoder"):
            validate_query_compatibility(
                query_strategy="sparse",
                build_config=self._build_config(
                    retrieval_capability="sparse",
                    sparse_encoder="Qdrant/bm25",
                ),
                runtime_sparse_encoder="other/model",
                context="arxiv_champion",
            )

    def test_sparse_query_accepts_sparse_build_with_matching_encoder(self):
        from rag.ops.meta import validate_query_compatibility

        validate_query_compatibility(
            query_strategy="sparse",
            build_config=self._build_config(
                retrieval_capability="sparse",
                sparse_encoder="Qdrant/bm25",
            ),
            runtime_sparse_encoder="Qdrant/bm25",
            context="arxiv_champion",
        )
