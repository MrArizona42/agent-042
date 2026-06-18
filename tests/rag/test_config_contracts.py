"""Tests for config contract validation.

Covers AliasConfig completeness, AdapterConfig validation, KB / task catalog
requirements, catalog reference validation, and validate_kb_alias() error
messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.catalog_samples import (
    write_chat_only_catalog,
    write_code_only_catalog,
)


@pytest.fixture(autouse=True)
def _reset_kb_catalog():
    import app_config.runtime as cfg

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

        from app_config.catalog import AliasConfig

        with pytest.raises(ValidationError, match="top_k"):
            AliasConfig(
                score_threshold=0.35,
                reranker=None,
            )

    def test_missing_score_threshold_raises(self):
        from pydantic import ValidationError

        from app_config.catalog import AliasConfig

        with pytest.raises(ValidationError, match="score_threshold"):
            AliasConfig(top_k=5, reranker=None)

    def test_missing_reranker_defaults_to_off(self):
        from app_config.catalog import AliasConfig

        cfg = AliasConfig(
            top_k=5,
            score_threshold=0.35,
            retrieval_strategy="dense",
            reranker_multiplier=1,
        )

        assert cfg.reranker is None

    def test_complete_alias_config_ok(self):
        from app_config.catalog import AliasConfig

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
        from app_config.catalog import AliasConfig

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
        from app_config.catalog import AdapterConfig

        cfg = AdapterConfig(name="", alias="", enabled=False)

        assert cfg.enabled is False
        assert cfg.name == ""
        assert cfg.alias == ""

    def test_enabled_adapter_requires_name(self):
        from pydantic import ValidationError

        from app_config.catalog import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="", alias="champion", enabled=True)

    def test_enabled_adapter_requires_alias(self):
        from pydantic import ValidationError

        from app_config.catalog import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="lora-chat", alias="", enabled=True)


# ---------------------------------------------------------------------------
# KBConfig.default_alias must point to declared alias
# ---------------------------------------------------------------------------


class TestKBConfigDefaultAlias:
    def test_default_alias_must_be_declared(self):
        from pydantic import ValidationError

        from app_config.catalog import KBConfig

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
                description="Selection text",
            )

    def test_valid_default_alias_ok(self):
        from app_config.catalog import KBConfig

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
            description="Selection text",
        )
        assert cfg.default_alias == "champion"
        assert cfg.selection_description == "Selection text"
        assert cfg.label == "Selection text"

    def test_description_is_required(self):
        from pydantic import ValidationError

        from app_config.catalog import KBConfig

        with pytest.raises(ValidationError, match="description"):
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
        from app_config.catalog import TaskConfig

        cfg = TaskConfig(
            task="summarize",
            description="Summarize user-provided content.",
            adapter={"name": "", "alias": "", "enabled": False},
            knowledge_bases=[],
        )

        assert cfg.task == "summarize"
        assert cfg.knowledge_bases == []
        assert cfg.routing_description == "Summarize user-provided content."
        assert cfg.label == "Summarize user-provided content."

    def test_task_config_requires_description(self):
        from pydantic import ValidationError

        from app_config.catalog import TaskConfig

        with pytest.raises(ValidationError, match="description"):
            TaskConfig(
                task="chat",
                knowledge_bases=[],
            )


# ---------------------------------------------------------------------------
# Duplicate KB names across tasks rejected
# ---------------------------------------------------------------------------


class TestRegistryReferenceValidation:
    def test_unknown_kb_ref_is_rejected(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        path = tmp_path / "invalid.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 3",
                    "",
                    "[[tasks]]",
                    'id = "chat"',
                    'description = "General chat about ML research."',
                    'knowledge_bases = ["missing_kb"]',
                    "lora_adapter = { enabled = false }",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="references unknown KB 'missing_kb'"):
            load_catalog(path)

    def test_source_instance_unknown_kb_is_rejected(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        path = tmp_path / "invalid.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 3",
                    "",
                    "[[source_adapters]]",
                    'id = "generic.http_html"',
                    'version = "1"',
                    'description = "d"',
                    'factory = "rag.adapters.sources:make_http_html_adapter"',
                    "",
                    "[[source_instances]]",
                    'id = "missing_kb.docs"',
                    'description = "d"',
                    'role = "corpus"',
                    'knowledge_base = "missing_kb"',
                    'adapter = { id = "generic.http_html", version = "1" }',
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="references unknown KB 'missing_kb'"):
            load_catalog(path)

    def test_source_instance_id_is_unique_within_kb(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        path = tmp_path / "invalid.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 3",
                    "",
                    "[[knowledge_bases]]",
                    'id = "pytorch_reference"',
                    'description = "PyTorch API reference."',
                    'default_alias = "champion"',
                    "",
                    "[knowledge_bases.aliases.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 1",
                    "",
                    "[[source_adapters]]",
                    'id = "generic.http_html"',
                    'version = "1"',
                    'description = "d"',
                    'factory = "rag.adapters.sources:make_http_html_adapter"',
                    "",
                    "[[source_instances]]",
                    'id = "pytorch_reference.docs"',
                    'description = "d"',
                    'role = "corpus"',
                    'knowledge_base = "pytorch_reference"',
                    'adapter = { id = "generic.http_html", version = "1" }',
                    "",
                    "[[source_instances]]",
                    'id = "pytorch_reference.docs"',
                    'description = "d"',
                    'role = "corpus"',
                    'knowledge_base = "pytorch_reference"',
                    'adapter = { id = "generic.http_html", version = "1" }',
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError, match="Duplicate source instance id 'pytorch_reference.docs'"
        ):
            load_catalog(path)


class TestKnowledgeBaseRegistryResolution:
    def test_gateway_settings_expose_grouped_sections(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.setenv("PLATFORM__VLLM_BASE_URL", "http://platform-vllm:8000")

        settings = load_settings({"rag": {"enabled": False}})

        assert settings.platform.vllm_base_url == "http://vllm:8000"
        assert settings.network.internal_url("vllm") == "http://vllm:8000"
        assert settings.network.host_url("vllm") == "http://localhost:8000"
        assert settings.rag.enabled is False
        assert settings.vllm.model == "/models/Qwen/Qwen3-0.6B"
        assert settings.gateway.service_name == "agent-042-gateway"
        assert settings.auth.session_ttl_seconds == 86400
        with pytest.raises(AttributeError):
            _ = settings.service_name
        with pytest.raises(AttributeError):
            _ = settings.rag.knowledge_bases_path

    def test_load_settings_merges_runtime_toml_with_explicit_overrides(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.setenv("GATEWAY__URL", "http://gateway-from-env:9001")
        monkeypatch.setenv("GATEWAY__BUDGET__MODEL_MAX_TOKENS", "4096")

        settings = load_settings(
            {
                "vllm": {"model": "override-model"},
                "gateway": {
                    "budget": {"min_response_budget": 1024},
                },
            }
        )

        assert settings.vllm.model == "override-model"
        assert settings.gateway.url == "http://gateway:9000"
        assert settings.gateway.budget.model_max_tokens == 32768
        assert settings.gateway.budget.min_response_budget == 1024

    def test_runtime_env_names_do_not_override_toml_values(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.setenv("GATEWAY__CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
        monkeypatch.setenv("ADAPTER_REGISTRY__SYNC_ALIASES", "champion,shadow")

        settings = load_settings()

        assert settings.gateway.cors_allow_origins == ("*",)
        assert settings.adapter_registry.sync_aliases == ("champion", "challenger")

    def test_runtime_path_is_required(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.delenv("CONFIG__RUNTIME_PATH", raising=False)

        with pytest.raises(RuntimeError, match="CONFIG__RUNTIME_PATH"):
            load_settings()

    def test_missing_runtime_toml_field_is_validation_error(self, tmp_path: Path):
        from pydantic import ValidationError

        from app_config.runtime import load_settings

        path = tmp_path / "runtime.toml"
        path.write_text("schema_version = 1\n", encoding="utf-8")

        with pytest.raises(ValidationError, match="gateway"):
            load_settings(runtime_path=path)

    def test_runtime_toml_rejects_vllm_launch_settings(self, tmp_path: Path):
        from pydantic import ValidationError

        from app_config.runtime import load_settings

        path = tmp_path / "runtime.toml"
        path.write_text(
            Path("runtime.toml").read_text(encoding="utf-8")
            + '\n[vllm]\nmodel = "/models/example"\n',
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="vllm.model"):
            load_settings(runtime_path=path)

    def test_runtime_toml_rejects_derived_or_env_only_keys(self, tmp_path: Path):
        from pydantic import ValidationError

        from app_config.runtime import load_settings

        path = tmp_path / "runtime.toml"
        runtime_toml = Path("runtime.toml").read_text(encoding="utf-8")
        runtime_toml = runtime_toml.replace(
            "[gateway]\n",
            '[gateway]\nurl = "http://gateway:9000"\n',
            1,
        )
        path.write_text(runtime_toml, encoding="utf-8")

        with pytest.raises(ValidationError, match="gateway.url"):
            load_settings(runtime_path=path)

    def test_config_catalog_path_env_sets_catalog_settings(self, tmp_path: Path, monkeypatch):
        from app_config.runtime import load_settings

        catalog_path = tmp_path / "catalog.toml"
        monkeypatch.setenv("CONFIG__CATALOG_PATH", str(catalog_path))

        settings = load_settings()

        assert settings.catalog.path == catalog_path

    def test_config_catalog_path_env_is_required(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.delenv("CONFIG__CATALOG_PATH", raising=False)

        with pytest.raises(RuntimeError, match="CONFIG__CATALOG_PATH"):
            load_settings()

    def test_legacy_flat_env_names_are_ignored(self, monkeypatch):
        from app_config.runtime import load_settings

        monkeypatch.delenv("PLATFORM__VLLM_BASE_URL", raising=False)
        monkeypatch.setenv("VLLM_BASE_URL", "http://legacy-vllm:8000")

        settings = load_settings()

        assert settings.platform.vllm_base_url == "http://vllm:8000"

    def test_catalog_settings_own_catalog_path(self):
        from app_config.catalog import resolve_catalog_path
        from app_config.runtime import CatalogConfig

        settings = CatalogConfig(path="configs/catalog.toml")

        assert settings.path == Path("configs/catalog.toml")
        assert resolve_catalog_path(settings) == Path.cwd() / "configs/catalog.toml"

    def test_get_catalog_prefers_catalog_settings_path(self, tmp_path: Path, monkeypatch):
        import app_config.runtime as cfg
        from app_config.catalog import get_catalog, get_kb_names

        path = write_code_only_catalog(tmp_path / "catalog.toml")

        monkeypatch.setenv("CONFIG__CATALOG_PATH", str(path))
        cfg.clear_knowledge_base_caches()

        catalog = get_catalog()

        assert list(catalog) == ["code"]
        assert get_kb_names() == ["pytorch_reference"]

    def test_clear_knowledge_base_caches_refreshes_registry_settings_path(
        self, tmp_path: Path, monkeypatch
    ):
        import app_config.runtime as cfg
        from app_config.catalog import get_kb_names

        first = write_chat_only_catalog(tmp_path / "catalog-first.toml")
        second = write_code_only_catalog(tmp_path / "catalog-second.toml")

        monkeypatch.setenv("CONFIG__CATALOG_PATH", str(first))
        cfg.clear_knowledge_base_caches()
        assert get_kb_names() == ["ml_papers_core"]

        monkeypatch.setenv("CONFIG__CATALOG_PATH", str(second))
        cfg.clear_knowledge_base_caches()
        assert get_kb_names() == ["pytorch_reference"]

    def test_legacy_catalog_env_names_are_ignored(self, tmp_path: Path, monkeypatch):
        from app_config.runtime import load_settings

        path = write_chat_only_catalog(tmp_path / "catalog.toml")

        monkeypatch.setenv("CATALOG_PATH", str(path))
        monkeypatch.setenv("CATALOG__PATH", str(path))
        settings = load_settings()

        assert settings.catalog.path != path

    def test_in_memory_catalog_override_bypasses_disk_loading(self):
        from app_config.catalog import (
            AdapterConfig,
            KBConfig,
            TaskConfig,
            catalog_override,
            get_kb_config,
            get_kb_names,
        )

        ml_papers_core = KBConfig(
            name="ml_papers_core",
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
            description="Research papers and theory.",
        )
        catalog = {
            "chat": TaskConfig(
                task="chat",
                description="General chat about ML research.",
                adapter=AdapterConfig(name="", alias="", enabled=False),
                knowledge_bases=[ml_papers_core],
            )
        }

        with catalog_override(catalog):
            assert get_kb_names() == ["ml_papers_core"]
            assert get_kb_config("ml_papers_core") is ml_papers_core

        assert get_kb_names() != ["ml_papers_core"]

    def test_get_catalog_reloads_when_settings_path_changes(self, tmp_path: Path):
        from app_config.catalog import get_catalog, get_kb_names
        from app_config.runtime import CatalogConfig

        first = write_chat_only_catalog(tmp_path / "kb-first.toml")

        second = write_code_only_catalog(tmp_path / "kb-second.toml")

        first_registry = get_catalog(settings=CatalogConfig(path=str(first)))
        second_registry = get_catalog(settings=CatalogConfig(path=str(second)))

        assert list(first_registry) == ["chat"]
        assert list(second_registry) == ["code"]
        assert get_kb_names(settings=CatalogConfig(path=str(second))) == ["pytorch_reference"]

    def test_load_catalog_from_canonical_toml(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        path = tmp_path / "catalog.toml"
        path.write_text(
            "\n".join(
                [
                    "schema_version = 3",
                    "",
                    "[[tasks]]",
                    'id = "chat"',
                    'description = "General chat about ML research."',
                    'knowledge_bases = ["ml_papers_core"]',
                    "lora_adapter = { enabled = false }",
                    "",
                    "[[tasks]]",
                    'id = "code"',
                    'description = "Programming help for ML systems."',
                    'knowledge_bases = ["ml_papers_core"]',
                    "lora_adapter = { enabled = false }",
                    "",
                    "[[knowledge_bases]]",
                    'id = "ml_papers_core"',
                    'description = "Research papers and theory."',
                    'default_alias = "champion"',
                    "",
                    "[knowledge_bases.aliases.champion]",
                    "top_k = 5",
                    "score_threshold = 0.35",
                    'retrieval_strategy = "dense"',
                    "reranker_multiplier = 1",
                    "",
                    "[knowledge_bases.aliases.challenger]",
                    "top_k = 5",
                    "score_threshold = 0.01",
                    'retrieval_strategy = "hybrid"',
                    'reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"',
                    "reranker_multiplier = 4",
                    "",
                    "[[source_adapters]]",
                    'id = "generic.arxiv_paper"',
                    'version = "1"',
                    'description = "d"',
                    'factory = "rag.adapters.sources:make_arxiv_paper_adapter"',
                    "",
                    "[[source_instances]]",
                    'id = "ml_papers_core.papers"',
                    'description = "d"',
                    'role = "corpus"',
                    'knowledge_base = "ml_papers_core"',
                    'adapter = { id = "generic.arxiv_paper", version = "1" }',
                ]
            ),
            encoding="utf-8",
        )

        catalog, index = load_catalog(path)

        assert list(catalog) == ["chat", "code"]
        assert catalog["chat"].knowledge_bases[0].name == "ml_papers_core"
        assert catalog["code"].knowledge_bases[0].name == "ml_papers_core"
        assert not hasattr(index["ml_papers_core"], "source_ref")
        assert (
            index["ml_papers_core"].aliases["challenger"].reranker
            == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


# ---------------------------------------------------------------------------
# validate_kb_alias() error messages
# ---------------------------------------------------------------------------


class TestValidateKbAlias:
    @pytest.fixture()
    def _loaded_registry(self, tmp_path: Path):
        from app_config.catalog import catalog_override, load_catalog

        path = write_chat_only_catalog(tmp_path / "kb.toml")
        catalog, index = load_catalog(path)
        with catalog_override(catalog, index=index):
            yield

    def test_unknown_kb_raises_valueerror(self, _loaded_registry):
        from app_config.catalog import validate_kb_alias

        with pytest.raises(ValueError, match="not found"):
            validate_kb_alias("nonexistent", "champion")

    def test_unknown_alias_raises_valueerror(self, _loaded_registry):
        from app_config.catalog import validate_kb_alias

        with pytest.raises(ValueError, match="not valid"):
            validate_kb_alias("ml_papers_core", "nonexistent")

    def test_valid_kb_and_alias_passes(self, _loaded_registry):
        from app_config.catalog import validate_kb_alias

        validate_kb_alias("ml_papers_core", "champion")  # no exception

    def test_kb_only_validation(self, _loaded_registry):
        from app_config.catalog import validate_kb_alias

        validate_kb_alias("ml_papers_core")  # alias=None is fine


# ---------------------------------------------------------------------------
# Query/build compatibility validation
# ---------------------------------------------------------------------------


class TestQueryBuildCompatibility:
    def test_dense_query_accepts_dense_collection(self):
        from rag.indexing.materialize import validate_strategy_supported

        validate_strategy_supported(
            retrieval_strategy="dense",
            retrieval_capability="dense",
        )

    def test_dense_query_accepts_hybrid_collection(self):
        from rag.indexing.materialize import validate_strategy_supported

        validate_strategy_supported(
            retrieval_strategy="dense",
            retrieval_capability="hybrid",
        )

    def test_hybrid_query_accepts_hybrid_collection(self):
        from rag.indexing.materialize import validate_strategy_supported

        validate_strategy_supported(
            retrieval_strategy="hybrid",
            retrieval_capability="hybrid",
        )

    def test_hybrid_query_rejects_dense_collection(self):
        from rag.indexing.materialize import validate_strategy_supported

        with pytest.raises(ValueError, match="not supported"):
            validate_strategy_supported(
                retrieval_strategy="hybrid",
                retrieval_capability="dense",
            )
