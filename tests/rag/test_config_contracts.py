"""Tests for config contract validation.

Covers AliasConfig completeness, AdapterConfig validation, KB / task registry
requirements, duplicate KB names across tasks, and validate_kb_alias() error
messages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_kb_registry():
    import shared.config as cfg

    cfg._KB_REGISTRY = None
    cfg._KB_INDEX = None
    yield
    cfg._KB_REGISTRY = None
    cfg._KB_INDEX = None


# ---------------------------------------------------------------------------
# AliasConfig validation
# ---------------------------------------------------------------------------


class TestAliasConfigValidation:
    """AliasConfig rejects incomplete JSON entries."""

    def test_missing_top_k_raises(self):
        from pydantic import ValidationError

        from shared.config import AliasConfig

        with pytest.raises(ValidationError, match="top_k"):
            AliasConfig(
                score_threshold=0.35,
                reranker=None,
            )

    def test_missing_score_threshold_raises(self):
        from pydantic import ValidationError

        from shared.config import AliasConfig

        with pytest.raises(ValidationError, match="score_threshold"):
            AliasConfig(top_k=5, reranker=None)

    def test_missing_reranker_raises(self):
        from pydantic import ValidationError

        from shared.config import AliasConfig

        with pytest.raises(ValidationError, match="reranker"):
            AliasConfig(top_k=5, score_threshold=0.35)

    def test_complete_alias_config_ok(self):
        from shared.config import AliasConfig

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
        from shared.config import AliasConfig

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
        from shared.config import AdapterConfig

        cfg = AdapterConfig(name="", alias="", enabled=False)

        assert cfg.enabled is False
        assert cfg.name == ""
        assert cfg.alias == ""

    def test_enabled_adapter_requires_name(self):
        from pydantic import ValidationError

        from shared.config import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="", alias="champion", enabled=True)

    def test_enabled_adapter_requires_alias(self):
        from pydantic import ValidationError

        from shared.config import AdapterConfig

        with pytest.raises(ValidationError, match="enabled adapter"):
            AdapterConfig(name="lora-chat", alias="", enabled=True)


# ---------------------------------------------------------------------------
# KBConfig.default_alias must point to declared alias
# ---------------------------------------------------------------------------


class TestKBConfigDefaultAlias:
    def test_default_alias_must_be_declared(self):
        from pydantic import ValidationError

        from shared.config import KBConfig

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
        from shared.config import KBConfig

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

        from shared.config import KBConfig

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
        from shared.config import TaskConfig

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

        from shared.config import TaskConfig

        with pytest.raises(ValidationError, match="routing_description"):
            TaskConfig(
                task="chat",
                label="General knowledge",
                knowledge_bases=[],
            )


# ---------------------------------------------------------------------------
# Duplicate KB names across tasks rejected
# ---------------------------------------------------------------------------


class TestDuplicateKBNames:
    def test_duplicate_kb_names_across_tasks_rejected(self, tmp_path: Path):
        from shared.config import _load_knowledge_bases

        data = [
            {
                "task": "chat",
                "label": "General",
                "routing_description": "General chat about ML research.",
                "adapter": {"name": "", "alias": "", "enabled": False},
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                                "retrieval_strategy": "dense",
                                "reranker_multiplier": 4,
                            },
                        },
                        "selection_description": "Research papers and theory.",
                    },
                ],
            },
            {
                "task": "code",
                "label": "Code",
                "routing_description": "Programming help for ML systems.",
                "adapter": {"name": "", "alias": "", "enabled": False},
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                                "retrieval_strategy": "dense",
                                "reranker_multiplier": 4,
                            },
                        },
                        "selection_description": "Research papers and theory.",
                    },
                ],
            },
        ]
        path = tmp_path / "dup.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Duplicate KB name"):
            _load_knowledge_bases(path)


class TestKnowledgeBaseRegistryResolution:
    def test_get_knowledge_bases_reloads_when_settings_path_changes(self, tmp_path: Path):
        from shared.config import Settings, get_knowledge_bases, get_kb_names

        first = tmp_path / "kb-first.json"
        first.write_text(
            json.dumps(
                [
                    {
                        "task": "chat",
                        "routing_description": "General chat about ML research.",
                        "adapter": {"name": "", "alias": "", "enabled": False},
                        "knowledge_bases": [
                            {
                                "name": "arxiv",
                                "default_alias": "champion",
                                "aliases": {
                                    "champion": {
                                        "top_k": 5,
                                        "score_threshold": 0.35,
                                        "reranker": None,
                                        "retrieval_strategy": "dense",
                                        "reranker_multiplier": 4,
                                    },
                                },
                                "selection_description": "Research papers and theory.",
                            },
                        ],
                    },
                ]
            )
        )

        second = tmp_path / "kb-second.json"
        second.write_text(
            json.dumps(
                [
                    {
                        "task": "code",
                        "routing_description": "Programming help for ML systems.",
                        "adapter": {"name": "", "alias": "", "enabled": False},
                        "knowledge_bases": [
                            {
                                "name": "pytorch_docs",
                                "default_alias": "champion",
                                "aliases": {
                                    "champion": {
                                        "top_k": 5,
                                        "score_threshold": 0.35,
                                        "reranker": None,
                                        "retrieval_strategy": "dense",
                                        "reranker_multiplier": 4,
                                    },
                                },
                                "selection_description": "PyTorch API reference.",
                            },
                        ],
                    },
                ]
            )
        )

        first_registry = get_knowledge_bases(settings=Settings(knowledge_bases_path=str(first)))
        second_registry = get_knowledge_bases(settings=Settings(knowledge_bases_path=str(second)))

        assert list(first_registry) == ["chat"]
        assert list(second_registry) == ["code"]
        assert get_kb_names(settings=Settings(knowledge_bases_path=str(second))) == [
            "pytorch_docs"
        ]


# ---------------------------------------------------------------------------
# validate_kb_alias() error messages
# ---------------------------------------------------------------------------


class TestValidateKbAlias:
    @pytest.fixture()
    def _loaded_registry(self, tmp_path: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        data = [
            {
                "task": "chat",
                "routing_description": "General chat about ML research.",
                "adapter": {"name": "", "alias": "", "enabled": False},
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                                "retrieval_strategy": "dense",
                                "reranker_multiplier": 4,
                            },
                        },
                        "selection_description": "Research papers and theory.",
                    },
                ],
            },
        ]
        path = tmp_path / "kb.json"
        path.write_text(json.dumps(data))
        cfg._KB_REGISTRY, cfg._KB_INDEX = _load_knowledge_bases(path)

    def test_unknown_kb_raises_valueerror(self, _loaded_registry):
        from shared.config import validate_kb_alias

        with pytest.raises(ValueError, match="not found"):
            validate_kb_alias("nonexistent", "champion")

    def test_unknown_alias_raises_valueerror(self, _loaded_registry):
        from shared.config import validate_kb_alias

        with pytest.raises(ValueError, match="not valid"):
            validate_kb_alias("arxiv", "nonexistent")

    def test_valid_kb_and_alias_passes(self, _loaded_registry):
        from shared.config import validate_kb_alias

        validate_kb_alias("arxiv", "champion")  # no exception

    def test_kb_only_validation(self, _loaded_registry):
        from shared.config import validate_kb_alias

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
