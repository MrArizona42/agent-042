"""Tests for catalog schema_version 4: nested alias build/retrieve blocks.

Covers schema_version rejection and the retrieval-strategy/encoder
compatibility rule from the declarative alias workflow plan's Phase 1
acceptance criteria.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _build(**overrides) -> dict:
    defaults = {
        "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        "dense_encoder": {"model": "minilm", "dimension": 384},
    }
    defaults.update(overrides)
    return defaults


def _retrieve(**overrides) -> dict:
    defaults = {"strategy": "dense", "top_k": 5, "score_threshold": 0.35}
    defaults.update(overrides)
    return defaults


class TestSchemaVersionRejection:
    def test_version_3_is_rejected(self):
        from app_config.catalog.schema import CatalogConfig

        with pytest.raises(ValidationError, match="schema_version 3 is not supported"):
            CatalogConfig(schema_version=3)

    def test_version_4_is_accepted(self):
        from app_config.catalog.schema import CatalogConfig

        cfg = CatalogConfig(schema_version=4)
        assert cfg.schema_version == 4

    def test_default_schema_version_is_4(self):
        from app_config.catalog.schema import CatalogConfig

        cfg = CatalogConfig()
        assert cfg.schema_version == 4

    def test_unsupported_future_version_is_rejected(self):
        from app_config.catalog.schema import CatalogConfig

        with pytest.raises(ValidationError, match="schema_version 5 is not supported"):
            CatalogConfig(schema_version=5)


class TestCatalogAliasConfigCompatibility:
    def test_dense_alias_without_sparse_encoder_is_valid(self):
        from app_config.catalog.schema import CatalogAliasConfig

        alias = CatalogAliasConfig(build=_build(), retrieve=_retrieve(strategy="dense"))
        assert alias.build.sparse_encoder is None

    def test_sparse_alias_without_sparse_encoder_is_rejected(self):
        from app_config.catalog.schema import CatalogAliasConfig

        with pytest.raises(ValidationError, match="sparse.*requires build.sparse_encoder"):
            CatalogAliasConfig(build=_build(), retrieve=_retrieve(strategy="sparse"))

    def test_hybrid_alias_without_sparse_encoder_is_rejected(self):
        from app_config.catalog.schema import CatalogAliasConfig

        with pytest.raises(ValidationError, match="hybrid.*requires build.sparse_encoder"):
            CatalogAliasConfig(build=_build(), retrieve=_retrieve(strategy="hybrid"))

    def test_sparse_alias_with_sparse_encoder_is_valid(self):
        from app_config.catalog.schema import CatalogAliasConfig

        alias = CatalogAliasConfig(
            build=_build(sparse_encoder={"model": "bm25"}),
            retrieve=_retrieve(strategy="sparse"),
        )
        assert alias.retrieve.strategy == "sparse"

    def test_hybrid_alias_with_sparse_encoder_is_valid(self):
        from app_config.catalog.schema import CatalogAliasConfig

        alias = CatalogAliasConfig(
            build=_build(sparse_encoder={"model": "bm25"}),
            retrieve=_retrieve(
                strategy="hybrid",
                reranker="cross-encoder/x",
                reranker_multiplier=4,
            ),
        )
        assert alias.build.sparse_encoder.model == "bm25"

    def test_unknown_top_level_field_is_rejected(self):
        from app_config.catalog.schema import CatalogAliasConfig

        with pytest.raises(ValidationError):
            CatalogAliasConfig(build=_build(), retrieve=_retrieve(), extra_field="x")


class TestCatalogToml:
    def test_repo_catalog_toml_loads_under_schema_v4(self):
        from app_config.catalog import load_catalog

        task_catalog, kb_index = load_catalog("catalog.toml")

        assert "pytorch_reference" in kb_index
        challenger = kb_index["pytorch_reference"].aliases["challenger"]
        assert challenger.retrieval_strategy == "hybrid"
