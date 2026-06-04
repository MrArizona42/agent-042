from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.config import Settings
from tests.catalog_samples import write_chat_and_code_catalog


@pytest.fixture(autouse=True)
def _reset_kb_catalog():
    import shared.config as cfg

    cfg.clear_knowledge_base_caches()
    yield
    cfg.clear_knowledge_base_caches()


@pytest.fixture()
def catalog_file(tmp_path: Path) -> Path:
    return write_chat_and_code_catalog(tmp_path / "catalog.toml")


@pytest.fixture()
def loaded_kb_catalog(catalog_file: Path):
    from shared.catalog import catalog_override, load_catalog

    catalog, index = load_catalog(catalog_file)
    with catalog_override(catalog, index=index):
        yield catalog


def _collection_meta_payload(kb_name: str = "ml_papers_core") -> dict[str, object]:
    return {
        "kb_name": kb_name,
        "created_at": "2026-04-01T12:00:00+00:00",
        "build_config": {
            "chunking_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "sparse_encoder": None,
            "retrieval_capability": "dense",
        },
    }


def _gateway_settings() -> Settings:
    return Settings(platform={"qdrant_host": "qdrant", "qdrant_port": 6333})


class TestValidateKbAlias:
    def test_rejects_unknown_knowledge_base(self, loaded_kb_catalog):
        from shared.catalog import validate_kb_alias

        with pytest.raises(ValueError, match="KB 'missing' not found"):
            validate_kb_alias("missing", "champion")

    def test_rejects_disallowed_alias(self, loaded_kb_catalog):
        from shared.catalog import validate_kb_alias

        with pytest.raises(ValueError, match="Alias 'production' not valid"):
            validate_kb_alias("ml_papers_core", "production")


class TestAssignAliasToCollection:
    def test_updates_alias_for_matching_collection(self, loaded_kb_catalog):
        from rag.ops.aliases import assign_alias_to_collection
        from rag.ops.meta import CollectionMeta

        meta = CollectionMeta.from_payload(_collection_meta_payload(), context="ml_papers_core_v1")

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta", return_value=meta),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = True

            result = assign_alias_to_collection(
                kb="ml_papers_core",
                alias="champion",
                collection_name="ml_papers_core_20260401",
            )

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="ml_papers_core_20260401",
        )
        mock_store.update_alias.assert_called_once_with(
            "ml_papers_core_champion",
            "ml_papers_core_20260401"
        )
        assert result == {
            "alias_name": "ml_papers_core_champion",
            "collection_name": "ml_papers_core_20260401",
            "meta": _collection_meta_payload(),
        }

    def test_raises_when_collection_does_not_exist(self, loaded_kb_catalog):
        from rag.ops.aliases import assign_alias_to_collection

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta") as read_meta,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = False

            with pytest.raises(RuntimeError, match="does not exist"):
                assign_alias_to_collection(
                    kb="ml_papers_core",
                    alias="champion",
                    collection_name="ml_papers_core_20260401",
                )

        read_meta.assert_not_called()

    def test_raises_when_collection_belongs_to_other_kb(self, loaded_kb_catalog):
        from rag.ops.aliases import assign_alias_to_collection
        from rag.ops.meta import CollectionMeta

        meta = CollectionMeta.from_payload(
            _collection_meta_payload(kb_name="pytorch_reference"),
            context="ml_papers_core_v1",
        )

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta", return_value=meta),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = True

            with pytest.raises(
                RuntimeError,
                match="belongs to 'pytorch_reference', not 'ml_papers_core'"
            ):
                assign_alias_to_collection(
                    kb="ml_papers_core",
                    alias="champion",
                    collection_name="ml_papers_core_20260401",
                )

        mock_store.update_alias.assert_not_called()


class TestPromoteAlias:
    def test_repoints_target_alias_to_source_collection(self, loaded_kb_catalog):
        from rag.ops.aliases import promote_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch(
                "rag.ops.aliases.assign_alias_to_collection",
                return_value={
                    "alias_name": "ml_papers_core_champion",
                    "collection_name": "ml_papers_core_20260401",
                    "meta": _collection_meta_payload(),
                },
            ) as assign_alias,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.resolve_alias.return_value = "ml_papers_core_20260401"

            result = promote_alias(
                kb="ml_papers_core",
                from_alias="challenger",
                to_alias="champion"
            )

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="ml_papers_core_challenger",
        )
        mock_store.resolve_alias.assert_called_once_with("ml_papers_core_challenger")
        assign_alias.assert_called_once_with(
            kb="ml_papers_core",
            alias="champion",
            collection_name="ml_papers_core_20260401",
            qdrant_host="qdrant",
            qdrant_port=6333,
        )
        assert result["source_alias_name"] == "ml_papers_core_challenger"

    def test_raises_when_source_alias_is_unresolved(self, loaded_kb_catalog):
        from rag.ops.aliases import promote_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.assign_alias_to_collection") as assign_alias,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.resolve_alias.return_value = None

            with pytest.raises(RuntimeError, match="does not resolve"):
                promote_alias(kb="ml_papers_core", from_alias="challenger", to_alias="champion")

        assign_alias.assert_not_called()


class TestDetachAlias:
    def test_deletes_alias_mapping(self, loaded_kb_catalog):
        from rag.ops.aliases import detach_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=_gateway_settings(),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
        ):
            mock_store = mock_store_cls.return_value

            result = detach_alias(kb="ml_papers_core", alias="champion")

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="ml_papers_core_champion",
        )
        mock_store.delete_alias.assert_called_once_with("ml_papers_core_champion")
        assert result == {"alias_name": "ml_papers_core_champion"}
