from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_kb_registry():
    import shared.config as cfg

    cfg._KB_REGISTRY = None
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()
    yield
    cfg._KB_REGISTRY = None
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()


@pytest.fixture()
def kb_json_file(tmp_path: Path) -> Path:
    data = [
        {
            "task": "chat",
            "label": "General knowledge",
            "knowledge_bases": [
                {
                    "name": "arxiv",
                    "aliases": ["champion", "challenger"],
                    "update_strategy": "incremental",
                    "label": "ArXiv papers",
                    "description": "ML papers",
                },
            ],
        },
        {
            "task": "code",
            "label": "Coding assistance",
            "knowledge_bases": [
                {
                    "name": "pytorch_docs",
                    "aliases": ["champion"],
                    "update_strategy": "replace",
                    "label": "PyTorch docs",
                    "description": "Coding docs",
                },
            ],
        },
    ]
    path = tmp_path / "knowledge_bases.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def loaded_kb_registry(kb_json_file: Path):
    import shared.config as cfg
    from shared.config import _load_knowledge_bases

    cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)
    return cfg._KB_REGISTRY


def _collection_meta_payload(kb_name: str = "arxiv") -> dict[str, object]:
    return {
        "kb_name": kb_name,
        "created_at": "2026-04-01T12:00:00+00:00",
        "build_config": {
            "chunking_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }


class TestValidateKbAlias:
    def test_rejects_unknown_knowledge_base(self, loaded_kb_registry):
        from rag.ops.aliases import _validate_kb_alias

        with pytest.raises(ValueError, match="Knowledge base 'missing' not found"):
            _validate_kb_alias("missing", "champion")

    def test_rejects_disallowed_alias(self, loaded_kb_registry):
        from rag.ops.aliases import _validate_kb_alias

        with pytest.raises(ValueError, match="Alias 'production' is not allowed"):
            _validate_kb_alias("arxiv", "production")


class TestAssignAliasToCollection:
    def test_updates_alias_for_matching_collection(self, loaded_kb_registry):
        from rag.ops.aliases import assign_alias_to_collection
        from rag.ops.meta import CollectionMeta

        meta = CollectionMeta.from_payload(_collection_meta_payload(), context="arxiv_v1")

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta", return_value=meta),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = True

            result = assign_alias_to_collection(
                kb="arxiv",
                alias="champion",
                collection_name="arxiv_20260401",
            )

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="arxiv_20260401",
        )
        mock_store.update_alias.assert_called_once_with("arxiv_champion", "arxiv_20260401")
        assert result == {
            "alias_name": "arxiv_champion",
            "collection_name": "arxiv_20260401",
            "meta": _collection_meta_payload(),
        }

    def test_raises_when_collection_does_not_exist(self, loaded_kb_registry):
        from rag.ops.aliases import assign_alias_to_collection

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta") as read_meta,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = False

            with pytest.raises(RuntimeError, match="does not exist"):
                assign_alias_to_collection(
                    kb="arxiv",
                    alias="champion",
                    collection_name="arxiv_20260401",
                )

        read_meta.assert_not_called()

    def test_raises_when_collection_belongs_to_other_kb(self, loaded_kb_registry):
        from rag.ops.aliases import assign_alias_to_collection
        from rag.ops.meta import CollectionMeta

        meta = CollectionMeta.from_payload(
            _collection_meta_payload(kb_name="pytorch_docs"),
            context="arxiv_v1",
        )

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.read_collection_meta", return_value=meta),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = True

            with pytest.raises(RuntimeError, match="belongs to 'pytorch_docs', not 'arxiv'"):
                assign_alias_to_collection(
                    kb="arxiv",
                    alias="champion",
                    collection_name="arxiv_20260401",
                )

        mock_store.update_alias.assert_not_called()


class TestPromoteAlias:
    def test_repoints_target_alias_to_source_collection(self, loaded_kb_registry):
        from rag.ops.aliases import promote_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch(
                "rag.ops.aliases.assign_alias_to_collection",
                return_value={
                    "alias_name": "arxiv_champion",
                    "collection_name": "arxiv_20260401",
                    "meta": _collection_meta_payload(),
                },
            ) as assign_alias,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.resolve_alias.return_value = "arxiv_20260401"

            result = promote_alias(kb="arxiv", from_alias="challenger", to_alias="champion")

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="arxiv_challenger",
        )
        mock_store.resolve_alias.assert_called_once_with("arxiv_challenger")
        assign_alias.assert_called_once_with(
            kb="arxiv",
            alias="champion",
            collection_name="arxiv_20260401",
            qdrant_host="qdrant",
            qdrant_port=6333,
        )
        assert result["source_alias_name"] == "arxiv_challenger"

    def test_raises_when_source_alias_is_unresolved(self, loaded_kb_registry):
        from rag.ops.aliases import promote_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
            patch("rag.ops.aliases.assign_alias_to_collection") as assign_alias,
        ):
            mock_store = mock_store_cls.return_value
            mock_store.resolve_alias.return_value = None

            with pytest.raises(RuntimeError, match="does not resolve"):
                promote_alias(kb="arxiv", from_alias="challenger", to_alias="champion")

        assign_alias.assert_not_called()


class TestDetachAlias:
    def test_deletes_alias_mapping(self, loaded_kb_registry):
        from rag.ops.aliases import detach_alias

        with (
            patch(
                "rag.ops.aliases.get_settings",
                return_value=SimpleNamespace(qdrant_host="qdrant", qdrant_port=6333),
            ),
            patch("rag.ops.aliases.QdrantVectorStore") as mock_store_cls,
        ):
            mock_store = mock_store_cls.return_value

            result = detach_alias(kb="arxiv", alias="champion")

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="arxiv_champion",
        )
        mock_store.delete_alias.assert_called_once_with("arxiv_champion")
        assert result == {"alias_name": "arxiv_champion"}
