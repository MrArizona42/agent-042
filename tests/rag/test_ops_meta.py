from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


def _valid_build_payload() -> dict[str, object]:
    return {
        "chunking_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sparse_encoder": None,
        "retrieval_strategy": "dense",
    }


def _valid_meta_payload() -> dict[str, object]:
    return {
        "type": "collection_meta",
        "kb_name": "arxiv",
        "created_at": "2026-04-01T12:00:00+00:00",
        "build_config": _valid_build_payload(),
        "implementation": {
            "module": "rag.ops.create.arxiv",
            "experimental": False,
            "identifier": "baseline",
            "git_sha": "abc123",
        },
    }


class TestBuildConfig:
    def test_roundtrip_from_payload(self):
        from rag.ops.meta import BuildConfig

        config = BuildConfig.from_payload(_valid_build_payload(), context="arxiv_champion")

        assert config.chunking_strategy == "recursive"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.to_payload() == _valid_build_payload()

    @pytest.mark.parametrize(
        ("payload", "error_message"),
        [
            ({**_valid_build_payload(), "chunk_size": None}, "chunk_size"),
            ({**_valid_build_payload(), "chunk_size": True}, "chunk_size"),
            (
                {**_valid_build_payload(), "chunk_overlap": -1},
                "chunk_overlap must be zero or greater",
            ),
            ({**_valid_build_payload(), "chunking_strategy": "   "}, "chunking_strategy"),
            ({**_valid_build_payload(), "embedding_model": ""}, "embedding_model"),
        ],
    )
    def test_from_payload_rejects_invalid_values(self, payload, error_message):
        from rag.ops.meta import BuildConfig

        with pytest.raises(ValueError, match=error_message):
            BuildConfig.from_payload(payload, context="collection_meta.build_config")


class TestCollectionMeta:
    def test_roundtrip_with_implementation(self):
        from rag.ops.meta import CollectionMeta

        meta = CollectionMeta.from_payload(_valid_meta_payload(), context="arxiv_v1")

        assert meta.kb_name == "arxiv"
        assert meta.created_at == "2026-04-01T12:00:00+00:00"
        assert meta.build_config.chunk_size == 512
        assert meta.implementation is not None
        assert meta.implementation.module == "rag.ops.create.arxiv"
        assert meta.to_payload() == {
            "kb_name": "arxiv",
            "created_at": "2026-04-01T12:00:00+00:00",
            "build_config": _valid_build_payload(),
            "implementation": {
                "module": "rag.ops.create.arxiv",
                "experimental": False,
                "identifier": "baseline",
                "git_sha": "abc123",
            },
        }

    def test_from_payload_requires_object_build_config(self):
        from rag.ops.meta import CollectionMeta

        payload = {**_valid_meta_payload(), "build_config": "invalid"}

        with pytest.raises(ValueError, match="build_config"):
            CollectionMeta.from_payload(payload, context="arxiv_v1")

    def test_build_collection_meta_generates_timestamp(self):
        from rag.ops.meta import BuildConfig, ImplementationInfo, build_collection_meta

        meta = build_collection_meta(
            kb_name="arxiv",
            build_config=BuildConfig.from_payload(_valid_build_payload()),
            implementation=ImplementationInfo(module="rag.ops.create.arxiv"),
        )

        assert meta.kb_name == "arxiv"
        assert meta.implementation is not None
        assert meta.implementation.module == "rag.ops.create.arxiv"
        assert datetime.fromisoformat(meta.created_at)


class TestMetadataStorageHelpers:
    def test_read_collection_meta_raises_when_missing(self):
        from rag.ops.meta import read_collection_meta

        vector_store = MagicMock()
        vector_store.collection_name = "arxiv_champion"
        vector_store.read_meta.return_value = None

        with pytest.raises(RuntimeError, match="Missing _meta"):
            read_collection_meta(vector_store)

    def test_write_collection_meta_serializes_payload(self):
        from rag.ops.meta import CollectionMeta, write_collection_meta

        vector_store = MagicMock()
        meta = CollectionMeta.from_payload(_valid_meta_payload(), context="arxiv_v1")

        write_collection_meta(vector_store, meta, dimension=384)

        vector_store.write_meta.assert_called_once_with(
            payload={
                "kb_name": "arxiv",
                "created_at": "2026-04-01T12:00:00+00:00",
                "build_config": _valid_build_payload(),
                "implementation": {
                    "module": "rag.ops.create.arxiv",
                    "experimental": False,
                    "identifier": "baseline",
                    "git_sha": "abc123",
                },
            },
            dimension=384,
        )

    def test_read_build_config_for_alias_reads_validated_meta(self):
        from rag.ops.meta import read_build_config_for_alias

        with patch("rag.ops.meta.QdrantVectorStore") as mock_store_cls:
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = True
            mock_store.collection_name = "arxiv_champion"
            mock_store.read_meta.return_value = _valid_meta_payload()

            config = read_build_config_for_alias(
                kb_name="arxiv",
                rag_alias="champion",
                qdrant_host="qdrant",
                qdrant_port=6333,
            )

        mock_store_cls.assert_called_once_with(
            host="qdrant",
            port=6333,
            collection_name="arxiv_champion",
        )
        assert config.chunking_strategy == "recursive"
        assert config.chunk_size == 512

    def test_read_build_config_for_alias_rejects_missing_alias(self):
        from rag.ops.meta import read_build_config_for_alias

        with patch("rag.ops.meta.QdrantVectorStore") as mock_store_cls:
            mock_store = mock_store_cls.return_value
            mock_store.collection_exists.return_value = False

            with pytest.raises(RuntimeError, match="does not resolve"):
                read_build_config_for_alias(
                    kb_name="arxiv",
                    rag_alias="champion",
                    qdrant_host="qdrant",
                    qdrant_port=6333,
                )
