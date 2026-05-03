from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient as RealQdrantClient

from rag.ops.materialize import create_collection_with_meta
from rag.ops.meta import BuildConfig, CollectionMeta


class TestLoadUpdateCollectionMeta:
    def test_loads_meta_from_named_vector_collection(self):
        from rag.ops.update.common import load_update_collection_meta

        in_memory = RealQdrantClient(":memory:")
        meta = CollectionMeta(
            kb_name="arxiv",
            build_config=BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                sparse_encoder=None,
                retrieval_capability="dense",
            ),
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc).isoformat(),
        )

        with patch("rag.vector_store.QdrantClient", return_value=in_memory):
            vector_store = create_collection_with_meta(
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="arxiv_20260402_074822",
                dimension=4,
                meta=meta,
            )

            loaded_meta = load_update_collection_meta(
                vector_store=vector_store,
                alias_name="arxiv_champion",
                collection_name="arxiv_20260402_074822",
                kb_name="arxiv",
            )

        assert loaded_meta == meta

    def test_wraps_legacy_build_config_errors_with_rebuild_guidance(self):
        from rag.ops.update.common import load_update_collection_meta

        vector_store = MagicMock()

        with patch(
            "rag.ops.update.common.read_collection_meta",
            side_effect=ValueError(
                "arxiv_20260402_074822.build_config: 'retrieval_capability' must be one of "
                "'dense', 'hybrid', 'sparse' (got None)"
            ),
        ):
            with pytest.raises(RuntimeError, match="cannot be refreshed in place"):
                load_update_collection_meta(
                    vector_store=vector_store,
                    alias_name="arxiv_champion",
                    collection_name="arxiv_20260402_074822",
                    kb_name="arxiv",
                )

    def test_wraps_missing_meta_errors_with_rebuild_guidance(self):
        from rag.ops.update.common import load_update_collection_meta

        vector_store = MagicMock()

        with patch(
            "rag.ops.update.common.read_collection_meta",
            side_effect=RuntimeError("Missing _meta for 'arxiv_20260402_074822'"),
        ):
            with pytest.raises(RuntimeError, match="without valid _meta"):
                load_update_collection_meta(
                    vector_store=vector_store,
                    alias_name="arxiv_champion",
                    collection_name="arxiv_20260402_074822",
                    kb_name="arxiv",
                )
