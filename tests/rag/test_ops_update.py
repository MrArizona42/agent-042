from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLoadUpdateCollectionMeta:
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
