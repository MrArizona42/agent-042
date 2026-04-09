"""Tests for config contract validation — Phase 5.

Covers AliasConfig completeness, KBConfig.default_alias pointing to a
declared alias, duplicate KB names across tasks, and validate_kb_alias()
error messages.
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
        )
        assert cfg.top_k == 5
        assert cfg.reranker is None


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
                    },
                },
                update_strategy="replace",
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
                },
            },
            update_strategy="replace",
        )
        assert cfg.default_alias == "champion"


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
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                            },
                        },
                    },
                ],
            },
            {
                "task": "code",
                "label": "Code",
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                            },
                        },
                    },
                ],
            },
        ]
        path = tmp_path / "dup.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Duplicate KB name"):
            _load_knowledge_bases(path)


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
                "knowledge_bases": [
                    {
                        "name": "arxiv",
                        "default_alias": "champion",
                        "aliases": {
                            "champion": {
                                "top_k": 5,
                                "score_threshold": 0.35,
                                "reranker": None,
                            },
                        },
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
