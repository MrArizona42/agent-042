"""Helpers for writing small operator-registry TOML fixtures in tests."""

from __future__ import annotations

from pathlib import Path


def _write_registry(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_chat_and_code_operator_registry(path: Path) -> Path:
    return _write_registry(
        path,
        [
            "schema_version = 2",
            "",
            "[tasks.chat]",
            'label = "General knowledge"',
            'routing_description = "General ML research discussion."',
            'kb_refs = ["arxiv"]',
            "",
            "[tasks.chat.adapter]",
            "enabled = false",
            "",
            "[tasks.code]",
            'label = "Coding assistance"',
            'routing_description = "Programming help for ML systems."',
            'kb_refs = ["pytorch_docs"]',
            "",
            "[tasks.code.adapter]",
            "enabled = false",
            "",
            "[knowledge_bases.arxiv]",
            'default_alias = "champion"',
            'update_strategy = "incremental"',
            'label = "ArXiv papers"',
            'description = "ML papers"',
            'selection_description = "Research papers and literature-grounded answers."',
            "",
            "[knowledge_bases.arxiv.aliases.champion]",
            'profile = "champion"',
            "",
            "[knowledge_bases.arxiv.aliases.challenger]",
            'profile = "challenger"',
            "",
            "[knowledge_bases.pytorch_docs]",
            'default_alias = "champion"',
            'update_strategy = "replace"',
            'label = "PyTorch docs"',
            'description = "Coding docs"',
            'selection_description = "PyTorch API reference and implementation guidance."',
            "",
            "[knowledge_bases.pytorch_docs.aliases.champion]",
            'profile = "champion"',
            "",
            "[alias_profiles.champion]",
            "top_k = 5",
            "score_threshold = 0.35",
            'retrieval_strategy = "dense"',
            "reranker_multiplier = 4",
            "",
            "[alias_profiles.challenger]",
            "top_k = 5",
            "score_threshold = 0.35",
            'retrieval_strategy = "dense"',
            "reranker_multiplier = 4",
        ],
    )


def write_chat_only_operator_registry(
    path: Path,
    *,
    retrieval_strategy: str = "dense",
) -> Path:
    return _write_registry(
        path,
        [
            "schema_version = 2",
            "",
            "[tasks.chat]",
            'label = "General knowledge"',
            'routing_description = "General ML research discussion."',
            'kb_refs = ["arxiv"]',
            "",
            "[tasks.chat.adapter]",
            "enabled = false",
            "",
            "[knowledge_bases.arxiv]",
            'default_alias = "champion"',
            'update_strategy = "incremental"',
            'label = "ArXiv papers"',
            'description = "ML papers"',
            'selection_description = "Research papers and literature-grounded answers."',
            "",
            "[knowledge_bases.arxiv.aliases.champion]",
            'profile = "champion"',
            "",
            "[alias_profiles.champion]",
            "top_k = 5",
            "score_threshold = 0.35",
            f'retrieval_strategy = "{retrieval_strategy}"',
            "reranker_multiplier = 4",
        ],
    )


def write_code_only_operator_registry(path: Path) -> Path:
    return _write_registry(
        path,
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
        ],
    )