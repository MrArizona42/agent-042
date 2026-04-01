"""Notebook-only convenience wrappers around production RAG operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.ops import (
    BuildConfig,
    ImplementationInfo,
    assign_alias_to_collection,
    create_arxiv_collection,
    create_pytorch_docs_collection,
    detach_alias,
    inspect_alias,
    inspect_collection,
    list_alias_mappings,
    promote_alias,
    update_arxiv_collection,
    update_pytorch_docs_collection,
)
from shared.config import bootstrap_local_settings_env, get_settings


def bootstrap_notebook_env() -> Path | None:
    """Load the repo-root `.env` for notebook sessions."""
    return bootstrap_local_settings_env(repo_root=Path(__file__).resolve().parents[2])


def build_config(
    *,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str | None = None,
) -> BuildConfig:
    """Construct a production build config using settings defaults when needed."""
    settings = get_settings()
    return BuildConfig(
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model or settings.embedding_model,
    )


def create_arxiv(
    *,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    arxiv_file: str = "assets/rag_data/arxiv/arxiv_papers.json",
    kb: str = "arxiv",
    alias: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Create a fresh ArXiv collection from production library code."""
    return create_arxiv_collection(
        build_config=build_config(
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        ),
        arxiv_file=arxiv_file,
        kb=kb,
        alias=alias,
        collection_name=collection_name,
        implementation=ImplementationInfo(module="rag.ops.create.arxiv", experimental=False),
    )


def create_pytorch_docs(
    *,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    pytorch_docs_file: str = "assets/rag_data/pytorch_docs/pytorch_docs.json",
    kb: str = "pytorch_docs",
    alias: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Create a fresh PyTorch docs collection from production library code."""
    return create_pytorch_docs_collection(
        build_config=build_config(
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        ),
        pytorch_docs_file=pytorch_docs_file,
        kb=kb,
        alias=alias,
        collection_name=collection_name,
        implementation=ImplementationInfo(module="rag.ops.create.pytorch_docs", experimental=False),
    )


def refresh_arxiv(
    *,
    arxiv_file: str = "assets/rag_data/arxiv/arxiv_papers.json",
    kb: str = "arxiv",
    alias: str = "champion",
) -> dict[str, Any]:
    """Run the production incremental ArXiv refresh from a notebook."""
    return update_arxiv_collection(arxiv_file=arxiv_file, kb=kb, alias=alias)


def refresh_pytorch_docs(
    *,
    pytorch_docs_file: str = "assets/rag_data/pytorch_docs/pytorch_docs.json",
    kb: str = "pytorch_docs",
    alias: str = "champion",
) -> dict[str, Any]:
    """Run the production replace-style PyTorch docs refresh from a notebook."""
    return update_pytorch_docs_collection(
        pytorch_docs_file=pytorch_docs_file,
        kb=kb,
        alias=alias,
    )


def assign_alias(*, kb: str, alias: str, collection_name: str) -> dict[str, Any]:
    """Attach an alias to an existing validated collection."""
    return assign_alias_to_collection(kb=kb, alias=alias, collection_name=collection_name)


def promote(*, kb: str, from_alias: str, to_alias: str) -> dict[str, Any]:
    """Promote one alias to another alias's target collection."""
    return promote_alias(kb=kb, from_alias=from_alias, to_alias=to_alias)


def detach(*, kb: str, alias: str) -> dict[str, str]:
    """Detach an alias from its current collection."""
    return detach_alias(kb=kb, alias=alias)


def inspect_kb_alias(*, kb: str, alias: str) -> dict[str, Any]:
    """Inspect an alias, its target collection, and validated metadata."""
    return inspect_alias(kb_name=kb, alias=alias)


def inspect_existing_collection(*, collection_name: str) -> dict[str, Any]:
    """Inspect a concrete collection and its validated metadata."""
    return inspect_collection(collection_name=collection_name)


def list_aliases(*, kb: str | None = None) -> list[dict[str, str]]:
    """List visible alias mappings, optionally filtered by knowledge base."""
    return list_alias_mappings(kb_name=kb)
