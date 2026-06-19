"""Helpers for writing small catalog TOML fixtures in tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def _write_catalog(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def write_chat_and_code_catalog(path: Path) -> Path:
    return _write_catalog(
        path,
        """
        schema_version = 4

        [[tasks]]
        id = "chat"
        description = "General ML research discussion."
        knowledge_bases = ["ml_papers_core"]
        lora_adapter = { enabled = false }

        [[tasks]]
        id = "code"
        description = "Programming help for ML systems."
        knowledge_bases = ["pytorch_reference"]
        lora_adapter = { enabled = false }

        [[knowledge_bases]]
        id = "ml_papers_core"
        default_alias = "champion"
        update_strategy = "replace"
        description = "Research papers and literature-grounded answers."

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384

        [knowledge_bases.aliases.champion.retrieve]
        top_k = 5
        score_threshold = 0.35
        strategy = "dense"
        reranker_multiplier = 1

        [knowledge_bases.aliases.challenger.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.challenger.build.dense_encoder]
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384

        [knowledge_bases.aliases.challenger.build.sparse_encoder]
        model = "Qdrant/bm25"

        [knowledge_bases.aliases.challenger.retrieve]
        top_k = 5
        score_threshold = 0.01
        reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        strategy = "hybrid"
        reranker_multiplier = 4

        [[knowledge_bases]]
        id = "pytorch_reference"
        default_alias = "champion"
        update_strategy = "replace"
        description = "PyTorch API reference and implementation guidance."

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384

        [knowledge_bases.aliases.champion.retrieve]
        top_k = 5
        score_threshold = 0.35
        strategy = "dense"
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.arxiv_paper"
        version = "1"
        description = "Fetches arXiv papers."
        factory = "rag.adapters.sources:make_arxiv_paper_adapter"

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
factory = "rag.adapters.sources:make_http_html_adapter"

        [[source_instances]]
        id = "ml_papers_core.papers"
        description = "Curated full-text ML/AI papers."
        role = "corpus"
        knowledge_base = "ml_papers_core"
        adapter = { id = "generic.arxiv_paper", version = "1" }

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official PyTorch documentation pages."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = { id = "generic.http_html", version = "1" }
        """,
    )


def write_chat_only_catalog(
    path: Path,
    *,
    retrieval_strategy: str = "dense",
) -> Path:
    sparse_encoder_block = (
        """
        [knowledge_bases.aliases.champion.build.sparse_encoder]
        model = "Qdrant/bm25"
        """
        if retrieval_strategy in ("sparse", "hybrid")
        else ""
    )
    return _write_catalog(
        path,
        f"""
        schema_version = 4

        [[tasks]]
        id = "chat"
        description = "General ML research discussion."
        knowledge_bases = ["ml_papers_core"]
        lora_adapter = {{ enabled = false }}

        [[knowledge_bases]]
        id = "ml_papers_core"
        default_alias = "champion"
        update_strategy = "replace"
        description = "Research papers and literature-grounded answers."

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384
        {sparse_encoder_block}
        [knowledge_bases.aliases.champion.retrieve]
        top_k = 5
        score_threshold = 0.35
        strategy = "{retrieval_strategy}"
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.arxiv_paper"
        version = "1"
        description = "Fetches arXiv papers."
        factory = "rag.adapters.sources:make_arxiv_paper_adapter"

        [[source_instances]]
        id = "ml_papers_core.papers"
        description = "Curated full-text ML/AI papers."
        role = "corpus"
        knowledge_base = "ml_papers_core"
        adapter = {{ id = "generic.arxiv_paper", version = "1" }}
        """,
    )


def write_code_only_catalog(path: Path) -> Path:
    return _write_catalog(
        path,
        """
        schema_version = 4

        [[tasks]]
        id = "code"
        description = "Programming help for ML systems."
        knowledge_bases = ["pytorch_reference"]
        lora_adapter = { enabled = false }

        [[knowledge_bases]]
        id = "pytorch_reference"
        default_alias = "champion"
        description = "PyTorch API reference."

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384

        [knowledge_bases.aliases.champion.retrieve]
        top_k = 5
        score_threshold = 0.35
        strategy = "dense"
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "rag.adapters.sources:make_http_html_adapter"

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official PyTorch documentation pages."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = { id = "generic.http_html", version = "1" }
        """,
    )
