"""HTTP source fetchers with immutable cache semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from rag.domain import SourceDocument
from rag.sources.cache import (
    SourceCachePaths,
    sha256_bytes,
    source_cache_paths,
    write_bytes_immutable,
    write_json_immutable,
)


class SourceFetchResult(BaseModel):
    """Result of fetching one source document into cache."""

    model_config = ConfigDict(extra="forbid")

    source_document: SourceDocument
    raw_path: Path
    metadata_path: Path
    checksum: str
    content_type: str | None = None
    from_cache: bool = False


class SourceFetcher(Protocol):
    """Fetch one source document into the project cache."""

    def fetch(
        self,
        source_document: SourceDocument,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
        force: bool = False,
    ) -> SourceFetchResult:
        """Fetch and cache one source document."""
        ...


class HttpSourceFetcher:
    """Base fetcher for simple HTTP source documents."""

    raw_filename: str
    expected_source_type: str

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = client
        self._timeout = timeout

    def _cache_paths(
        self,
        source_document: SourceDocument,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
    ) -> SourceCachePaths:
        return source_cache_paths(
            rag_data_root=rag_data_root,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            source_document=source_document,
            raw_filename=self.raw_filename,
        )

    def _fetch_bytes(self, uri: str) -> tuple[bytes, str | None]:
        if self._client is not None:
            response = self._client.get(uri)
        else:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(uri)
        response.raise_for_status()
        return response.content, response.headers.get("content-type")

    def _metadata_payload(
        self,
        source_document: SourceDocument,
        *,
        raw_path: Path,
        checksum: str,
        content_type: str | None,
    ) -> dict[str, object]:
        return {
            "id": source_document.id,
            "source_type": source_document.source_type,
            "uri": source_document.uri,
            "title": source_document.title,
            "raw_path": raw_path.as_posix(),
            "checksum": checksum,
            "content_type": content_type,
        }

    def fetch(
        self,
        source_document: SourceDocument,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
        force: bool = False,
    ) -> SourceFetchResult:
        """Fetch and cache one source document."""
        if source_document.source_type != self.expected_source_type:
            raise ValueError(
                f"{self.__class__.__name__} expected source_type "
                f"'{self.expected_source_type}' (got '{source_document.source_type}')"
            )

        paths = self._cache_paths(
            source_document,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            rag_data_root=rag_data_root,
        )
        from_cache = paths.raw_path.exists() and paths.metadata_path.exists() and not force
        if from_cache:
            content = paths.raw_path.read_bytes()
            checksum = sha256_bytes(content)
            return SourceFetchResult(
                source_document=source_document.model_copy(
                    update={
                        "raw_path": paths.raw_path.as_posix(),
                        "checksum": checksum,
                    }
                ),
                raw_path=paths.raw_path,
                metadata_path=paths.metadata_path,
                checksum=checksum,
                from_cache=True,
            )

        content, content_type = self._fetch_bytes(source_document.uri)
        checksum = sha256_bytes(content)
        write_bytes_immutable(paths.raw_path, content, force=force)
        write_json_immutable(
            paths.metadata_path,
            self._metadata_payload(
                source_document,
                raw_path=paths.raw_path,
                checksum=checksum,
                content_type=content_type,
            ),
            force=force,
        )
        return SourceFetchResult(
            source_document=source_document.model_copy(
                update={
                    "raw_path": paths.raw_path.as_posix(),
                    "checksum": checksum,
                }
            ),
            raw_path=paths.raw_path,
            metadata_path=paths.metadata_path,
            checksum=checksum,
            content_type=content_type,
        )


class HtmlDocsFetcher(HttpSourceFetcher):
    """Fetch HTML documentation pages into raw cache."""

    raw_filename = "page.html"
    expected_source_type = "html_docs"


class ArxivPaperFetcher(HttpSourceFetcher):
    """Fetch ArXiv PDFs into raw cache."""

    raw_filename = "paper.pdf"
    expected_source_type = "arxiv_paper"

    def _pdf_url(self, source_document: SourceDocument) -> str:
        arxiv_id = str(source_document.metadata.get("arxiv_id") or "").strip()
        if not arxiv_id:
            raise ValueError(f"ArXiv source document '{source_document.id}' is missing arxiv_id")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    def fetch(
        self,
        source_document: SourceDocument,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
        force: bool = False,
    ) -> SourceFetchResult:
        fetch_document = source_document
        if source_document.uri.startswith("arxiv:"):
            fetch_document = source_document.model_copy(
                update={"uri": self._pdf_url(source_document)}
            )
        return super().fetch(
            fetch_document,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            rag_data_root=rag_data_root,
            force=force,
        )
