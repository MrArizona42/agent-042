"""HTTP source fetchers with immutable cache semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import httpx
from llama_index.core import Document
from pydantic import BaseModel, ConfigDict

from rag.contracts.metadata import require_document_metadata
from rag.sources.cache import (
    SourceCachePaths,
    sha256_bytes,
    source_cache_paths,
    write_bytes_immutable,
    write_json_immutable,
)


DEFAULT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


class SourceFetchResult(BaseModel):
    """Result of fetching one source document into cache."""

    model_config = ConfigDict(extra="forbid")

    source_document: Document
    raw_path: Path
    metadata_path: Path
    checksum: str
    content_type: str | None = None
    from_cache: bool = False


class SourceFetcher(Protocol):
    """Fetch one source document into the project cache."""

    def fetch(
        self,
        source_document: Document,
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
        source_document: Document,
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
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers=DEFAULT_FETCH_HEADERS,
            ) as client:
                response = client.get(uri)
        response.raise_for_status()
        return response.content, response.headers.get("content-type")

    def _metadata_payload(
        self,
        source_document: Document,
        *,
        raw_path: Path,
        checksum: str,
        content_type: str | None,
    ) -> dict[str, object]:
        return {
            "id": source_document.id_,
            "source_uri": source_document.metadata["source_uri"],
            "title": source_document.metadata["title"],
            "raw_path": raw_path.as_posix(),
            "checksum": checksum,
            "content_type": content_type,
        }

    def fetch(
        self,
        source_document: Document,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
        force: bool = False,
    ) -> SourceFetchResult:
        """Fetch and cache one source document."""
        require_document_metadata(source_document)

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
                source_document=_with_fetch_metadata(
                    source_document,
                    raw_path=paths.raw_path,
                    checksum=checksum,
                ),
                raw_path=paths.raw_path,
                metadata_path=paths.metadata_path,
                checksum=checksum,
                from_cache=True,
            )

        content, content_type = self._fetch_bytes(str(source_document.metadata["source_uri"]))
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
            source_document=_with_fetch_metadata(
                source_document,
                raw_path=paths.raw_path,
                checksum=checksum,
            ),
            raw_path=paths.raw_path,
            metadata_path=paths.metadata_path,
            checksum=checksum,
            content_type=content_type,
        )


class HtmlDocsFetcher(HttpSourceFetcher):
    """Fetch HTML documentation pages into raw cache."""

    raw_filename = "page.html"


class ArxivPaperFetcher(HttpSourceFetcher):
    """Fetch ArXiv PDFs into raw cache."""

    raw_filename = "paper.pdf"

    def _pdf_url(self, source_document: Document) -> str:
        arxiv_id = str(source_document.metadata.get("arxiv_id") or "").strip()
        if not arxiv_id:
            uri = str(source_document.metadata["source_uri"])
            for prefix in ("arxiv_paper:", "arxiv:"):
                if uri.startswith(prefix):
                    arxiv_id = uri[len(prefix) :]
                    break
        if not arxiv_id:
            raise ValueError(f"ArXiv source document '{source_document.id_}' is missing arxiv_id")
        return f"https://arxiv.org/pdf/{arxiv_id}"

    def fetch(
        self,
        source_document: Document,
        *,
        kb_id: str,
        source_instance_id: str,
        rag_data_root: Path | str,
        force: bool = False,
    ) -> SourceFetchResult:
        fetch_document = source_document
        source_uri = str(source_document.metadata["source_uri"])
        if source_uri.startswith(("arxiv:", "arxiv_paper:")):
            fetch_document = source_document.model_copy(
                update={
                    "metadata": {
                        **source_document.metadata,
                        "source_uri": self._pdf_url(source_document),
                    }
                }
            )
        return super().fetch(
            fetch_document,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            rag_data_root=rag_data_root,
            force=force,
        )


def _with_fetch_metadata(
    document: Document,
    *,
    raw_path: Path,
    checksum: str,
) -> Document:
    return document.model_copy(
        update={
            "metadata": {
                **document.metadata,
                "raw_path": raw_path.as_posix(),
                "checksum": checksum,
            }
        }
    )
