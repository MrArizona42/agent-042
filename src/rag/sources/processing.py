"""Source processing orchestration for fetch/extract/artifact lifecycle."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import SourceDocument
from rag.sources.artifacts import (
    extracted_artifact_from_result,
    extracted_artifact_path,
    read_extracted_artifact,
    write_extracted_artifact,
)
from rag.sources.extractors import ArxivPdfExtractor, HtmlDocsExtractor, SourceExtractor
from rag.sources.fetchers import ArxivPaperFetcher, HtmlDocsFetcher, SourceFetcher
from rag.sources.manifests import load_source_manifest


class SourceProcessingFailure(BaseModel):
    """One failed source document processing attempt."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    error_type: str
    message: str


class SourceProcessingSummary(BaseModel):
    """Summary for one source instance processing run."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    source_type: str
    total_selected: int = Field(ge=0)
    fetched: int = Field(default=0, ge=0)
    fetched_from_cache: int = Field(default=0, ge=0)
    extracted: int = Field(default=0, ge=0)
    extracted_from_cache: int = Field(default=0, ge=0)
    failed: list[SourceProcessingFailure] = Field(default_factory=list)


DEFAULT_SOURCE_FETCHERS: dict[str, SourceFetcher] = {
    "arxiv_paper": ArxivPaperFetcher(),
    "html_docs": HtmlDocsFetcher(),
}

DEFAULT_SOURCE_EXTRACTORS: dict[str, SourceExtractor] = {
    "arxiv_paper": ArxivPdfExtractor(),
    "html_docs": HtmlDocsExtractor(),
}


def _source_document_matches(
    source_document: SourceDocument,
    selected_ids: set[str],
) -> bool:
    return source_document.id in selected_ids


def _select_source_documents(
    source_documents: list[SourceDocument],
    *,
    document_ids: list[str] | None,
    limit: int | None,
) -> list[SourceDocument]:
    selected = source_documents
    if document_ids is not None:
        selected_ids = set(document_ids)
        selected = [
            source_document
            for source_document in selected
            if _source_document_matches(source_document, selected_ids)
        ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    return selected


def process_source_instance(
    *,
    kb_id: str,
    source_instance_id: str,
    source_type: str,
    manifest_path: Path | str,
    rag_data_root: Path | str,
    limit: int | None = None,
    document_ids: list[str] | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    fetchers: dict[str, SourceFetcher] | None = None,
    extractors: dict[str, SourceExtractor] | None = None,
) -> SourceProcessingSummary:
    """Fetch, extract, and persist artifacts for one source instance."""
    manifest = load_source_manifest(manifest_path)
    if manifest.source_type != source_type:
        raise ValueError(
            f"Source manifest '{manifest_path}' has source_type '{manifest.source_type}' "
            f"(expected '{source_type}')"
        )

    fetcher_registry = fetchers or DEFAULT_SOURCE_FETCHERS
    extractor_registry = extractors or DEFAULT_SOURCE_EXTRACTORS
    if source_type not in fetcher_registry:
        raise ValueError(f"No source fetcher registered for source_type '{source_type}'")
    if source_type not in extractor_registry:
        raise ValueError(f"No source extractor registered for source_type '{source_type}'")

    selected_documents = _select_source_documents(
        manifest.to_source_documents(),
        document_ids=document_ids,
        limit=limit,
    )
    summary = SourceProcessingSummary(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_type,
        total_selected=len(selected_documents),
    )
    fetcher = fetcher_registry[source_type]
    extractor = extractor_registry[source_type]

    for source_document in selected_documents:
        try:
            artifact_path = extracted_artifact_path(
                rag_data_root=rag_data_root,
                kb_id=kb_id,
                source_instance_id=source_instance_id,
                source_document_id=source_document.id,
            )
            if artifact_path.exists() and not force_extract:
                read_extracted_artifact(artifact_path)
                summary.extracted_from_cache += 1
                continue

            fetch_result = fetcher.fetch(
                source_document,
                kb_id=kb_id,
                source_instance_id=source_instance_id,
                rag_data_root=rag_data_root,
                force=force_fetch,
            )
            summary.fetched += 1
            if fetch_result.from_cache:
                summary.fetched_from_cache += 1

            extracted_document = extractor.extract(fetch_result)
            artifact = extracted_artifact_from_result(
                kb_id=kb_id,
                source_instance_id=source_instance_id,
                fetch_result=fetch_result,
                extracted_document=extracted_document,
            )
            write_extracted_artifact(artifact_path, artifact, force=force_extract)
            summary.extracted += 1
        except Exception as exc:  # noqa: BLE001 - lifecycle summary owns per-doc failures.
            summary.failed.append(
                SourceProcessingFailure(
                    document_id=source_document.id,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            )

    return summary
