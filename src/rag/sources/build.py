"""High-level source build orchestration before index materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag.sources.chunks import (
    ChunkingConfig,
    SourceInstanceChunkingSummary,
    chunk_source_instance,
)
from rag.sources.extractors import SourceExtractor
from rag.sources.fetchers import SourceFetcher
from rag.sources.processing import SourceProcessingSummary, process_source_instance

SourceBuildStatus = Literal["empty", "success", "partial", "failed"]


class SourceBuildSummary(BaseModel):
    """Summary for one complete pre-index source instance build."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    source_type: str
    status: SourceBuildStatus
    processing: SourceProcessingSummary
    chunking: SourceInstanceChunkingSummary


def _build_status(
    *,
    processing: SourceProcessingSummary,
    chunking: SourceInstanceChunkingSummary,
) -> SourceBuildStatus:
    if processing.total_selected == 0:
        return "empty"

    failures = len(processing.failed) + len(chunking.failed)
    successes = chunking.chunked + chunking.from_cache
    if successes == 0:
        return "failed"
    if failures:
        return "partial"
    return "success"


def build_source_instance(
    *,
    kb_id: str,
    source_instance_id: str,
    source_type: str,
    manifest_path: Path | str,
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    fetchers: dict[str, SourceFetcher] | None = None,
    extractors: dict[str, SourceExtractor] | None = None,
) -> SourceBuildSummary:
    """Run fetch/extract/chunk lifecycle for one source instance."""
    processing = process_source_instance(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_type,
        manifest_path=manifest_path,
        rag_data_root=rag_data_root,
        limit=limit,
        document_ids=document_ids,
        force_fetch=force_fetch,
        force_extract=force_extract,
        fetchers=fetchers,
        extractors=extractors,
    )
    chunking_summary = chunk_source_instance(
        rag_data_root=rag_data_root,
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        document_ids=document_ids,
        limit=limit,
        config=chunking,
        force=force_chunk,
    )
    return SourceBuildSummary(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_type,
        status=_build_status(processing=processing, chunking=chunking_summary),
        processing=processing,
        chunking=chunking_summary,
    )

