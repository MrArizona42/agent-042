-- Immutable, content-identified RAG releases. A release carries no alias
-- field -- which alias serves it is recorded in rag_alias_deployments.

CREATE TABLE IF NOT EXISTS rag_releases (
    id                        TEXT PRIMARY KEY,
    kb_id                     TEXT NOT NULL,
    collection_name           TEXT NOT NULL UNIQUE,
    manifest_id               TEXT NOT NULL UNIQUE,
    manifest_path             TEXT NOT NULL,
    release_fingerprint       TEXT NOT NULL UNIQUE,
    catalog_digest            TEXT NOT NULL,
    build_config_digest       TEXT NOT NULL,
    source_declaration_digest TEXT NOT NULL,
    source_snapshot_id        TEXT NOT NULL,
    build_config              JSONB NOT NULL,
    source_manifest_digests   JSONB NOT NULL,
    source_adapter_versions   JSONB NOT NULL,
    document_count            INTEGER NOT NULL,
    chunk_count               INTEGER NOT NULL,
    created_at                TIMESTAMPTZ NOT NULL,
    retired_at                TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_rag_releases_kb_created ON rag_releases (kb_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_releases_build_config_digest
    ON rag_releases (build_config_digest);
CREATE INDEX IF NOT EXISTS idx_rag_releases_source_declaration_digest
    ON rag_releases (source_declaration_digest);
CREATE INDEX IF NOT EXISTS idx_rag_releases_source_snapshot_id
    ON rag_releases (source_snapshot_id);
