-- Execution and failure record for one RAG release build attempt.
-- Not a runtime source of truth -- see rag_releases and rag_alias_deployments.

CREATE TABLE IF NOT EXISTS rag_release_builds (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id                     TEXT NOT NULL,
    requested_alias           TEXT NOT NULL,
    status                    TEXT NOT NULL,                 -- running | failed | completed
    catalog_digest            TEXT NOT NULL,
    build_config_digest       TEXT NOT NULL,
    retrieval_config_digest   TEXT NOT NULL,
    source_declaration_digest TEXT NOT NULL,
    source_snapshot_id        TEXT,
    release_id                TEXT,
    collection_name           TEXT,
    started_at                TIMESTAMPTZ NOT NULL,
    finished_at               TIMESTAMPTZ,
    error                     TEXT,
    details                   JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_rag_release_builds_kb_started
    ON rag_release_builds (kb_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_release_builds_status ON rag_release_builds (status);
CREATE INDEX IF NOT EXISTS idx_rag_release_builds_release_id ON rag_release_builds (release_id);
