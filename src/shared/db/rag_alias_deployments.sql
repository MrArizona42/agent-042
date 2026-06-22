-- Applied alias deployment history. At most one row per (kb_id, alias) may
-- be 'active'; older rows are superseded, never deleted. This table -- not
-- the Qdrant alias -- is the runtime serving source of truth.

CREATE TABLE IF NOT EXISTS rag_alias_deployments (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id                    TEXT NOT NULL,
    alias                    TEXT NOT NULL,
    release_id               TEXT NOT NULL REFERENCES rag_releases(id),
    collection_name          TEXT NOT NULL,
    catalog_digest           TEXT NOT NULL,
    build_config_digest      TEXT NOT NULL,
    retrieval_config_digest  TEXT NOT NULL,
    retrieval_config         JSONB NOT NULL,
    status                   TEXT NOT NULL,           -- pending | active | superseded | failed
    created_at               TIMESTAMPTZ NOT NULL,
    applied_at               TIMESTAMPTZ,
    superseded_at            TIMESTAMPTZ,
    error                    TEXT,
    details                  JSONB NOT NULL DEFAULT '{}'
);

ALTER TABLE rag_alias_deployments
    ADD COLUMN IF NOT EXISTS details JSONB NOT NULL DEFAULT '{}';

-- At most one active deployment per (kb_id, alias).
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_alias_deployments_active
    ON rag_alias_deployments (kb_id, alias)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_rag_alias_deployments_release_id
    ON rag_alias_deployments (release_id);
CREATE INDEX IF NOT EXISTS idx_rag_alias_deployments_kb_alias_created
    ON rag_alias_deployments (kb_id, alias, created_at DESC);
