-- Add RAG observability fields to existing eval_runs tables.
-- Safe to run multiple times on deployed agent042 databases.

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS qdrant_alias TEXT,
    ADD COLUMN IF NOT EXISTS rag_manifest_id TEXT,
    ADD COLUMN IF NOT EXISTS eval_verdict TEXT;

CREATE INDEX IF NOT EXISTS idx_eval_runs_qdrant_alias ON eval_runs (qdrant_alias);
CREATE INDEX IF NOT EXISTS idx_eval_runs_rag_manifest ON eval_runs (rag_manifest_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_verdict ON eval_runs (eval_verdict);
