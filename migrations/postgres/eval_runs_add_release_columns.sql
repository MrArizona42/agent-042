-- Add release/deployment/execution identity to existing eval_runs tables,
-- for the declarative alias workflow's release-aware benchmark service.
-- Safe to run multiple times on deployed agent042 databases.

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS benchmark_execution_id UUID,
    ADD COLUMN IF NOT EXISTS rag_release_id TEXT REFERENCES rag_releases(id),
    ADD COLUMN IF NOT EXISTS alias_deployment_id UUID REFERENCES rag_alias_deployments(id),
    ADD COLUMN IF NOT EXISTS build_config_digest TEXT,
    ADD COLUMN IF NOT EXISTS retrieval_config_digest TEXT;

CREATE INDEX IF NOT EXISTS idx_eval_runs_benchmark_execution_id
    ON eval_runs (benchmark_execution_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_rag_release_id ON eval_runs (rag_release_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_alias_deployment_id ON eval_runs (alias_deployment_id);
