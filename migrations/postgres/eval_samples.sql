-- Per-sample evaluation results.
-- One row per (eval_run, sample) combination.
-- Links to eval_runs via eval_run_id for the aggregate context
-- (model, RAG config, dataset, etc.).

CREATE TABLE IF NOT EXISTS eval_samples (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id     UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    sample_idx      INTEGER NOT NULL,                    -- ordinal position in the dataset
    sample_id       TEXT,                                -- dataset's own identifier (e.g. HumanEval/42)

    input           TEXT,                                -- prompt / question / query
    output          TEXT,                                -- model-generated text
    reference       TEXT,                                -- gold answer / expected output

    detail          JSONB NOT NULL DEFAULT '{}',         -- task-specific per-sample data

    UNIQUE (eval_run_id, sample_idx)
);

-- Query patterns: "show me all failed samples for eval run X",
-- "aggregate pass-rate per sample_id across runs".
CREATE INDEX IF NOT EXISTS idx_eval_samples_run ON eval_samples (eval_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_samples_sample_id ON eval_samples (sample_id);
CREATE INDEX IF NOT EXISTS idx_eval_samples_detail ON eval_samples USING gin (detail);
