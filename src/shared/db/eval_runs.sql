-- Evaluation runs table for tracking eval metrics across configurations.
-- One row per (task, dataset, metric, rag_alias, lora_alias) combination.

CREATE TABLE IF NOT EXISTS eval_runs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at           TIMESTAMPTZ,
    status                TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed

    task                  TEXT NOT NULL,                     -- chat | summarize | code | retrieval
    dataset_name          TEXT NOT NULL,
    metric_name           TEXT NOT NULL,
    metric_value          DOUBLE PRECISION NOT NULL,

    -- Model
    base_model            TEXT NOT NULL,
    adapter_name          TEXT,                              -- e.g., lora-chat, lora-code
    adapter_version       INTEGER,
    adapter_mlflow_run_id TEXT,                              -- read from MLflow Model Registry
    lora_alias            TEXT,                              -- champion | challenger | none

    -- RAG
    rag_enabled           BOOLEAN NOT NULL DEFAULT false,
    rag_alias             TEXT,                              -- champion | challenger | null
    knowledge_base        TEXT,                              -- arxiv | pytorch_docs | null
    qdrant_collection     TEXT,                              -- resolved collection name
    embedding_model       TEXT,
    chunking_strategy     TEXT,
    chunk_size            INTEGER,
    chunk_overlap         INTEGER,
    retrieval_top_k       INTEGER,
    score_threshold       DOUBLE PRECISION,
    qdrant_snapshot_id    TEXT,                              -- snapshot taken before eval
    dataset_dvc_hash      TEXT,                              -- from .dvc file at eval start
    reranking_strategy    TEXT,                              -- none | cross_encoder | llm

    -- Judge & metrics config
    judge_model           TEXT,
    bert_score_model      TEXT,

    -- Generation params
    temperature           DOUBLE PRECISION,
    max_tokens            INTEGER,

    extra                 JSONB NOT NULL DEFAULT '{}',

    error_message         TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_task ON eval_runs (task);
CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset ON eval_runs (dataset_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_adapter ON eval_runs (adapter_name, adapter_version);
CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_base_model ON eval_runs (base_model);
CREATE INDEX IF NOT EXISTS idx_eval_runs_rag_alias ON eval_runs (rag_alias);
CREATE INDEX IF NOT EXISTS idx_eval_runs_lora_alias ON eval_runs (lora_alias);
CREATE INDEX IF NOT EXISTS idx_eval_runs_extra ON eval_runs USING gin (extra);
