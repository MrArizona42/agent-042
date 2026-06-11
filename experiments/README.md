# Experiments

This directory contains notebooks and scripts for model training, evaluation,
RAG inspection, and operator analysis.

Most production-safe workflows are implemented in `src/`, `dags/`, and
`scripts/`. The `experiments/` tree is for reproducible research code,
notebooks, and analysis work around those workflows.

## Main Paths

- `training/`: LoRA training code and Hydra configs.
- `training/lora_ops.ipynb`: MLflow Model Registry operations for LoRA
  adapters.
- `eval/`: evaluation notebooks and scripts.
- `eval/failure_analysis.ipynb`: failure analysis over `eval_runs` and
  `eval_samples`.
- `eval/eval_results.ipynb`: existing eval result exploration.
- `eval/debug_eval.ipynb`: eval pipeline debugging.
- `rag/`: RAG operator notebooks for Qdrant diagnostics and observability.
- `rag/rag_ops.ipynb`: manual Qdrant inspection, aliases, attestations,
  samples, cleanup checks, and danger-zone maintenance cells.
- `rag/sandboxes/`: notebook-only experimental forks.
- `misc_ops/`: asset prefetch, MLflow, PostgreSQL, and other operational
  notebooks.

## Start Here

- [Evaluation results](../docs/analytics/evaluation-results.md): eval tables,
  fields, metrics, and common SQL.
- [Failure analysis workflow](../docs/analytics/observability-evaluation-workflow.md):
  how runtime observability and offline evals fit together.
- [Training and model registry](../docs/experiments/training-and-model-registry.md):
  LoRA training, Hydra, DVC, MLflow, and adapter registry details.
- [RAG operations](../docs/operations/rag-operations.md): production-safe RAG
  build, materialize, promote, inspect, and rollback workflow.

## Rules Of Thumb

- Use Airflow or `scripts/rag_ops.sh` for production RAG lifecycle actions.
- Use notebooks for inspection, review, and analysis.
- Keep notebook-only experiments in `rag/sandboxes/` until they are ready to be
  promoted into production code.
- Keep large datasets, models, and generated artifacts out of Git; use DVC or
  the configured artifact roots.
