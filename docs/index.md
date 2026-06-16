# Agent 042 Documentation

This directory holds detailed project documentation. The main entry point stays
in the repository root: [README.md](../README.md).

## Start Here

- [Architecture](architecture/system-design.md): detailed system design and
  component relationships.
- [Deployment And Operations](../infra/README.md): server layout, `.env`,
  Docker Compose, shared roots, ports, and operational setup.
- [RAG Operations](operations/rag-operations.md): build, materialize, promote,
  inspect, and roll back RAG collections.
- [Observability, Evaluation, And Analytics Workflow](analytics/observability-evaluation-workflow.md):
  one-request diagnostics and the overall control loop.
- [Observability](analytics/observability.md): logs, traces, Grafana Explore,
  and request correlation.
- [Durable Inference Events](analytics/inference-events.md): Redpanda topic,
  event schema, and topic inspection.
- [ClickHouse Analytics](analytics/clickhouse-analytics.md): ingestion from
  Redpanda and starter analytics queries.
- [Evaluation Results](analytics/evaluation-results.md): eval tables, metrics,
  field meanings, and common SQL.
- [Experiments](../experiments/README.md): notebooks, evaluation scripts, LoRA
  training, and model registry workflows.
- [Improvement Plan](planning/improvements.md): staged roadmap; Phase 1 is
  complete and Phase 2 RAG quality work is next.

## Documentation Ownership

- `architecture/`: why the system is shaped this way.
- `operations/`: how to run operational workflows on the deployed system.
- `analytics/`: observability, monitoring, inference events, ClickHouse
  analytics, evaluation workflow, and failure analysis.
- `evaluation/`: reserved for evaluation-specific references if they grow large
  enough to split out of `analytics/`.
- `experiments/`: experiment and notebook workflow docs.
- `planning/`: active planning documents.
- `legacy/`: older project planning material kept for reference.
