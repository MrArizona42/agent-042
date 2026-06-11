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
- [Observability](operations/observability.md): logs, traces, Grafana Explore,
  and request correlation.
- [Durable Inference Events](operations/inference-events.md): Redpanda topic,
  event schema, and topic inspection.
- [ClickHouse Analytics](operations/clickhouse-analytics.md): ingestion from
  Redpanda and starter analytics queries.
- [Experiments](../experiments/README.md): notebooks, evaluation scripts, LoRA
  training, and model registry workflows.
- [Improvement Plan](planning/improvements.md): current staged work plan.

## Documentation Ownership

- `architecture/`: why the system is shaped this way.
- `operations/`: how to run, inspect, and debug the deployed system.
- `evaluation/`: evaluation workflow, metric interpretation, and failure
  analysis.
- `experiments/`: experiment and notebook workflow docs.
- `planning/`: active planning documents.
- `legacy/`: older project planning material kept for reference.
