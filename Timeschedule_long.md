# Project Timeline

## ✅ 0. (Brief) MLOps Side Project

**Focus:** reproducibility + adapter training + basic local serving.

### Minimal checklist

- `setup/` folder with:
  - `local-dev/` (docker files, compose, makefile)
  - `infra-setup/` (Terraform later)
  - Hydra config template (`conf/`)
- Training script with adapters: `train_adapter.py`
- Minimal inference script: `run_adapter.py`
- Base Compose:
  - API service
  - model service
  - vector DB (optional)

This phase is short — just build enough to move forward with the real agent system.

---

## 🟦 1. Infra Prep (CI/CD, Compose, Nginx, Data Gathering)

**Goal:** A stable environment for experiments + remote access + logs.

### Checklist

#### 1.1 Local + remote environment

- Finish `docker-compose.dev.yaml` and `docker-compose.staging.yaml`
- Make `commands.py` or `make dev` for:
  - up, down, logs, rebuild, migrate
- Use environment-based Hydra config.

#### 1.2 Networking + Nginx

- Nginx as reverse proxy with:
  - `/api` → API container
  - `/agent` → agent interface (future)
  - `/static` → UI assets (optional)
- Internal network separation:
  - `backend_net`
  - optionally `db_net`

#### 1.3 CI/CD pipeline

- Basic GitHub Actions:
  - lint + tests
  - build Docker images
  - push to registry
  - deploy to staging (using SSH runner or Kubernetes later)

#### 1.4 Logging & data gathering

- Start logging early:
  - raw queries
  - tool calls
  - intermediate reasoning traces (system messages)
  - adapter usage
  - errors
- Design simple schemas:
  ```
  logs/
    queries.jsonl
    tool_calls.jsonl
    rag_events.jsonl
    errors.jsonl
  datasets/
    embeddings/
    training/
  ```

#### 1.5 Observability (initial)

- Structured logs with correlation IDs
- Prometheus endpoint or OpenTelemetry basic exporter

---

## 🟩 2. Data Processing Pipelines & Base RAG

**Aim:** functional RAG that supports retrieval for agents.

### Checklist

#### 2.1 Data ingestion

- "Collectors": scripts pulling data from:
  - your notes,
  - source documents,
  - internal project structure (repos, docs).
- Normalize content to `documents/` in YAML/Markdown.

#### 2.2 Preprocessing pipeline

- chunking (recursive, semantic)
- cleaning (dedup, HTML stripping)
- metadata assignment (source, tags)

#### 2.3 Embeddings pipeline

- embed + store in vector DB (Chroma, Qdrant, pgvector)

#### 2.4 RAG service

- `/retrieve` endpoint
- Retrieval config in Hydra (top-k, rerankers optional)
- Caching layer (simple LRU or Redis)

#### 2.5 Evaluation

- minimal correctness testing:
  - keyword queries
  - semantic tests
  - latency benchmark

This is your first working RAG.

---

## 🟧 3. Tools Orchestration (MCP, routing, reasoning)

This is the core of your thesis-grade agent.

### Checklist

#### 3.1 MCP server architecture

- Build your tool layer as standalone MCP servers.
- Minimal set:
  - filesystem (read/write project files)
  - command (run shell inside container)
  - search (local search)
  - rag (retrieve)
  - data (query logs, metrics, small DB)
- Each server separate so you can scale.

#### 3.2 Agent → Tools interface

- registry of available tools
- tool metadata (name, schema)
- dynamic enabling/disabling

#### 3.3 Reasoning layer

Your core agent pipeline:

**Input → pre-processing → RAG? → adapter? → choose tools → execute → post-process**

Implement step-by-step:

- **Heuristic-first router:**
  - if query contains "search" → search tool
  - if technical → call "code" adapter
- **LLM-based router:**
  - evaluate: do we need RAG?
  - do we need a tool?
  - which adapter is optimal?
- Reasoning traces persisted to logs.

#### 3.4 Chained tool execution

Add:
- multiple tool calls
- recursive/looped reasoning
- error recovery

#### 3.5 Safety policies

- limits on tool depth
- timeout management
- resource budgets

#### 3.6 Agent evaluation

- correctness tests
- tool-call accuracy tests
- pass/fail scenarios

This stage gives you a solid agent capable of autonomous routing.

---

## 🟨 4. k3s Setup and IaC (Terraform, Helm)

Only after agent architecture stabilizes.

### Checklist

#### 4.1 k3s cluster

- Single-node or HA cluster
- Let Terraform install:
  - VM instance / local cluster
  - Traefik or Nginx ingress
  - Cert-manager (TLS)
  - StorageClass

#### 4.2 Helm charts

- Create your own charts for:
  - agent backend
  - rag service
  - vector DB
  - mcp servers
  - nginx/ingress
  - monitoring stack
- Follow simple chart structure:
  ```
  charts/
    agent/
    rag/
    mcp-filesystem/
    mcp-search/
    ...
  ```

#### 4.3 Terraform

Manage:
- DNS
- certificates
- container registry
- cluster creation
- secrets storage

#### 4.4 CI/CD to k8s

- GitHub Action:
  - build & push images
  - helm upgrade --install

#### 4.5 Scaling tests

- HPA for backend
- stress test RAG
- load test agent tool loops

---

## 🟫 5. Monitoring, Benchmarking, UI Polishing

Final polishing layer.

### Checklist

#### 5.1 Observability

- Grafana dashboards:
  - response latency
  - tool call counts
  - RAG success rate
  - adapter selection stats
- Tracing using OTEL
- Log correlation (per request, per tool chain)

#### 5.2 Benchmarking

- agent correctness benchmark
- tool selection benchmark
- RAG retrieval accuracy
- latency under load

#### 5.3 UI

- Minimal UI:
  - chat interface with trace view
  - logs browser
  - RAG inspector
  - tool analytics

This closes the loop from infra → reasoning → monitoring.