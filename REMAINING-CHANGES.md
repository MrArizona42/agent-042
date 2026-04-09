# Remaining Changes

Этот документ содержит подробное описание незавершенных изменений и планируемых системных
доработок. Документ разделён на два раздела: конкретные bugfix-ы и улучшения существующего кода,
а также крупные системные изменения и расширения функционала.

> **Ревизия**: документ актуализирован по текущему состоянию кодовой базы.
> Оставлены только подтверждённо нерешённые задачи.

---

## 1. Bugfix-ы и улучшения существующего кода

Конкретные, скоупированные задачи по исправлению или улучшению уже реализованных компонентов.

На момент текущей сверки подтверждённых незакрытых bugfix-пунктов в этом разделе не осталось.

---

## 2. Системные изменения и расширения

Крупные многофайловые изменения и новый функционал, требующий планирования и поэтапной
реализации.

### 2.1 RAG: reranking

**Текущее состояние**: RAG система не имеет reranking-стадии. Multi-KB results сливаются и
все передаются в промпт. `EvalRun` модель содержит поле `reranking_strategy`, но реализации нет.

**Целевое**: добавить cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) в
retrieval pipeline. Реализовать через factory pattern `get_reranker(strategy)` по аналогии с
`get_chunker()`. Одна строка в `rag_service.py` для включения/отключения. Провести benchmark
champion vs champion+reranker используя существующую eval-платформу.

**Новые файлы**: `src/rag/rerankers.py`.
**Затрагиваемые файлы**: `src/rag/retriever.py`, `src/gateway/services/rag_service.py`.

### 2.2 RAG: hybrid search

**Текущее состояние**: только dense retrieval (`VectorParams` с `Distance.COSINE`).

**Целевое**: Qdrant поддерживает sparse+dense fusion нативно. Добавить sparse vectors
(BM25 или SPLADE) при индексации, использовать Qdrant fusion при поиске. Eval framework уже
содержит retrieval benchmarks — чистое A/B сравнение.

**Затрагиваемые файлы**: `src/rag/vector_store.py`, `src/rag/ops/materialize.py`,
`src/rag/retriever.py`.

### 2.3 Observability: Prometheus + Grafana

**Текущее состояние**: нет метрик, нет alerting. Мониторинг ограничен UI dashboards
(Flower, RedisInsight, Airflow) и health checks.

**Целевое**:
- Prometheus metrics endpoint в gateway (request latency, error rates, queue depth)
- vLLM native Prometheus metrics (token throughput, GPU utilization)
- Grafana dashboards для inference pipeline, RAG retrieval, Celery workers
- Docker Compose services для Prometheus и Grafana

**Новые файлы**: Compose service definitions, Grafana dashboard JSONs, Prometheus config.
**Затрагиваемые файлы**: gateway main.py (metrics middleware).

### 2.4 Observability: LLM-specific tracing

**Текущее состояние**: нет prompt/response logging, нет latency per step, нет cost tracking.

**Целевое**: интеграция Langfuse или Arize Phoenix (self-hosted) для:
- Prompt/response logging с latency breakdown
- Token cost tracking per user / per session
- Retrieval quality tracking
- Minimal code changes: decorator calls в gateway

**Альтернатива**: OpenTelemetry auto-instrumentation (FastAPI + Celery + httpx) для distributed
tracing через Jaeger/Tempo.

### 2.5 CI/CD: hosted workflows

**Текущее состояние**: quality gates только через локальные pre-commit hooks, ruff, pytest.
Нет hosted CI/CD.

**Целевое** (minimum viable CI):
1. Pre-commit hooks run on push
2. `pytest` on push
3. Docker image builds on merge to `main` (validates all Dockerfiles)
4. Optional: push images to container registry (e.g. GitHub Container Registry)

**Новые файлы**: `.github/workflows/ci.yml` (или GitLab CI equivalent).

### 2.6 Database: Alembic migrations

**Текущее состояние**: schema bootstrap через ORM `Base.metadata.create_all` в gateway startup.
Нет version-controlled migrations.

**Целевое**: настроить Alembic для `agent042` database. Генерировать initial migration из
текущей ORM schema. Все последующие schema changes через managed migrations.

**Новые файлы**: `alembic.ini`, `alembic/` directory, initial migration.
**Затрагиваемые файлы**: gateway startup (run migrations on boot).

### 2.7 Token / cost tracking

**Текущее состояние**: online inference stores prompt/completion usage in the
ORM model and threads prompt token counts through the sync/async gateway paths.
Существующие `agent042` БД применяют
`src/shared/db/chat_messages_add_usage_columns.sql`, чтобы схема
`chat_messages` включала эти колонки.

**Целевое**:
- Применить `src/shared/db/chat_messages_add_usage_columns.sql` на все
    существующие `agent042` БД
- Агрегация per-user / per-session
- Grafana dashboard для token throughput и cost estimation

**Затрагиваемые файлы**: `src/shared/db/`, gateway processing, `chat_messages` schema.

### 2.8 Security: rate limiting and input validation

**Текущее состояние**: нет rate limiting, нет ограничений на длину input.

**Целевое**:
- FastAPI + `slowapi` или Redis-based rate limiter (per-user, per-IP)
- Length limits на user input в prompt builder
- Content/length validation на `rag_sources` request field

**Затрагиваемые файлы**: gateway middleware, `src/gateway/services/processing.py`.

### 2.9 Data validation in Airflow DAGs

**Текущее состояние**: RAG DAGs (download → dvc_version → build_index) скачивают данные и
индексируют без проверки качества. Eval datasets не проходят schema validation.

**Целевое**: добавить validation tasks в Airflow DAGs:
- Record count checks после download
- Schema validation (expected fields, types)
- Embedding dimension verification после индексации
- Quality gates перед promotion

**Затрагиваемые файлы**: `dags/arxiv_rag_update.py`, `dags/pytorch_docs_rag_update.py`,
eval DAGs.

### 2.10 Agent layer: dynamic tool selection (Stage 4)

**Текущее состояние**: task routing — rule-based по keywords
(`RuleBasedTaskRouter.decide()` проверяет `any(k in t for k in [...])`).

**Целевое**: function-calling based agent layer:
- User query → LLM решает tool calls → Execute tools → LLM синтезирует ответ
- Registered tools: `search_knowledge_base(query, kb_name)`,
  `summarize_document(text)`, `generate_code(description)`,
  `web_search(query)` (optional)
- vLLM поддерживает function calling для многих моделей
- Абстракция между gateway и task router / prompt builder

**Затрагиваемые файлы**: `src/gateway/services/task_router.py`,
`src/gateway/services/processing.py`, новые tool registry modules.

### 2.11 Kubernetes / Helm / Terraform (deferred)

**Статус**: отложено. Docker Compose + Nginx TLS — достаточный production-grade deployment для
single-node setup. Kubernetes оправдан при потребности в horizontal scaling или multi-node HA,
что не является требованием thesis.

Существующие директории `infra/helm/`, `infra/k3s/`, `infra/terraform/` зарезервированы.
Документирование как future work в thesis.
