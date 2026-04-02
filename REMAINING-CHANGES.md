# Remaining Changes

Этот документ содержит подробное описание незавершенных изменений и планируемых системных
доработок. Документ разделён на два раздела: конкретные bugfix-ы и улучшения существующего кода,
а также крупные системные изменения и расширения функционала.

> **Ревизия**: документ актуализирован по результатам сверки каждого пункта с кодовой базой.
> Ранее фиксированные пункты (training pipeline: sequence budget, best-checkpoint export,
> run-scoped artifacts, QLoRA setup, precision, Hydra override examples, MLflow logging,
> notebook drift, Airflow log streaming, minor runtime issues, config duplication,
> Compose defaults, gateway startup validation, Hydra restructuring, LoRA hot-swap,
> chunking factory docstring/code mismatch) удалены.
> Оставлены только подтверждённо нерешенные задачи.

---

## 1. Bugfix-ы и улучшения существующего кода

Конкретные, скоупированные задачи по исправлению или улучшению уже реализованных компонентов.

### 1.1 KB discovery: flat API loses task grouping

**Проблема**: `knowledge_bases.json` организован по task-ам, но `GET /v1/knowledge-bases`
возвращает flat list, а UI читает `KNOWLEDGE_BASES` как flat dict через `_KBProxy`. Ни один
consumer не отражает task-grouped структуру; информация о принадлежности KB к task-у теряется
на уровне API.

**Решение**: обновить discovery endpoint для возврата task-grouped структуры (или добавить
поле `task` в каждый элемент flat list). Обновить UI при необходимости.

### 1.2 Chat history: streaming persistence gap

**Проблема**: `stream_chat()` в `src/gateway/services/processing.py` принимает параметры
`user_id` и `chat_session_id`, но нигде их не использует. Streaming-токены не накапливаются
и не сохраняются — большинство conversation history теряется. Non-streaming `chat()` корректно
вызывает `_persist_exchange()`.

**Решение**: собирать streaming-токены в буфер на gateway-стороне и вызывать
`_persist_exchange()` по завершении stream-а.

---

## 2. Системные изменения и расширения

Крупные многофайловые изменения и новый функционал, требующий планирования и поэтапной
реализации.

### 2.1 Training → Eval pipeline integration

**Текущее состояние**: `train -> inspect/promote` (eval выполняется вручную).

**Целевое**: `train -> evaluate -> human decision (inspect/promote)`. Promotion остаётся ручной,
но оценка должна запускаться автоматически после тренировки. DAG `train_lora` должен запускать
eval DAG-и после успешного завершения тренировки, передавая `run_id` и `lora_alias="challenger"`.

**Затрагиваемые файлы**: `dags/train_lora.py`, eval DAGs, `lora_ops.ipynb`.

### 2.2 Knowledge bases: task-first API contract

**Текущее состояние**: `knowledge_bases.json` уже организован по tasks; internal helpers
(`TaskConfig`, `KBConfig`, `get_kb_config()`, `_KBProxy`) полностью адаптированы. Однако
discovery API и UI по-прежнему работают с flat-представлением (см. §1.2).

**Целевое**: полный переход на task-first contract в API:
- `GET /v1/knowledge-bases` возвращает task-grouped структуру
- Обновить `openai_compat.py`, `rag_service.py`, `knowledge_bases.py` endpoint
- Обновить daily RAG DAGs для итерации task config

**Затрагиваемые файлы**: `src/gateway/api/v1/knowledge_bases.py`,
`src/gateway/services/rag_service.py`, DAGs.

### 2.3 RAG: reranking

**Текущее состояние**: RAG система не имеет reranking-стадии. Multi-KB results сливаются и
все передаются в промпт. `EvalRun` модель содержит поле `reranking_strategy`, но реализации нет.

**Целевое**: добавить cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) в
retrieval pipeline. Реализовать через factory pattern `get_reranker(strategy)` по аналогии с
`get_chunker()`. Одна строка в `rag_service.py` для включения/отключения. Провести benchmark
champion vs champion+reranker используя существующую eval-платформу.

**Новые файлы**: `src/rag/rerankers.py`.
**Затрагиваемые файлы**: `src/rag/retriever.py`, `src/gateway/services/rag_service.py`.

### 2.4 RAG: hybrid search

**Текущее состояние**: только dense retrieval (`VectorParams` с `Distance.COSINE`).

**Целевое**: Qdrant поддерживает sparse+dense fusion нативно. Добавить sparse vectors
(BM25 или SPLADE) при индексации, использовать Qdrant fusion при поиске. Eval framework уже
содержит retrieval benchmarks — чистое A/B сравнение.

**Затрагиваемые файлы**: `src/rag/vector_store.py`, `src/rag/ops/materialize.py`,
`src/rag/retriever.py`.

### 2.5 Observability: Prometheus + Grafana

**Текущее состояние**: нет метрик, нет alerting. Мониторинг ограничен UI dashboards
(Flower, RedisInsight, Airflow) и health checks.

**Целевое**:
- Prometheus metrics endpoint в gateway (request latency, error rates, queue depth)
- vLLM native Prometheus metrics (token throughput, GPU utilization)
- Grafana dashboards для inference pipeline, RAG retrieval, Celery workers
- Docker Compose services для Prometheus и Grafana

**Новые файлы**: Compose service definitions, Grafana dashboard JSONs, Prometheus config.
**Затрагиваемые файлы**: gateway main.py (metrics middleware).

### 2.6 Observability: LLM-specific tracing

**Текущее состояние**: нет prompt/response logging, нет latency per step, нет cost tracking.

**Целевое**: интеграция Langfuse или Arize Phoenix (self-hosted) для:
- Prompt/response logging с latency breakdown
- Token cost tracking per user / per session
- Retrieval quality tracking
- Minimal code changes: decorator calls в gateway

**Альтернатива**: OpenTelemetry auto-instrumentation (FastAPI + Celery + httpx) для distributed
tracing через Jaeger/Tempo.

### 2.7 CI/CD: hosted workflows

**Текущее состояние**: quality gates только через локальные pre-commit hooks, ruff, pytest.
Нет hosted CI/CD.

**Целевое** (minimum viable CI):
1. Pre-commit hooks run on push
2. `pytest` on push
3. Docker image builds on merge to `main` (validates all Dockerfiles)
4. Optional: push images to container registry (e.g. GitHub Container Registry)

**Новые файлы**: `.github/workflows/ci.yml` (или GitLab CI equivalent).

### 2.8 Database: Alembic migrations

**Текущее состояние**: schema bootstrap через ORM `Base.metadata.create_all` в gateway startup.
Нет version-controlled migrations.

**Целевое**: настроить Alembic для `agent042` database. Генерировать initial migration из
текущей ORM schema. Все последующие schema changes через managed migrations.

**Новые файлы**: `alembic.ini`, `alembic/` directory, initial migration.
**Затрагиваемые файлы**: gateway startup (run migrations on boot).

### 2.9 Token / cost tracking

**Текущее состояние**: нет track-инга token usage. В production LLM системе это критично для
cost control, capacity planning и abuse prevention. Модель `chat_messages` не имеет полей
`prompt_tokens` / `completion_tokens`.

**Целевое**:
- Добавить `token_count` поля в `chat_messages` (prompt_tokens, completion_tokens)
- Парсить usage из vLLM response (OpenAI-compatible API возвращает `usage` в ответе)
- Агрегация per-user / per-session
- Grafana dashboard для token throughput и cost estimation

**Затрагиваемые файлы**: `src/shared/db/`, gateway processing, `chat_messages` schema.

### 2.10 Security: rate limiting and input validation

**Текущее состояние**: нет rate limiting, нет ограничений на длину input.

**Целевое**:
- FastAPI + `slowapi` или Redis-based rate limiter (per-user, per-IP)
- Length limits на user input в prompt builder
- Content/length validation на `rag_sources` request field

**Затрагиваемые файлы**: gateway middleware, `src/gateway/services/processing.py`.

### 2.11 Data validation in Airflow DAGs

**Текущее состояние**: RAG DAGs (download → dvc_version → build_index) скачивают данные и
индексируют без проверки качества. Eval datasets не проходят schema validation.

**Целевое**: добавить validation tasks в Airflow DAGs:
- Record count checks после download
- Schema validation (expected fields, types)
- Embedding dimension verification после индексации
- Quality gates перед promotion

**Затрагиваемые файлы**: `dags/arxiv_rag_update.py`, `dags/pytorch_docs_rag_update.py`,
eval DAGs.

### 2.12 Agent layer: dynamic tool selection (Stage 4)

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

### 2.13 Kubernetes / Helm / Terraform (deferred)

**Статус**: отложено. Docker Compose + Nginx TLS — достаточный production-grade deployment для
single-node setup. Kubernetes оправдан при потребности в horizontal scaling или multi-node HA,
что не является требованием thesis.

Существующие директории `infra/helm/`, `infra/k3s/`, `infra/terraform/` зарезервированы.
Документирование как future work в thesis.
