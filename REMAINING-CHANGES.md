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

---

### 2.1 RAG: reranking

**Текущее состояние**: RAG система не имеет reranking-стадии. Multi-KB results сливаются и
все передаются в промпт. `EvalRun` модель содержит поле `reranking_strategy`, но реализации нет.

**Целевое**: добавить cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) в
retrieval pipeline. Реализовать через factory pattern `get_reranker(strategy)` по аналогии с
`get_chunker()`. Одна строка в `rag_service.py` для включения/отключения. Провести benchmark
champion vs champion+reranker используя существующую eval-платформу.

**Новые файлы**: `src/rag/rerankers.py`.
**Затрагиваемые файлы**: `src/rag/retriever.py`, `src/gateway/services/rag_service.py`.

---

### 2.2 RAG: hybrid search

**Текущее состояние**: только dense retrieval (`VectorParams` с `Distance.COSINE`).

**Целевое**: Qdrant поддерживает sparse+dense fusion нативно. Добавить sparse vectors
(BM25 или SPLADE) при индексации, использовать Qdrant fusion при поиске. Eval framework уже
содержит retrieval benchmarks — чистое A/B сравнение.

**Затрагиваемые файлы**: `src/rag/vector_store.py`, `src/rag/ops/materialize.py`,
`src/rag/retriever.py`.

---

### 2.3 Delivery workflow and CI/CD

**Текущее состояние**: quality gates только через локальные pre-commit hooks, ruff, pytest.
Server deployment по-прежнему завязан на live git checkout: ветки переключаются и pull-ятся на
одном узле, а runtime tree одновременно является и deployment target, и Git workspace.

**Целевое**:
1. Hosted CI: pre-commit и полный `pytest` на push / PR для любой ветки
2. Один deploy entrypoint для same-node deployment: выбрать ветку и развернуть clean release без
  `.git` в runtime tree
3. Manual branch deploy workflow/script с input `branch`, который делает export выбранной ветки в
  fresh release dir вместо ручной серии `git switch` + `git pull` + compose rebuild
4. Optional server-side admin workspace только для Git-mutating operator tasks, но не для runtime
  deployment
5. Optional image builds только как branch-aware optimization поверх clean-release flow
6. Persistent state mounts и ownership cleanup только там, где данные должны переживать release
  replacement

**Детальный план**: `DELIVERY-WORKFLOW-PLAN.md`.

**Новые файлы**: `.github/workflows/ci.yml`, `scripts/deploy_branch.sh`, optional
`scripts/export_release.sh`, optional `.github/workflows/deploy-branch.yml`, optional
`.github/workflows/images.yml`.

---

### 2.4 Observability: Prometheus + Grafana

**Текущее состояние**: нет метрик, нет alerting. Мониторинг ограничен UI dashboards
(Flower, RedisInsight, Airflow) и health checks.

**Целевое**:
- Prometheus metrics endpoint в gateway (request latency, error rates, queue depth)
- vLLM native Prometheus metrics (token throughput, GPU utilization)
- Grafana с двумя datasource: **Prometheus** (инфраструктурные метрики) и **ClickHouse**
  (ML-аналитика — см. 2.6). Разделение принципиально: Prometheus отвечает за SLI/SLO
  инфраструктуры, ClickHouse — за качество модели и тренды eval
- Docker Compose services для Prometheus и Grafana

**Новые файлы**: Compose service definitions, Grafana dashboard JSONs, Prometheus config.
**Затрагиваемые файлы**: `src/gateway/main.py` (metrics middleware).

---

### 2.5 Kafka: inference event streaming

**Текущее состояние**: gateway не сохраняет inference-события как durable event log. После
завершения генерации данные существуют только в `chat_messages` PostgreSQL.

**Целевое**: Gateway публикует событие в топик `inference-events` после каждой завершённой
генерации. Payload: `{query, response, lora_adapter, rag_docs_retrieved, latency_ms, token_counts,
user_id, session_id, timestamp}`. Отдельный топик `feedback-events` зарезервирован для будущих
пользовательских сигналов.

**Важно**: Kafka **не заменяет** RabbitMQ. RabbitMQ — job queue для task dispatch (удаляет
сообщение после забора воркером). Kafka — durable, replayable audit log с несколькими
консьюмерами: ClickHouse consumer (аналитика, см. 2.6) и Spark batch jobs (данные для обучения,
см. 2.8).

**Новые файлы**: `src/gateway/services/event_publisher.py`, Compose service для Kafka + Zookeeper
(или Redpanda как lightweight альтернатива).
**Затрагиваемые файлы**: `src/gateway/services/processing.py` (publish after completion).

---

### 2.6 ClickHouse: OLAP аналитика

**Текущее состояние**: per-sample eval метрики (ROUGE-L, BERTScore, Recall@k по каждому
документу) хранятся в PostgreSQL вместе с транзакционными данными. Inference-аналитика
отсутствует.

**Целевое**: ClickHouse как аналитический backend для двух потоков данных:
- **Inference analytics**: Kafka consumer из топика `inference-events` → ClickHouse таблица
  `inference_log`. Запросы: latency percentiles по LoRA-адаптерам, RAG hit rate динамика,
  token throughput тренды
- **Eval analytics**: per-sample eval метрики (тяжёлые OLAP-запросы — сравнения сотен
  evaluation runs) уходят из PostgreSQL в ClickHouse. Сводная `eval_runs` таблица остаётся
  в PostgreSQL

Grafana получает ClickHouse как второй datasource (см. 2.4).

**Новые файлы**: Compose service, ClickHouse schema DDL, Kafka→ClickHouse consumer
(materialized view или отдельный consumer).
**Затрагиваемые файлы**: `experiments/eval/eval_scripts/` (запись метрик), `src/shared/db/`.

---

### 2.7 Kubernetes: k3s + Helm + KEDA

**Текущее состояние**: `infra/helm/`, `infra/k3s/`, `infra/terraform/` зарезервированы.
Docker Compose — единственный deployment manifest.

**Целевое**: перенести Docker Compose конфигурацию в Helm charts для k3s. Основные задачи:
- Helm chart для каждого сервиса (или umbrella chart для всего стека)
- **KEDA** (Kubernetes Event-Driven Autoscaling) для Celery workers, привязанный к глубине
  очереди RabbitMQ. При queue depth > N k3s автоматически запускает дополнительные воркеры —
  прямая демонстрация заявленной scalability
- GPU resource requests/limits для vLLM pod
- Rolling update при смене LoRA `champion` alias → zero-downtime model swap

Директория `infra/k3s/` становится активной. `infa/helm/` — Helm charts.

**Новые файлы**: Helm chart templates, k3s manifests, KEDA ScaledObject definitions.
**Затрагиваемые файлы**: `infra/helm/`, `infra/k3s/`.

---

### 2.8 Spark: distributed data pipeline в Airflow

**Текущее состояние**: RAG update DAGs (download → dvc_version → build_index) запускают
single-process Python для chunking и обработки. Training data (open-code-instruct,
arxiv-summarization) используется as-is без deduplication / filtering шага. Validation
в DAGs отсутствует. Нет обратной связи от production трафика к RAG pipeline.

**Целевое**: добавить `SparkSubmitOperator` шаг в RAG update и training data DAGs, а также
два новых аналитических Spark job с Airflow расписанием.

*RAG pipeline*: `download → [Spark: dedup + filter + chunk] → write Parquet/S3 → embed → Qdrant upsert`.
Spark job заменяет single-process chunking и добавляет data quality checks (record count gates,
schema validation, embedding dimension verification) как нативные Spark assertions.

*Training data pipeline*: Spark job для дедупликации и фильтрации датасетов перед LoRA
training DAG. Особенно актуально для open-code-instruct (большой объём).

*KB gap detection* (еженедельный Airflow DAG): Spark job читает из Kafka `inference-events`
за скользящее окно 7 дней, фильтрует запросы с нулевым или низким RAG retrieval score,
embeds их тексты (batch inference через embedding microservice), кластеризует (HDBSCAN или
K-Means), авто-лейблит кластеры по top-N query terms. Результат — отчёт в ClickHouse
таблице `kb_gap_report`: кластер, количество запросов, топ-термины, средний retrieval score.
Grafana dashboard показывает динамику по неделям. Оператор видит: "340 запросов о JAX не
нашли релевантных документов" и принимает решение о расширении KB. Замыкает контур обратной
связи production → RAG update pipeline.

*Query drift detection* (еженедельный Airflow DAG): сравнение распределения кластеров
текущей недели с baseline (первые 4 недели после деплоя). Если JS-дивергенция между
распределениями превышает порог — Airflow alert. Сигнализирует о тематическом дрейфе
запросов: пользователи начали спрашивать о том, для чего система не проектировалась.

Spark работает в local/standalone режиме на одном сервере. Архитектура cluster-ready.

**Новые файлы**: `src/spark/rag_preprocessing.py`, `src/spark/training_data_prep.py`,
`src/spark/kb_gap_detection.py`, `src/spark/query_drift_detection.py`,
Compose service для Spark standalone.
**Новые DAGs**: `dags/kb_gap_detection.py`, `dags/query_drift_detection.py`.
**Затрагиваемые файлы**: `dags/arxiv_rag_update.py`, `dags/pytorch_docs_rag_update.py`,
`dags/train_lora.py`.

---

### 2.9 A/B model evaluation framework

**Текущее состояние**: champion/challenger alias система существует в Qdrant и MLflow Model
Registry как инфраструктура. Однако решение о promotion принимается только на основе offline
benchmarks (`eval_runs`). Нет production traffic split, нет статистики по реальным запросам,
nет формального критерия для promotion.

**Целевое**: замкнуть champion/challenger систему production данными.

*Gateway traffic split*: конфигурируемый параметр `challenger_traffic_pct` (default: 0,
включается оператором) маршрутизирует N% запросов к challenger LoRA-адаптеру. Каждое событие
в Kafka `inference-events` содержит поле `ab_variant: "champion" | "challenger"`. Нет
необходимости в отдельной инфраструктуре — split реализуется в одной строке
`task_router.py`.

*ClickHouse аналитика по вариантам*: запросы типа
`SELECT ab_variant, quantile(0.95)(latency_ms), avg(rag_hit_rate) FROM inference_log
GROUP BY ab_variant` дают сравнение вариантов по production метрикам.

*Статистическое решение о promotion*: расширение `experiments/training/lora_ops.ipynb`
новым разделом. Ноутбук запрашивает ClickHouse, запускает Mann-Whitney U-test на
распределениях latency и RAG hit rate, проверяет guardrail метрики (latency и error rate
не должны регрессировать), выводит p-value и рекомендацию. Оператор видит не просто
"challenger лучше на 2% ROUGE-L", а "challenger значимо лучше по rag_hit_rate (p=0.03),
latency не регрессировала (p=0.41), рекомендуется promotion".

Это закрывает разрыв между "у нас есть champion/challenger aliases" и "у нас есть
формальный production-grounded процесс promotion".

**Новые файлы**: расширение `experiments/training/lora_ops.ipynb` (A/B decision section).
**Затрагиваемые файлы**: `src/gateway/services/task_router.py`,
`src/gateway/services/processing.py` (ab_variant в event payload), `src/shared/config.py`
(challenger_traffic_pct setting).

---

### 2.10 Observability: LLM-specific tracing

**Текущее состояние**: нет prompt/response logging, нет latency per step, нет cost tracking.

**Целевое**: интеграция Langfuse или Arize Phoenix (self-hosted) для:
- Prompt/response logging с latency breakdown по шагам (RAG retrieval / prompt build / generation)
- Token cost tracking per user / per session
- Retrieval quality tracking

**Альтернатива**: OpenTelemetry auto-instrumentation (FastAPI + Celery + httpx) для distributed
tracing через Jaeger/Tempo.

---

### 2.11 Token / cost tracking

**Текущее состояние**: online inference stores prompt/completion usage in the ORM model and
threads prompt token counts through the sync/async gateway paths. Существующие `agent042` БД
применяют `src/shared/db/chat_messages_add_usage_columns.sql`, чтобы схема `chat_messages`
включала эти колонки.

**Целевое**:
- Применить `src/shared/db/chat_messages_add_usage_columns.sql` на все существующие `agent042` БД
- Агрегация per-user / per-session
- Grafana dashboard для token throughput и cost estimation (из ClickHouse inference_log)

**Затрагиваемые файлы**: `src/shared/db/`, gateway processing, `chat_messages` schema.

---

### 2.12 Security: rate limiting and input validation

**Текущее состояние**: нет rate limiting, нет ограничений на длину input.

**Целевое**:
- FastAPI + `slowapi` или Redis-based rate limiter (per-user, per-IP)
- Length limits на user input в prompt builder
- Content/length validation на `rag_sources` request field

**Затрагиваемые файлы**: gateway middleware, `src/gateway/services/processing.py`.

---

### 2.13 Database: Alembic migrations

**Текущее состояние**: schema bootstrap через ORM `Base.metadata.create_all` в gateway startup.
Нет version-controlled migrations.

**Целевое**: настроить Alembic для `agent042` database. Генерировать initial migration из
текущей ORM schema. Все последующие schema changes через managed migrations.

**Новые файлы**: `alembic.ini`, `alembic/` directory, initial migration.
**Затрагиваемые файлы**: gateway startup (run migrations on boot).

---

### 2.14 Agent layer: dynamic tool selection

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
