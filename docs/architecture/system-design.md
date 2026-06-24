# Разработка и исследование интеллектуального ассистента для исследователей с использованием генерации на основе поиска и эффективного дообучения моделей

## Содержание

1. [Введение](#1-введение)
2. [Высокоуровневая архитектура](#2-высокоуровневая-архитектура)
3. [Каталог сервисов](#3-каталог-сервисов)
4. [Инференс-пайплайн (runtime)](#4-инференс-пайплайн-runtime)
5. [RAG-система](#5-rag-система)
6. [LoRA fine-tuning pipeline](#6-lora-fine-tuning-pipeline)
7. [Конфигурационная архитектура](#7-конфигурационная-архитектура)
8. [MLOps и автоматизация](#8-mlops-и-автоматизация)
9. [Оценка качества (Evaluation)](#9-оценка-качества-evaluation)
10. [Инфраструктура и деплой](#10-инфраструктура-и-деплой)
11. [Безопасность](#11-безопасность)
12. [Структура репозитория](#12-структура-репозитория)
13. [Итоги и выводы](#13-итоги-и-выводы)

---

## 1. Введение

### 1.1 Цель проекта и позиционирование

Целью данного проекта является разработка интеллектуального ассистента для исследователей в области ML/AI с двумя ключевыми техническими компонентами: системой Retrieval-Augmented Generation (RAG) и pipeline'ом эффективного дообучения языковой модели на основе метода LoRA (Low-Rank Adaptation).

Запустить LLM и получить от неё ответы — тривиальная задача. Прикрутить к LLM базовый RAG — задача одного вечера. Настроить простой LoRA training pipeline — ещё один вечер. Однако поднять полноценную production-систему с надёжным хранением данных, воспроизводимыми экспериментами, автоматизированными пайплайнами обучения и оценки, прозрачным мониторингом и отлаженными процессами CI/CD — это оставшиеся 99% усилий. Данный проект сосредоточен именно на этих 99%.

Система разработана в парадигме **single-repository**: весь код — runtime-сервисы, инфраструктура, эксперименты, operator workflows и тесты — живёт в одном репозитории. Это обеспечивает единую точку входа для разработчика и операционного инженера, упрощает рефакторинг и гарантирует консистентность между экспериментальным и production-кодом.

**Целевая аудитория** — команды исследователей, которым нужен AI-ассистент для работы с собственными (в том числе конфиденциальными) базами знаний. Система разворачивается на выделенном сервере организации, что обеспечивает полный контроль над данными и позволяет работать с NDA-материалами.

### 1.2 Ключевые проектные решения

* **Single-repository** - единая точка входа
* **Docker Compose (single-node)** - система контейнеризации для запуска в режиме single-node
* **vLLM как inference engine** - OpenAI-совместимый API, hot-loading LoRA адаптеров, высокая производительность
* **Alias-based управление RAG и LoRA** - Возможность тестировать различные конфигурации без downtime и с возможностью откатиться
* **Celery + Redis pub/sub для streaming** - Отвязка latency инференса от HTTP-соединения, снижение TTFT (time to first token), поддержка длинных ответов
* **Airflow для автоматизации** - Основной бэкенд для стандартизованных и периодичных задач (обновление RAG, бенчмарки, пайплайн обучения LoRA)

---

## 2. Архитектура и каталог сервисов

### 2.1 Платформа инференса

| Сервис | Технология | Роль |
|---|---|---|
| `gateway` | FastAPI + uvicorn | API Gateway: аутентификация, task routing, RAG, сборка промпта, async dispatch |
| `vllm` | vLLM | Inference engine с OpenAI-совместимым API и hot-loading LoRA адаптеров |
| `celery-worker` | Celery + RabbitMQ | Асинхронный inference: стримит токены обратно через Redis pub/sub |
| `vllm-adapter-sync` | Python | Синхронизация LoRA артефактов из MLflow Model Registry в vLLM |
| `embeddings` | FastAPI (sentence-transformers) | Сервис dense и sparse embeddings |
| `reranker` | FastAPI (cross-encoder) | Cross-encoder reranking для улучшения качества RAG |
| `qdrant` | Qdrant | Векторное хранилище для RAG-коллекций |
| `redis` | Redis | Сессии, pub/sub для streaming-ответов, вспомогательное состояние |
| `rabbitmq` | RabbitMQ | Брокер сообщений для Celery task queue |
| `postgres` | PostgreSQL | Пользовательские данные, история диалогов, backend Airflow и MLflow |
| `ui` | Streamlit | Пользовательский интерфейс с Google-авторизацией и streaming-инференсом |
| `nginx` | nginx | TLS termination, reverse proxy, маршрутизация в UI и Gateway |

### 2.2 Платформа экспериментов

| Сервис | Технология | Роль |
|---|---|---|
| `airflow-webserver / scheduler / dag-processor` | Apache Airflow | Оркестрация пайплайнов |
| `airflow-worker` | Celery (CPU) | Бенчмарки, RAG-обновления, оценка качества |
| `airflow-worker-gpu` | Celery (GPU) | Обучение LoRA адаптеров |
| `rag-ops` | one-shot CLI container | Ручные RAG build/materialize/promote операции внутри Docker network |
| `jupyter` | JupyterLab | Интерактивные эксперименты и operator workflows |
| `mlflow` | MLflow | Трекинг экспериментов, Model Registry |
| `code-sandbox` | Docker (изолированный) | Безопасное выполнение кода при code evaluation |

### 2.3 Платформа мониторинга

| Сервис | Роль |
|---|---|
| `prometheus` | Сбор технических метрик со всех сервисов |
| `grafana` | Дашборды инфраструктурной observability и ML-процессов |
| `flower` | Мониторинг Celery workers и очередей задач |
| `redisinsight` | Инспекция Redis-ключей и pub/sub активности |

---

## 4. Инференс-пайплайн (runtime)

В этом разделе описан полный путь пользовательского запроса от браузера до ответа модели.

### 4.1 Аутентификация (Google OAuth2 / OIDC + PKCE)

Система использует Google OAuth 2.0 с PKCE (Proof Key for Code Exchange) — рекомендуемый стандарт для веб-приложений, защищающий от атак перехвата кода авторизации.

**Поток аутентификации:**

1. Неаутентифицированный запрос попадает в `AuthMiddleware` Gateway.
2. Gateway генерирует `code_verifier` (случайная строка), `code_challenge` (SHA-256 хэш verifier'а) и `state` (CSRF-токен).
3. Пользователь перенаправляется на Google Authorization Endpoint.
4. После успешного входа Google возвращает `code` на callback URL Gateway.
5. Gateway обменивает `code` + `code_verifier` на ID Token и Access Token через Google Token Endpoint.
6. ID Token верифицируется через Google JWKS (публичные ключи). Из него извлекается `email` и `sub` пользователя.
7. Создаётся сессия: уникальный `session_id` сохраняется в Redis с TTL. Cookie с `session_id` устанавливается в браузере.

**Хранение сессий:**
Сессии хранятся в Redis. Streamlit UI при каждом запросе передаёт cookie `session_id` в Gateway, который верифицирует сессию через Redis. При отсутствии или истечении сессии пользователь перенаправляется на повторный вход.

### 4.2 Task Routing — классификация типа задачи

После аутентификации Gateway определяет тип задачи: `chat`, `code` или `summarize`. Это влияет на выбор RAG-коллекции и LoRA-адаптера.

**Embedding-based routing:**
Основной метод — `EmbeddingTaskRouter`. Для каждой задачи в catalog (`catalog.toml`) задано `routing_description` — текстовое описание задачи. При инициализации Gateway вычисляет эмбеддинги всех `routing_description` и кэширует их. При запросе:

1. Вычисляется эмбеддинг последнего сообщения пользователя.
2. Считается косинусное сходство с эмбеддингами каждой задачи.
3. Выбирается задача с наибольшим сходством. Если сходство ниже порогового значения (`task_classification_threshold`) — роутер возвращает `chat` как безопасный fallback.

Примеры `routing_description` из конфигурации:
- `chat`: "Open-ended ML/DL/AI/LLM research discussion, conceptual explanation..."
- `code`: "Programming help for ML systems: writing code, debugging tracebacks..."
- `summarize`: "Summarize or condense user-provided content into a shorter form..."

### 4.3 RAG Retrieval — поиск в базах знаний

На основе определённой задачи Gateway запрашивает соответствующую базу знаний через `RAGService`. Детальное описание — в разделе 5.

Gateway поддерживает три режима RAG:

- `auto` — автоматический выбор базы знаний по задаче.
- `explicit` — пользователь явно указывает базу знаний (для eval-пайплайна).
- `off` — RAG отключён (для задачи `summarize` нет баз знаний).

### 4.4 Сборка промпта и token budget management

`PromptBuilder` собирает итоговый промпт из системного промпта, истории диалога, RAG-контекста и текущего сообщения пользователя. Критически важна задача бюджетирования токенов: контекстное окно модели ограничено, нужно уместить все части с нужными приоритетами.

**Бюджеты токенов (значения по умолчанию):**

| Бюджет | Значение | Описание |
|---|---|---|
| `model_max_tokens` | 32 768 | Размер контекстного окна модели |
| `budget_guard` | 512 | Резерв на overhead chat-template |
| `budget_system` | 768 | Системный промпт |
| `budget_turn` | 10 240 | Текущий запрос пользователя |
| `min_budget_history` | 4 096 | Минимум для истории диалога |
| `budget_rag` | 6 144 | RAG-контекст |
| `min_response_budget` | 256 | Минимум для ответа модели |

#### Приближённый подсчёт токенов (символьная эвристика)

На этапе сборки промпта в Gateway вызов реального токенизатора для каждого входящего запроса был бы дорогостоящим сетевым обращением к vLLM. Поэтому все проверки и обрезки на стороне Gateway используют **символьную аппроксимацию**: функция `estimate_tokens(text, chars_per_token)` вычисляет `⌈len(text) / chars_per_token⌉`. Параметр `chars_per_token` по умолчанию равен `4.0` (характерное значение для латинского текста). Использование `ceil` намеренно консервативно — Gateway предпочитает недооценить бюджет, чем переполнить контекстное окно.

#### Точный подсчёт токенов (реальный токенизатор)

Перед непосредственной генерацией Celery worker отправляет уже собранный промпт в vLLM через `/tokenize` и получает точное количество токенов `prompt_tokens`. На основе этого вычисляется окончательный бюджет ответа:

```
max_tokens = model_max_tokens - prompt_tokens - budget_guard
```

Если запрошен явный `max_tokens` пользователем, берётся `min(budget_cap, requested_max_tokens)`. Если оставшийся бюджет меньше `min_response_budget` — выбрасывается `ResponseBudgetExceededError` и генерация не начинается.

#### Алгоритм формирования промпта

**Шаг 1 — текущий запрос.** Оценивается длина текущего сообщения пользователя. Если `current_turn_tokens > budget_turn` → немедленная `BudgetValidationError`.

**Шаг 2 — динамический бюджет истории.** Эффективный бюджет истории вычисляется как:

```
history_budget = min_budget_history + (budget_turn - current_turn_tokens)
```

Если запрос короткий, неиспользованная часть `budget_turn` передаётся истории. Это обеспечивает, что короткие запросы могут «видеть» больше контекста диалога.

**Шаг 3 — обрезка истории.** `trim_history_pairs()` обходит историю **с конца** (от новых сообщений к старым). Сообщения группируются в «единицы»: пара `user + assistant` всегда обрабатывается совместно, чтобы не разрывать завершённые обмены. Одиночные сообщения (без пары) обрабатываются как отдельные единицы. Единицы добавляются к результату, пока не исчерпан бюджет; первая не уместившаяся единица и все более старые — отбрасываются.

**Шаг 4 — системный промпт.** Оценивается размер системного промпта (включая все `system`-сообщения клиента). Если `system_tokens > budget_system` → `BudgetValidationError`.

**Шаг 5 — обрезка RAG-чанков.** `trim_rag_chunks()` делит `budget_rag` **поровну** между источниками: `section_budget = budget_rag // num_sources`. Внутри каждого источника чанки берутся в порядке убывания релевантности (как вернул retriever) до исчерпания `section_budget`. Заголовок секции (`### Knowledge Base: ...`) учитывается в бюджете источника.

**Шаг 6 — финальный промпт.** Возможны три варианта системного промпта в зависимости от результата RAG:

- RAG вернул чанки → в системный промпт добавляется блок `--- RETRIEVED CONTEXT ---` с отформатированными документами (имя базы знаний, alias, источник, score, текст).
- RAG запрашивался, но ничего не найдено → добавляется текст `(No relevant context was found in the knowledge base for this query.)`.
- RAG не запрашивался (задача `summarize` или режим `off`) → системный промпт без изменений.

#### Два типа ошибок бюджета

| Ошибка | Где возникает | Причина |
|---|---|---|
| `BudgetValidationError` | Gateway, сборка промпта | Текущий запрос или системный промпт превышают символьный бюджет |
| `ResponseBudgetExceededError` | Celery worker, перед генерацией | Точный размер промпта оставляет меньше `min_response_budget` токенов для ответа |

`BudgetValidationError` срабатывает до отправки задачи в Celery и возвращает HTTP-ошибку клиенту немедленно. `ResponseBudgetExceededError` срабатывает уже в worker'е — задача завершается с ошибкой, которая передаётся клиенту через Redis pub/sub.

Системный промпт варьируется по задаче:
- `chat`: базовый промпт исследовательского ассистента.
- `code`: акцент на корректных, запускаемых решениях.
- `summarize`: инструкция на структурированное сжатие.

Если RAG нашёл релевантные чанки, они добавляются в системный промпт в виде секций `[Source: <kb_name>]`. Если RAG включён, но чанки не найдены — добавляется явное предупреждение об отсутствии контекста.

### 4.5 Асинхронный инференс: Celery + RabbitMQ

Gateway не обращается к vLLM напрямую. Вместо этого:

1. Gateway отправляет задачу в RabbitMQ через Celery.
2. Celery worker получает задачу и открывает streaming-соединение с vLLM (`/v1/chat/completions` с `stream=true`).
3. Каждый полученный токен worker публикует в Redis-канал с уникальным `request_id`.
4. Gateway подписывается на Redis-канал и проксирует токены в браузер через Server-Sent Events (SSE).

**Обработка thinking-токенов:**
Если модель поддерживает режим "extended thinking" (теги `<think>...</think>`), worker разделяет поток на `thinking_token` и `answer_token` события. UI отображает thinking-контент в отдельном раскрывающемся блоке "💭 Thinking...".

**Детектирование зацикливания:**
Worker отслеживает повторяющиеся последовательности символов (регулярное выражение на последних 1024 символах ответа). При обнаружении зацикливания генерация прерывается и пользователю сообщается об усечении.

**Response token budget:**
Перед генерацией worker запрашивает у vLLM количество токенов промпта через `/tokenize` и вычисляет допустимый `max_tokens` для ответа: `model_max_tokens - prompt_tokens - budget_guard`. Это предотвращает усечение ответа из-за переполнения контекстного окна.

---

## 5. RAG-система

### 5.1 Концептуальная модель

RAG в проекте разделён на несколько независимых понятий:

- **KB id** — логическая база знаний из `catalog.toml`, например
  `ml_papers_core` или `pytorch_reference`.
- **Source adapter** — catalog-declared source lifecycle behavior from
  `[[source_adapters]]`. Adapter identity is the behavior selector; source
  manifests do not carry a separate `source_type` behavior field.
- **Benchmark adapter** — catalog-declared adapter from `[[benchmark_adapters]]`
  that implements the normal source lifecycle plus benchmark preparation.
- **Source instance** — globally meaningful source id, for example
  `ml_papers_core.papers` or `pytorch_reference.docs`.
- **Source role** — `role = "corpus"` participates in normal KB builds;
  `role = "benchmark"` produces benchmark cases/labels and is excluded from
  normal materialization.
- **Source manifest** — curated document list or adapter-specific config at
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
- **Release** — immutable, content-addressed build result. Its id
  (`ragrel_<kb>_<16hex>`) and Qdrant collection name (`rag__<kb>__<16hex>`)
  are derived from a fingerprint of the build config, source declaration,
  and source snapshot — not a timestamp. Release manifest JSON provenance
  lives at `assets/rag_data/knowledge_bases/<kb>/releases/<release_id>.json`.
- **Alias deployment** — the Postgres row (`rag_alias_deployments`) recording
  which release is active for a (kb_id, alias) pair. This, not the Qdrant
  alias, is the runtime serving source of truth.
- **Physical collection** — реальная Qdrant collection вида
  `rag__<kb_id>__<16-hex fingerprint>`. Имя физической коллекции не содержит
  alias.
- **Qdrant alias** — runtime pointer вида `rag__<kb_id>__<alias>`, например
  `rag__pytorch_reference__champion`. Зеркало applied state, обновляется
  после активации deployment в Postgres.
- **Release attestation** — компактная запись metadata внутри collection
  (schema version 2), позволяющая runtime/cleanup проверить release id, KB
  id, collection name, manifest id, encoder identity и retrieval capability.

Связи:

```text
Task 1-to-many KB
KB 1-to-many SourceInstance
SourceInstance -> SourceAdapter | BenchmarkAdapter
KB 1-to-many AliasConfig
AliasConfig -> AliasService.diff/apply -> Release (Postgres rag_releases)
Release -> Alias deployment (Postgres rag_alias_deployments) -> Qdrant alias (mirror)
Release -> Qdrant attestation -> Release manifest
```

Source instance ids are global. Legacy `[[sources]]` and KB-local `--source`
selectors are removed; operator workflows use `[[source_instances]]` and global
`--source-instance` values.

### 5.2 Архитектура retrieval

Retrieval pipeline реализован в `src/rag/` и состоит из четырёх слоёв:

```
Запрос
  │
  ▼
[Embedding / Sparse Encoding]
  │
  ▼
[LlamaIndex VectorStoreIndex + QdrantVectorStore]
  ── dense / sparse / hybrid search
  │
  ▼
[Reranker] (опционально) ── cross-encoder re-scoring
  │
  ▼
[Score threshold filtering]
  │
  ▼
Релевантные документы (top_k)
```

**Стратегии поиска:**

- **Dense retrieval** — поиск по векторному расстоянию (cosine) от dense эмбеддинга запроса. Модель эмбеддингов: `sentence-transformers/all-MiniLM-L6-v2` (по умолчанию).
- **Sparse retrieval** — поиск по разреженным векторам (BM25). В проекте используется модель `Qdrant/bm25` из библиотеки fastembed. Подходит для точных терминологических запросов.
- **Hybrid retrieval** — комбинация dense и sparse поиска через Reciprocal Rank Fusion (RRF). Обеспечивает баланс между семантическим и лексическим поиском.

**Reranking:**
При включённом reranker'е первый этап извлекает `top_k × reranker_multiplier` кандидатов (с расширенным порогом), второй этап пересортировывает их cross-encoder'ом (`cross-encoder/ms-marco-MiniLM-L-6-v2`), после чего применяется финальный score threshold.

**Chunking:**
Current chunking uses LlamaIndex `SentenceSplitter` through
`src/rag/sources/chunks.py`. Source adapters and extractors emit LlamaIndex
`Document`; node artifacts persist native `TextNode` objects. `TextNode.id_` is
a deterministic UUID used as the Qdrant point id, while the readable
`chunk_id` remains in node metadata. Project `Document` / `Chunk` contracts are
not part of the active source/build path.

### 5.3 Базы знаний и источники

Система содержит две базы знаний, каждая из которых привязана к типу задачи:

**`ml_papers_core`** (задача `chat`)
- Содержимое: curated full-text ML/AI papers.
- Source instance: `ml_papers_core.papers`.
- Manifest: `assets/rag_data/source_instances/ml_papers_core.papers/manifest.toml`.
- Стратегия обновления: **replace** — новая physical collection собирается
  целиком и затем может быть продвинута через alias.
- Активный champion: dense retrieval, `top_k=5`, `score_threshold=0.35`.
- Challenger-конфиг: hybrid retrieval с cross-encoder reranking.

**`pytorch_reference`** (задача `code`)
- Содержимое: официальная документация PyTorch.
- Source instance: `pytorch_reference.docs`.
- Manifest: `assets/rag_data/source_instances/pytorch_reference.docs/manifest.toml`.
- Стратегия обновления: **replace** — при каждом обновлении коллекция
  пересоздаётся полностью (документация версионируется целиком).
- Идентичная схема champion/challenger.

Задача `summarize` не использует базы знаний — модель работает исключительно с содержимым, предоставленным пользователем.

### 5.4 Source/build pipeline

Source/build lifecycle is split across production modules:

- `src/rag/adapters/` — adapter contracts, catalog factory loading, and source
  implementations;
- `src/rag/sources/` — generic source manifests, fetch/extract/chunk artifacts,
  source builds, benchmark preparation, and source bundle collection;
- `src/rag/indexing/` — LlamaIndex Qdrant materialization, collection metadata,
  collection manifests, and immutable release attestation;
- `src/rag/control_plane/` — content-addressed release builder
  (`release_builder.py`), fingerprint helpers (`fingerprints.py`), the
  Postgres-backed release/deployment registry (`repositories.py`,
  `postgres.py`), and `AliasService` (`alias_service.py`), the diff/apply
  reconciliation engine;
- `src/rag/cli/` — the `rag` Typer CLI (`catalog`, `alias`, `release`,
  `benchmark`, `source` command groups), the only supported operator
  entrypoint.

The data flow for `rag alias apply` is:

```text
catalog.toml alias declaration (build + retrieve config)
  -> AliasService.diff(): compare against Postgres applied state
  -> on drift: fetch raw artifacts (cache permitting)
  -> extract LlamaIndex Document artifacts
  -> parse native TextNode artifacts
  -> collect SourceNodeBundle
  -> VectorStoreIndex + LlamaIndex QdrantVectorStore, named by content fingerprint
  -> write immutable release manifest + insert rag_releases row
  -> insert/activate rag_alias_deployments row, update the Qdrant alias
```

A release is reused across this flow whenever its build config and source
declaration fingerprint match an existing, non-retired `rag_releases` row —
fetching/materializing only happens on an actual cache or fingerprint miss.

Политика артефактов:

- Source instance `manifest.toml` files stay in Git as curated operator input.
- Raw cache (`assets/rag_data/source_instances/<source_instance_id>/raw/`) immutable по умолчанию и не
  DVC-tracked; force flags нужны для повторного fetch/extract/chunk.
- Generated source-instance artifacts and knowledge-base manifests/metadata can
  be synchronized through DVC when needed.
- Source manifests are generic. Adapter-specific validation, document listing,
  fetcher selection, and extractor selection live behind the source adapter.
  Ordinary new datasets should not require editing generic manifest unions or
  source-type enums.

Benchmark source instances use the same source-instance identity model but
`role = "benchmark"`. `prepare-benchmark` writes normalized benchmark artifacts:

```text
assets/rag_data/source_instances/<benchmark_source_instance_id>/benchmark/
  corpus.jsonl
  cases.jsonl
  labels.jsonl
  metadata.json
```

Benchmark results are not stored as report files; Postgres `eval_runs` and
`eval_samples.detail` are the source of truth.

### 5.5 Declarative alias control plane (desired vs. applied state, паттерн champion / challenger)

Ключевой механизм для управления качеством RAG — declarative reconciliation
между `catalog.toml` (desired state) и Postgres (`rag_releases`,
`rag_alias_deployments` — applied state), по аналогии с Terraform/Kubernetes.
Qdrant alias — лишь зеркало applied state, не источник истины: runtime
resolution и cleanup решают liveness через Postgres, а не через то, на что
сейчас указывает Qdrant alias.

Физическая коллекция — immutable, content-addressed release:
`rag__<kb_id>__<16-hex fingerprint>` (не timestamp). Тот же fingerprint
(build config + source declaration + source snapshot) всегда резолвится в тот
же `release_id`/collection name, поэтому одна и та же release может быть
переиспользована несколькими alias'ами без пересборки.

```
rag_releases (Postgres):
  ragrel_pytorch_reference_<fp1>  ── collection rag__pytorch_reference__<fp1>
  ragrel_pytorch_reference_<fp2>  ── collection rag__pytorch_reference__<fp2>

rag_alias_deployments (Postgres, applied state):
  (pytorch_reference, champion)   -> active   -> ragrel_pytorch_reference_<fp1>
  (pytorch_reference, challenger) -> active   -> ragrel_pytorch_reference_<fp2>

Qdrant aliases (mirror, updated by AliasService._activate()):
  rag__pytorch_reference__champion   -> rag__pytorch_reference__<fp1>
  rag__pytorch_reference__challenger -> rag__pytorch_reference__<fp2>
```

Activation (`AliasService.apply()`) сначала пишет/обновляет
`rag_alias_deployments`, затем обновляет Qdrant alias — атомарная для
runtime операция (Postgres решает what serves traffic), не требующая
перезапуска Gateway. В catalog каждый alias задаёт `build` (chunking,
dense/sparse encoder) и `retrieve` (`top_k`, `score_threshold`,
`strategy`, `reranker`) профили. Runtime резолвит alias через активный
deployment, не через collection metadata attestation.

Совместимость alias `retrieve.strategy` и release capability:

- dense-запрос к dense release — разрешён;
- dense-запрос к hybrid release — разрешён;
- hybrid/sparse-запрос к hybrid release — разрешён;
- hybrid/sparse-запрос к dense-only release — отклоняется `AliasService`
  при apply (`AliasApplyError`), retrieval не достигается.

UI может использовать `default_alias`; API/eval могут явно запросить любой
declared alias KB. Это даёт основу для champion/challenger сравнений и будущих
A/B тестов. `default_alias` защищён evaluation-coverage gate: apply
отказывается собрать/активировать неоцененный release для default alias без
явного `--allow-build-default`/`--allow-unevaluated`.

**Reconciliation workflow:**

1. **Diff** (`rag alias diff <kb> <alias>`) — desired (catalog digest) vs.
   applied (Postgres digest) comparison, без side effects.
2. **Apply** (`rag alias apply <kb> <alias>`) — резолвит/собирает (fetch при
   cache miss или `--refresh-sources`) release через `AliasService`,
   проверяет provider identity и retrieval compatibility, активирует
   deployment, обновляет Qdrant alias.
3. **Benchmark** (`rag benchmark run ...`) — прогон attached benchmarks
   против активного release alias'а; результаты пишутся в `eval_runs`/
   `eval_samples` с `rag_release_id`/`alias_deployment_id`.
4. **Inspection** (`rag release list/show`, `rag alias status`) — сравнение
   declared/applied state, без прямого обращения к Qdrant.
5. **Cleanup** (`dags/rag_collection_cleanup.py`, `@daily`) — retire release
   в Postgres перед удалением её Qdrant collection; никогда не удаляет
   collection без соответствующей retired `rag_releases` строки.

LlamaIndex-built collections are serving-compatible: runtime resolves the
active deployment's release, validates provider identity, reopens
`VectorStoreIndex.from_vector_store()`, and queries native nodes. Alias
activation remains a project-owned, Postgres-then-Qdrant operation inside
`AliasService`.

### 5.6 Runtime retrieval и observability

Gateway вызывает `RAGService`, который делегирует lookup в
`rag.runtime.RagRuntime`. Runtime выполняет:

1. normalizes requested `(knowledge_base, alias)`;
2. находит active `AliasDeployment` в Postgres (`rag_alias_deployments`) —
   не Qdrant alias, который остаётся лишь зеркалом;
3. резолвит deployment's `RagRelease` (`rag_releases`) и проверяет, что его
   Qdrant collection существует и несёт matching schema-version-2 release
   attestation (`compare_release_attestation`);
4. проверяет live embedding provider identity (`embedding_service.model`,
   не статичный `runtime.toml` config) и vector dimension против release;
5. reopens the LlamaIndex vector index and retrieves `NodeWithScore` objects;
6. applies optional project reranking and score filtering as LlamaIndex node
   postprocessors;
7. maps nodes to citation-ready compatibility hits and observability payload.

Любое несовпадение (missing release, missing/mismatched collection,
attestation mismatch, provider identity drift) — `fail-closed` для этого
KB/alias: `strict` режим поднимает исключение, non-strict пропускает источник
и помечает его в `skipped_sources`.

`RagRuntime.query()` is the standalone generation path for benchmarks and
future inference integration. It uses `RetrieverQueryEngine`, returns answer
plus source nodes, and records `prompt_id`, `prompt_version`, `prompt_digest`,
and prompt parameters. Gateway chat keeps its existing streaming generation
path and consumes the compatibility retrieval mapping.

`RagRuntimeResult` содержит:

- `hits` — документы для prompt builder / citations;
- `skipped_sources` — KB/alias пары, которые нельзя было запросить;
- `provenance` — selected KB/alias, Qdrant alias, physical collection,
  manifest id, retrieval strategy/capability, score summary, hit count,
  no-hit flag и per-source timings;
- `timings_ms` — total runtime retrieval latency;
- `diagnostics` — requested/resolved/skipped source counts, total hit count,
  no-hit flag.

Gateway пишет эти поля в structured logs. Eval pipeline также сохраняет
resolved collection и manifest id, чтобы сравнения champion/challenger были
traceable.

### 5.7 Валидация конфигурации при старте

При запуске Gateway автоматически валидирует catalog: проверяется наличие Qdrant-коллекций для всех объявленных aliases. Если коллекция не найдена, Gateway либо завершается с ошибкой (при `RAG__RAG_STRICT_STARTUP=true`), либо логирует предупреждение и продолжает работу.

---

## 6. LoRA Fine-Tuning Pipeline

### 6.1 Стек и архитектура

Fine-tuning реализован поверх следующего стека:

| Компонент | Библиотека | Роль |
|---|---|---|
| PEFT | `peft` | Реализация LoRA, управление адаптерами |
| Training loop | PyTorch Lightning | Абстракция над GPU, gradient accumulation, checkpointing |
| Config management | Hydra | Иерархическое конфигурирование экспериментов |
| Experiment tracking | MLflow | Логирование метрик, параметров и артефактов |
| Model registry | MLflow | Версионирование и alias-based продвижение адаптеров |
| Data versioning | DVC | Версионирование датасетов через Yandex Cloud S3 |

**Целевая модель (текущая конфигурация):** Qwen3-0.6B — компактная модель для обучения на RTX 3060 12GB.

**LoRA-конфигурация:**
- Rank `r=8`, `lora_alpha=16`, `lora_dropout=0.05`.
- Target modules: все проекционные слои attention и feed-forward (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Квантизация: 4-bit NF4 (bitsandbytes) с double quantization — позволяет обучать модель при ограниченном VRAM.
- Mixed precision: fp16.

**Обучающие данные (текущая конфигурация):** `arxiv-summarization` датасет. Задача — суммаризация научных статей. Формат промпта: `Summarize the following article into an abstract: [article] Abstract:`. Токены промпта исключены из loss (параметр `train_on_inputs: false`).

### 6.2 Конфигурирование экспериментов (Hydra)

Конфиги хранятся в `experiments/training/conf/`. Иерархия:

```
conf/
  config.yaml              # Базовый config (defaults: task, dataset, model, lora, data, training, ...)
  experiment/
    arxiv_summarization.yaml   # Optional thin preset: overrides top-level groups
    open_code_instruct_qwen.yaml
  task/
    summarization.yaml
    coding_sft.yaml
  dataset/
    arxiv_summarization.yaml
    open_code_instruct.yaml
  model/
    qwen3_0_6b.yaml
  lora/
    qwen_attention_mlp.yaml
  data/
    sft_768_tokens.yaml
  training/
    adapter_default.yaml
  scheduler/
    linear_warmup.yaml
  paths/
    paths_config.yaml      # Пути проекта (project_root)
```

Параметры теперь разделены по ответственности: задача, датасет, модель, LoRA, data budget,
training и scheduler выбираются как top-level Hydra groups. Optional `experiment/*.yaml`
пресеты лишь переопределяют эти группы одной строкой. Переопределение через CLI:
`python -m ... training.lr=2e-5 lora.r=16` или `python -m ... +experiment=open_code_instruct_qwen`.

Scheduler: linear warmup (100 шагов, start_factor=0.05) с последующим линейным decay. Gradient accumulation: 8 шагов (effective batch size = 8 при batch_size=1). Gradient clipping: 1.0.

#### Способы запуска обучения

Обучение можно запустить двумя способами:

**Через Airflow DAG** (`dags/train_lora.py`) — основной production-способ. Обучение LoRA на GPU занимает значительное время, и держать открытую консоль или JupyterLab-сессию на всё это время непрактично. Airflow DAG работает по принципу «запустил и забыл»: задача ставится в очередь GPU worker'а через RabbitMQ, выполняется в фоне, а результаты (путь к `training_summary.json`, `run_id`) доступны в Airflow UI после завершения. Параметры передаются через интерфейс Airflow: optional experiment preset и произвольные Hydra-переопределения в виде JSON-списка. Если DAG завершается с ошибкой, полный лог тренировки сохраняется в Airflow.

**Через CLI** (`python -m experiments.training.train_adapter.start_train ...`) — вспомогательный способ для быстрых локальных экспериментов. Удобен при отладке конфигурации или запуске на локальной машине с GPU, когда нет доступа к Airflow.

В обоих случаях используется один и тот же training entry point, поэтому результаты воспроизводимы.

### 6.3 Трекинг экспериментов и Model Registry (MLflow)

Каждый training run логирует в MLflow:
- Все Hydra-параметры эксперимента.
- Метрики: `train_loss`, `val_loss`, `learning_rate`, `tokens_per_second`, `gpu_memory_allocated_mb`, `zero_target_ratio`.
- Артефакты: веса адаптера, `training_summary.json` с лучшей метрикой и путём к checkpoint'у.

После завершения обучения run доступен в MLflow UI. Оператор инспектирует результаты в `experiments/training/lora_ops.ipynb` и принимает решение о промоушене:

```python
# В lora_ops.ipynb:
registry.register("lora-summarize", run_id="...", version_tag="v2")
registry.promote("lora-summarize", version=2, alias="challenger")
# После валидации:
registry.promote("lora-summarize", version=2, alias="champion")
```

### 6.4 Adapter Sync и Hot-Loading в vLLM

`vllm-adapter-sync` — выделенный сервис, который периодически:

1. Запрашивает MLflow Model Registry все адаптеры с aliases `champion` и `challenger`.
2. Скачивает недостающие версии в локальную директорию: `/adapters/{model_name}/v{version}/model/`.
3. Загружает новые адаптеры в vLLM через `/v1/load_lora_adapter` API.
4. Выгружает устаревшие адаптеры через `/v1/unload_lora_adapter` API.

Адаптеры именуются в vLLM как `{model_name}-{alias}`, например `lora-summarize-champion`. Gateway при выборе адаптера обращается к вLLM по этому имени. Процесс не требует перезапуска vLLM.

---

## 7. Архитектура конфигов

### 7.1 Принцип единственного владельца (Config Ownership)

Каждый аспект конфигурации системы описан ровно в одном месте:

| Конфиг | Назначение |
|---|---|
| `.env` | Операторский env-файл. Runtime settings используют nested имена вида `SECTION__FIELD`; инфраструктурные bootstrap/env-переменные Compose могут оставаться flat |
| `src/app_config/runtime/` | Root runtime settings: `Settings(BaseSettings)`, cache helpers, и safe startup logging для Python-сервисов.|
| `src/app_config/catalog/` + `catalog.toml` | Catalog schema, loader и operator catalog для задач, баз знаний и источников |
| `infra/compose/docker-compose.yaml` | Topology всей системы: сети, port bindings, volumes, health checks, зависимости между сервисами |
| `infra/docker/**/Dockerfile` | Определения образов: базовые образы, установка зависимостей, process defaults |
| `infra/nginx/*.conf` | TLS termination, reverse proxy rules и маршрутизация между UI и Gateway |
| `experiments/training/conf/**` | Иерархические Hydra-конфиги для LoRA-экспериментов: модель, LoRA, данные, trainer, scheduler |
| `pyproject.toml` | Зависимости Python-пакета, настройки linting (ruff, mypy) и dev-tooling |

### 7.2 `catalog.toml` — catalog задач, баз знаний и источников

- Списка задач: `id`, `description`, `knowledge_bases`, and optional
  `lora_adapter`.
- Списка баз знаний: `id`, `description`, update strategy, default alias, and
  alias retrieval profiles.
- Связей `task -> knowledge_bases`.
- Per-KB alias retrieval profiles (`top_k`, `score_threshold`, `retrieval_strategy`, `reranker`).
- Task-level LoRA/model adapter config (`lora_adapter`).
- Source and benchmark adapter declarations.
- Source instances for corpus builds and benchmark preparation.

Файл загружается через `src/app_config/catalog/` и валидируется через
Pydantic-модели. Нарушения схемы (например, отсутствующий `default_alias` или
source instance, ссылающийся на неизвестный KB/adapter) приводят к отказу при
старте.

Canonical catalog v3 uses list sections:

```text
[[tasks]]
[[knowledge_bases]]
[[source_adapters]]
[[benchmark_adapters]]
[[source_instances]]
```

Source instances declare:

```text
id
description
role                  "corpus" or "benchmark"
knowledge_base
adapter               { id = "...", version = "..." }
benchmark.suites      only for role = "benchmark"
```

Tasks and knowledge bases use only `id` and `description` for descriptive
strings. Do not add catalog `label`, `routing_description`, or
`selection_description` fields.

`benchmark.suites` allowed values are:

```text
retrieval_quality
context_quality
generation_quality
```

Legacy `[[sources]]` is no longer supported.

### 7.3 Runtime Settings (`src/app_config/runtime/`)

Python-конфигурация реализована через `pydantic-settings` с одним root loader'ом: `Settings(BaseSettings)` в `src/app_config/runtime/`.

Все модели и load-функции живут в `app_config.runtime.models` и `app_config.runtime.loaders`;
`src/shared/` теперь ограничен cross-cutting infrastructure: database, events, logging, telemetry, service helpers.

Ключевые свойства текущей схемы:

- env читает только root `Settings`, а nested sections являются plain `BaseModel`
- canonical runtime env names используют nested contract с delimiter `__`
- flat compatibility aliases для runtime env names больше не поддерживаются
- catalog models/loaders живут в `app_config.catalog`; не реэкспортируются ни через `shared.config`, ни через `app_config.runtime`

Основные секции runtime settings:

- `PlatformSettings` — shared platform endpoints и broker URLs
- `GatewayConfig` + `BudgetSettings` — gateway behavior и budgeting knobs
- `RagSettings` — embedding/retrieval runtime knobs
- `AuthSettings` — OAuth/session/database auth settings
- `CatalogConfig` — путь к task/KB catalog
- `AdapterRegistryConfig` — MLflow adapter materialization и alias sync policy
- `EvalConfig` — judge, metrics и sandbox settings
- `WorkerConfig` — Celery worker runtime defaults
- `UIConfig` — UI timeouts и related knobs

Примеры canonical env names:

- `NETWORK__VLLM__INTERNAL_HOST`
- `NETWORK__VLLM__INTERNAL_PORT`
- `VLLM__MODEL`
- `AUTH__INTERNAL_API_KEY`
- `RABBITMQ_DEFAULT_PASS`
- `POSTGRES_APP_DB`
- `EVAL__JUDGE__MODEL`
- `WORKER__CONCURRENCY`

Функция `get_settings()` кэширует root settings через `@lru_cache`. Для settings-driven тестов и локальных override-сценариев используется `load_settings({...})`; для catalog override используется `catalog_override(...)` из `app_config.catalog`.

Финальное решение по naming convention: текущий mix class names `*Settings` / `*Config` сохраняется, чтобы не делать churn-only rename pass. Канонизирована именно field/env surface, а не имена всех классов.

### 7.4 Maintainer Checklist

При добавлении нового runtime settings field:

1. объявите поле в существующей nested section или добавьте новую nested model под root `Settings`
2. используйте canonical env shape `SECTION__FIELD`, если значение должно быть operator-facing
3. обновите `.env.example`, `infra/README.md` и focused tests только если поле действительно должно настраиваться оператором
4. не добавляйте flat compatibility alias

При добавлении нового catalog field:

1. меняйте schema/models в `src/app_config/catalog/`
2. обновляйте `catalog.toml` и sample/contract tests
3. используйте `catalog_override(...)` в тестах вместо manual global mutation
4. не добавляйте catalog helper re-exports в `shared.config` или `app_config.runtime`
5. do not reintroduce behavior selection through `source_type`; use adapter ids
   and adapter capability checks.

---

## 8. MLOps и автоматизация

### 8.1 Airflow DAG-и

Все автоматизированные пайплайны реализованы как Airflow DAG-и в директории `dags/`. Запускаются на выделенных Celery workers (CPU и GPU).

**`train_lora.py`** — обучение LoRA адаптера на GPU worker'е.
- Запускает `experiments.training.train_adapter.start_train` как subprocess (изоляция от Hydra global state).
- Параметры через Airflow UI: имя конфига эксперимента (`experiment_config`) и список Hydra параметров, которые нужно переопределить.
- Возвращает путь к `training_summary.json` через XCom.
- Не выполняет регистрацию и продвижение — это ручной шаг в `lora_ops.ipynb`.

**`rag_alias_apply.py`** — make one KB alias match its `catalog.toml`
declaration.
- Вызывает `AliasService.apply()` напрямую через те же factories, что и `rag`
  CLI — не через subprocess/CLI-процесс.
- Параметризуется через `kb_id`, `alias`, optional `release_id`,
  `refresh_sources`, `allow_unevaluated`, `allow_build_default`, и
  DVC sync settings (`sync_dvc`, `dvc_base_branch`, `dvc_bot_branch`).
- Optional follow-up `sync_dvc` task синхронизирует generated source-instance
  artifacts через DVC после успешного apply.
- Airflow остаётся orchestration layer; reconciliation logic живёт в
  `src/rag/control_plane/`, source build code in `src/rag/sources/`, and
  materialization code in `src/rag/indexing/`.

**`eval_dags.py`** — оценка качества (подробнее в разделе 9).

**`rag_collection_cleanup.py`** — release/deployment-aware cleanup: marks a
release retired in Postgres (`rag_releases.retired_at`) when it has no
active/pending/recently-superseded `rag_alias_deployments` row, then deletes
its Qdrant collection. Never deletes a collection with no matching
`rag_releases` row at all (it may still be mid-build).

### 8.2 Operator Notebooks (JupyterLab)

JupyterLab — точка входа для ручных операций оператора. Ноутбуки не содержат production-логику напрямую: они вызывают production entrypoints из `src/`.

| Ноутбук | Назначение |
|---|---|
| `experiments/training/lora_ops.ipynb` | LoRA операции: регистрация, промоушен, синхронизация |
| `experiments/training/lora_training.ipynb` | Интерактивный запуск обучения |
| `experiments/eval/eval_results.ipynb` | Анализ результатов оценки |
| `experiments/misc_ops/prefetch_assets.ipynb` | Загрузка моделей и датасетов |
| `experiments/misc_ops/postgres_diagnostics.ipynb` | Диагностика БД |

RAG production operations and diagnostics use `python -m rag.cli.app` in
the `rag-ops` container or Airflow `rag_alias_apply`. Direct collection
metadata, alias, point, and snapshot inspection uses the Qdrant API/dashboard.

### 8.3 Версионирование данных (DVC)

Все датасеты и крупные артефакты находятся под контролем DVC с remote-хранилищем в Yandex Cloud S3. В репозитории хранятся только `.dvc`-файлы (метаданные); сами данные загружаются через `dvc pull`.

Поддерживаемые датасеты:
- **Для обучения:** `arxiv-summarization`, `open-code-instruct`.
- **Для RAG benchmark'ов:** `beir-nfcorpus`, `beir-scifact`, `hotpotqa`, `msmarco`, `natural-questions`.
- **Для code evaluation:** `humaneval`.
- **Для RAG коллекций:** generated artifacts under
  `assets/rag_data/source_instances/` and
  `assets/rag_data/knowledge_bases/`.

Для RAG коллекций policy отличается от eval/training datasets:

- curated source instance `manifest.toml` files stay in Git;
- generated source-instance artifacts, KB manifests, build metadata, and
  benchmark normalized artifacts can be DVC-tracked;
- raw cache (`raw/`, включая PDF/HTML downloads) остаётся server-local по
  умолчанию, чтобы не раздувать DVC без явного требования offline rebuild.

---

## 9. Оценка качества (Evaluation)

### 9.1 Архитектура eval pipeline

Оценка реализована как двухэтапный процесс, управляемый Airflow DAG-ами в `dags/eval_dags.py`:

**Этап 1 — `fetch_predictions`:** Gateway запрашивается с тестовыми вопросами из датасета. Ответы модели (предсказания) сохраняются в временный JSON-файл. Промежуточное хранение через файл (а не XCom) необходимо из-за размера данных.

**Этап 2 — `calculate_metrics`:** По сохранённым предсказаниям вычисляются выбранные метрики и результаты логируются в PostgreSQL для последующего анализа.

### 9.2 Метрики по типу задачи

Каждый eval-suite — уникальная тройка `(task, dataset, metric)`. Eval runner принимает одну метрику за вызов, что позволяет запускать метрики параллельно или независимо.

**`chat`** — оценка качества ответов ассистента на вопросы по ML/AI.
- Датасеты: `hotpotqa`, `natural-questions`.
- **relevance**, **correctness** — LLM-as-judge оценки (через внешний API); оценивают релевантность и фактическую корректность ответа.
- **BERTScore F1** — семантическое сходство ответа и reference через языковую модель.
- **ROUGE-L** — F1 на основе Longest Common Subsequence между ответом и reference.
- При включённом RAG дополнительно вычисляется **groundedness** — LLM-judge оценка того, подкреплён ли ответ retrieved контекстом.

**`summarize`** — оценка качества суммаризации научных статей.
- Датасет: `arxiv-summarization`.
- **faithfulness**, **coverage** — LLM-as-judge оценки связности и полноты резюме.
- **BERTScore F1**, **ROUGE-L** — автоматические метрики совпадения с reference-абстрактом.

**`code`** — оценка функциональной корректности сгенерированного кода.
- Датасет: `humaneval`.
- **pass@1** — задача решена, если сгенерированный код проходит все unit-тесты с первой попытки.
- **executable_rate** — доля ответов, которые вообще исполняются без синтаксических ошибок (более мягкая метрика).
- Код выполняется в изолированном `code-sandbox` контейнере (read-only filesystem, tmpfs, без сети).

**`retrieval`** — независимый бенчмарк качества retrieval pipeline без участия генерации.
- Датасеты: `beir-scifact`, `msmarco`, `beir-nfcorpus`.
- **Recall@k** — доля релевантных документов среди top-k retrieved.
- **nDCG@k** — нормализованный дисконтированный кумулятивный выигрыш; учитывает порядок ранжирования.
- **MRR@k** (Mean Reciprocal Rank) — среднее обратное значение ранга первого релевантного документа.

Замечание: на текущем этапе LLM-as-judge метрики не реализованы.

### 9.3 Параметры запуска DAG-ов

Eval DAG-и параметризованы через Airflow UI: выбор датасета, метрик, knowledge
base, alias и режима RAG (`auto` vs `explicit`). RAG alias options берутся из
catalog KB aliases, а LoRA alias options — из adapter registry settings. Это
позволяет гибко сравнивать champion/challenger RAG setups без изменения кода.

Eval rows сохраняют resolved Qdrant alias, physical collection, RAG manifest id,
dataset DVC hash и optional `eval_verdict` (`pass`, `warn`, `fail`,
`unscored`). Verdict помогает читать результаты, но не является promotion gate.

### 9.4 Alias-based продвижение по результатам оценки

Результаты оценки являются основанием для ручного решения о продвижении
`challenger` в `champion`. Eval не блокирует promotion автоматически; он даёт
traceable evidence для operator decision. Workflow:

```
Обучение (Airflow DAG)
    │
    ▼
Регистрация в MLflow (lora_ops.ipynb)
    │
    ▼
Промоушен: version → alias "challenger"
    │
    ▼
Eval DAG с alias="challenger"
    │
    ▼
Анализ в eval_results.ipynb
    │
    ▼ (если operator принимает решение)
Промоушен RAG/Qdrant alias или LoRA/MLflow alias в champion
```

---

## 10. Инфраструктура и деплой

### 10.1 Docker Compose topology

Всё приложение описано в одном `infra/compose/docker-compose.yaml`. Сервисы разделены на внутренние сети:
- `app-network` — инференс-сервисы.
- `infra-network` — инфраструктурные сервисы (БД, брокеры).
- `monitoring-network` — observability стек.

Volumes хранят персистентные данные: qdrant-storage, postgres-data, redis-data,
mlflow-artifacts и airflow-logs; модели, датасеты и project artifacts
монтируются из host-side `PROJECT_ROOT`.

### 10.2 Release-based деплой

На production-сервере реализован release-based deployment с симлинком:

```
/home/anton-m/agent-042/
  .env                          # Операторский env-файл (вне релизов)
  .dvc/config.local             # DVC remote credentials
  assets/                       # Модели и датасеты (вне релизов)
  artifacts/                    # Checkpoints и artifacts (вне релизов)
  releases/
    <sha1>/                     # Релиз 1 (код из GitHub)
    <sha2>/                     # Релиз 2 (код из GitHub)
  current -> releases/<sha2>/   # Симлинк на активный релиз
```

**Скрипт деплоя** (`ops/deploy_release.sh`):
1. Создаёт новую директорию `releases/<sha>` с кодом нового релиза.
2. Переключает симлинк `current` на новый релиз.
3. Запускает `docker compose up -d --build` с новым `IMAGE_TAG`.
4. Выполняет smoke-тесты (health check). При неудаче — откат симлинка на предыдущий релиз.
5. Удаляет старые релизы (хранит последние N).

### 10.3 CI/CD

Автоматизированный pipeline (GitHub Actions):
1. **Lint** — `ruff` (linter) + `mypy` (type checking) + `pre-commit`.
2. **Tests** — `pytest` (unit и integration тесты).
3. **Build** — сборка Docker образов и push в registry с тегом `<branch>-<sha12>`.
4. **Deploy** — SSH-деплой на production-сервер через `deploy_release.sh`.

Зависимости управляются через `uv` (ultrafast package manager). Группы зависимостей разделены по сервисам: `gateway`, `ui`, `worker`, `airflow-worker`, `airflow-worker-gpu`, `training`, `rag`.

### 10.4 Минимальные требования к инфраструктуре

| Ресурс | Минимум |
|---|---|
| RAM | 16 GB |
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| CPU | 4+ ядра |
| Диск | 30+ GB (модели, кэши, volumes) |
| ПО | Docker Engine, Docker Compose v2, NVIDIA Container Toolkit |

### 10.5 Мониторинг

**Prometheus + Grafana:**
Gateway использует `prometheus-fastapi-instrumentator` — автоматически экспортирует метрики HTTP-запросов (latency, status codes, throughput). Grafana предоставляет дашборды инфраструктурной observability (CPU, GPU, память) и ML-специфические дашборды (очереди, inference latency).

**OpenTelemetry + Tempo + Loki:**
Python-сервисы пишут structured JSON logs с `request_id` и, когда есть активный span,
`trace_id`/`span_id`. OpenTelemetry traces отправляются в `otel-collector`, затем в Tempo.
Grafana Alloy читает Docker logs через Docker socket и отправляет их в Loki. Grafana
provisioning подключает Postgres, Prometheus, Loki и Tempo. Рабочий workflow описан в
`docs/analytics/observability.md`.

**Redpanda:**
Gateway и Celery worker публикуют durable inference lifecycle events в
`inference.events.v1`. Redpanda Console включён в Compose для инспекции topic'ов.
Схема и workflow описаны в `docs/analytics/inference-events.md`.

**ClickHouse:**
ClickHouse Kafka Engine читает `inference.events.v1` из Redpanda и materialized view
записывает события в `inference_events_raw` (`MergeTree`). Это первый слой
аналитики по inference lifecycle; workflow и SQL-примеры описаны в
`docs/analytics/clickhouse-analytics.md`.

**Flower:** Мониторинг Celery workers — активные задачи, история, статистика очередей.

**RedisInsight:** Инспекция Redis-ключей, pub/sub топиков, памяти.

**MLflow UI:** Полная история всех training runs с метриками, параметрами и артефактами. Сравнение запусков, анализ трендов.

---

## 11. Безопасность

### 11.1 Аутентификация

Система использует Google OAuth 2.0 Authorization Code Flow с PKCE:
- `code_verifier` — криптографически случайная строка (256 бит).
- `code_challenge` — SHA-256 хэш `code_verifier`, передаётся в authorization request.
- Это предотвращает атаки перехвата кода авторизации (authorization code interception attack).

ID Token верифицируется через Google JWKS (JSON Web Key Sets) с проверкой подписи, audience (`aud`) и времени истечения (`exp`).

### 11.2 Управление сессиями

- Сессии хранятся в Redis с TTL (не в cookie напрямую).
- Cookie содержит только непредсказуемый `session_id` (CSRF-защита через `state` parameter в OAuth flow).
- `AuthMiddleware` в Gateway проверяет сессию при каждом запросе.

### 11.3 Управление секретами

- Все секреты (OAuth credentials, API keys, DB passwords) хранятся в `.env` файле на сервере.
- `.env` никогда не коммитится в репозиторий (есть `.env.example` с документацией переменных).
- Переменные окружения инжектируются в контейнеры через Docker Compose.

---

## 12. Структура репозитория

```
agent-042/
├── src/                        # Production runtime код
│   ├── gateway/                # FastAPI Gateway
│   │   ├── api/                # HTTP routes
│   │   ├── auth/               # OAuth2/OIDC middleware и router
│   │   ├── schemas/            # Pydantic schemas (OpenAI-совместимые)
│   │   └── services/           # Business logic (processing, RAG, prompt, Celery, Redis)
│   ├── rag/                    # RAG pipeline
│   │   ├── adapters/           # Catalog-declared source/benchmark adapters
│   │   ├── sources/            # Native Document/TextNode source lifecycle
│   │   ├── indexing/           # LlamaIndex Qdrant materialization and aliases
│   │   ├── runtime/            # LlamaIndex retrieval/query runtime
│   │   ├── evaluation/         # RAG benchmark preparation and evaluation
│   │   ├── lifecycle/          # Shared CLI/Airflow build stages
│   │   ├── embeddings.py       # Embedding service
│   │   ├── reranker.py         # Cross-encoder reranker
│   │   ├── sparse_encoder.py   # Sparse (BM25) encoding
│   ├── shared/                 # Общий код для всех сервисов
│   │   ├── config.py           # Pydantic settings
│   │   ├── catalog.toml       # Task / KB / source catalog
│   │   ├── model_registry.py   # MLflow adapter sync
│   │   └── db/                 # SQLAlchemy модели и engine
│   ├── embeddings/             # Embeddings microservice
│   ├── reranker/               # Reranker microservice
│   ├── ui/                     # Streamlit UI
│   └── worker/                 # Celery worker (inference tasks)
├── experiments/                # Эксперименты и operator workflows
│   ├── eval/                   # Eval ноутбуки и скрипты
│   ├── rag/                    # RAG ноутбуки и sandboxes
│   ├── training/               # LoRA обучение
│   │   ├── conf/               # Hydra конфиги
│   │   └── train_adapter/      # Training pipeline (Lightning, PEFT, MLflow)
│   └── misc_ops/               # Прочие операции
├── dags/                       # Airflow DAG-и
├── infra/                      # Инфраструктура
│   ├── compose/                # docker-compose.yaml
│   ├── docker/                 # Dockerfiles по сервисам
│   ├── grafana/                # Grafana provisioning
│   └── nginx/                  # nginx конфиги
├── tests/                      # Unit и integration тесты
├── migrations/                 # Postgres schema migrations (SQL)
├── bootstrap/                  # One-time environment bring-up scripts
├── ops/                        # Recurring operational scripts (деплой, утилиты)
├── assets/                     # Модели и датасеты (DVC-tracked)
└── artifacts/                  # Training checkpoints и Hydra runs
```

---

## 13. Итоги и выводы

Проект Agent-042 представляет собой полноценную production-систему AI-ассистента, разработанную как выпускная квалификационная работа магистратуры по направлению «Искусственный Интеллект».

**Основные реализованные компоненты:**

- **Инференс-платформа** с поддержкой streaming, аутентификацией через Google OAuth2/PKCE и асинхронной обработкой запросов через Celery.
- **RAG-система** с тремя стратегиями retrieval (dense, sparse, hybrid), cross-encoder reranking, alias-based управлением коллекциями и автоматической маршрутизацией по типу задачи через embedding-based классификацию.
- **LoRA fine-tuning pipeline** с полным циклом от подготовки данных до serving: Hydra-конфигурирование, PyTorch Lightning обучение, MLflow трекинг и Model Registry, hot-loading адаптеров в vLLM.
- **MLOps автоматизация** через Airflow DAG-и: обновление RAG, обучение адаптеров, eval pipeline.
- **Evaluation framework** с метриками ROUGE-L, BERTScore, Recall@k, nDCG@k и поддержкой LLM-as-judge.
- **Наблюдаемость** через Prometheus, Grafana, Flower, RedisInsight и MLflow.
- **CI/CD** с автоматической проверкой кода, тестированием, сборкой образов и release-based деплоем.

**Ключевая идея:** значительная часть усилий при разработке AI-систем — не алгоритмы, а инфраструктура. Воспроизводимость экспериментов, надёжность сервисов, прозрачность процессов и автоматизация рутинных задач формируют основу, на которой можно итеративно улучшать качество системы.
