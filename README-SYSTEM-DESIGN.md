# Разработка и исследование интеллектуального ассистента для исследователей с использованием генерации на основе поиска и эффективного дообучения моделей

## Инструкции

* `./infra/README.md` - настройка окружения и инфраструктуры
* `./experiments/README.md` - как проводить эксперименты и operator workflow
* `./CONFIG-CONTRACT.md` - краткий актуальный контракт конфигурации
* `./src/gateway/README.md` - документация Gateway (FastAPI)
* `./src/ui/README.md` - документация UI (Streamlit)
* `./REMAINING-CHANGES.md` - список незавершенных изменений и планируемых доработок

### Experiment Workflow

Операции с экспериментами разделены на два entrypoint-а: Airflow для тяжелых вычислений и
Jupyter-ноутбуки в `experiments/` для ручных операторских решений. Инфраструктурные
shell-утилиты расположены в `scripts/`.

* **LoRA**: обучение через Airflow DAG `train_lora`, регистрация/промоушен через
  `experiments/training/lora_ops.ipynb`
* **RAG**: обновление индексов через DAG-и `arxiv_rag_update` / `pytorch_docs_rag_update`,
  production entrypoints в `src/rag/ops`, manual create/promote/inspect через
  `experiments/rag/rag_ops.ipynb` (напрямую вызывает `src/rag/ops`),
  notebook-only experimental forks в `experiments/rag/sandboxes/`
* **Eval**: запуск через `dags/eval_dags.py`, просмотр результатов и сравнение конфигураций через
  `experiments/eval/eval_results.ipynb`
* **Misc**: `experiments/misc_ops/` — prefetch, MLflow quickref, PostgreSQL diagnostics

## Структура репозитория

Проект организован как монорепозиторий, содержащий все компоненты системы: inference-сервисы,
RAG-пайплайн, обучение адаптеров, платформу оценки и инфраструктурные конфигурации. Ключевой
принцип — **один репозиторий, один сервер**, приближенные к production практики.

```
├── src/                          # Все runtime-сервисы
│   ├── gateway/                  # FastAPI API Gateway
│   │   ├── api/v1/              # REST API endpoints (OpenAI-compat, discovery, sessions, KBs)
│   │   ├── auth/                # OAuth2/OIDC, middleware, session management
│   │   ├── schemas/             # Pydantic модели запросов и ответов
│   │   └── services/            # Бизнес-логика (processing, task_router, prompt_builder,
│   │                            #   rag_service, vllm_client, celery_client, redis_stream)
│   ├── rag/                     # RAG-система
│   │   ├── chunking.py          # Стратегии разбиения (fixed_token, code, section_aware)
│   │   ├── embeddings.py        # HTTP-клиент к embedding microservice
│   │   ├── retriever.py         # Orchestrator: embed → search → format
│   │   ├── vector_store.py      # Qdrant wrapper: collections, aliases, _meta sentinel
│   │   └── ops/                 # Production lifecycle (create/, update/, aliases, inspect, meta)
│   ├── worker/                  # Celery worker для async LLM inference
│   ├── embeddings/              # Standalone embedding microservice (FastAPI + sentence-transformers)
│   ├── shared/                  # Общая конфигурация, DB models, model registry
│   │   ├── config.py            # Единый источник settings (Platform, Gateway, RAG, Auth, ...)
│   │   ├── knowledge_bases.json # Runtime registry: task → KB → aliases, update_strategy
│   │   ├── model_registry.py    # MLflow-based LoRA adapter management + vLLM hot-load
│   │   └── db/                  # SQLAlchemy models + SQL schema (users, sessions, eval_runs, ...)
│   └── ui/                      # Streamlit чат-интерфейс
├── experiments/                  # Operator notebooks и скрипты экспериментов
│   ├── training/                # LoRA обучение (PyTorch Lightning + Hydra + PEFT)
│   │   ├── train_adapter/       # Lightning Module, Data Module, модели, MLflow интеграция
│   │   └── conf/                # Hydra конфигурации (config.yaml, experiment/, paths/)
│   ├── eval/                    # Evaluation framework
│   │   └── eval_scripts/        # Runner, datasets loader, metrics (automatic, llm_judge, code_exec)
│   ├── rag/                     # RAG operator notebooks + experimental sandboxes
│   └── misc_ops/                # Prefetch, MLflow quickref, PostgreSQL diagnostics
├── dags/                        # Airflow DAG-и (train, eval, RAG update, cleanup)
├── infra/                       # Инфраструктура
│   ├── compose/                 # Docker Compose (основной deployment manifest)
│   ├── docker/                  # Dockerfiles для всех сервисов
│   ├── nginx/                   # Nginx reverse proxy config
│   ├── helm/                    # (зарезервировано для будущих Helm charts)
│   ├── k3s/                     # (зарезервировано для k3s)
│   └── terraform/               # (зарезервировано для IaC)
├── assets/                      # DVC-управляемые данные (datasets/, rag_data/, adapters/)
├── artifacts/                   # Gitignored runtime outputs (training runs, hydra, logs)
├── tests/                       # Тесты (api, auth, eval, rag, training)
└── scripts/                     # Shell-утилиты (update_locks, dump_logs, fetch_logs)
```

## Постановка задачи. Scope / Область исследования.

Целью данной работы является создать агентскую систему production уровня, которая способна
выполнять роль полноценного AI-помощника для исследователей, работая в условиях ограниченных
ресурсов и с высокой конфиденциальностью пользовательских данных. Технологии, которые будут
использоваться, и которые предположительно смогут помочь в достижении таких целей:

* LoRA адаптеры. Система будет работать с одной базовой LLM, при этом используя различные LoRA
  под разные цели. Вопрос для исследования: возможно ли улучшить работу сервиса с одной LLM,
  обучив отдельные LoRA под разные нужды.
* RAG система. Расширение пользовательских промптов дополнительной информацией из баз знаний,
  документацией, примерами кода и т.д. - один из эффективных способов повысить качество работы
  LLM. В рамках проекта предстоит построить RAG систему и ответить на следующие вопросы:
    * Какие данные следует использовать в RAG для разных задач?
    * Какие методы retrieval будут оптимальными с точки зрения баланса производительности /
      качества?
    * Какие методы reranking выбрать?
    * Влияние Chunking стратегий на качество работы сервиса.
* Полноценный агентский сервис. Пункт на случай успешного выполнения предыдущих двух и наличия
  дополнительного времени. Выбирать LoRA и задействовать RAG можно разными способами: вручную,
  через простые rule-based стратегии, либо доверить это отдельной LLM. В рамках данного пункта
  будет выполнена попытка организовать полноценных агентский сервис, который получает только
  запрос от пользователя и генерирует ответ, используя все доступные инструменты, когда это
  необходимо.

## Бизнес описание работы агентской системы

Цель проекта: Создать AI-ассистента на базе агентской системы с RAG для исследователей в областях
ML/DL/AI/LLM, который ускоряет поиск информации, суммаризацию научных материалов и генерацию
кода, повышая продуктивность и воспроизводимость исследований. Система работает локально и сохраняет
конфиденциальность пользовательских данных. Она контекстно осведомлена о текущем проекте и
участниках, учитывает историю запросов и выполняемых задач, а при необходимости дополняет знания
внешним поиском — без раскрытия приватной информации. По сути, это «коллега‑ассистент», который
понимает, где и над чем он работает, и помогает принимать решения быстро и безопасно.

### Целевая аудитория

* Исследователи: быстро извлекают суть статей, находят релевантные цитаты и идеи для
  экспериментов.
* ML-инженеры: получают примеры кода, рефакторинг и помощь с интеграцией моделей в пайплайны.
* Студенты и стажеры: получают объяснения концепций и примеры с минимальным входным порогом.
* Руководитель группы: получает обзор прогресса и агрегированные знания по проекту.

### Ключевые сценарии использования

* Чат с поддержкой поиска по внутренним и внешним источникам (RAG): ответы с указанием источников и
  цитат.
* Суммаризация статей и длинных документов (multi-level: от краткого «TL;DR» до подробной
  структуры).
* Генерация и доработка кода (шаблоны, тесты, советы по оптимизации).
* Поиск по корпоративным/локальным репозиториям, базам знаний и коду.

### Основные ценности

* Экономия времени на обзор литературы и поиск решений.
* Быстрая генерация кода и примеров, релевантных имеющимся базам знаний, проектам и репозиториям.

### Ожидаемые возможности системы:

* Чат-бот. Ответы на вопросы про разные области и аспекты ML / DL / AI / LLM.
* Суммаризация документов / статей как отдельная функция.
* Генерация кода как отдельная функция.
* Поддержка контекста с базой знаний (статьи, документация, кодовые базы и т.д.) на базе RAG
  системы как отдельная функция.

### Возможные расширения функционала, возможностей RAG системы и решение проблемы cutoff-date:

* Агентская система с автоматизированным выбором инструментов: LoRA для суммаризации / генерации
  кода, задействование RAG, web-search.
* Хранение и динамическое обновление информации о пользователе и истории переписки. Поддержка
  агента в состоянии постоянного пребывания в контексте той системы, в которой он работает.

## Техническое описание и постановка задачи

Сервис имеет 2 основные платформы:

1. Платформа для экспериментирования и обучения моделей и адаптеров
2. Платформа с работающим LLM сервисом. Сам сервис строится в 4 этапа:
    1. Базовая LLM.
    2. Базовая LLM + RAG система. Фиксированный или rule-based выбор, когда использовать RAG.
    3. Базовая LLM + RAG + LoRA адаптеры. Фиксированный или rule-based выбор адаптеров.
    4. Агентский сервис с динамическим выбором задействуемых инструментов.

### Микросервисная архитектура

Вся система развёрнута как набор Docker-контейнеров, управляемых через Docker Compose, с единой
точкой входа через Nginx reverse proxy. Архитектура следует принципам production-систем:
изоляция по сетям, health check-и, горячая перезагрузка адаптеров, crash-recovery для worker-ов.

**Сервисы inference-платформы:**

| Сервис | Технология | Порт | Назначение |
|--------|-----------|------|------------|
| `gateway` | FastAPI | 9000 | API Gateway: маршрутизация задач, сборка промптов, RAG, аутентификация |
| `vllm` | vLLM v0.16.0 | 8000 | OpenAI-compatible LLM inference с multi-LoRA и hot-reload |
| `embeddings` | FastAPI + sentence-transformers | 8100 | Standalone embedding microservice |
| `celery-worker` | Celery | — | Async LLM inference с token streaming через Redis Pub/Sub |
| `ui` | Streamlit | 8501 | Чат-интерфейс с OAuth2 аутентификацией |
| `vllm-adapter-sync` | Python | — | Синхронизация MLflow Model Registry адаптеров в vLLM |
| `code-sandbox` | Python 3.13 (изолированный) | 8200 | Безопасное выполнение кода для HumanEval eval |

**Сервисы платформы экспериментов:**

| Сервис | Технология | Порт | Назначение |
|--------|-----------|------|------------|
| `airflow-webserver` | Apache Airflow 3 | 8080 | Web UI и API для DAG-ов |
| `airflow-scheduler` | Apache Airflow 3 | — | Расписание DAG-ов |
| `airflow-dag-processor` | Apache Airflow 3 | — | Парсинг DAG-файлов |
| `airflow-worker` | Airflow Celery Worker | — | CPU worker: evals, RAG updates, cleanup (concurrency 2) |
| `airflow-worker-gpu` | Airflow Celery Worker + CUDA | — | GPU worker: LoRA training (concurrency 1) |
| `jupyter` | JupyterLab | 8888 | Operator notebooks для экспериментов |
| `mlflow` | MLflow Tracking | 5050 | Experiment tracking и Model Registry |

**Инфраструктурные сервисы:**

| Сервис | Технология | Порт | Назначение |
|--------|-----------|------|------------|
| `postgres` | PostgreSQL 15 | 5432 | Airflow metadata, MLflow backend, agent042 app DB |
| `qdrant` | Qdrant v1.17.0 | 6333/6334 | Векторная БД для RAG (HTTP/gRPC) |
| `redis` | Redis 7 | 6379 | Sessions, Pub/Sub streaming, кэш |
| `rabbitmq` | RabbitMQ 3 + Management | 5672/15672 | Celery broker (Airflow workers и gateway worker) |
| `flower` | Flower | 5555 | Мониторинг Celery worker-ов |
| `redisinsight` | RedisInsight | 5540 | Мониторинг Redis |

**Docker network isolation:**

| Сеть | Содержит | Назначение |
|------|----------|------------|
| `mlflow_db_net` | PostgreSQL, MLflow, Airflow, workers | Данные и experiment tracking |
| `backend_net` | vLLM, Qdrant, RabbitMQ, Redis, Gateway, workers | Ядро inference |
| `frontend_net` | UI ↔ Gateway | Клиентский слой |
| `sandbox_net` | code-sandbox (без интернета) | Изолированное выполнение кода |

Каждый сервис подключён только к тем сетям, которые необходимы для его работы. `code-sandbox`
полностью изолирован: read-only root filesystem, tmpfs `/tmp`, ограничение 1 CPU / 256 MB RAM,
доступ только из `sandbox_net` — без выхода в интернет.

### Платформа для экспериментов и обучение LoRA

* DVC with Yandex Cloud S3 remote
* MLFlow with Yandex Cloud S3 remote
* **MLflow Model Registry** — реестр версионированных LoRA-адаптеров с alias-based promotion
  (champion / challenger) для перехода из экспериментов в production
* **Qdrant aliases для RAG-индексов** — alias-based promotion для retrieval-конфигураций
  (champion / challenger) без полного релиза в production
* Hydra для конфигурирования тренировок
* Lightning AI (Pytorch Lightning) для организации тренировочных пайплайнов
* **Airflow + Jupyter split** — тяжелые compute workload-ы идут через DAG-и, а register / promote /
  sync / inspect выполняются из operator notebooks
* **Run-scoped training artifacts** — каждый training run пишет артефакты в
  `artifacts/training/runs/<timestamp>-<uuid>/` с поддиректориями `checkpoints/`, `export/`,
  `metadata/` и `evaluation/`
* **Export from best checkpoint** — deployable LoRA-артефакт восстанавливается из лучшего
  checkpoint-а перед экспортом и регистрацией
* **`artifacts/` как runtime-корень** — gitignored runtime outputs складываются в `artifacts/`,
  а hot-loaded serving adapters остаются в `assets/adapters/`

### Контракт конфигурации

* `src/shared/config.py` — единый источник истины для cross-service конфигурации. В нём
  выделены `PlatformSettings`, `GatewayBehaviorSettings`, `RagSettings`, `AuthSettings`,
  `RegistrySettings`, `EvalSettings` и `UISettings`; `GatewaySettings` агрегирует gateway-facing
  настройки в один flat-контракт.
* Канонический локальный шаблон конфигурации — корневой `.env.example` репозитория. Локальные
  entrypoint-ы bootstrap-ят корневой `.env` из корня проекта через `src/shared/local_env.py`.
  Контейнеризированные сервисы получают env vars напрямую из Docker Compose.
* Канонические shared endpoint env vars: `VLLM_BASE_URL`, `EMBEDDINGS_URL`, `QDRANT_HOST`,
  `QDRANT_PORT`, `MLFLOW_TRACKING_URI`, `REDIS_URL`, `CELERY_BROKER_URL`. Eval-specific endpoint
  остаётся `EVAL_GATEWAY_URL`.
* Service-specific settings используют префиксные семейства: `GATEWAY_*`, `EVAL_*`, `UI_*`,
  `REGISTRY_*`, `AIRFLOW_*`, `VLLM_*` и т.д.
* Краткий операционный reference по ownership и canonical env names живёт в
  `CONFIG-CONTRACT.md`.

### Этап 1. Базовая LLM.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* FastAPI использует Task Router для функции chat
* FastAPI использует Prompt Builder, который собирает промпт
    * Промпт содержит базовый system message и task-specific суффикс
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * адаптеры не используются
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Этап 2. Базовая LLM + RAG система.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* FastAPI использует Task Router, который определяет, что нужно сделать: chat / summarize /
  generate code
    * таска определяется rule-based по кейвордам или выбирается вручную в UI
* FastAPI использует Prompt Builder, который собирает промпт
    * **в UI можно выбрать, использовать ли RAG и какие knowledge base задействовать**
    * **Промпт может дополняться retrieved context из RAG**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * адаптеры не используются
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Этап 3. Базовая LLM + RAG + LoRA адаптеры.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* **FastAPI использует Task Router, который определяет задействуемую функцию: chat / summarize /
  generate code**
    * **таска определяется rule-based по кейвордам или выбирается вручную в UI**
    * **под каждую таску существует свой LoRA**
* FastAPI использует Prompt Builder, который собирает промпт
    * в UI можно выбрать, использовать ли RAG
    * **Промпт фиксированный, но разный для каждой функции**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * **получает информацию, какой адаптер подгружать (или никакой)**
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Этап 4. Агентский сервис с динамическим выбором задействуемых инструментов.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* **Между FastAPI и Task Router / Prompt Builder есть отдельный слой абстракции с
  LLM, которая автоматизирует выбор адаптеров и RAG, а также может задействовать другие
  инструменты.**
    * **Подробности TBD**
* FastAPI использует Task Router, который определяет, что нужно сделать: chat / summarize /
  generate code
    * **в UI можно вручную выбрать, какую задачу нужно выполнять**
    * под каждую таску существует свой LoRA
* FastAPI использует Prompt Builder, который собирает промпт
    * **в UI можно выбрать, использовать ли RAG**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * получает информацию, какой адаптер подгружать (или никакой)
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

> **Примечание**: Agent layer с динамическим выбором инструментов — запланированное расширение
> (см. `REMAINING-CHANGES.md` §2.12).

### Конвейер inference запроса

При поступлении запроса на `POST /v1/chat/completions` Gateway выполняет следующий pipeline:

```
Client Request   (POST /v1/chat/completions)
      │
      ▼
AuthMiddleware   (X-Api-Key → service auth | session_id cookie → user auth)
      │
      ▼
_ProcessChat     (основной оркестратор в services/processing.py)
      │
      ├── TaskRouter.decide(user_text) → "chat" | "summarize" | "code"
      │     └── Rule-based по ключевым словам (summarize/tldr → summarize;
      │         code/python/bug/traceback/refactor → code; default → chat)
      │
      ├── RAGService.retrieve_documents(query, kb, alias, top_k)
      │     ├── EmbeddingService.embed_query(text)  →  HTTP POST embeddings:8100
      │     └── QdrantVectorStore.search(embedding)  →  Qdrant collection через alias
      │
      ├── PromptBuilder.build_system_prompt(task, rag_mode, retrieved_context)
      │     ├── Base: "You are an AI assistant for ML/DL/AI/LLM researchers."
      │     ├── + Task-specific суффикс (chat / summarize / code)
      │     └── + Retrieved RAG context block (if available)
      │
      └── Inference (два режима):
            ├── Sync:  VllmOpenAIClient → POST vllm:8000/v1/chat/completions
            └── Async: CeleryClient.enqueue() → Celery Worker → Redis Pub/Sub → SSE
```

**Sync-режим** (по умолчанию): Gateway напрямую вызывает vLLM через HTTP. Если `stream=true`,
Gateway проксирует SSE-поток от vLLM к клиенту. Timeout по умолчанию: 60 секунд (стриминг —
без таймаута).

**Async-режим** (через Celery, опционально): Gateway ставит задачу `generate_response` в
RabbitMQ. Celery worker получает задачу, стримит ответ от vLLM и публикует токены в Redis
Pub/Sub канал `tokens:{conversation_id}`. Gateway подписывается на канал и возвращает SSE-поток
клиенту. Async-режим включается конфигурацией `GATEWAY_ASYNC_ENABLED=true`. Worker использует
`task_acks_late=True` для crash recovery и автоматический retry с exponential backoff.

**Формат запроса** совместим с OpenAI Chat Completions API и дополнен полями:
* `rag_sources: [{knowledge_base, alias}]` — выбор knowledge bases для RAG
* `chat_session_id` — привязка к сессии для персистентной истории
* `extra` — pass-through поля для OpenAI-совместимых параметров

**Формат ответа** совместим с OpenAI API, но дополнен полем `rag_context` с информацией об
использованных документах (content, score, source).

### Embeddings Microservice

Embedding-сервис (`src/embeddings/main.py`) вынесен в отдельный контейнер для изоляции тяжелых
зависимостей (PyTorch, sentence-transformers) от Gateway и Airflow worker-ов. Gateway, RAG
pipeline и eval runner обращаются к нему по HTTP, что позволяет масштабировать
embedding-вычисления независимо.

**API:**

| Endpoint | Назначение |
|----------|------------|
| `GET /health` | Health check |
| `GET /v1/dimension` | Размерность эмбеддингов и имя модели |
| `POST /v1/embeddings` | Batch embedding текстов → float vectors |

По умолчанию используется модель `sentence-transformers/all-MiniLM-L6-v2` с поддержкой CPU,
CUDA и MPS devices. Batch size настраивается через `EMBEDDING_BATCH_SIZE` (по умолчанию 32).

### Gateway API Endpoints

| Endpoint | Method | Назначение | Auth |
|----------|--------|------------|------|
| `/health` | GET | Health check | нет |
| `/config` | GET | Текущая конфигурация gateway | нет |
| `/v1/models` | GET | Proxy к vLLM: список доступных моделей | да |
| `/v1/chat/completions` | POST | Основной inference endpoint (sync/stream) | да |
| `/v1/chat/sessions` | POST | Создание новой chat session | да |
| `/v1/chat/sessions` | GET | Список сессий пользователя | да |
| `/v1/chat/sessions/{id}/messages` | GET | Сообщения в сессии | да |
| `/v1/chat/sessions/{id}` | DELETE | Удаление сессии (cascade) | да |
| `/v1/knowledge-bases` | GET | Список доступных KB + alias-ов | нет |
| `/auth/login` | GET | Начало OAuth2 PKCE flow | нет |
| `/auth/callback` | GET | Обработка OAuth2 callback | нет |
| `/auth/logout` | GET | Завершение сессии | нет |
| `/auth/me` | GET | Профиль текущего пользователя | да |

### Аутентификация и пользовательское состояние

Система использует Google OAuth2 / OpenID Connect Authorization Code Flow with PKCE.

* Nginx reverse proxy публикует все сервисы на одном домене с path-based routing:
  `/` → Streamlit UI, `/auth/` → Gateway auth routes, `/api/` → Gateway API,
  `/airflow/` → Airflow, `/jupyter/` → JupyterLab, `/mlflow/` → MLflow,
  `/flower/` → Flower, `/redis-insight/` → RedisInsight, `/rabbitmq/` → RabbitMQ Management.
  Внутренние сервисы (vLLM, Qdrant, Redis, PostgreSQL) доступны только через Docker-сеть.
* Браузер получает только `HttpOnly` cookie `session_id`; access token, refresh token и срок их
  жизни хранятся server-side в Redis (TTL сессии по умолчанию 24 часа).
* **CSRF-защита** реализована через одноразовый `state` параметр в OAuth flow: при начале
  авторизации генерируется криптографически стойкий `state`, который сохраняется в Redis с TTL
  10 минут. При callback `state` потребляется атомарно через Redis pipeline, предотвращая replay
  атаки. PKCE flow использует `code_verifier` / `code_challenge` (SHA-256) для защиты от
  перехвата authorization code.
* Gateway `AuthMiddleware` поддерживает два метода аутентификации:
  1. **API Key** (`X-Api-Key` header) — для service-to-service вызовов (Airflow eval runner,
     adapter sync). Проверка через `hmac.compare_digest()`. Запрос получает
     `user_id = "__service__"`.
  2. **Session Cookie / Bearer Token** — для пользовательских запросов. При необходимости
     middleware тихо refresh-ит Google access token (за 120 секунд до истечения).
* Публичные пути без аутентификации: `/health`, `/auth/*`, `/docs`, `/openapi.json`, `/redoc`.
* PostgreSQL database `agent042` хранит `users`, `chat_sessions`, `chat_messages`, а также
  evaluation tables `eval_runs` и `eval_samples`.
* Streamlit UI позволяет создавать, переключать и удалять chat sessions; завершённые
  non-streaming диалоги сохраняются gateway-ем в историю пользователя.

> **Примечание**: chat history частично server-side only — gateway сохраняет завершенные
> non-streaming exchanges, но streaming responses не персистятся (`stream_chat()` принимает
> `user_id`/`chat_session_id`, но не использует их) (см. `REMAINING-CHANGES.md` §1.2).

### Streamlit UI

Streamlit-приложение (`src/ui/app.py`) реализует чат-интерфейс:

* **Авторизация**: OAuth2 redirect на Google через Gateway `/auth/login`, session cookie
  forwarding через `GatewayClient`.
* **Chat sessions**: боковая панель со списком сессий, создание/удаление/переключение.
  Lazy creation — сессия создаётся только при первом сообщении.
* **Knowledge base selector**: выбор RAG knowledge bases для текущего запроса (arXiv для chat,
  PyTorch docs для code).
* **Thinking visualization**: извлечение `<think>...</think>` блоков из ответа LLM и рендер в
  collapsible Markdown expanders.
* **GatewayClient** (`src/ui/client.py`) — HTTP wrapper над Gateway API с пробросом session
  cookie как Bearer token.

### RAG пайплайны

RAG-система разрабатывается для двух ключевых функций агента: **чат** и **генерация кода**.
**Суммаризация** всегда работает без RAG, так как суммаризация работает непосредственно
с предоставленным пользователем документом.

Для каждой RAG-подсистемы проводятся эксперименты по следующим направлениям:

* **Данные**: какие типы источников и знаний необходимо включать в векторную БД для конкретной
  задачи. Выбор knowledge base (arXiv, PyTorch docs и др.) — вопрос версионирования эксперимента;
  переключение между ними выполняется вручную.
* **Chunking стратегии**: способы разбиения документов на чанки и их влияние на качество retrieval и
  генерации.
* **Retrieval стратегии**: сравнение sparse, dense и hybrid подходов.
* **Reranking стратегии**: методы переупорядочивания извлечённых чанков для повышения целевых метрик
  качества.

> **Примечание**: RAG hybrid-search и reranking benchmarks являются запланированными, но ещё не
> реализованными улучшениями (см. `REMAINING-CHANGES.md` §2.3–2.4).

#### Реализованные chunking-стратегии

В `src/rag/chunking.py` реализованы три стратегии разбиения документов на чанки:

| Стратегия | Описание | Применение |
|-----------|----------|------------|
| `fixed_token` | `RecursiveCharacterTextSplitter` с иерархией разделителей (`\n\n`, `\n`, `. `, ` `) | Baseline для большинства документов |
| `code` | Разделение по структуре кода (regex: `def`, `class`, `async def`) с fallback на текстовое разбиение | PyTorch docs и код |
| `section_aware` | Разделение по markdown-заголовкам (`#{1,6}`) с учётом ограничений на размер чанка | Научные статьи с выраженной структурой |

Фабричная функция `get_chunker(strategy, **kwargs)` создаёт экземпляр нужной стратегии.
Параметры `chunk_size` и `chunk_overlap` передаются при создании и сохраняются в `_meta`
sentinel коллекции.

#### RAG pipeline компоненты

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `EmbeddingService` | `src/rag/embeddings.py` | HTTP-клиент к embeddings microservice |
| `QdrantVectorStore` | `src/rag/vector_store.py` | Обёртка над Qdrant: collections, aliases, `_meta` sentinel, search |
| `Retriever` | `src/rag/retriever.py` | Orchestrator: embed query → vector search → format context |
| `RAGService` | `src/gateway/services/rag_service.py` | Gateway-side: multi-KB retrieval с alias-based routing |

#### Production RAG operations (`src/rag/ops/`)

Вся production lifecycle логика для RAG-коллекций сосредоточена в `src/rag/ops/`:

| Модуль | Назначение |
|--------|------------|
| `ops/meta.py` | `BuildConfig`, `CollectionMeta`, `ImplementationInfo` — метаданные и валидация |
| `ops/materialize.py` | Создание коллекций, batch embed & upsert, генерация timestamped имён |
| `ops/create/arxiv.py` | Bootstrap ArXiv коллекции с deterministic point IDs |
| `ops/create/pytorch_docs.py` | Bootstrap PyTorch docs коллекции |
| `ops/update/arxiv.py` | Инкрементальное обновление ArXiv (upsert в ту же коллекцию) |
| `ops/update/pytorch_docs.py` | Полная замена PyTorch docs (blue-green deployment) |
| `ops/aliases.py` | Назначение / промоушен / отвязка alias-ов с валидацией из `knowledge_bases.json` |
| `ops/inspect.py` | Инспекция коллекций и alias-ов |

`experiments/rag/rag_ops.ipynb` напрямую вызывает production entrypoints из `src/rag/ops` для
ручного управления коллекциями. `experiments/rag/sandboxes/` предназначен исключительно для
notebook-only экспериментального кода; Gateway, Airflow и production evals не импортируют его.

**Данные для RAG**

| Knowledge Base | Тип данных | Задачи | Стратегия обновления |
|---|---|---|---|
| arXiv | Научные статьи (cs.LG, cs.AI, NeurIPS, ICML, ICLR) | Chat | Инкрементальная (upsert по deterministic UUID) |
| PyTorch docs | Документация библиотек, туториалы, примеры кода | Code generation | Полная замена (blue-green с staging alias) |

## Метрики качества

* Chat: Relevance (1–5), Correctness (1–5), BERTScore, ROUGE-L
* Summarization: Faithfulness (1–5), Coverage (1–5), BERTScore, ROUGE-L
* Code generation: Executable rate, pass@1
* RAG-specific: Recall@k, nDCG@k, Groundedness

## Данные и датасеты

В проекте используются три категории данных с различным назначением.

### 1. Данные для обучения LoRA адаптеров

| Задача | Датасет | Назначение |
|---|---|---|
| Суммаризация | `ccdv/arxiv-summarization` (train, 203k примеров) | Fine-tuning LoRA для summarization |
| Генерация кода | `nvidia/OpenCodeInstruct` (train, 5M примеров, фильтр: Python + ML/DL) | Fine-tuning LoRA для code generation |

### 2. Данные для оценки (Evaluation Datasets)

Все перечисленные датасеты используются **на каждой** оценке соответствующей задачи.

**Генерация:**

| Задача | Датасет | Используется в этапах |
|---|---|---|
| Chat (QA) | HotpotQA (validation) | Этапы 1–4 |
| Chat (QA) | Natural Questions (validation) | Этапы 1–4 |
| Summarization | `ccdv/arxiv-summarization` (validation, 6.4k примеров) | Этапы 1–4 |
| Code generation | `openai/openai_humaneval` (test, 164 примера) | Этапы 1–4 |

**Retrieval (без генерации):**

| Датасет | Метрики | Используется в этапах |
|---|---|---|
| MS MARCO (validation) | Recall@k, nDCG@k | Этапы 2–4 |
| BEIR‑SciFact (corpus) | Recall@k, nDCG@k | Этапы 2–4 |
| BEIR‑NFCorpus (corpus) | Recall@k, nDCG@k | Этапы 2–4 |

### 3. Knowledge Corpora для RAG-системы

RAG использует недетерминированные, обновляемые источники, не входящие в supervised-датасеты.

| Knowledge Base | Тип данных | Задачи |
|---|---|---|
| arXiv | Научные статьи (NeurIPS, ICML, ICLR), блоги по ML/DL | Chat |
| PyTorch docs | Документация библиотек, туториалы, примеры кода | Code generation |

### 4. Управление данными и воспроизводимость

* **DVC** (Yandex Cloud S3 remote): хранение датасетов, фиксация preprocessing шагов,
  воспроизводимость экспериментов.
* **MLflow**: хранение метрик, логирование параметров обучения, сравнение LoRA-адаптеров.

### 5. Model Registry и управление адаптерами (MLflow Model Registry)

Для обеспечения плавного перехода от экспериментов к production используется **MLflow Model
Registry** — единый реестр версионированных LoRA-адаптеров с alias-based lifecycle management.

#### Жизненный цикл адаптера

```
train_adapter                  lora_ops.ipynb                sync (model_registry.py)
─────────────                  ────────────────────          ──────────────────────────
  Обучение LoRA                  Просмотр метрик              Скачивание aliased
       ↓                        в MLflow UI                   адаптеров из S3
  Логирование метрик                   ↓                              ↓
  и артефактов в MLflow          register run → v4             Hot-load в vLLM
  Tracking                             ↓                       через REST API
                                 promote v4 → champion         (без рестарта)
```

#### Ключевые концепции

* **Registered Model** — именованная группа адаптеров (например, `lora-summarize`,
  `lora-code`, `lora-chat`). Имя соответствует задачам в `TaskRouter`.
* **Model Version** — каждая регистрация создаёт новую версию. Версии иммутабельны.
* **Aliases** — метки жизненного цикла:
  * `champion` — production-адаптер, загружаемый в vLLM.
  * `challenger` — кандидат на A/B-тестирование или ручную оценку.

#### Инфраструктура

* **Registry backend**: PostgreSQL (тот же, что для MLflow Tracking).
* **Artifact storage**: Yandex Object Storage (S3) — адаптеры хранятся рядом с MLflow-артефактами.
* **Hot-load sync**: `python -m shared.model_registry sync`
  скачивает aliased-адаптеры в `assets/adapters/{model}/v{N}/` и загружает их в работающий
  vLLM через `POST /v1/load_lora_adapter`. В vLLM адаптер регистрируется как `{model}-{alias}`
  (например, `lora-summarize-champion`).
* **vLLM multi-LoRA**: запускается с `--enable-lora` и `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`;
  адаптеры загружаются/выгружаются без рестарта сервера.
* **Adapter sync service**: Docker-контейнер `vllm-adapter-sync` автоматически синхронизирует
  MLflow Model Registry с vLLM при изменении alias-ов.

**Подробности использования**: `./experiments/README.md` → раздел «Model Registry».

> **Примечание**: Training orchestration пока заканчивается на `train → inspect/promote`;
> автоматический шаг `train → evaluate → human decision` не подключен

### 6. Версионирование RAG-индексов

`src/shared/knowledge_bases.json` — runtime registry knowledge bases. Он сгруппирован по task-ам и
хранит только пользовательский каталог KB: task → knowledge base → aliases, `update_strategy`,
label и description. Параметры сборки (`chunking_strategy`, `chunk_size`, `chunk_overlap`,
embedding model) не лежат в JSON и сохраняются только в `_meta` sentinel внутри Qdrant
collection, откуда их читают production refresh workflows и retrieval-only evals.

#### `_meta` sentinel

Каждая RAG-коллекция содержит служебную точку `_meta` — запись в Qdrant с детерминированным
UUID (`uuid.uuid5` от фиксированного namespace и ключа `"_meta"`), нулевым вектором (не влияет
на similarity search) и метаданными в payload:

```json
{
  "type": "collection_meta",
  "kb_name": "arxiv",
  "build_config": {
    "chunking_strategy": "fixed_token",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "created_at": "2025-01-01T12:00:00Z",
  "implementation": {
    "module": "rag.ops.create.arxiv",
    "experimental": false,
    "identifier": null,
    "git_sha": "abc1234"
  }
}
```

Поисковые запросы автоматически исключают `_meta` через фильтр
`must_not: [{"key": "type", "match": {"value": "collection_meta"}}]`.

Это позволяет DAG-ам и production workflows не хранить конфиг внешне — вся информация для
rebuild-а читается из `_meta` существующей production-коллекции.

#### Версионирование и снятие снимков

Для обеспечения воспроизводимости RAG-системы, векторные индексы Qdrant также подлежат
версионированию:

* **Qdrant snapshots**: встроенный механизм снимков коллекций (`POST /collections/{name}/snapshots`).
* **DVC**: снимки индексов хранятся в Yandex Cloud S3 через DVC, аналогично датасетам.
* **Связь с адаптерами**: в тегах model version в MLflow фиксируется версия RAG-индекса,
  которая использовалась при оценке адаптера, обеспечивая полную воспроизводимость.

#### Alias-based lifecycle

Для безопасного сравнения retrieval-архитектур без полного release вводится alias-based lifecycle:

* **RAG alias `champion`** — production retrieval-конфигурация.
* **RAG alias `challenger`** — кандидатная retrieval-конфигурация (новый индекс, chunking,
  reranking, top-k и т.д.).
* **Atomic switch** — promotion выполняется через переключение alias на новую коллекцию.
* **Naming convention**: `{kb_name}_{alias}` (например, `arxiv_champion`, `pytorch_docs_challenger`).
  Physical collection names: `{kb_name}_{YYYYMMDD_HHMMSS}`.
* **Production entrypoints** — создание, refresh, alias-management и inspection живут только в
  `src/rag/ops/`.
* **Sandbox boundary** — `experiments/rag/sandboxes/` предназначен только для notebook-only
  экспериментального кода; Gateway, Airflow и production evals не импортируют его.

#### Политика обновления индексов

* `arxiv_rag_update` вызывает `rag.ops.update.update_arxiv_collection(kb="arxiv", alias="champion")`.
  Функция читает `_meta` champion-коллекции и инкрементально обновляет именно её.
  ArXiv использует deterministic point IDs (`uuid.uuid5` от `arxiv_id:chunk_idx`), что позволяет
  перезаписывать существующие чанки без дубликатов.
  Непродовые alias-ы можно refresh-ить вручную только из notebook path.
* `pytorch_docs_rag_update` вызывает
  `rag.ops.update.update_pytorch_docs_collection(kb="pytorch_docs", alias="champion")`.
  Функция читает `_meta` champion-коллекции, создаёт successor collection, вешает staging alias
  `{kb}_{alias}_staging` и затем атомарно перепривязывает champion. Старая коллекция удаляется,
  если на неё не указывают другие alias-ы.
* Gateway при старте валидирует все `(kb, alias)` из runtime registry против Qdrant и заранее
  помечает отсутствующие alias-ы как unavailable с warning-логами.

Выбор стратегии обновления:
* **Инкрементальная (arXiv)**: статьи иммутабельны, новые добавляются upsert-ом. Безопасно
  обновлять коллекцию in-place.
* **Полная замена (PyTorch docs)**: документация может значительно меняться. Blue-green deployment
  гарантирует атомарный переход без stale данных mid-update.

#### Автоматическая очистка коллекций

DAG `rag_collection_cleanup` (расписание: `@daily`) удаляет orphan-коллекции в Qdrant, которые
не привязаны ни к одному alias-у и чей timestamp старше 7 дней. Legacy коллекции (`chat_documents`,
`code_documents`) находятся в skip-list и не удаляются автоматически.

#### Production policy

* На production **всегда должен существовать alias `champion`**.
* На production **могут существовать дополнительные alias-ы** (`challenger` и др.) для тестов и
  валидации.
* Параметры `top_k`, `score_threshold`, `context_max_length`, `reranker` являются
  alias-owned конфигурацией в `knowledge_bases.json`. Каждый alias несёт свои
  параметры — champion и challenger могут различаться без рестарта сервиса.
* Изменение query-config: редактирование `knowledge_bases.json` +
  `POST /v1/admin/reload-config` (authenticated).
* Build-time параметры (`chunking_strategy`, `embedding_model`, `sparse_encoder`,
  `retrieval_strategy`) хранятся в Qdrant `_meta.build_config` и меняются только
  через rebuild коллекции.

#### Политика маршрутизации трафика

* Production inference по умолчанию использует `default_alias` из `knowledge_bases.json`
  (обычно `"champion"`).
* Непродовые alias-ы (`challenger` и др.) используются для eval/тестов/ручных проверок.
* Создание новых challenger-коллекций и alias promotion выполняются через
  `experiments/rag/rag_ops.ipynb`, а не отдельным CLI.

## Архитектура оценки

Пайплайн оценки разделён на generation evals и retrieval-only evals.

* **Generation evals** (`chat`, `summarize`, `code`) вызывают Gateway API
  `POST /v1/chat/completions`. Gateway остаётся единственным источником истины для prompt
  assembly, LoRA selection, RAG retrieval и inference.
* **Retrieval-only evals** строят временные benchmark collections в Qdrant из corpora BEIR /
  MS MARCO, используя те же build params, что и выбранный production alias: конфиг читается из
  `_meta` соответствующей collection.

### Модули метрик

Метрики реализованы в `experiments/eval/eval_scripts/metrics/` тремя независимыми модулями:

| Модуль | Метрики | Способ вычисления |
|--------|---------|-------------------|
| `automatic.py` | ROUGE-L, BERTScore, Recall@k, nDCG@k | Локальное вычисление, без внешних API |
| `llm_judge.py` | Relevance, Correctness, Faithfulness, Coverage, Groundedness (1–5) | Google Gemini 2.0 Flash API с rate limiting (15 RPM) |
| `code_exec.py` | pass@1, executable_rate | Извлечение кода из ответа LLM и исполнение в изолированном `code-sandbox` контейнере |

**Code sandbox** для безопасного выполнения кода:
* Отдельный Docker-контейнер (`code-sandbox`) на базе `python:3.13-slim`
* Read-only root filesystem, tmpfs `/tmp`
* Ограничения: 1 CPU, 256 MB RAM
* Изолирован в `sandbox_net` — без доступа к интернету или другим сервисам
* Запускается как non-root пользователь (`sandbox`)

### Airflow orchestration

`dags/eval_dags.py` создаёт шесть Airflow DAG-ов, по одному на каждую пару `(task, dataset)`:

| DAG | Task | Dataset |
|-----|------|---------|
| `eval_chat_hotpotqa` | chat | HotpotQA |
| `eval_chat_nq` | chat | Natural Questions |
| `eval_summarize_arxiv` | summarize | ArXiv summarization |
| `eval_code_humaneval` | code | HumanEval |
| `eval_retrieval_beir_scifact` | retrieval | BEIR-SciFact |
| `eval_retrieval_beir_nfcorpus` | retrieval | BEIR-NFCorpus |

Каждый DAG содержит два последовательных шага:
1. **`fetch_predictions`**: вызывает Gateway API (для generation) или Qdrant напрямую (для
   retrieval). Predictions передаются между шагами через временные JSON-файлы (избегает
   ограничений XCom на размер).
2. **`calculate_metrics`**: вычисляет выбранную метрику. Результаты записываются в PostgreSQL
   таблицы `eval_runs` и `eval_samples`.

Конкретная метрика и матрица по `rag_aliases` / `lora_aliases` выбираются при trigger-time
через параметры Airflow UI.

### Two-step runner

`experiments/eval/eval_scripts/runner.py` реализует standalone runner для оценки:
* Разделяет `fetch_predictions()` и `calculate_metrics()`
* Поддерживает три этапа оценки (stage 1: base LLM, stage 2: + RAG, stage 3: + LoRA)
* Результаты пишет в PostgreSQL tables `eval_runs` и `eval_samples`
* CLI: `python -m experiments.eval.eval_scripts.runner --task chat --dataset hotpotqa --metric rouge_l`

### Operator path

Просмотр eval-таблиц, сравнение конфигураций и трендов находится в
`experiments/eval/eval_results.ipynb`. Debug-notebook `experiments/eval/debug_eval.ipynb`
используется для отладки отдельных шагов eval pipeline.

## Обучение LoRA адаптеров

Обучение адаптеров построено на стеке **PyTorch Lightning + Hydra + PEFT + BitsAndBytes**
и расположено в `experiments/training/`.

### Архитектура training pipeline

```
Hydra Config (conf/)
      │
      ▼
start_train.py          @hydra.main → инициализация конфига
      │
      ├── modeling.py       загрузка базовой LLM + 4-bit quantization + PEFT LoRA
      ├── data_module.py    ArxivDataModule: tokenization, sequence budget, batching
      ├── lit_module.py     PeftCausalLMModule: training/validation step, optimizer, scheduler
      └── mlflow_utils.py   логирование в MLflow Tracking
      │
      ▼
PyTorch Lightning Trainer
      │
      ├── Training loop с gradient checkpointing
      ├── Validation с мониторингом val_loss
      ├── Checkpointing лучших моделей
      └── MLflow: метрики (train_loss, val_loss, lr, tokens/sec, GPU memory)
```

### Ключевые компоненты

* **`PeftCausalLMModule`** (`lit_module.py`): Lightning Module, оборачивающий PEFT-fine-tuned
  causal LM. Поддерживает AdamW optimizer с cosine/linear scheduling, tracking tokens/sec
  и GPU memory usage.
* **`ArxivDataModule`** (`data_module.py`): Lightning DataModule для on-the-fly tokenization
  ArXiv article/abstract пар. Контролирует sequence budget (source + target + EOS ≤ max_seq_length).
* **Quantization**: 4-bit quantization через BitsAndBytes (NF4) для обучения на ограниченных
  GPU ресурсах.
* **PEFT**: LoRA адаптеры с конфигурируемыми `r`, `lora_alpha`, `lora_dropout`, `target_modules`.
* **Gradient checkpointing**: для уменьшения memory footprint при обучении.

### Hydra конфигурация

Конфиги в `experiments/training/conf/`:
* `config.yaml` — точка входа, default experiment
* `experiment/train_adapter.yaml` — полная спецификация обучения
  (model, lora, data, trainer, scheduler, logger, tracking, evaluation)
* `paths/paths_config.yaml` — `project_root` (переопределяется за CLI для каждой машины)
* Output directory: `artifacts/training/hydra/{date}/{time}/`
* Поддержка multi-run sweeps: `python -m ... -m experiment.training.lr=1e-4,5e-5`

### MLflow интеграция

`mlflow_utils.py` логирует гиперпараметры, метрики и артефакты в MLflow Tracking.
Регистрация адаптера в Model Registry и promotion alias-ов выполняются **отдельным шагом**
из `experiments/training/lora_ops.ipynb`, а не автоматически при обучении.

### Airflow DAG

DAG `train_lora` (`dags/train_lora.py`) запускает тренировку как subprocess (для изоляции
Hydra global state) на GPU worker (`queue="gpu"`, concurrency 1). Параметры
`experiment_config` и `hydra_overrides` (JSON array) задаются через Airflow UI при trigger-time.

## Airflow DAG-и

Airflow используется для оркестрации тяжёлых вычислительных задач. Все DAG-и расположены в `dags/`:

| DAG | Расписание | Worker Queue | Назначение |
|-----|-----------|-------------|------------|
| `train_lora` | Manual trigger | `gpu` (concurrency 1) | Обучение LoRA адаптера |
| `eval_chat_hotpotqa` | Manual trigger | default (CPU) | Eval: chat на HotpotQA |
| `eval_chat_nq` | Manual trigger | default | Eval: chat на Natural Questions |
| `eval_summarize_arxiv` | Manual trigger | default | Eval: summarization на ArXiv |
| `eval_code_humaneval` | Manual trigger | default | Eval: code на HumanEval |
| `eval_retrieval_beir_scifact` | Manual trigger | default | Eval: retrieval на BEIR-SciFact |
| `eval_retrieval_beir_nfcorpus` | Manual trigger | default | Eval: retrieval на BEIR-NFCorpus |
| `arxiv_rag_update` | `@daily` | default | Скачивание статей, DVC, обновление RAG |
| `pytorch_docs_rag_update` | `@weekly` | default | Скрапинг docs, DVC, обновление RAG |
| `rag_collection_cleanup` | `@daily` | default | Удаление orphan-коллекций Qdrant (retention 7 дней) |
| `simple_dag` | `@daily` | default | Тестовый DAG (hello world) |

**Worker routing**: Airflow использует Celery executor с двумя очередями:
* **Default queue** (CPU, concurrency 2): evals, RAG updates, cleanup
* **GPU queue** (concurrency 1): LoRA training

## База данных

PostgreSQL 15 обслуживает три логических домена:

### application database `agent042`

| Таблица | Назначение |
|---------|------------|
| `users` | Google OIDC пользователи (provider, sub, email, name, picture) |
| `chat_sessions` | Per-user сессии чата (title, timestamps) |
| `chat_messages` | Сообщения (role: user/assistant, content, timestamps) |
| `eval_runs` | Метрики оценки: task, dataset, metric, model/adapter info, RAG config, status |
| `eval_samples` | Per-sample eval details: input, output, reference, JSONB detail |

`eval_runs` хранит полный контекст каждого eval run: task, dataset, metric name/value, base model,
adapter name/version/MLflow run ID, lora_alias, rag_alias, knowledge base, qdrant collection,
chunking strategy, chunk_size, top_k, score_threshold, generation params (temperature, max_tokens),
judge model, status (running/completed/failed), timestamps и JSONB `extra` для расширяемости.

### MLflow database

Хранит experiment metadata, run tracking и Model Registry (адаптеры и их версии).

### Airflow metadata database

Хранит DAG definitions, task instances, XCom и scheduler state.

Все три домена используют один PostgreSQL-инстанс. ORM table creation используется для bootstrap-а
schema.

> **Примечание**: Alembic migrations для `agent042` DB не заведены — schema bootstrap по-прежнему
> опирается на ORM `Base.metadata.create_all` в gateway startup (см. `REMAINING-CHANGES.md` §2.8).

## Развёртывание и инфраструктура

### Docker Compose

Основной deployment manifest — `infra/compose/docker-compose.yaml`. Он описывает все сервисы,
сети, volumes и зависимости. Compose оркестрирует:

* **Инициализацию**: `airflow-prepare-dirs` создаёт writable директории для RAG data, training
  artifacts, DVC. `airflow-init` выполняет database creation, admin user seeding, Airflow
  migrations.
* **Health checks**: каждый сервис имеет health check (vLLM проверяет `/v1/models`, Gateway —
  `/health`, Qdrant — `/healthz`), с dependency ordering через `depends_on.condition`.
* **GPU support**: `airflow-worker-gpu` и `vllm` используют `deploy.resources.reservations` с
  `capabilities: [gpu]`.
* **Environment**: Compose env vars определяют container-to-container wiring (внутренние URL,
  порты, credentials). Operator-editable values (секреты, порты, модель) берутся из `.env`.

### Dockerfile-ы

Каждый сервис имеет свой Docker-образ в `infra/docker/`:

| Образ | Base image | Особенности |
|-------|-----------|-------------|
| `gateway` | `python:3.12-slim` | FastAPI, qdrant-client, authlib, sqlalchemy |
| `embeddings` | `python:3.12-slim` | sentence-transformers, PyTorch |
| `airflow` / `airflow-worker` | `apache/airflow:3.1.8` | bert-score, torch (CPU), qdrant, DVC, arxiv |
| `airflow-worker-gpu` | `apache/airflow:3.1.8` | + CUDA torch, PEFT, Lightning, Hydra |
| `code-sandbox` | `python:3.13-slim` | Минимальный образ, non-root user, без лишних пакетов |
| `ui` | `python:3.12-slim` | Streamlit |
| `celery` | `python:3.12-slim` | Celery worker для async inference |
| `adapter-sync` | `python:3.12-slim` | MLflow client, vLLM REST API |

Lock-файлы зависимостей (`requirements-*.lock`) перегенерируются через `scripts/update_locks.sh`.

### Nginx Reverse Proxy

Nginx (`infra/nginx/agent.antonlab.ru.conf`) обеспечивает единую точку входа с HTTPS
(Let's Encrypt) и path-based routing:

| Path | Upstream | Auth | Назначение |
|------|----------|------|------------|
| `/` | Streamlit UI :8501 | Application | Чат-интерфейс (WebSocket support) |
| `/auth/` | Gateway :9000 | нет | OAuth2 flow |
| `/api/` | Gateway :9000 | Application | REST API (prefix strip, long timeouts, no buffering) |
| `/airflow/` | Airflow :8080 | Airflow own | DAG management |
| `/jupyter/` | Jupyter :8888 | Token | Operator notebooks (WebSocket support) |
| `/mlflow/` | MLflow :5050 | Basic auth | Experiment tracking |
| `/flower/` | Flower :5555 | Basic auth | Celery monitoring |
| `/redis-insight/` | RedisInsight :5540 | Basic auth | Redis monitoring |
| `/rabbitmq/` | RabbitMQ :15672 | RabbitMQ own | Queue management |

Security headers: `X-Frame-Options: SAMEORIGIN`, `X-Content-Type-Options: nosniff`,
`X-XSS-Protection: 1; mode=block`. SSL: TLSv1.2/1.3, modern cipher suites.

Внутренние сервисы (vLLM, Qdrant, Redis, PostgreSQL, RabbitMQ AMQP) привязаны к `127.0.0.1`
и недоступны извне.

## Мониторинг и Observability

| Инструмент | Доступ | Назначение |
|-----------|--------|------------|
| **MLflow** | `/mlflow/` | Просмотр метрик обучения, сравнение run-ов, Model Registry |
| **Airflow** | `/airflow/` | Мониторинг DAG-ов, history, логи task-ов |
| **Flower** | `/flower/` | Celery worker status, active tasks, task history |
| **RedisInsight** | `/redis-insight/` | Redis keys, Pub/Sub каналы, memory usage |
| **RabbitMQ Management** | `/rabbitmq/` | Queues, exchanges, connections, message rates |
| **Gateway `/health`** | `/api/health` | Health check для мониторинг систем |
| **Gateway `/config`** | `/api/config` | Текущая runtime-конфигурация gateway |

> **Примечание**: token/cost tracking и более полная observability для LLM path являются
> запланированными улучшениями (см. `REMAINING-CHANGES.md` §2.5–2.6, §2.9).

## Тестирование

Тесты расположены в `tests/` и покрывают ключевые компоненты системы:

| Модуль | Файлы | Что проверяет |
|--------|-------|---------------|
| API | `tests/api/test_rag_lifecycle.py` | KB config loader, RAGSource schema, ChatCompletionRequest с rag_sources, metadata exclusion filter |
| Auth | `tests/auth/test_middleware.py` | AuthMiddleware routing, public/protected paths, session validation |
| Auth | `tests/auth/test_csrf.py` | CSRF protection: state parameter validation в OAuth callback, missing/invalid state rejection |
| Auth | `tests/auth/test_oidc.py` | OIDC client integration |
| Eval | `tests/eval/test_eval_workflow.py` | EvalRun model, required DB columns (40+ fields), default status |
| RAG | `tests/rag/test_ops_meta.py` | BuildConfig serialization/deserialization, CollectionMeta with ImplementationInfo |
| RAG | `tests/rag/test_ops_aliases.py` | Alias operations |
| Training | `tests/training/test_local_env.py` | Local environment setup, repo root detection |

Тесты запускаются через `pytest`. Pre-commit hooks обеспечивают linting и formatting при каждом
коммите.

## Зависимости и tooling проекта

### pyproject.toml

Проект использует `pyproject.toml` с **модульными extras** для изоляции зависимостей по
сервисам:

| Extra | Назначение | Ключевые пакеты |
|-------|------------|-----------------|
| `gateway` | FastAPI сервис | FastAPI, uvicorn, qdrant-client, authlib, sqlalchemy[asyncio], asyncpg |
| `ui` | Streamlit UI | Streamlit, requests |
| `worker` | Celery worker | celery, redis, httpx |
| `rag` | RAG pipeline | sentence-transformers, langchain-text-splitters, arxiv, pypdf |
| `embeddings` | Embedding service | sentence-transformers |
| `training` | LoRA training | torch, transformers, peft, pytorch-lightning, hydra-core, bitsandbytes, datasets |
| `airflow-worker` | CPU Airflow worker | bert-score, rouge-score, torch (CPU), qdrant-client, dvc[s3], arxiv |
| `airflow-worker-gpu` | GPU Airflow worker | + CUDA torch, peft, pytorch-lightning, hydra |
| `mlflow` | MLflow server | mlflow, psycopg2-binary, boto3 |

Python version: 3.12–3.13.

### Утилитные скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/update_locks.sh` | Перегенерация pip lock-файлов для Docker-образов |
| `scripts/dump_docker_logs.sh` | Экспорт логов из работающих контейнеров |
| `scripts/fetch_logs_ssh.sh` | Удалённое извлечение логов через SSH |

## Workflow automation and CI/CD

### Branch: experiments

Ветка Experiments используется для экспериментов с обучением LoRA и аналогичных. Пайплайны
экспериментов, которые подтвердили свою успешность, могут быть смержены в Main.

**Pre-commit:**

* ruff check (linting + import sorting, с автофиксом)
* ruff format (форматирование кода)
* проверка YAML/JSON
* trailing whitespace, end-of-file-fixer, mixed-line-ending, check-case-conflict
* check-added-large-files

### Branch: develop (inference dev)

Эта ветка используется для разработки всей Inference части сервиса. Push в develop ветку, merge
request из develop в main.

**Pre-commit:**

По сути то же самое, что и в Experiments (единый `.pre-commit-config.yaml` в корне проекта).

### Branch: main

Эта ветка содержит рабочую версию inference-сервиса.

**CI:** quality gates выполняются через локальные `pre-commit`, `ruff`, `pytest` и ручную
валидацию сервисов перед деплоем.

**CD:** деплой выполняется через Docker Compose и Nginx; Airflow DAG-и и notebook-ы входят в
операционный контур развёртывания и сопровождения.

> **Примечание**: Hosted CI/CD workflows (GitHub Actions и аналоги) являются запланированным
> улучшением (см. `REMAINING-CHANGES.md` §2.7).

## Незавершенные изменения и будущая работа

Полный список незавершенных изменений и планируемых доработок ведётся в `REMAINING-CHANGES.md`.
