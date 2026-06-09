# Agent 042

## Инструкции

* `./infra/README.md` - настройка окружения и инфраструктуры
* `./experiments/README.md` - как проводить эксперименты и operator workflows
* `./CONFIG-REFACTOR-PLAN.md` - план упрощения и перестройки runtime-конфигов

## Контракт конфигурации

Runtime-конфигурация Python-сервисов теперь фиксируется так:

* единственная env-loading boundary - root `Settings(BaseSettings)` в `src/shared/config.py`
* канонические runtime env names используют nested shape `SECTION__FIELD`
* task/KB catalog schema и loader живут в `src/shared/catalog/`, а не в `shared.config`

Практически это означает:

* для runtime-настроек используйте имена вроде `GATEWAY__DEFAULT_MODEL`, `RAG__EMBEDDING_MODEL`, `CATALOG__PATH`, `ADAPTER_REGISTRY__SYNC_ALIASES`, `EVAL__JUDGE__MODEL`
* flat compatibility aliases для runtime env больше не поддерживаются
* если нужен operator-facing env key, он должен быть задокументирован в `.env.example` и `infra/README.md`

## Цель проекта и позиционирование

Цель проекта - создать ИИ-ассистента с RAG и LoRA.

Запустить LLM и заставить ее отвечать на вопросы - тривиальная задача. Прикрутить к LLM RAG систему тоже не трудно. Настроить LoRA training pipeline для базовой LLM можно за вечер. Поднять полноценное приложение на выделенном сервере с прописанными принципами разработки и экспериментов, надежным хранением всех данных, автоматизированными пайплайнами обучения, обновления и оценки, прозрачным мониторингом - оставшиеся 99% усилий.

Данный проект направлен на разработку инфраструктуры и workflows для ИИ-ассистента production-уровня, сохраняя при этом парадигму single-repository.

Пример целевого использования - команды исследователей, которым нужен ИИ-ассистент для работы с NDA данными и базами знаний.

## Возможности системы

<img src="schema.png" alt="Architecture overview" width="1600"/>

**UX и пользовательские возможности.**

UI с ИИ-ассистентом. Поддерживается Google-авторизация, история сообщений и streaming-инференс в UI. Основные типы задач: чат, суммаризация документов и генерация кода. В зависимости от типа задач бэкенд может задействовать поиск в RAG коллекциях и соответвтующий LoRA адаптер.

**Backend и инфраструктура.**

* Инференс платформа. Основная часть, которая обеспечивает работу ИИ-ассистента. Состоит из следующих микросервисов:
    * `gateway` - API gateway для аутентификации, маршрутизации запросов, prompt assembly, RAG и streaming-ответов.
    * `vllm` - основной inference engine с OpenAI-совместимым API и поддержкой hot-loading LoRA адаптеров.
    * `celery-worker` - асинхронное выполнение inference-задач и стриминг токенов обратно в пользовательский контур.
    * `embeddings` - отдельный сервис для dense и sparse embeddings.
    * `reranker` - отдельный сервис для cross-encoder reranking.
    * `vllm-adapter-sync` - синхронизация LoRA артефактов из registry в serving-контур.
    * `qdrant` - векторное хранилище для RAG коллекций
    * `redis` - сессии, вспомогательное состояние и pub/sub для streaming-ответов.
    * `rabbitmq` - брокер очередей для фоновых inference и workflow-задач.
    * `postgres` - БД для пользовательских сущностей, истории диалогов, а также для backend микросервисов.

* Платформа для экспериментов с RAG и LoRA. Включает все необходимое для проведения экспериментов с логированием, версионированием, автоматизацией отдельных шагов и бенчмарками. Состоит из следующих микросервисов:
    * `airflow-webserver`, `airflow-scheduler`, `airflow-dag-processor` - основные Airflow сервисы.
    * `airflow-worker` - CPU-воркер для бенчмарков, обновления RAG и прочих тяжелых фоновых задач без GPU.
    * `airflow-worker-gpu` - GPU-воркер для обучения LoRA.
    * `rag-ops` - одноразовый CLI-runner для ручных RAG-операций внутри docker network; использует тот же образ и маунты, что и `airflow-worker`.
    * `jupyter` - точка входа для ручных экспериментов, анализа результатов и точечных операций.
    * `mlflow` - трекинг экспериментов, model registry.
    * `code-sandbox` - изолированная среда выполнения кода для бенчмарков с запуском кода.

* Платформа для мониторинга и аналитики. Observability обеспечивается следующими микросервисами:
    * `prometheus` - сбор технических метрик с inference и инфраструктурных сервисов.
    * `grafana` - дашборды для инфраструктурной observability и аналитики по ML-процессам.
    * `flower` - мониторинг очередей и состояния Celery workers.
    * `redisinsight` - мониторинг Redis-состояния, ключей и pub/sub активности.
    * `mlflow` - аналитика по training runs, параметрам, метрикам и артефактам экспериментов.

**Workflows и проработанные пайплайны.**

* Эксперименты с RAG
    * Загрузка и подготовка данных, версионирование датасетов
    * Создание коллекций
    * Alias-based конфиги RAG-а

**RAG operations entry point.**

Manual RAG lifecycle commands should run through the `rag-ops` Compose service
so they execute inside the same Docker network and dependency image as Airflow
worker tasks:

```bash
bash scripts/rag_ops.sh python -m rag.sources.cli build-source \
  --catalog src/shared/catalog.toml \
  --kb pytorch_reference \
  --source docs \
  --rag-data-root assets/rag_data \
  --limit 1
```

Use Jupyter for artifact inspection and curation, and Airflow for scheduled
orchestration; both should call the same Python/CLI lifecycle code.

See [docs/rag-operations.md](docs/rag-operations.md) for the full RAG lifecycle,
Airflow params, naming glossary, promotion, inspection, and rollback notes.
    * Бенчмарки
    * Автоматическое обновление коллекций

* Эксперименты с LoRA
    * Загрузка и подготовка данных, версионирование датасетов
    * Обучение. Конфигурирование и трекинг экспериментов, версионирование моделей
    * Бенчмарки, alias-based продвижение моделей в прод

* Разработка и CI/CD процессы. Автоматизироавны шаги:
    * проверка кода
    * тестирование
    * сборка образов
    * deploy

## Структура проекта

Проект собран как единое рабочее пространство, в котором рядом живут runtime-сервисы, инфраструктура, эксперименты и operator workflows. Основные разделы репозитория:

* `infra` - инфраструктура
* `src` - основной runtime код сервиса
* `experiments` - notebooks, training-код, eval-скрипты и операторские entrypoint'ы для ручной работы с RAG и LoRA
* `scripts` - служебные shell-скрипты
* `artifacts` и `assets` - весь "state" проекта
* `tests` - unit и integration тесты
* `dags` - Airflow DAG-и для обучения, evaluation, обновления RAG коллекций и других автоматизированных задач.
