# Настройка окружения

Этот репозиторий поддерживает запуск через Docker Compose.
В одном `docker-compose.yaml` собраны:
- MLflow Tracking Server + PostgreSQL (backend store)
- vLLM (OpenAI-compatible LLM inference server) с доступом к GPU
- Qdrant (векторная БД)
- RabbitMQ (брокер сообщений для Celery)
- Redis (pub/sub для потоковой передачи токенов)
- Celery worker (асинхронное выполнение LLM-задач)
- Gateway (FastAPI)
- UI (Streamlit)
- Flower (мониторинг Celery)
- RedisInsight (мониторинг Redis)
- Apache Airflow (LocalExecutor) — оркестрация пайплайнов, использует общий PostgreSQL
- JupyterLab — интерактивная среда для экспериментов

## Контракт Phase 1: серверный корень

Phase 1 фиксирует операторский контракт для single-node deployment, не меняя пока текущий
checkout-based запуск Compose. Каноническая серверная раскладка теперь считается такой:

```text
/home/anton-m/agent-042/
  .env
  .dvc/
    config.local
  assets/
  artifacts/
  releases/
    <sha>/
  current -> /home/anton-m/agent-042/releases/<sha>/
```

Что это означает на практике:
- repo-root `.env` остаётся активным env-файлом для локального checkout и текущего Compose запуска
- release-based deploy будет использовать внешний `/home/anton-m/agent-042/.env`, а не `.env` внутри релиза
- переменные `GITHUB_REPOSITORY`, `GITHUB_DATA_SYNC_TOKEN` и `IMAGE_TAG` остаются частью
  server contract; release code/config монтируются через `PROJECT_ROOT`, а persistent project data
  монтируется через `SHARED_ROOT`

## Требования

### Минимальные требования к железу

Текущий минимум для запуска полного стека (vLLM + эмбеддинги + Qdrant + UI):
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 12GB VRAM
- CPU: 4+ ядра (рекомендация)
- Диск: 30+ GB свободного места (модели + кэши + volume'ы; сильно зависит от выбранной модели)

Примечания:
- В Gateway по умолчанию используется лёгкая модель эмбеддингов `sentence-transformers/all-MiniLM-L6-v2`.

### Требования к ПО

- Docker Engine и Docker Compose v2 (`docker compose ...`)
- NVIDIA driver + NVIDIA Container Toolkit (для GPU внутри контейнеров)

Проверка:
```bash
docker --version
docker compose version
nvidia-smi
```

## Настройка инфраструктуры

### Yandex Cloud S3

См. отдельный файл: `infra/SETUP-YANDEX-CLOUD.md`.

### UV-менеджер

Нужен для запуска экспериментов/скриптов локально (вне Docker):

* Установить [UV-менеджер](https://docs.astral.sh/uv/getting-started/installation/)
* Запустить синхронизацию только нужных групп зависимостей:
```bash
uv sync --extra training --extra rag --group dev
```

Примеры выборочной установки:
```bash
# только gateway + worker + UI для локального сервиса
uv sync --extra gateway --extra worker --extra ui --group dev

# только инфраструктура MLflow
uv sync --extra mlflow
```

Сборка lock-файлов для Docker-сервисов (выполнять из корня репозитория):
```bash
# Обновить все lock-файлы разом:
ops/update_locks.sh

# Или только конкретные сервисы:
ops/update_locks.sh gateway airflow-worker

# Посмотреть список сервисов:
ops/update_locks.sh --list

# Проверить команды без выполнения:
ops/update_locks.sh --dry-run
```

## Docker / Docker Compose

### Что разворачивается в Compose

Файл: `infra/compose/docker-compose.yaml`.

Сервисы:
- `postgres` — PostgreSQL 15 для MLflow backend store и Airflow metadata (volume: `mlflow_pg_data`)
- `mlflow` — MLflow Tracking Server (порт хоста по умолчанию `5050` → контейнер `5000`)
- `vllm` — vLLM OpenAI server (порт хоста по умолчанию `8000` → контейнер `8000`), использует GPU
- `qdrant` — Qdrant (порт хоста по умолчанию `6333` → контейнер `6333`, volume: `qdrant_data`)
- `rabbitmq` — RabbitMQ (порт хоста по умолчанию `5672` → контейнер `5672`, Management UI: `15672`)
- `redis` — Redis 7 (порт хоста по умолчанию `6379` → контейнер `6379`, volume: `redis_data`)
- `redpanda` — Kafka-compatible broker для durable inference events (порт хоста по умолчанию `19092` → контейнер `19092`)
- `redpanda-console` — UI для инспекции Kafka/Redpanda topics (порт хоста по умолчанию `8081` → контейнер `8080`)
- `clickhouse` — аналитическое хранилище inference events из Redpanda (HTTP порт `8123`, native порт `9000`)
- `celery-worker` — Celery worker для асинхронного выполнения LLM-задач (1 процесс, GPU-bound)
- `gateway` — FastAPI gateway (порт хоста по умолчанию `9001` → контейнер `9000`)
- `ui` — Streamlit UI (порт хоста по умолчанию `8501` → контейнер `8501`)
- `embeddings` — Embedding microservice: dense vectors (`/v1/embeddings`) и sparse BM25 (`/v1/sparse-embeddings`) (порт хоста по умолчанию `8100`)
- `reranker` — Reranker microservice: cross-encoder scoring (`/v1/rerank`) (порт хоста по умолчанию `8101`)
- `flower` — Flower мониторинг Celery (порт хоста по умолчанию `5555` → контейнер `5555`)
- `redisinsight` — RedisInsight мониторинг Redis (порт хоста по умолчанию `5540` → контейнер `5540`)
- `prometheus` — Prometheus (порт хоста по умолчанию `9090` → контейнер `9090`); scrapes gateway, vLLM, RabbitMQ
- `loki` — Loki для хранения Docker/application logs (порт хоста по умолчанию `3100` → контейнер `3100`)
- `tempo` — Tempo для хранения OpenTelemetry traces (порт хоста по умолчанию `3200` → контейнер `3200`)
- `otel-collector` — OpenTelemetry Collector; принимает OTLP от Python-сервисов и отправляет traces в Tempo
- `alloy` — Grafana Alloy; читает Docker logs через Docker socket и отправляет их в Loki
- `grafana` — Grafana (порт хоста по умолчанию `3000` → контейнер `3000`); доступна через nginx `/grafana/`; datasources: Postgres, Prometheus, Loki, Tempo
- `airflow-init` — одноразовая миграция БД Airflow и создание admin-пользователя
- `airflow-webserver` — Airflow UI (порт хоста по умолчанию `8080` → контейнер `8080`)
- `airflow-scheduler` — Airflow Scheduler (LocalExecutor)
- `jupyter` — JupyterLab (порт хоста по умолчанию `8888` → контейнер `8888`)

### Подготовка переменных окружения

1) Для локального checkout создайте `.env` в корне репозитория на основе примера:
```bash
cp .env.example .env
```

2) Для серверного deploy используйте тот же набор ключей во внешнем файле
`/home/anton-m/agent-042/.env`.

3) При необходимости скорректируйте `PROJECT_ROOT` и остальные значения `# CHANGE ME!`.

Ключевые переменные:
- `PROJECT_ROOT` — host-side путь, который Compose использует для checkout-based mount'ов или для
  активного релиза
  - Linux пример: `/home/user/agent-042`
  - Windows пример (как в шаблоне): `C:/Users/user/MyGitRepos/agent-042`
- `SHARED_ROOT` — host-side путь к durable state root с `assets/`, `artifacts/` и `.dvc`;
  для release-based deploy это обычно `/home/anton-m/agent-042`, а не `current`
- `runtime.toml` и `catalog.toml` лежат в `PROJECT_ROOT` и монтируются в контейнеры
  как `/opt/agent/runtime.toml` и `/opt/agent/catalog.toml`
- `GITHUB_REPOSITORY` / `GITHUB_DATA_SYNC_TOKEN` — используются Airflow temp-clone DVC/Git sync
  helper'ом для push в bot branch и для открытия или обновления PR в GitHub
- `IMAGE_TAG` — будущий deployment-scoped tag для CI-built images; в текущем checkout-based
  Compose ещё не используется
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — креды для Yandex Object Storage (нужны MLflow)
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` — опциональная auth-пара для локальных MLflow-клиентов и ноутбуков
- `MLFLOW_*` — конфиг MLflow (backend store + artifact root)
- `VLLM__*` — bootstrap/env переменные самого vLLM контейнера (модель, dtype, quantization, scheduler caps)
- `NETWORK__*` — internal host/internal port/host port/scheme primitives для Compose-managed сервисов
- `PUBLIC__*` — публичный base URL и path prefixes, из которых Compose строит adapter env для сервисов
- `GATEWAY__API_KEY`, `AUTH__*`, `EVAL__JUDGE__API_KEY` — runtime secrets, которые Compose явно передаёт Python-контейнерам
- `RABBITMQ_DEFAULT_*` — native RabbitMQ логин/пароль
- `CLICKHOUSE_*` — native ClickHouse база, пользователь и пароль
- host/internal порты задаются через соответствующие `NETWORK__...__HOST_PORT` и
  `NETWORK__...__INTERNAL_PORT`
- `OTEL_TRACES_SAMPLER_ARG` — доля трассировки для `parentbased_traceidratio`; на старте `1.0`
- `GF_SECURITY_ADMIN_PASSWORD` — native пароль admin-пользователя Grafana
- `AIRFLOW__*` / `AIRFLOW_ADMIN_*` — native/env конфиг Airflow
- `JUPYTER_TOKEN` — native токен JupyterLab

Замечание:
- Канонический шаблон `.env.example` не содержит полные внутренние endpoint'ы.
  Python derives project-owned URLs from `NETWORK__...` via `app_config.runtime`.
- Compose derives code/config mounts from `PROJECT_ROOT`, durable data mounts from `SHARED_ROOT`;
  `IMAGE_TAG` controls image tags for deploy/local runs.
- Workflow для logs/traces/metrics описан в `docs/analytics/observability.md`; durable inference events описаны в `docs/analytics/inference-events.md`; ClickHouse analytics описана в `docs/analytics/clickhouse-analytics.md`.

### Разделение env surfaces

В проекте теперь есть жёсткое разделение между двумя типами переменных:

- native env keys для service/bootstrap concerns: image/build параметры, credentials и конфиг внешних сервисов вроде `RABBITMQ_DEFAULT_PASS`, `GF_SECURITY_ADMIN_PASSWORD`, `AWS_ACCESS_KEY_ID`
- project-owned nested env keys для topology/secrets: `NETWORK__...`, `PUBLIC__...`, `AUTH__...`
- runtime behavior lives in root `runtime.toml`

Практические правила:

- если ключ имеет upstream/native имя, сохраняйте native имя
- если ключ описывает project-owned endpoint topology, используйте `NETWORK__...`
- новые runtime settings добавляйте в `runtime.toml`, а не в env
- catalog-specific helpers и schema не должны документироваться как часть `app_config.runtime`; их владелец — `src/app_config/catalog/`

Типичные operator-facing nested keys из текущего контракта:

- `NETWORK__VLLM__INTERNAL_HOST`
- `NETWORK__VLLM__INTERNAL_PORT`
- `AUTH__INTERNAL_API_KEY`
- `RABBITMQ_DEFAULT_PASS`
- `POSTGRES_APP_DB`

### Подготовка shared roots и прав доступа (Phase 2)

В текущем rollout Phase 2 Compose всё ещё запускается из checkout, но mutable state уже должен
жить во внешнем `/home/anton-m/agent-042`. После удаления `airflow-prepare-dirs` эти bind mount'ы нужно
подготовить на хосте заранее; иначе Airflow и Jupyter увидят каталоги, но не смогут в них писать.

Рекомендуемый путь — helper из корня checkout:

```bash
sudo bash bootstrap/setup_shared_root_permissions.sh --deploy-user "<server-login>"
```

Полезные override-флаги:
- `--server-root /home/anton-m/agent-042`
- `--env-file .env`
- `--compose-file infra/compose/docker-compose.yaml`
- `--airflow-uid <uid>` / `--airflow-gpu-uid <uid>` / `--jupyter-uid <uid>` если UID нужно
  передать вручную, не полагаясь на Compose
- `--skip-usermod`, если пользователя не нужно добавлять в host-группу

Ниже — ручной эквивалент того, что делает helper:

1) Создайте shared roots и отдельную host-группу для операторского обслуживания:
```bash
export DEPLOY_USER="<server-login>"
export CHECKOUT_ROOT="<server-checkout-path>"

sudo apt-get install -y acl
sudo groupadd --force agent042
sudo usermod -aG agent042 "$DEPLOY_USER"

sudo install -d -o "$DEPLOY_USER" -g agent042 -m 2775 \
  /home/anton-m/agent-042 \
  /home/anton-m/agent-042/assets \
  /home/anton-m/agent-042/assets/models \
  /home/anton-m/agent-042/assets/adapters \
  /home/anton-m/agent-042/assets/datasets \
  /home/anton-m/agent-042/assets/rag_data \
  /home/anton-m/agent-042/artifacts \
  /home/anton-m/agent-042/artifacts/training \
  /home/anton-m/agent-042/.dvc
```

2) Узнайте реальные UID контейнерных пользователей. Для нового сервера удобнее делать это через
`docker compose run --no-deps`, а если стек уже поднят, можно заменить команды на `docker compose exec`.
```bash
AIRFLOW_UID="$(docker compose --env-file .env -f infra/compose/docker-compose.yaml run --rm --no-deps --entrypoint id airflow-worker -u | tr -d '\r')"
JUPYTER_UID="$(docker compose --env-file .env -f infra/compose/docker-compose.yaml run --rm --no-deps --entrypoint id jupyter -u | tr -d '\r')"
echo "airflow uid: $AIRFLOW_UID"
echo "jupyter uid: $JUPYTER_UID"
```

3) Назначьте group ownership, setgid и default ACL на writable shared roots:
```bash
sudo chgrp -R agent042 /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts /home/anton-m/agent-042/.dvc
sudo find /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts -type d -exec chmod 2775 {} +
sudo setfacl -R -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts
sudo setfacl -R -d -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts
```

Если `airflow-worker-gpu` возвращает другой UID, добавьте для него те же ACL отдельной командой
`setfacl -m` и `setfacl -d -m`.

4) Перенесите machine-local DVC config во внешний путь и дайте Airflow read-only доступ:
```bash
sudo install -o "$DEPLOY_USER" -g agent042 -m 640 "$CHECKOUT_ROOT/.dvc/config.local" /home/anton-m/agent-042/.dvc/config.local
sudo setfacl -m u:${AIRFLOW_UID}:r /home/anton-m/agent-042/.dvc/config.local
```

5) После смены групп у host user перезайдите в shell или выполните:
```bash
newgrp agent042
```

6) Проверка после старта Compose:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml exec airflow-worker test -r /opt/airflow/project/.dvc/config.local && echo ok

docker compose --env-file .env -f infra/compose/docker-compose.yaml exec -T airflow-worker python - <<'PY'
from pathlib import Path

path = Path('/opt/airflow/project/assets/rag_data/_perm_smoke')
path.mkdir(parents=True, exist_ok=True)
probe = path / 'ok.txt'
probe.write_text('ok\n', encoding='utf-8')
print(probe.read_text().strip())
probe.unlink()
path.rmdir()
PY
```

Если эти проверки падают с `Permission denied`, это host-side проблема подготовки ACL/ownership, а
не причина возвращать `airflow-prepare-dirs`.

Практический тюнинг vLLM для локальной GPU:
- `VLLM__MAX_NUM_SEQS` — жёсткий верхний предел числа последовательностей, которые vLLM одновременно держит в scheduler batch. Для 12 GB GPU и длинного контекста безопасно начинать с `1-2`.
- `VLLM__MAX_NUM_BATCHED_TOKENS` — верхний предел числа токенов в одном scheduler/pre-fill шаге. Это не размер полного контекста; при chunked prefill длинный prompt просто режется на куски такого размера. Для старта разумно держать `1024-2048`.
- Соотношение параметров: `prompt_tokens + final_generation_budget` должны помещаться в `max_model_len`, а `VLLM__MAX_NUM_BATCHED_TOKENS` обычно должен быть заметно меньше `max_model_len`, потому что он ограничивает пик памяти на шаг, а не общий размер одного запроса.
- Если vLLM падает именно на `Capturing CUDA graphs`, сначала снижайте `VLLM__GPU_UTILIZATION` или `VLLM__MAX_NUM_BATCHED_TOKENS`; только потом повышайте `VLLM__MAX_NUM_SEQS`.

Важно:
- MLflow в текущей конфигурации подключён к S3, но не проксирует артефакты (опция `--serve-artifacts` отключена).
  Поэтому при логировании/чтении артефактов из MLflow-клиента нужны S3 креды в окружении процесса.

### Запуск полного стека

Ниже показаны текущие команды запуска из рабочего checkout. Это ещё не release-based deploy из
`/home/anton-m/agent-042/current`.

  - серверный deploy пример: `/home/anton-m/agent-042/assets`
```bash
  - серверный deploy пример: `/home/anton-m/agent-042/artifacts`
```
  - серверный deploy пример: `/home/anton-m/agent-042/.dvc/config.local`
Полезные URL (если используете значения портов по умолчанию из `.env.example`):
жить во внешнем `/home/anton-m/agent-042`. После удаления `airflow-prepare-dirs` эти bind mount'ы нужно
- vLLM OpenAI API: `http://<host>:8000/v1/models`
- `--server-root /home/anton-m/agent-042`
- UI (Streamlit): `http://<host>:8501`
  /home/anton-m/agent-042 \
- RedisInsight (мониторинг Redis): `http://<host>:5540`
sudo chgrp -R agent042 /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts /home/anton-m/agent-042/.dvc
sudo find /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts -type d -exec chmod 2775 {} +
sudo setfacl -R -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts
sudo setfacl -R -d -m u:${DEPLOY_USER}:rwx,u:${AIRFLOW_UID}:rwx,u:${JUPYTER_UID}:rwx,g:agent042:rwx /home/anton-m/agent-042/assets /home/anton-m/agent-042/artifacts

### Модели для vLLM

Контейнер vLLM монтирует папку `${SHARED_ROOT}/assets/models` как `/models`.
Чтобы использовать локальную модель из репозитория, укажите:
- `VLLM__MODEL=/models/<vendor>/<model>`

Если хотите использовать HuggingFace model id (без локальных файлов), укажите `VLLM__MODEL=<org>/<name>`.
В этом случае модели будут скачиваться в кэш внутри контейнера (см. `HF_HOME=/models/.cache`).

## MLFlow tracking server

MLFlow разворачивается в докере (см. compose выше).

Особенности:
- Backend store: PostgreSQL (доступен внутри docker network)
- Artifact store: S3 (Yandex Object Storage)
- Артефакты не проксируются через MLflow server (по умолчанию)

## Apache Airflow

Airflow развёрнут с CeleryExecutor и использует общий PostgreSQL с отдельной базой `airflow` и RabbitMQ как broker.

Сервисы:
- `airflow-init` — одноразовая инициализация: автоматическое создание БД (если не существует), миграция и создание admin-пользователя
- `airflow-webserver` — веб-интерфейс Airflow (порт `8080`)
- `airflow-dag-processor` — отдельный процесс разбора DAG-ов
- `airflow-scheduler` — планировщик задач
- `airflow-worker` — CPU worker для PythonOperator/BashOperator задач без GPU
- `airflow-worker-gpu` — worker с доступом к GPU для задач, которым он нужен

DAG-файлы размещаются в директории `dags/` в корне репозитория и монтируются в контейнеры Airflow.
Корень проекта так же монтируется как `/opt/airflow/project`. Shared state остаётся
project-relative, а Compose строит bind mount'ы от `SHARED_ROOT`:
- `${SHARED_ROOT}/assets/datasets` → `/opt/airflow/project/assets/datasets`
- `${SHARED_ROOT}/assets/rag_data` → `/opt/airflow/project/assets/rag_data`
- `${SHARED_ROOT}/artifacts/training` → `/opt/airflow/project/artifacts/training`
- `${SHARED_ROOT}/.dvc/config.local` → `/opt/airflow/project/.dvc/config.local`

Это даёт DAG'ам стабильные project-relative пути, но убирает зависимость от записи в checkout-backed
`assets/`, `artifacts/` и `.dvc`.

Важно: после Phase 2 отдельного `airflow-prepare-dirs` больше нет. Права на эти bind mount'ы
должны быть подготовлены на хосте заранее через setgid + ACL из раздела выше.

### Доступные DAG'и

| DAG | Расписание | Описание |
|-----|-----------|----------|
| `rag_alias_apply` | manual | Calls `AliasService.apply()` directly to make one KB alias match its `catalog.toml` declaration |
| `rag_collection_cleanup` | `@daily` | Release/deployment-aware cleanup: retires releases with no active/recent deployment in Postgres before deleting their Qdrant collection |

`rag_alias_apply` принимает параметры Airflow UI:

- `kb_id`, `alias` — target KB id (например `pytorch_reference`) и alias
  (например `champion` или `challenger`)
- `release_id` — optional, disambiguates an ambiguous reusable release
- `refresh_sources` — force a re-fetch of source content even without
  catalog drift
- `allow_unevaluated`, `allow_build_default` — default-alias bootstrap
  overrides; use sparingly
- `sync_dvc`, `dvc_base_branch`, `dvc_bot_branch` — if `sync_dvc=true`, a
  follow-up task DVC-syncs the KB's generated source artifacts after apply

Основной server entrypoint для тех же операций без Airflow:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias diff pytorch_reference challenger
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference challenger
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference champion --release ragrel_pytorch_reference_<fingerprint>
```

RAG DVC policy: curated source instance `manifest.toml` files stay in Git;
generated `extracted`, `chunks`, and `benchmark` artifacts can be
DVC-tracked; raw cache is server-local by default.

### Зависимости DAG'ов

Зависимости для выполнения DAG-задач задаются в `pyproject.toml` в группе `airflow-worker` и устанавливаются при сборке образа Celery-воркера (`infra/docker/airflow-worker/Dockerfile`). Scheduler, webserver и dag-processor используют лёгкий базовый образ без лишних пакетов.

Чтобы обновить зависимости:
```bash
# 1. Отредактируйте группу airflow-worker в pyproject.toml
# 2. Пересоберите lock:
ops/update_locks.sh airflow-worker

# 3. Пересоберите образ:
cd infra/compose
docker compose build airflow-worker
docker compose up -d airflow-worker
```

Переменные окружения (`.env`):
- `AIRFLOW_DB` — имя базы в PostgreSQL (по умолчанию `airflow`)
- `AIRFLOW_PORT` — порт веб-интерфейса (по умолчанию `8080`)
- `AIRFLOW_FERNET_KEY` — ключ шифрования; сгенерировать: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `AIRFLOW_JWT_SECRET` — JWT-секрет для Execution API между scheduler/webserver/dag-processor (Airflow 3.x); сгенерировать: `python -c "import secrets; print(secrets.token_hex(32))"` (по умолчанию `airflow_jwt_secret`)
- `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` — логин/пароль admin-пользователя

DAG'и также используют следующие переменные (передаются через `x-airflow-common-env`):
- `NETWORK__QDRANT_HTTP__INTERNAL_HOST` / `NETWORK__QDRANT_HTTP__INTERNAL_PORT` — адрес Qdrant для пересборки индексов
- `CONFIG__RUNTIME_PATH` — путь к mounted `runtime.toml` с runtime policy
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` — для `dvc push` в Yandex Cloud S3
- `GITHUB_REPOSITORY` / `GITHUB_DATA_SYNC_TOKEN` — для temp-clone Git push и GitHub PR API

Примечание: `airflow-init` автоматически проверяет наличие базы `airflow` в PostgreSQL и создаёт её при необходимости — никаких ручных шагов не требуется.

## JupyterLab

JupyterLab предоставляет интерактивную среду для экспериментов и анализа данных.

Монтируемые директории:
- `${PROJECT_ROOT}/src` → `/home/jovyan/src` (ro) — production-модули для импортов `shared/*`, `rag/*`, `gateway/*`
- `experiments/` → `/home/jovyan/experiments` (rw) — скрипты и конфиги экспериментов
- `${SHARED_ROOT}/assets` → `/home/jovyan/assets` (rw) — shared root для данных, моделей и адаптеров
- `dags/` → `/home/jovyan/dags` (rw) — Airflow DAG-файлы

Переменные окружения (`.env`):
- `JUPYTER_PORT` — порт JupyterLab (по умолчанию `8888`)
- `JUPYTER_TOKEN` — токен для аутентификации (по умолчанию `agent042`)

Дополнительно контейнер получает сервисные переменные:
- `CONTAINER__PROJECT_ROOT=/home/jovyan`
- `PYTHONPATH=/home/jovyan:/home/jovyan/src`
- `NETWORK__...` primitives for internal service discovery
- `CONFIG__RUNTIME_PATH=/opt/agent/runtime.toml`
- `CONFIG__CATALOG_PATH=/opt/agent/catalog.toml`

Важно: `CONTAINER__PROJECT_ROOT=/home/jovyan` относится только к контейнеру Jupyter. Это не тот же
самый `PROJECT_ROOT`, который оператор задаёт в repo-root `.env` или в `/home/anton-m/agent-042/.env` для
Compose interpolation на хосте.

Этого достаточно, чтобы ноутбуки и `experiments/rag/*.py` подключались к Qdrant/embeddings внутри Docker-сети, импортировали код из `src/`, но не получали rw-доступ ко всему репозиторию.

RAG operator boundary:
- Direct Qdrant diagnostics use the Qdrant API/dashboard.
- Production operations запускаются через `rag-ops` container и `python -m rag.cli.app`, либо через Airflow `rag_alias_apply`.

## DVC с бэкэндом Yandex Cloud S3

* После команды `uv sync --extra training` dvc должен быть уже установлен
* Добавить креды Yandex Cloud в файл `agent-042/.dvc/config.local`:
```text
['remote "ycloud"']
    access_key_id = YCA...
    secret_access_key = YCM...
```
