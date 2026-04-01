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
scripts/update_locks.sh

# Или только конкретные сервисы:
scripts/update_locks.sh gateway airflow-worker

# Посмотреть список сервисов:
scripts/update_locks.sh --list

# Проверить команды без выполнения:
scripts/update_locks.sh --dry-run
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
- `celery-worker` — Celery worker для асинхронного выполнения LLM-задач (1 процесс, GPU-bound)
- `gateway` — FastAPI gateway (порт хоста по умолчанию `9001` → контейнер `9000`)
- `ui` — Streamlit UI (порт хоста по умолчанию `8501` → контейнер `8501`)
- `flower` — Flower мониторинг Celery (порт хоста по умолчанию `5555` → контейнер `5555`)
- `redisinsight` — RedisInsight мониторинг Redis (порт хоста по умолчанию `5540` → контейнер `5540`)
- `airflow-init` — одноразовая миграция БД Airflow и создание admin-пользователя
- `airflow-webserver` — Airflow UI (порт хоста по умолчанию `8080` → контейнер `8080`)
- `airflow-scheduler` — Airflow Scheduler (LocalExecutor)
- `jupyter` — JupyterLab (порт хоста по умолчанию `8888` → контейнер `8888`)

### Подготовка переменных окружения

1) Из корня репозитория создайте `.env` на основе примера и заполните значениями:
```bash
cp .env.example .env
```

2) При необходимости скорректируйте `PROJECT_ROOT` и остальные значения `# CHANGE ME!`.

Ключевые переменные:
- `PROJECT_ROOT` — абсолютный путь к корню репозитория на машине, где запускается Compose
  - Linux пример: `/home/user/agent-042`
  - Windows пример (как в шаблоне): `C:/Users/user/MyGitRepos/agent-042`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — креды для Yandex Object Storage (нужны MLflow)
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` — опциональная auth-пара для локальных MLflow-клиентов и ноутбуков
- `MLFLOW_*` — конфиг MLflow (backend store + artifact root)
- `VLLM_*` — модель/квантизация/параметры GPU для vLLM
- `GATEWAY_*` — настройки Gateway (RAG, auth, async mode)
- `RABBITMQ_*` — логин/пароль и порты RabbitMQ (брокер для Celery)
- `REDIS_*` — порт Redis (pub/sub для потоковой передачи токенов)
- `FLOWER_*` — порт Flower (мониторинг Celery)
- `REDISINSIGHT_*` — порт RedisInsight (мониторинг Redis)
- `AIRFLOW_*` — конфиг Airflow (порт, БД, Fernet-ключ, admin-пользователь)
- `JUPYTER_*` — конфиг JupyterLab (порт, токен)

Замечание:
- Канонический шаблон `.env.example` намеренно не содержит внутренние endpoint'ы вроде `VLLM_BASE_URL`, `EMBEDDINGS_URL`, `GATEWAY_URL`, `QDRANT_HOST` или `MLFLOW_TRACKING_URI`.
  Compose подставляет свои Docker-network значения напрямую из `infra/compose/docker-compose.yaml`, а локальные Python-запуски используют значения по умолчанию из `shared.config`.

Важно:
- MLflow в текущей конфигурации подключён к S3, но не проксирует артефакты (опция `--serve-artifacts` отключена).
  Поэтому при логировании/чтении артефактов из MLflow-клиента нужны S3 креды в окружении процесса.

### Запуск полного стека

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml up --build -d
```

Проверка статуса:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml ps
```

Полезные URL (если используете значения портов по умолчанию из `.env.example`):
- MLflow UI: `http://<host>:5050`
- vLLM OpenAI API: `http://<host>:8000/v1/models`
- Gateway health: `http://<host>:9001/health`
- UI (Streamlit): `http://<host>:8501`
- Flower (мониторинг Celery): `http://<host>:5555`
- RedisInsight (мониторинг Redis): `http://<host>:5540`
- RabbitMQ Management: `http://<host>:15672`
- Airflow UI: `http://<host>:8080`
- JupyterLab: `http://<host>:8888`

### Запуск только части сервисов

Только MLflow + Postgres:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml up --build -d postgres mlflow
```

Только inference + RAG (vLLM + Qdrant + Gateway + UI):
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml up --build -d vllm qdrant gateway ui
```

### Остановка / перезапуск / логи

Остановить:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml down
```

Остановить и удалить volume'ы (удалит Postgres/Qdrant данные):
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml down -v
```

Логи всех сервисов:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml logs -f
```

Логи конкретного сервиса:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml logs -f vllm
```

Пересобрать и перезапустить один сервис:
```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml up --build -d gateway
```

### Модели для vLLM

Контейнер vLLM монтирует папку `assets/models` как `/models`.
Чтобы использовать локальную модель из репозитория, укажите:
- `VLLM_MODEL=/models/<vendor>/<model>`

Если хотите использовать HuggingFace model id (без локальных файлов), укажите `VLLM_MODEL=<org>/<name>`.
В этом случае модели будут скачиваться в кэш внутри контейнера (см. `HF_HOME=/models/.cache`).

## MLFlow tracking server

MLFlow разворачивается в докере (см. compose выше).

Особенности:
- Backend store: PostgreSQL (доступен внутри docker network)
- Artifact store: S3 (Yandex Object Storage)
- Артефакты не проксируются через MLflow server (по умолчанию)

## Apache Airflow

Airflow развёрнут с LocalExecutor (без Celery/Redis) и использует общий PostgreSQL с отдельной базой `airflow`.

Сервисы:
- `airflow-init` — одноразовая инициализация: автоматическое создание БД (если не существует), миграция и создание admin-пользователя
- `airflow-webserver` — веб-интерфейс Airflow (порт `8080`)
- `airflow-scheduler` — планировщик задач (LocalExecutor)

DAG-файлы размещаются в директории `dags/` в корне репозитория и монтируются в контейнеры Airflow.
Корень проекта так же монтируется как `/opt/airflow/project` — это даёт DAG'ам доступ к скриптам, DVC-конфигурации и каталогам данных.

### Доступные DAG'и

| DAG | Расписание | Описание |
|-----|-----------|----------|
| `arxiv_rag_update` | `@daily` | Загрузка новых ArXiv статей → `dvc add/push` → пересборка индекса `chat_documents` в Qdrant |
| `pytorch_docs_rag_update` | `@weekly` | Скрейпинг документации PyTorch → `dvc add/push` → пересборка индекса `code_documents` в Qdrant |

Каждый DAG состоит из трёх задач:
```
download / scrape  >>  dvc_version  >>  build_index
```

- **download / scrape** — PythonOperator: загрузка данных (ArXiv API или web scraping)
- **dvc_version** — BashOperator: `dvc add` + `dvc push` для версионирования данных
- **build_index** — BashOperator: запуск `build_arxiv_index.py` / `build_pytorch_docs_index.py` для пересборки вектор-индекса

### Зависимости DAG'ов

Зависимости для выполнения DAG-задач задаются в `pyproject.toml` в группе `airflow-worker` и устанавливаются при сборке образа Celery-воркера (`infra/docker/airflow-worker/Dockerfile`). Scheduler, webserver и dag-processor используют лёгкий базовый образ без лишних пакетов.

Чтобы обновить зависимости:
```bash
# 1. Отредактируйте группу airflow-worker в pyproject.toml
# 2. Пересоберите lock:
scripts/update_locks.sh airflow-worker

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
- `QDRANT_HOST` / `QDRANT_PORT` — адрес Qdrant для пересборки индексов
- `EMBEDDING_MODEL` — модель эмбеддингов (берётся из `GATEWAY_EMBEDDING_MODEL`)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` — для `dvc push` в Yandex Cloud S3

Примечание: `airflow-init` автоматически проверяет наличие базы `airflow` в PostgreSQL и создаёт её при необходимости — никаких ручных шагов не требуется.

## JupyterLab

JupyterLab предоставляет интерактивную среду для экспериментов и анализа данных.

Монтируемые директории:
- `${PROJECT_ROOT}/src` → `/home/jovyan/src` (ro) — production-модули для импортов `shared/*`, `rag/*`, `gateway/*`
- `experiments/` → `/home/jovyan/experiments` (rw) — скрипты и конфиги экспериментов
- `assets/` → `/home/jovyan/assets` (rw) — данные, модели, адаптеры
- `dags/` → `/home/jovyan/dags` (rw) — Airflow DAG-файлы

Переменные окружения (`.env`):
- `JUPYTER_PORT` — порт JupyterLab (по умолчанию `8888`)
- `JUPYTER_TOKEN` — токен для аутентификации (по умолчанию `agent042`)

Дополнительно контейнер получает сервисные переменные:
- `PROJECT_ROOT=/home/jovyan`
- `PYTHONPATH=/home/jovyan:/home/jovyan/src`
- `QDRANT_HOST=qdrant`, `QDRANT_PORT=${QDRANT_PORT}`
- `EMBEDDINGS_URL=http://embeddings:8100`
- `EVAL_GATEWAY_URL=http://gateway:9000`

Этого достаточно, чтобы ноутбуки и `experiments/rag/*.py` подключались к Qdrant/embeddings внутри Docker-сети, импортировали код из `src/`, но не получали rw-доступ ко всему репозиторию.

## DVC с бэкэндом Yandex Cloud S3

* После команды `uv sync --extra training` dvc должен быть уже установлен
* Добавить креды Yandex Cloud в файл `agent-042/.dvc/config.local`:
```text
['remote "ycloud"']
    access_key_id = YCA...
    secret_access_key = YCM...
```
