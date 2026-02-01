# Настройка окружения

Этот репозиторий поддерживает запуск через Docker Compose.
В одном `docker-compose.yaml` собраны:
- MLflow Tracking Server + PostgreSQL (backend store)
- vLLM (OpenAI-compatible LLM inference server) с доступом к GPU
- Qdrant (векторная БД)
- Gateway (FastAPI)
- UI (Streamlit)

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
* Запустить команду для синхронизации окружения:
```bash
uv sync
```

## Docker / Docker Compose

### Что разворачивается в Compose

Файл: `infra/compose/docker-compose.yaml`.

Сервисы:
- `postgres` — PostgreSQL 15 для MLflow backend store (volume: `mlflow_pg_data`)
- `mlflow` — MLflow Tracking Server (порт хоста по умолчанию `5050` → контейнер `5000`)
- `vllm` — vLLM OpenAI server (порт хоста по умолчанию `8000` → контейнер `8000`), использует GPU
- `qdrant` — Qdrant (порт хоста по умолчанию `6333` → контейнер `6333`, volume: `qdrant_data`)
- `gateway` — FastAPI gateway (порт хоста по умолчанию `9001` → контейнер `9000`)
- `ui` — Streamlit UI (порт хоста по умолчанию `8501` → контейнер `8501`)

### Подготовка переменных окружения

1) Перейдите в папку compose:
```bash
cd ./infra/compose/
```

2) Создайте `.env` на основе примера и заполните значениями:
```bash
cp .env.example .env
```

Ключевые переменные:
- `PROJECT_ROOT` — абсолютный путь к корню репозитория на машине, где запускается Compose
  - Linux пример: `/home/user/agent-042`
  - Windows пример (как в шаблоне): `C:/Users/user/MyGitRepos/agent-042`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — креды для Yandex Object Storage (нужны MLflow)
- `MLFLOW_*` — конфиг MLflow (backend store + artifact root)
- `VLLM_*` — модель/квантизация/параметры GPU для vLLM
- `GATEWAY_*` — настройки Gateway (RAG, Qdrant, vLLM)

Важно:
- MLflow в текущей конфигурации подключён к S3, но не проксирует артефакты (опция `--serve-artifacts` отключена).
  Поэтому при логировании/чтении артефактов из MLflow-клиента нужны S3 креды в окружении процесса.

### Запуск полного стека

Из папки `infra/compose/`:
```bash
docker compose up --build -d
```

Проверка статуса:
```bash
docker compose ps
```

Полезные URL (если используете значения портов по умолчанию из `.env.example`):
- MLflow UI: `http://<host>:5050`
- vLLM OpenAI API: `http://<host>:8000/v1/models`
- Gateway health: `http://<host>:9001/health`
- UI (Streamlit): `http://<host>:8501`

### Запуск только части сервисов

Только MLflow + Postgres:
```bash
docker compose up --build -d postgres mlflow
```

Только inference + RAG (vLLM + Qdrant + Gateway + UI):
```bash
docker compose up --build -d vllm qdrant gateway ui
```

### Остановка / перезапуск / логи

Остановить:
```bash
docker compose down
```

Остановить и удалить volume'ы (удалит Postgres/Qdrant данные):
```bash
docker compose down -v
```

Логи всех сервисов:
```bash
docker compose logs -f
```

Логи конкретного сервиса:
```bash
docker compose logs -f vllm
```

Пересобрать и перезапустить один сервис:
```bash
docker compose up --build -d gateway
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

## DVC с бэкэндом Yandex Cloud S3

* После команды `uv sync` dvc должен быть уже установлен
* Добавить креды Yandex Cloud в файл `agent-042/.dvc/config.local`:
```text
['remote "ycloud"']
    access_key_id = YCA...
    secret_access_key = YCM...
```

Про обращение с данными см. в `agent-042/experiments/README.md`.
