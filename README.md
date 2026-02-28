# agent-042

Платформа для RAG-инференса и экспериментов с LoRA-адаптерами для задач ML/AI.

## Что есть в проекте сейчас

- **Gateway (FastAPI)** с OpenAI-совместимыми endpoint'ами (`/v1/models`, `/v1/chat/completions`).
- **UI (Streamlit)** с выбором базы знаний для RAG.
- **vLLM** как inference backend.
- **Qdrant** для векторного поиска.
- **Async inference** через **Celery + RabbitMQ + Redis** (включено по умолчанию).
- **MLflow + PostgreSQL + S3 (Yandex Object Storage)** для трекинга и реестра моделей.
- **Airflow DAGs** для регулярного обновления RAG-данных.
- **JupyterLab** для интерактивных экспериментов.

## Навигация по документации

- `infra/README.md` — запуск и сопровождение всей инфраструктуры.
- `src/gateway/README.md` — устройство и запуск Gateway.
- `src/ui/README.md` — устройство и запуск UI.
- `experiments/README.md` — обучение, Hydra, DVC, Model Registry.

## Быстрый запуск (Docker Compose)

```bash
cd /home/runner/work/agent-042/agent-042/infra/compose
cp .env.example .env
# заполните секреты и PROJECT_ROOT в .env
docker compose up --build -d
```

Проверка:

```bash
docker compose ps
```

Основные URL (по умолчанию из `.env.example`):

- Gateway health: `http://localhost:9001/health`
- UI: `http://localhost:8501`
- vLLM models: `http://localhost:8000/v1/models`
- MLflow: `http://localhost:5050`
- Airflow: `http://localhost:8080/airflow`
- JupyterLab: `http://localhost:8888/jupyter`
- Flower: `http://localhost:5555/flower`
- RedisInsight: `http://localhost:5540/redis-insight`
- RabbitMQ management: `http://localhost:15672/rabbitmq`

## Локальная разработка (без Docker)

```bash
uv sync --extra gateway --extra ui --extra worker --extra rag --extra dev
```

Запуск сервисов:

```bash
# Gateway
PYTHONPATH=src uvicorn gateway.main:app --reload --port 9000

# UI (в отдельном терминале)
PYTHONPATH=src GATEWAY_URL=http://localhost:9000 streamlit run src/ui/app.py
```

## Проверки качества кода

- `pre-commit` (хуки из `.pre-commit-config.yaml`)
- `ruff` (настройки в `pyproject.toml`)

