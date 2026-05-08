# Config Contract

Полный список конфиг файлов в проекте: инфраструктурные, runtime, специфические для сервисов, etc.

1. `src/shared/config.py` owns the Python runtime config surface for gateway, worker, eval, registry, and UI services.
2. Repo-root `.env.example` is the canonical operator template for both local checkout `.env` files and the deployed `/home/anton-m/agent-042/.env` contract.
3. `infra/compose/docker-compose.yaml` owns deployment topology, container env injection, internal URLs, public port bindings, networks, volumes, and health checks.
4. `infra/docker/**/Dockerfile` owns image build instructions and image-local process defaults.
5. `src/shared/knowledge_bases.json` owns the RAG query registry: tasks, knowledge bases, aliases, and per-alias retrieval parameters.
6. `src/ui/.streamlit/config.toml` owns Streamlit-native runtime behavior for the UI process.
7. `infra/docker/rabbitmq/rabbitmq.conf` owns RabbitMQ-native broker configuration.
8. `infra/nginx/agent.antonlab.ru.conf` owns public routing, TLS termination, and reverse-proxy behavior.
9. `experiments/training/conf/**` and `experiments/training/train_adapter/config.py` own training-only Hydra and typed experiment configuration.
10. `pyproject.toml` and `.pre-commit-config.yaml` own packaging, dependency groups, linting, and developer tooling configuration.
