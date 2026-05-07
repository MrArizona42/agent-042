# Config Contract

Short operator/developer reference for the active configuration contract.

## Big Picture

Each active config source appears once below.

1. `src/shared/config.py` is the canonical shared Python settings schema and defaults layer for gateway, registry, eval, UI, embeddings, and shared runtime code.
2. `src/worker/config.py` holds worker-only Python settings that should not leak into the shared runtime contract.
3. `src/gateway/config.py` and `src/ui/config.py` are compatibility shims that re-export shared settings for existing imports.
4. Repo-root `.env.example` is the canonical operator template for both local checkout `.env` files and the deployed `/home/anton-m/agent-042/.env` contract.
5. Repo-root `.env` is the actual local value set used by checkout-based Compose interpolation and host-side Python entrypoints from a working checkout.
6. `/home/anton-m/agent-042/.env` is the actual deployment value set used by release-based server deploys and server-side Compose interpolation.
7. `infra/compose/docker-compose.yaml` owns deployment topology, container env injection, internal URLs, public port bindings, networks, volumes, and health checks.
8. `infra/docker/**/Dockerfile` owns image build instructions and image-local process defaults.
9. `src/shared/knowledge_bases.json` owns the RAG query registry: tasks, knowledge bases, aliases, and per-alias retrieval parameters.
10. `src/ui/.streamlit/config.toml` owns Streamlit-native runtime behavior for the UI process.
11. `infra/docker/rabbitmq/rabbitmq.conf` owns RabbitMQ-native broker configuration.
12. `infra/nginx/agent.antonlab.ru.conf` owns public routing, TLS termination, and reverse-proxy behavior.
13. `experiments/training/conf/**` and `experiments/training/train_adapter/config.py` own training-only Hydra and typed experiment configuration.
14. `pyproject.toml` and `.pre-commit-config.yaml` own packaging, dependency groups, linting, and developer tooling configuration.
15. `infra/helm/` and `infra/terraform/` are reserved deployment surfaces and are currently inactive.

## Source Reference

| Source | Owner / primary reader | What it controls | Change it when | Notes |
| --- | --- | --- | --- | --- |
| `src/shared/config.py` | Application runtime code; read by gateway, UI, worker, embeddings, eval, and registry helpers | Shared Python settings schema, validation, and defaults: `PlatformSettings`, `GatewayBehaviorSettings`, `BudgetSettings`, `RagSettings`, `AuthSettings`, `RegistrySettings`, `EvalSettings`, `UISettings` | A Python service needs a new validated setting, a default must exist outside Compose, or a shared env contract changes | Canonical shared endpoint env names are `VLLM_BASE_URL`, `EMBEDDINGS_URL`, `QDRANT_HOST`, `QDRANT_PORT`, `MLFLOW_TRACKING_URI`, `REDIS_URL`, and `CELERY_BROKER_URL`; gateway-local behavior uses `GATEWAY_*`; `EMBEDDINGS_*` aliases are compatibility-only |
| `src/worker/config.py` | Worker runtime | Worker-only operational behavior such as broker URL, task timeout, retry count, and retry delay | The Celery worker needs a setting that is not part of the shared cross-service contract | Keep shared endpoints and gateway/runtime defaults in `src/shared/config.py`, not here |
| `src/gateway/config.py`, `src/ui/config.py` | Import compatibility layer for service code | Re-export shared settings and helpers; they do not define an independent config contract | Import paths or public compatibility shims need to change | Do not add new defaults, env parsing, or source-of-truth values here |
| Repo-root `.env.example` | Repo maintainers and operators | Template for secrets, public ports, credentials, path roots, model choices, feature toggles, and other operator-edited values for both local and deployed environments | An operator is expected to supply or override a value per machine or deployment | Intentionally omits internal container URLs such as `GATEWAY_URL`; container wiring belongs in Compose |
| Repo-root `.env` | Developers and operators working from a checkout | Actual values used for local development, checkout-based Compose interpolation, and host-side Python runs | A machine-specific local value changes or a local checkout needs a different path/secret | Local Python entrypoints load repo-root `.env` through `src/shared/local_env.py`; this is the active local dotenv source, not the release-deploy env source |
| `/home/anton-m/agent-042/.env` | Operators on the deployment server | Actual values used by release-based deploy automation and Compose interpolation on the target host | A deployment-specific value changes on the single-node server | Passed to Compose explicitly from outside the release tree; should not be copied into each release |
| `infra/compose/docker-compose.yaml` | Deployment topology | Services, networks, volumes, health checks, public port bindings, internal service discovery, and per-container env injection | Service wiring, internal URLs, dependency graph, or public port exposure changes | `x-shared-endpoints` injects canonical shared endpoint env vars; the UI container's `GATEWAY_URL=http://gateway:9000` belongs here |
| `infra/docker/**/Dockerfile` | Image build and image runtime | Image dependencies, build steps, working directories, startup commands, and image-local defaults | A service image needs different system packages, Python extras, or startup behavior | Not the place for operator secrets, public ports, or deployment-specific URLs |
| `src/shared/knowledge_bases.json` | RAG query contract | Task grouping, KB registry, per-task `routing_description` (for embedding-based task router), per-KB `selection_description` (for KB auto-selection), per-task `adapter` block (`name`, `alias`, `enabled` — for LoRA auto-selection), `default_alias`, alias-level `top_k`, `score_threshold`, `reranker`, `retrieval_strategy`, `reranker_multiplier`, plus labels and descriptions | Query-time RAG behavior, task routing text, KB selection text, or the visible KB registry changes | This is declarative app data, not a dotenv surface; build-time collection facts such as `retrieval_capability` live in Qdrant `_meta.build_config`, not here |
| `src/ui/.streamlit/config.toml` | Streamlit UI runtime | Streamlit-native server settings such as bind address, port, browser address, WebSocket behavior, and upload/message limits | Streamlit itself needs a native runtime change | Do not put gateway business logic, auth policy, or shared env contracts here |
| `infra/docker/rabbitmq/rabbitmq.conf` | RabbitMQ server runtime | Broker-native configuration that RabbitMQ reads directly | RabbitMQ server behavior or management-path behavior changes | Not a substitute for app-level env vars such as `CELERY_BROKER_URL` |
| `infra/nginx/agent.antonlab.ru.conf` | Edge deployment / reverse proxy | TLS termination, upstream routing, path prefixes, and external publication of services | Public routes, hostnames, TLS, or proxy behavior changes | Must stay aligned with the public ports exposed by the active operator env file and Compose |
| `experiments/training/conf/**`, `experiments/training/train_adapter/config.py` | Training pipeline and operator workflows | Hydra composition, experiment hyperparameters, training paths, model/data locations, and tracking behavior | Training behavior changes | This is intentionally separate from online serving runtime config; do not treat it as an override layer for gateway, UI, or worker behavior |
| `pyproject.toml`, `.pre-commit-config.yaml` | Packaging and developer tooling | Project metadata, optional dependency groups, lint settings, and developer hooks | Dependencies, extras, lint rules, or local developer workflow change | These files are project/tooling config, not serving runtime config |
| `infra/helm/`, `infra/terraform/` | Future deployment work | Nothing active today | The repo adopts Helm or Terraform as a real deployment layer | Until then, do not split active runtime config across these directories |
