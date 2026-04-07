# Config Contract

Short operator/developer reference for the active configuration contract.

For historical context and migration notes, see `CONFIG-AUDIT.md`.

## Scope

This document describes the current source-of-truth split for configuration:

- where validated application defaults live
- where deployment topology lives
- where deploy-time/operator-edited values live
- which env names are canonical now
- which old names and files are retired

## Ownership

### Application defaults and validation

- Shared cross-service config schema lives in `src/shared/config.py`.
- Canonical shared endpoint settings are defined by `PlatformSettings`.
- Gateway-facing behavior is grouped by `GatewayBehaviorSettings`, `RagSettings`, `AuthSettings`, and aggregated by `GatewaySettings`.
- Registry, eval, and UI-specific settings live in `RegistrySettings`, `EvalSettings`, and `UISettings` in the same file.
- Service-local settings that are not shared stay in service-local config files, for example `src/worker/config.py` for Celery-only behavior.

Defaults rule:

- If a value is an application default or validation rule, it belongs in a settings class.
- If a value describes container-to-container wiring, it does not belong in Python defaults.

### Local dotenv loading

- Canonical local env file is repo-root `.env`.
- Canonical local template is repo-root `.env.example`.
- Local dotenv loading helpers live in `src/shared/local_env.py`.
- Local Python entrypoints and notebooks explicitly load repo-root `.env`; containerized deployments should inject env vars directly.

### Deployment topology

- Docker Compose topology lives in `infra/compose/docker-compose.yaml`.
- Internal Docker-network addresses, container-only URLs, and stable service wiring belong in Compose.
- Future Helm or k8s manifests should inject the same application env contract, not redefine it.

### Operator-edited deploy-time values

- Human-edited local/deploy-time values live in repo-root `.env`.
- The template in `.env.example` is intentionally limited to values operators are expected to change:
  - secrets
  - public ports
  - model choice and runtime tuning
  - feature flags
  - external credentials and selected URLs

## Canonical env names

### Shared cross-service endpoint names

Use these names for shared platform connectivity:

- `VLLM_BASE_URL`
- `EMBEDDINGS_URL`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `MLFLOW_TRACKING_URI`
- `REDIS_URL`
- `CELERY_BROKER_URL`

`EVAL_GATEWAY_URL` remains eval-specific rather than platform-wide.

### Service-specific env families

Use service prefixes for behavior that is local to one service:

- `GATEWAY_*`
- `EVAL_*`
- `UI_*`
- `REGISTRY_*`
- `AIRFLOW_*`
- `RABBITMQ_*`
- `REDIS_*`
- `VLLM_*`

Important exception:

- `GATEWAY_URL` is still a UI-facing variable, but the UI container's internal value is owned by Compose (`http://gateway:9000`). It is intentionally omitted from repo-root `.env.example` so host-side overrides do not leak into container topology.

## Where values should live

### Put it in `src/shared/config.py` when

- multiple services need the same validated value
- the value is part of the application contract
- the value should have a Python default for local host-side runs

### Put it in a service-local config file when

- the value is only relevant to one service
- it is operational behavior, not a shared platform contract
- example: Celery worker retry/timeouts in `src/worker/config.py`

### Put it in `.env.example` and `.env` when

- an operator is expected to supply or override it per machine/deployment
- it is a secret, public port, model selection, or credential

### Put it in Compose or Helm when

- the value describes service discovery or in-cluster/container networking
- the value should stay stable for a given deployment topology
- example: Docker-network URLs such as `http://gateway:9000` or `http://embeddings:8100`

## Retired names and files

These are no longer part of the active runtime contract:

- `GATEWAY_VLLM_BASE_URL`
- `GATEWAY_EMBEDDINGS_URL`
- `GATEWAY_QDRANT_HOST`
- `GATEWAY_QDRANT_PORT`
- `GATEWAY_TOP_K` — now alias-owned in `knowledge_bases.json`
- `GATEWAY_SCORE_THRESHOLD` — now alias-owned in `knowledge_bases.json`
- `GATEWAY_CONTEXT_MAX_LENGTH` — now alias-owned in `knowledge_bases.json`
- `GATEWAY_DEFAULT_ALIAS` — now `default_alias` field in `knowledge_bases.json` per KB
- `REGISTRY_VLLM_BASE_URL`
- `REGISTRY_MLFLOW_TRACKING_URI`
- `experiments/.env`
- `infra/compose/.env.example`
- `experiments/.env.example`

Do not add new runtime logic that depends on these names or files.

## Change Checklist

When adding or changing config:

1. Decide whether the value is shared contract, service-local behavior, operator input, or deployment topology.
2. If it is a shared endpoint, use the canonical env names in this document.
3. If an operator must edit it, expose it through repo-root `.env.example`.
4. If it is container/service wiring, keep it in Compose or future Helm defaults.
5. Update the closest service README if the change affects local run instructions.
