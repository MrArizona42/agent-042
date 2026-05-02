# Observability Implementation Plan

Grafana + Postgres analytics (visible immediately), then Prometheus backend instrumentation.
No Kafka, no ClickHouse dependency.

---

## Scope decision: Postgres instead of ClickHouse

The original `REMAINING-CHANGES.md §2.4` planned ClickHouse as Grafana's second datasource for ML
analytics. ClickHouse depends on Kafka (§2.5) and the ClickHouse setup (§2.6), both of which are
cut. Postgres already holds the data that matters for analytics at this scale:

| Table | Content | Use in Grafana |
|---|---|---|
| `eval_runs` | benchmark summary, adapter alias, timestamp | score trend by adapter |
| `eval_samples` | per-sample ROUGE-L, BERTScore, Recall@k | per-metric distribution |
| `chat_sessions` | user sessions | daily active users, session volume |
| `chat_messages` | individual messages | chat volume, length trends |

Grafana's native PostgreSQL datasource can query all of this directly. No extra infrastructure.

---

## What gets added

```
Grafana     ──reads──▶    Postgres    (ML analytics dashboards)   ← Phase 1–3: working UI first
                          Prometheus  (infra dashboards)           ← Phase 4–7: backend tracing

Prometheus  ──scrapes──▶  gateway /metrics        (new FastAPI middleware)
                          vllm :8000/metrics      (native, no code change)
                          rabbitmq :15692/metrics  (enable rabbitmq_prometheus plugin)
```

Two new Compose services. All provisioned as code.

---

## Compose convention

All `${VAR}` references in Compose are mandatory — no `${VAR:-default}` fallback syntax.
`.env.example` (and the operator's `.env`) is the single source of defaults for every variable.

---

## Network topology

| Service | Networks needed |
|---|---|
| `prometheus` | `backend_net` (scrapes gateway, vLLM, RabbitMQ) |
| `grafana` | `backend_net` (reaches Prometheus) + `mlflow_db_net` (reaches Postgres) |

No new Docker network needed; both services join existing ones.

---

## Implementation phases

### Phase 1 — Grafana Compose service

**Compose service:**

```yaml
grafana:
  image: grafana/grafana-oss:13.0.1
  environment:
    GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
    GF_SERVER_ROOT_URL: https://agent.antonlab.ru:8443/grafana
    GF_SERVER_SERVE_FROM_SUB_PATH: "true"
    POSTGRES_USER: ${POSTGRES_USER}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
  volumes:
    - ${PROJECT_ROOT}/infra/grafana/provisioning:/etc/grafana/provisioning:ro
    - ${PROJECT_ROOT}/infra/grafana/dashboards:/var/lib/grafana/dashboards:ro
    - grafana_data:/var/lib/grafana
  ports:
    - "127.0.0.1:${GRAFANA_PORT}:3000"
  depends_on:
    postgres:
      condition: service_healthy
  networks:
    - backend_net
    - mlflow_db_net
  healthcheck:
    test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:3000/api/health || exit 1"]
    interval: 15s
    timeout: 5s
    retries: 5
  logging: *default-logging
  restart: unless-stopped
```

`POSTGRES_USER` and `POSTGRES_PASSWORD` are passed to the Grafana container explicitly so
Grafana's provisioning interpolation can substitute them in the datasource config.

**Add `grafana_data` to the volumes block.**

No `prometheus` dependency yet — Grafana starts with Postgres only and Prometheus datasource
is added later.

---

### Phase 2 — Nginx proxy for Grafana

```nginx
location /grafana/ {
    proxy_pass http://127.0.0.1:${GRAFANA_PORT}/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

After this phase Grafana is accessible at `https://agent.antonlab.ru:8443/grafana` with the
ML analytics dashboard populated from real Postgres data.

---

### Phase 3 — Grafana provisioning files

All provisioning files are mounted read-only. Grafana reads them on startup and registers
datasources and dashboard providers automatically.

**`infra/grafana/provisioning/datasources/datasources.yml`**

```yaml
apiVersion: 1
datasources:
  - name: PostgreSQL
    type: postgres
    uid: postgres
    url: postgres:5432
    database: agent042
    user: ${POSTGRES_USER}
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
    jsonData:
      sslmode: disable
      postgresVersion: 1500
      timescaledb: false
    isDefault: true
    editable: false
```

Prometheus datasource is omitted here and added in Phase 5 once the Prometheus service exists.

**`infra/grafana/provisioning/dashboards/dashboards.yml`**

```yaml
apiVersion: 1
providers:
  - name: Default
    type: file
    updateIntervalSeconds: 60
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

---

### Phase 4 — ML Analytics dashboard (Postgres)

Build the dashboard JSON via the Grafana UI, then export and commit it to
`infra/grafana/dashboards/ml-analytics.json`. The provisioning provider picks it up
on the next `updateIntervalSeconds` cycle without restarting Grafana.

Panels and queries:

| Panel | SQL | Viz |
|---|---|---|
| Eval score trends (ROUGE-L) | `SELECT date_trunc('day', er.created_at) AS time, AVG(es.rouge_l) FROM eval_samples es JOIN eval_runs er ON es.eval_run_id = er.id GROUP BY 1 ORDER BY 1` | Time series |
| Score by adapter | `SELECT er.adapter_alias, AVG(es.rouge_l), AVG(es.bert_score) FROM eval_samples es JOIN eval_runs er ON es.eval_run_id = er.id GROUP BY er.adapter_alias` | Bar chart |
| BERTScore distribution | `SELECT es.bert_score FROM eval_samples es JOIN eval_runs er ON es.eval_run_id = er.id WHERE er.created_at > NOW() - INTERVAL '30 days'` | Histogram |
| Recent eval runs | `SELECT er.id, er.adapter_alias, er.created_at, er.dataset_name FROM eval_runs er ORDER BY er.created_at DESC LIMIT 20` | Table |
| Chat volume (daily) | `SELECT date_trunc('day', created_at) AS time, COUNT(*) FROM chat_messages GROUP BY 1 ORDER BY 1` | Bar chart |
| Active sessions (daily) | `SELECT date_trunc('day', created_at) AS time, COUNT(DISTINCT session_id) FROM chat_messages GROUP BY 1 ORDER BY 1` | Time series |

Column names should be verified against `src/shared/db/models.py` and the SQL DDL files before
finalising queries.

---

### Phase 5 — Prometheus service

**New file: `infra/docker/prometheus/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: gateway
    static_configs:
      - targets: ['gateway:9000']
    metrics_path: /metrics

  - job_name: vllm
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: /metrics

  - job_name: rabbitmq
    static_configs:
      - targets: ['rabbitmq:15692']
    metrics_path: /metrics
```

**Compose service:**

```yaml
prometheus:
  image: prom/prometheus:v3.4.0
  volumes:
    - ${PROJECT_ROOT}/infra/docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - prometheus_data:/prometheus
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
    - '--storage.tsdb.retention.time=30d'
    - '--web.enable-lifecycle'
  ports:
    - "127.0.0.1:${PROMETHEUS_PORT}:9090"
  networks:
    - backend_net
  healthcheck:
    test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/ready"]
    interval: 15s
    timeout: 5s
    retries: 5
  logging: *default-logging
  restart: unless-stopped
```

**Add `prometheus_data` to the volumes block.**

Add Prometheus as a `depends_on` condition in the `grafana` service and add the Prometheus
datasource to `infra/grafana/provisioning/datasources/datasources.yml`:

```yaml
  - name: Prometheus
    type: prometheus
    uid: prometheus
    url: http://prometheus:9090
    editable: false
```

---

### Phase 6 — Gateway instrumentation

Add `prometheus-fastapi-instrumentator` to the gateway.

**`pyproject.toml`** — add to gateway dependencies:
```
prometheus-fastapi-instrumentator>=7.0.0
```

**`src/gateway/main.py`** — after `app = FastAPI(...)`, before routes are registered:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

The `/metrics` endpoint is unauthenticated by design — Prometheus scrapes it on the internal
Docker network; it is not exposed through nginx to the public.

Metrics available after this change:
- `http_requests_total{method, handler, status}` — request rate and error rate
- `http_request_duration_seconds{handler}` — latency histograms (p50/p95/p99)
- `http_requests_in_progress{method, handler}` — concurrency

---

### Phase 7 — RabbitMQ Prometheus plugin

The `rabbitmq:3-management` image ships with `rabbitmq_prometheus` disabled by default.
Enable it via a plugin configuration file:

**New file: `infra/docker/rabbitmq/enabled_plugins`**
```
[rabbitmq_management,rabbitmq_prometheus].
```

Mount it in the RabbitMQ Compose service (add to its `volumes` list):
```yaml
- ${PROJECT_ROOT}/infra/docker/rabbitmq/enabled_plugins:/etc/rabbitmq/enabled_plugins:ro
```

The plugin exposes queue depth, message rates, and consumer counts at `:15692/metrics`.

---

### Phase 8 — Infrastructure dashboard (Prometheus)

Build via Grafana UI, export to `infra/grafana/dashboards/infrastructure.json`.

| Panel | Query | Viz |
|---|---|---|
| Request rate | `rate(http_requests_total[5m])` by handler | Time series |
| Error rate | `rate(http_requests_total{status=~"5.."}[5m])` | Time series |
| Latency p95 | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` | Time series |
| Requests in-progress | `http_requests_in_progress` | Stat |
| vLLM: pending requests | `vllm:num_requests_waiting` | Time series |
| vLLM: GPU KV-cache usage | `vllm:gpu_cache_usage_perc` | Gauge |
| vLLM: generation throughput | `rate(vllm:generation_tokens_total[1m])` | Time series |
| vLLM: e2e latency p95 | `histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))` | Time series |
| RabbitMQ: queue depth | `rabbitmq_queue_messages_ready_total` | Time series |
| RabbitMQ: consumers | `rabbitmq_queue_consumers` | Stat |

---

## Environment variables

Add to `.env.example` and `.env`:

```dotenv
# Observability
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=changeme
```

`POSTGRES_USER` and `POSTGRES_PASSWORD` already exist in `.env.example`.

---

## Files to create

| File | Note |
|---|---|
| `infra/grafana/provisioning/datasources/datasources.yml` | Postgres only initially; Prometheus added in Phase 5 |
| `infra/grafana/provisioning/dashboards/dashboards.yml` | |
| `infra/grafana/dashboards/ml-analytics.json` | Export from UI after Phase 3 |
| `infra/docker/prometheus/prometheus.yml` | Phase 5 |
| `infra/docker/rabbitmq/enabled_plugins` | Phase 7 |
| `infra/grafana/dashboards/infrastructure.json` | Export from UI after Phase 8 |

## Files to modify

| File | Change |
|---|---|
| `infra/compose/docker-compose.yaml` | Add `grafana` service + `grafana_data` volume (Phase 1); add `prometheus` service + `prometheus_data` volume + update grafana `depends_on` (Phase 5); add RabbitMQ volume mount (Phase 7) |
| `.env.example` | Add `PROMETHEUS_PORT`, `GRAFANA_PORT`, `GRAFANA_ADMIN_PASSWORD` |
| `infra/nginx/agent.antonlab.ru.conf` | Add `/grafana/` proxy block (Phase 4) |
| `src/gateway/main.py` | Add instrumentator (Phase 6) |
| `pyproject.toml` | Add `prometheus-fastapi-instrumentator` (Phase 6) |
