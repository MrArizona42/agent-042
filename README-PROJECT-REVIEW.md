# Project Review: agent-042

## 1. Current State — What's Actually Built

For a master's thesis, the amount of implemented, working infrastructure is genuinely impressive.
This isn't a notebook demo — it's a **13-service Docker Compose stack** with real service separation,
async processing, and production-like patterns.

### What works well

* **Service architecture** is sound: gateway (FastAPI) / worker (Celery) / embeddings (separate
  microservice) / vLLM / Qdrant / Redis / RabbitMQ / PostgreSQL. This is a real microservice
  topology with appropriate technology choices for each role. The decision to isolate embeddings from
  the gateway to manage GPU resources is a genuinely good engineering call.
* **RAG system** is the strongest part. Alias-based versioning (champion/challenger), multiple
  chunking strategies (fixed-token, code-aware, section-aware), incremental vs. replace update
  policies, collection metadata stored as sentinel points — this is well-thought-out and goes beyond
  typical thesis RAG implementations. The `knowledge_bases.json` registry + production
  `src/rag/ops` entrypoints, surfaced to operators via `experiments/rag/notebook_ops.py`, is
  practical and keeps notebooks aligned with the runtime used by DAGs.
* **MLflow + Model Registry** lifecycle for LoRA adapters is real MLOps. The
  `train → register → promote → sync → serve` pipeline with alias-based promotion mirrors how
  production ML teams work.
* **Auth** is correctly implemented: OIDC + PKCE + HttpOnly cookies + server-side token storage. No
  common security shortcuts.
* **Airflow DAGs** for RAG data pipelines (arxiv daily, pytorch_docs weekly, cleanup) + eval
  orchestration show genuine workflow automation thinking.
* **DVC** for data versioning with S3 remote gives reproducibility. Separate lock files per service
  prevent dependency bloat.
* **Eval framework** is comprehensive: generation evals (chat/summarize/code with LLM-as-Judge +
  automatic metrics), retrieval-only evals (benchmark corpora → temporary Qdrant collections),
  sandboxed code execution in Docker.

### What's incomplete or weak

* **Task routing** is keyword-based only. The system design promises a progression toward LLM-based
  routing (Stage 4), but even the rule-based router is bare-bones.
* **Streaming response persistence** is missing — chat messages are only saved for non-streaming
  responses, which in a real system means you lose most conversation history.
* **Token usage tracking** is absent. In any production LLM system, token metering is essential for
  cost control, capacity planning, and abuse prevention.
* **No CI/CD pipelines exist at all.** The system design document describes CI/CD triggers for
  `develop` and `main` branches but the actual workflow files don't exist. For a project that
  emphasizes "resembling real development platforms," this is a significant gap.
* **No reranking** is implemented in RAG despite being listed as a research direction. This is one
  of the highest-impact RAG quality levers and would strengthen the thesis experimentally.
* **Tests** are partial — auth and basic API validation are covered, but there are no integration
  tests, no RAG pipeline tests with actual embeddings, no end-to-end workflow tests.

---

## 2. Planned Steps — Realistic Assessment

Per `Timeschedule.md`:

| Phase | Target | Realistic? |
|-------|--------|------------|
| Stage 4 (Apr 1–5) | Agent routing OR k3s+Helm+Terraform | **Overloaded.** Doing both is unrealistic. Pick one. |
| Stage 5 (May 10) | Prometheus + Grafana + benchmarking | **Achievable** if scoped to basic service metrics. |
| Stage 6 (May 20 – Jun 10) | Final polish + defense | Standard. |

### Concerns about Stage 4

The "agent with dynamic tool selection" (Stage 4 of the system design) is described vaguely
("details TBD"). This is the hardest part of the entire project — essentially building a
ReAct/function-calling agent layer. Simultaneously, Stage 4 of the timeline also mentions
k3s + Helm + Terraform migration. These are two completely different, each individually large
projects. **Pick one.** The agent layer is recommended since it's more directly relevant to the
thesis topic (AI assistant for researchers) and more publishable.

The k3s/Helm/Terraform migration is impressive infrastructure work but adds minimal thesis value. A
well-documented Docker Compose deployment with Nginx TLS is already production-grade for a
single-node setup. Kubernetes matters when you need horizontal scaling or multi-node HA — neither is
a thesis requirement.

---

## 3. What's Missing or Could Be Better

### A. Observability & Monitoring (Critical Gap)

This is the biggest gap relative to "real systems." There is **zero observability** beyond service
health checks and UI dashboards (Flower, RedisInsight).

What real systems have:

| Layer | Tool | Purpose | Effort |
|-------|------|---------|--------|
| Metrics | Prometheus + Grafana | Request latency, error rates, queue depth, GPU utilization, token throughput | Medium |
| Structured logging | Loki or ELK | Centralized log aggregation, search, correlation | Medium |
| Distributed tracing | OpenTelemetry + Jaeger/Tempo | Request flow across gateway → worker → vLLM → Qdrant | Medium-High |
| LLM-specific observability | Phoenix (Arize) or Langfuse | Prompt/response logging, latency per step, retrieval quality tracking, cost tracking | Low-Medium |

**Recommendation:** At minimum, add **Prometheus + Grafana** (already planned for Stage 5) and
**Langfuse** (open-source, self-hostable, designed exactly for LLM app observability). Langfuse
would give prompt tracing, token cost tracking, and retrieval quality dashboards with minimal code
changes — and it's directly relevant to the thesis topic. It's a single Docker container + a few
decorator calls in the gateway.

OpenTelemetry instrumentation of FastAPI + Celery + httpx would give distributed tracing "for free"
with `opentelemetry-instrument` auto-instrumentation — ~20 lines of setup code.

### B. CI/CD (Important Gap)

The system design document describes CI triggers but nothing is implemented. For a project
emphasizing real-world practices:

**Minimum viable CI (GitHub Actions or GitLab CI):**

1. `pre-commit` hooks run on push (config already exists)
2. `pytest` on push (tests already exist)
3. Docker image builds on merge to `main` (validates all Dockerfiles)
4. Optional: push to a container registry (even GitHub Container Registry)

This is ~50 lines of YAML and would dramatically strengthen the "real platform" angle of the thesis.

### C. RAG Pipeline Improvements

The RAG system is good but missing several techniques that would make strong thesis content:

* **Reranking.** Listed as a research question but not implemented. Adding a cross-encoder reranker
  (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) is straightforward and measurable. This is likely
  the single highest-impact RAG improvement.
* **Hybrid search.** Qdrant supports sparse+dense fusion natively. The eval framework already has
  retrieval benchmarks — this is a clean A/B comparison.
* **Query expansion / HyDE.** Use the LLM to generate a hypothetical answer, embed that, search.
  Cheap experiment, measurable impact.
* **Metadata filtering at query time.** Chunks have metadata but the retriever doesn't expose
  filtering to the user (e.g., "search only papers from 2025" or "search only PyTorch 2.x docs").

### D. Data Quality & Validation

No data validation exists anywhere in the pipeline:

* **RAG data pipeline:** The Airflow DAGs download arxiv papers and pytorch docs and feed them
  directly into the vector store. There's no validation that the downloaded content is well-formed,
  no deduplication check, no quality gate.
* **Eval data:** No schema validation on datasets before running evaluations.

In real ML systems, tools like **Great Expectations**, **Pandera**, or even simple assertion checks
in DAGs are standard. Adding basic data validation tasks to Airflow DAGs (record count checks,
schema validation, embedding dimension verification) would be low effort and high thesis value.

### E. Cost & Resource Tracking

A real LLM platform tracks:

* Token usage per user / per session
* GPU utilization and inference latency percentiles
* Vector DB query latency and hit rates
* Embedding service throughput

None of this exists. Even a simple `token_count` column on `chat_messages` + a Grafana dashboard
would add significant value.

### F. Security Hardening (Minor)

* **Rate limiting** is absent. FastAPI + `slowapi` or a simple Redis-based rate limiter would
  prevent abuse.
* **Input sanitization** — the prompt builder concatenates user input with RAG context directly.
  In a production system you'd want at minimum length limits and potentially content filtering.
* **Secrets management** — all secrets are in `.env` files. Fine for a thesis, but worth mentioning
  in the writeup that production systems use HashiCorp Vault, AWS Secrets Manager, etc.

### G. Agent Layer Design (Stage 4)

The "agent with dynamic tool selection" is described as "details TBD." Concrete suggestion:

Instead of building a custom agent framework, use **function calling** (which vLLM supports for many
models) with a simple tool registry:

```
User query → LLM decides tool calls → Execute tools → LLM synthesizes answer
```

**Tools to register:**

* `search_knowledge_base(query, kb_name)` — RAG retrieval
* `summarize_document(text)` — invoke summarization LoRA
* `generate_code(description)` — invoke code LoRA
* `web_search(query)` — external search (if time permits)

This is a cleaner design than inserting "another LLM layer" between the gateway and task router.
It's also more aligned with how real agent systems work (OpenAI function calling, Anthropic tool use,
LangChain agents).

### H. Big Data / Streaming (Minor, for Thesis Context)

The system design doesn't mention:

* **Apache Kafka** (or even Redis Streams used as a log) for event sourcing — every user
  interaction, RAG retrieval, and LLM generation could be events in a log. This enables offline
  analytics, replay, and audit trails.
* **Apache Spark** or **DuckDB** for batch analytics on evaluation results. Right now eval metrics
  go into MLflow, but there's no analytical layer for cross-experiment comparison beyond the MLflow
  UI.
* **Feature store** concepts — while not directly applicable to LLM systems, the knowledge base
  registry you built is conceptually similar. Worth drawing the parallel in the thesis.

These aren't necessary to implement, but mentioning them in the thesis as "what a scaled version
would add" strengthens the system design narrative.

---

## Summary: Prioritized Recommendations

Prioritized list for the remaining ~2.5 months:

| Priority | Action | Thesis Value | Effort |
|:--------:|--------|:------------:|:------:|
| 1 | Implement reranking in RAG + benchmark it | Very High | Low |
| 2 | Add Langfuse for LLM observability | High | Low |
| 3 | Implement basic CI pipeline | High | Low |
| 4 | Fix streaming response persistence | Medium | Low |
| 5 | Add token usage tracking | Medium | Low |
| 6 | Implement hybrid search + benchmark | High | Medium |
| 7 | Design & implement agent layer (function calling) | Very High | Medium |
| 8 | Add Prometheus + Grafana | Medium | Medium |
| 9 | Add data validation in Airflow DAGs | Medium | Low |
| 10 | Drop k3s/Helm/Terraform (document as future work) | — | Saves time |

**Bottom line:** This is a strong project with real engineering substance. The main risk is scope
creep — trying to do Kubernetes migration AND agent routing AND monitoring all at once. Focus on what
makes the thesis unique (the RAG experiments, the agent layer, the LoRA lifecycle) and treat
infrastructure as "good enough" at Docker Compose level. The RAG system and eval framework are the
strongest assets — double down on those with reranking, hybrid search, and proper observability.
