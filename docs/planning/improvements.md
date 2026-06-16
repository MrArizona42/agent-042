# Agent 042 Improvement Plan

This is the top-level planning document for the next project stage. Phase 1 is
complete; the active next stage is Phase 2 RAG quality work. The plan is
organized around the core AI / LLM / RAG system goals:

1. prove that the system can be observed, evaluated, and analyzed;
2. improve RAG answer quality and research usability;
3. expand platform functionality where it directly supports production LLM/RAG
   workflows;
4. keep lower-priority infrastructure and operational ideas deferred.

RAG evaluation should be scoped to the thing being measured. Public IR
benchmarks validate general retrieval configurations; KB-scoped gold and
failure sets support promotion of a specific knowledge-base alias; global
regression checks validate platform behavior across KBs. Not every benchmark is
meaningful for every knowledge base.

Before the RAG experiments start, complete the
[RAG Pipeline Isolation Plan](rag-pipeline-isolation.md). The project should
keep reusable RAG platform code in `src/`, production dataset pipelines in a
dedicated pipeline layer, and exploratory notebooks/reports in `experiments/`.

### 1. Source Citations In RAG Answers

Current state: RAG retrieval can provide context to the model, and runtime
observability tracks provenance internally, but user-facing answers do not yet
make source grounding explicit enough for research workflows.

Problem to solve:

- researchers need to know which documents support an answer;
- RAG answers should expose source provenance without forcing the user to inspect
  backend logs or Qdrant metadata;
- citations should make hallucinations and weak retrieval easier to spot.

Target state:

- Preserve source metadata through retrieval, prompt construction, generation,
  streaming, and persisted chat history.
- Ask the model to cite retrieved sources in answers when RAG context is used.
- Return structured citation metadata in API responses:
  - source title or document id;
  - source URI;
  - chunk id or section metadata when available;
  - rank/score where useful;
  - knowledge base and alias.
- Render citations in the UI in a compact, inspectable way.
- Track citation coverage for later analytics.

Acceptance criteria:

- A RAG answer can be traced from visible citation to retrieved chunk metadata.
- Non-RAG answers do not invent citations.
- API and UI behavior remain useful even when some source metadata is missing.
- Tests cover citation metadata propagation and no-citation behavior.

Likely files:

- `src/rag/runtime/models.py`
- `src/gateway/services/rag_service.py`
- `src/gateway/services/prompt_builder.py`
- `src/gateway/services/processing.py`
- `src/gateway/schemas/openai_chat.py`
- `src/ui/app.py`


### 2. User Feedback Tracking

Current state: there is no simple way for users to tell the system whether an
answer was useful. Quality signals come mostly from offline evals and operator
inspection.

Target state:

- Add simple feedback capture:
  - thumbs up/down on an answer;
  - optional short reason or category;
  - optional "choose between two answers" workflow for comparison tasks.
- Associate feedback with request id, chat session id, model/adapter metadata,
  RAG sources, citations, and timestamp.
- Publish feedback to `feedback-events` and store it in ClickHouse for
  analytics.
- Keep feedback UI minimal and non-blocking.
- Define privacy rules for any free-text feedback.

Acceptance criteria:

- Users can submit feedback for a completed answer.
- Feedback can be joined to inference events by request id.
- Feedback analytics can be queried in ClickHouse.
- The system supports thumbs feedback first, with pairwise answer choice as a
  later extension if needed.

### 3. A/B Champion/Challenger Evaluation

Current state: champion/challenger aliases exist for RAG collections and MLflow
model registry workflows, but promotion decisions are mostly offline.

Target state:

- Add `challenger_traffic_pct` config with default `0`.
- Route a controlled share of requests to challenger variants.
- Add `ab_variant` and variant metadata to inference events.
- Compare variants in ClickHouse and a notebook or script.
- Use guardrails before promotion:
  - latency;
  - error rate;
  - token budget;
  - RAG hit rate;
  - citation quality;
  - user feedback;
  - offline eval deltas.

Acceptance criteria:

- Champion/challenger comparison can use production inference data.
- Promotion recommendations include guardrail checks, not only quality deltas.
- The process is documented as an operator workflow.

## Future Ideas

These are valuable, but they should not interrupt the phases above.

### Spark For Data Quality And Feedback Loops

Current idea:

- RAG source dedup/filter/chunk quality gates.
- Training data dedup/filter before LoRA training.
- Weekly KB gap detection from low-hit inference events.
- Weekly query drift detection against a baseline.
- Write gap/drift reports into ClickHouse for Grafana dashboards.

This should wait until ClickHouse inference analytics exists and there is enough
real or synthetic traffic to justify batch data jobs.

### Kubernetes Later: k3s, Helm, KEDA

Current idea:

- Add k3s/Helm after the production data loop exists.
- Use KEDA to autoscale Celery workers from RabbitMQ queue depth.
- Add GPU resource requests/limits for vLLM.
- Support rolling updates for model or adapter changes.

This should wait until Compose-based operations are observable enough that
Kubernetes solves a visible problem instead of adding infrastructure for its own
sake.

### Product And Agent Ideas

- Function-calling agent layer on top of current task routing.
- Web search as an optional tool.
- Broader user feedback UX after simple feedback is proven useful.
- More advanced cost accounting once ClickHouse inference analytics exists.
