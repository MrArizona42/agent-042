# Agent 042 Improvement Plan

This is the single planning document for the next project stage. Phase 1 is
complete; the active next stage is Phase 2 RAG quality work. The plan is
organized around the core AI / LLM / RAG system goals:

1. prove that the system can be observed, evaluated, and analyzed;
2. improve RAG answer quality and research usability;
3. expand platform functionality where it directly supports production LLM/RAG
   workflows;
4. keep lower-priority infrastructure and operational ideas deferred.

## Phase 2: RAG Quality Improvements

Phase 2 should improve the core RAG product: grounded answers, citations,
retrieval evaluation, and judge-based quality checks. Evaluation groundwork
started in Phase 1, but RAG-specific datasets and metrics belong here.

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

### 2. RAG Evaluation Datasets

Current state: the project has evaluation infrastructure, but RAG-specific
quality depends on having curated questions, expected sources, and expected
answer properties.

Target state:

- Add small curated RAG evaluation datasets for the main KBs.
- Include examples that test:
  - exact source lookup;
  - multi-document synthesis;
  - no-answer or insufficient-context behavior;
  - citation correctness;
  - questions that should prefer one KB over another.
- Store dataset provenance and versioning so RAG eval results are reproducible.
- Reuse Phase 1 failure analysis to turn real failures into new eval examples.

Acceptance criteria:

- Each core KB has at least a small representative eval set.
- Dataset rows include enough metadata to evaluate retrieval and citation
  quality, not only final answer text.
- Eval datasets can be run from the existing eval workflow.

Likely files:

- `assets/datasets/`
- `experiments/eval/eval_scripts/datasets.py`
- `experiments/eval/eval_scripts/runner.py`

### 3. RAG Metrics

Current state: automatic metrics exist, but RAG quality should be decomposed
into retrieval quality, citation quality, and final answer quality.

Target state:

- Add or formalize retrieval metrics:
  - Recall@k;
  - MRR;
  - nDCG where labels support it;
  - hit/no-hit rate;
  - expected-source coverage.
- Add citation metrics:
  - citation presence when RAG is used;
  - citation precision where expected sources are known;
  - unsupported citation detection;
  - answer sentences with/without cited support where feasible.
- Add answer quality metrics:
  - existing automatic metrics where appropriate;
  - LLM-as-judge relevance;
  - LLM-as-judge faithfulness/groundedness;
  - refusal/no-answer correctness for insufficient context.
- Store metric outputs in `eval_runs` / `eval_samples` with enough detail for
  the failure analysis notebook.

Acceptance criteria:

- RAG eval can show whether a failure came from retrieval, citation behavior, or
  answer generation.
- LLM-as-judge prompts are versioned and documented.
- Metrics can compare KB aliases such as champion/challenger.

Likely files:

- `experiments/eval/eval_scripts/metrics/automatic.py`
- `experiments/eval/eval_scripts/metrics/llm_judge.py`
- `experiments/eval/eval_scripts/retrieval_bench.py`
- `experiments/eval/eval_scripts/runner.py`

### 4. RAG Regression And Promotion Workflow

Current state: Qdrant aliases and eval tables support comparison, but the
promotion workflow should explicitly connect RAG builds, eval metrics, and
failure analysis.

Target state:

- Define a repeatable workflow before promoting a new KB alias:
  - build/materialize candidate collection;
  - run RAG eval dataset;
  - inspect retrieval/citation/answer metrics;
  - review failure analysis notebook;
  - promote or reject alias with a short operator note.
- Document guardrails:
  - retrieval quality must not regress;
  - citation quality must not regress;
  - answer quality must improve or stay neutral;
  - latency impact should be visible.

Acceptance criteria:

- A RAG alias promotion can be justified with eval results and failure analysis.
- The workflow is documented in RAG operations docs or a dedicated eval doc.

## Phase 3: Functionality And Platform Expansion

Phase 3 adds new platform capabilities incrementally. These should support the
Phase 1 and Phase 2 quality loops rather than distract from them.

### 1. LLM Observability Product Evaluation

Current state: OpenTelemetry/Tempo/Loki/Prometheus/Grafana can provide a strong
vendor-neutral observability stack. Specialized LLM observability tools may
still be useful for prompt and retrieval review workflows.

Target state:

- Evaluate Langfuse or Arize Phoenix as part of the platform expansion.
- Decide whether either tool adds enough value on top of the base stack.
- Define what may be captured if adopted:
  - prompt/response metadata;
  - retrieval context metadata;
  - latency per LLM/RAG step;
  - feedback labels;
  - redacted prompt/response samples only if explicitly allowed.
- Define privacy, redaction, and retention rules before storing prompt or
  response text.
- Use these tools for prompt/retrieval review, not as replacements for the core
  observability stack.

Acceptance criteria:

- There is a documented recommendation: adopt one product, defer adoption, or
  explicitly skip for now.
- The recommendation explains what problem the product solves that the base
  stack does not.
- Any prompt/response capture plan includes explicit redaction and retention
  rules.

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

## Phase 4: Future Ideas

These are valuable, but they should not interrupt the phases above.

### Operational Hardening

- Add Alembic migrations for the `agent042` Postgres database and remove
  startup `Base.metadata.create_all`.
- Add Compose health inspection for an already-running deployment.
- Add gateway abuse protection with Redis-backed rate limiting.
- Document local reproduction of CI jobs.
- Add a project quickstart once the new observability/RAG workflows exist.

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
