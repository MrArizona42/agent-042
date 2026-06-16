# LoRA Experiment Series

LoRA work should support the RAG and inference platform instead of competing
with it. Adapter experiments begin after `inference_baseline_v1` exists and
after the RAG evaluation harness can measure whether an adapter helps or hurts
grounded answers.

## 1. Training Parameter Grid

Vary:

- dataset mix;
- LoRA rank;
- alpha;
- dropout;
- learning rate;
- epochs;
- sequence length;
- target modules.

Measure:

- task metrics;
- validation loss;
- code or summarization task success where relevant;
- RAG answer groundedness if the adapter is used in RAG;
- training time;
- artifact size.

Decision:

- choose adapter training defaults per task family.

## 2. Serving Overhead Grid

Vary:

- base model without adapter;
- warm adapter already loaded;
- cold adapter load path;
- number of simultaneously loaded adapters;
- adapter rank;
- alias switching.

Measure:

- VRAM delta;
- adapter load time;
- time to first token;
- tokens per second;
- p50/p95 response latency;
- missing alias behavior;
- rollback behavior.

Decision:

- choose warm adapter count, max adapter rank, and alias validation policy.

## 3. LoRA And RAG Interaction

Vary:

- no adapter;
- task adapter;
- style/instruction adapter;
- champion vs challenger adapter aliases.

Measure:

- answer correctness;
- groundedness;
- citation compliance;
- hallucination rate;
- latency;
- regression against baseline RAG metrics.

Decision:

- define when adapters are allowed for RAG and when the base model is safer.

## Promotion Rule

An adapter can be promoted only if it:

- improves or preserves target task quality;
- does not regress RAG groundedness when used with RAG;
- does not exceed serving latency or VRAM limits;
- has a rollback alias;
- has MLflow metadata sufficient to reproduce the run.
