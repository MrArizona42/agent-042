# Inference Baseline

This plan defines the minimum stable inference envelope needed before serious
RAG experiments. The goal is not to optimize every serving parameter up front.
The goal is to choose one reliable profile that later RAG grids can use as a
fixed dependency.

## Design Questions

- Which base model can the single-node deployment serve reliably?
- Which quantization mode leaves enough VRAM for context, LoRA adapters,
  embeddings, reranking, and the surrounding services?
- What context window and token budgets are realistic?
- What concurrency can the Gateway, Celery worker, Redis stream, and vLLM path
  support without unstable latency or memory pressure?

## Parameters To Decide

- base model family and size;
- quantization format;
- vLLM memory settings;
- max context window;
- default generation parameters;
- default response budget;
- system prompt budget;
- chat history budget;
- RAG context budget;
- Celery worker concurrency;
- vLLM concurrent sequence limits;
- default timeout values;
- acceptable p50/p95 time to first token;
- acceptable p50/p95 full response latency.

## Candidate Profiles

| Profile | Base model | Quantization | Context | LoRA | Intended role |
| --- | --- | --- | --- | --- | --- |
| Minimal | 3B-4B instruct | 4-bit or 8-bit | 8k-16k | 1 warm adapter | reliable fallback |
| Balanced | 7B-8B instruct | 4-bit AWQ/GPTQ or similar | 16k-32k if stable | 2-3 warm adapters | default target |
| Stress | 7B-8B instruct | best available 4-bit profile | largest stable context | more adapters | boundary finding |

## Metrics

- model load success or OOM;
- idle and peak VRAM;
- full stack RAM usage;
- cold start time;
- time to first token;
- output tokens per second;
- p50/p95 latency for short chat;
- p50/p95 latency for RAG chat using the baseline RAG harness;
- queue wait under low concurrency;
- failure mode under GPU saturation.

## Result Table

| Decision | Selected value | Evidence | Tradeoff |
| --- | --- | --- | --- |
| Base model | TBD | TBD | TBD |
| Quantization | TBD | TBD | TBD |
| Context window | TBD | TBD | TBD |
| Response budget | TBD | TBD | TBD |
| RAG budget | TBD | TBD | TBD |
| Concurrency | TBD | TBD | TBD |
| Latency target | TBD | TBD | TBD |

## Output Contract

The selected profile becomes `inference_baseline_v1`. RAG and LoRA experiments
should use it as a fixed dependency unless an experiment explicitly studies
inference parameters.
