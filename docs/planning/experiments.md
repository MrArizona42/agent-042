# Experiment Planning Index

Phase 1 made the platform observable enough to measure design choices. The next
planning layer is a controlled experiment program for choosing inference, RAG,
LoRA, and operating parameters.

The main principle is:

- establish a stable baseline pipeline;
- evaluate against fixed datasets and metrics from the start;
- vary one parameter family at a time;
- record the chosen value, evidence, and tradeoff as a design decision.

RAG is the priority track. Inference work comes first only far enough to define
the serving envelope that RAG experiments must fit inside.

## Documents

1. [Inference Baseline](inference-baseline.md)
   - Defines the model, quantization, context, token budget, and concurrency
     envelope used by later RAG experiments.
2. [RAG Evaluation Harness](rag-evaluation-harness.md)
   - Defines the datasets, baseline pipeline, metrics, result schema, and
     repeatability rules used by every RAG experiment.
3. [RAG Experiment Series](rag-experiment-series.md)
   - Defines the controlled grids for corpus processing, chunking, embeddings,
     retrieval, reranking, prompt assembly, citations, and generation settings.
4. [LoRA Experiment Series](lora-experiment-series.md)
   - Defines adapter training, serving, promotion, and RAG-interaction
     experiments.
5. [Operating Limits](operating-limits.md)
   - Defines load, degradation, fallback, and production guardrail experiments.

## Dependency Order

```text
Inference baseline
  -> RAG evaluation harness
    -> RAG experiment series
      -> RAG promotion rules

Inference baseline
  -> LoRA experiment series
    -> LoRA promotion rules

RAG + LoRA baselines
  -> Operating limits
  -> Quality loop and release guardrails
```

## Shared Experiment Template

Each experiment should include:

- baseline pipeline name and version;
- changed parameter group;
- fixed parameters;
- grid values;
- datasets and dataset versions;
- metrics;
- decision rule;
- promotion or rollback rule;
- result table;
- follow-up work.
