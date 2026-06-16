# RAG Experiment Series

RAG experiments are whole-system experiments. The pipeline should run end to
end for every grid: ingest or select a corpus, retrieve evidence, assemble
context, generate an answer, persist results, and evaluate against the harness.

The default rule is:

- baseline pipeline: current `rag_baseline_vN`;
- changed parameter group: one family at a time;
- fixed parameters: all other RAG and inference settings;
- datasets: all required tiers from the RAG evaluation harness;
- result: promote a new `rag_baseline_vN+1` only when the decision rule passes.

## Experiment Template

```text
Baseline pipeline:
Changed parameter group:
Fixed parameters:
Grid:
Datasets:
Metrics:
Decision rule:
Promotion rule:
Rollback rule:
```

## 1. Corpus And Parsing Grid

Vary:

- source format handling;
- markdown and header-aware parsing;
- PDF text extraction mode;
- code block handling;
- table/list handling;
- metadata extraction;
- document filtering.

Measure:

- indexing success rate;
- empty or malformed chunk rate;
- retrieval metrics;
- answer groundedness;
- source attribution quality;
- build time.

Decision:

- choose the source normalization and metadata policy used by later chunking and
  citation experiments.

## 2. Chunking Grid

Vary:

- chunk size;
- overlap;
- section boundary policy;
- minimum and maximum chunk length;
- semantic or header-aware chunking;
- code-aware chunking where relevant.

Measure:

- Recall@K, MRR, and nDCG;
- relevant chunks in final prompt;
- duplicate context rate;
- prompt tokens used;
- answer groundedness;
- prompt overflow or trimming rate;
- index size and build time.

Decision:

- choose default chunking parameters per corpus type if one global default is
  too weak.

## 3. Embedding And Index Grid

Vary:

- dense embedding model;
- sparse encoder setting;
- vector normalization;
- Qdrant distance metric;
- payload indexes;
- embedding batch size.

Measure:

- retrieval quality;
- embedding latency;
- Qdrant query latency;
- collection size;
- build time;
- memory footprint.

Decision:

- choose default embedding/index settings for the baseline RAG profile.

## 4. Retrieval And Fusion Grid

Vary:

- dense only;
- sparse only;
- hybrid retrieval;
- fusion method;
- dense/sparse weights;
- candidate top-k;
- metadata filters;
- query preprocessing.

Measure:

- retrieval quality;
- no-hit rate;
- final-context quality;
- downstream answer quality;
- retrieval latency.

Decision:

- choose the default retrieval strategy and top-k values.

## 5. Reranking Grid

Vary:

- reranker on/off;
- reranker model;
- candidate count before reranking;
- final top-k after reranking;
- rerank score threshold;
- fallback behavior when reranking fails.

Measure:

- post-rerank retrieval quality;
- answer groundedness;
- latency cost;
- cases helped;
- cases hurt.

Decision:

- choose when reranking is enabled by default and when it is skipped.

## 6. Context Assembly And Prompt Grid

Vary:

- number of final chunks;
- per-source budget;
- ordering by score, source, or document section;
- source headers;
- prompt template;
- no-context behavior;
- uncertainty instruction.

Measure:

- answer correctness;
- groundedness;
- prompt token usage;
- hallucination rate;
- refusal correctness;
- citation readiness.

Decision:

- choose context formatting, prompt policy, and no-answer behavior.

## 7. Citation And Provenance Grid

Vary:

- citation response schema;
- citation prompt instruction;
- model-generated citations;
- system-attached source metadata;
- claim-to-source validation strategy;
- UI rendering format.

Measure:

- citation precision;
- citation recall;
- unsupported citation rate;
- unsupported claim rate;
- user-inspectable source coverage;
- API and persistence complexity.

Decision:

- choose the citation contract that Phase 2 implements in API, persistence, and
  UI.

## 8. Generation Parameter Grid For RAG

Vary:

- temperature;
- top-p;
- max response tokens;
- repetition penalty if used;
- instruction variants;
- answer length constraints.

Measure:

- correctness;
- groundedness;
- citation compliance;
- verbosity;
- latency;
- token usage.

Decision:

- choose default generation parameters for RAG requests.

