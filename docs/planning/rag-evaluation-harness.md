# RAG Evaluation Harness

The RAG evaluation harness is step zero for RAG work. RAG experiments should not
start by changing chunking, retrieval, reranking, or prompts. They start by
defining fixed datasets, metrics, a baseline pipeline, and a result schema.

The project can stay domain-free while still using realistic data. The plan is
to adopt open datasets as product-like knowledge bases instead of inventing a
made-up private domain.

## Benchmark Scope

Benchmarks are not global by default. Each dataset must declare what it is
allowed to prove.

- KB-scoped benchmarks evaluate a specific knowledge base. Their corpus and
  labels correspond to that KB, so they can support KB alias promotion.
- Config-scoped benchmarks evaluate a RAG configuration across public retrieval
  tasks. They help choose parser, chunking, embedding, retrieval, fusion, and
  reranking defaults, but they do not directly prove quality for every
  production KB.
- Global regression benchmarks evaluate platform behavior that should hold for
  every KB, such as no-answer behavior, citation formatting, metadata
  propagation, prompt-budget overflow, reranker fallback, and timeout handling.

Every knowledge base should declare its related evaluation sets. A benchmark
result is valid for promotion only when the benchmark is KB-scoped for that
knowledge base, or when the promotion rule explicitly treats the result as a
config-scoped sanity check rather than direct product quality evidence.

Example mapping:

| Knowledge base | KB-scoped eval sets | Config/global eval sets |
| --- | --- | --- |
| `research_papers` | QASPER, Open RAG Benchmark subset, local failures | BEIR/MS MARCO sanity, global regressions |
| `general_wikipedia` | HotpotQA, KILT subsets, local failures | BEIR sanity, global regressions |
| `pytorch_docs` | local PyTorch gold set, local failures | BEIR/MS MARCO sanity, global regressions |

## Evaluation Tiers

### Tier A: Public IR Sanity Benchmarks

Scope: config-scoped.

Purpose: compare generic retrieval and reranking behavior.

Recommended datasets:

- [BEIR](https://github.com/beir-cellar/beir), starting with SciFact, NFCorpus,
  and FiQA subsets;
- [MS MARCO](https://microsoft.github.io/msmarco/), preferably a manageable
  passage-ranking subset.

Use these datasets with their own corpora and relevance labels. Do not query a
PyTorch or arXiv knowledge base with BEIR/MS MARCO questions and treat the
result as product RAG quality.

### Tier B: Product-Like RAG Gold Sets

Scope: KB-scoped when the benchmark corpus is mounted as a project knowledge
base with matching labels.

Purpose: evaluate the full RAG system against realistic knowledge bases.

Primary recommendation:

- [QASPER](https://huggingface.co/datasets/allenai/qasper): research-paper QA
  over NLP papers, with answers and supporting evidence.

Secondary recommendation:

- [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench):
  arXiv PDF-based RAG benchmark for ingestion, parsing, and PDF-grounded QA.
  Use a text-first subset until the project explicitly supports multimodal PDF
  evidence.

Ready-made end-to-end expansion:

- [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench): large RAG
  benchmark across several industry-like domains. Use it when its example format
  can be normalized into the project harness without hiding retrieval details.

Optional expansion:

- [HotpotQA](https://hotpotqa.github.io/) for multi-hop evidence retrieval;
- [KILT](https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/)
  for Wikipedia-grounded knowledge-intensive tasks.

Existing PyTorch docs and local arXiv corpora can remain useful operational
knowledge bases, but they are not gold evaluation datasets until they have
question rows, expected answer properties, and source labels.

### Tier C: Synthetic Or Weakly Labeled Expansion

Scope: usually KB-scoped when generated from a specific KB and reviewed enough
to trust for that KB.

Purpose: expand coverage after Tier B is working.

Generate additional questions from selected QASPER papers, Open RAG Benchmark
documents, or local corpora. Keep source document ids and candidate evidence
spans. Review a small sample manually before trusting the synthetic rows.

Synthetic rows are useful for coverage and regression pressure. They should not
replace human-labeled or benchmark-provided gold data.

### Tier D: Failure Regression Set

Scope: KB-scoped for failures tied to one corpus; global when the failure is a
platform behavior bug.

Purpose: preserve hard cases discovered during development.

Rows should come from:

- failed benchmark cases;
- no-hit retrievals;
- wrong or weak citations;
- hallucinated answers;
- prompt-budget truncation;
- cases where reranking or chunking made the answer worse;
- production-like manual questions that expose real weaknesses.

This set grows naturally as experiments run.

## Baseline Pipeline

The first harness output is `rag_baseline_v1`.

It should fix:

- inference profile: `inference_baseline_v1`;
- corpus loader and metadata contract;
- chunking strategy;
- embedding model;
- sparse encoder setting;
- retrieval method;
- fusion method;
- candidate top-k;
- reranker setting;
- final context top-k;
- prompt template;
- generation parameters;
- no-context behavior;
- result persistence format.

Later RAG experiments vary one parameter family while keeping the rest at the
current baseline.

## Dataset Row Contract

Each normalized row should include:

- `dataset_name`;
- `dataset_version`;
- `benchmark_scope`: `kb`, `config`, or `global`;
- `knowledge_base`;
- `applicable_knowledge_bases`;
- `corpus_version` or manifest id;
- `query_id`;
- `query`;
- `query_type`;
- `expected_answer` or answer rubric;
- `relevant_doc_ids`;
- `relevant_chunk_ids` or evidence spans when available;
- `difficulty`;
- `notes`.

## Metrics

Retrieval:

- Recall@K;
- Hit@K;
- MRR;
- nDCG where graded labels exist;
- no-hit rate;
- expected-source coverage.

Context:

- relevant chunks in final prompt;
- context precision;
- context recall;
- duplicate chunk rate;
- source diversity;
- prompt tokens used;
- overflow and trimming rate.

Answer:

- correctness;
- relevance;
- faithfulness or groundedness;
- refusal correctness for no-answer rows;
- answer completeness.

Citation, after citation support is implemented:

- citation presence when RAG is used;
- citation precision;
- citation recall;
- unsupported citation rate;
- unsupported claim rate.

Operations:

- retrieval latency;
- reranking latency;
- prompt assembly latency;
- total request latency;
- token usage;
- error and timeout rate.

## Tooling Position

Ragas is implementation tooling for selected metrics and possible synthetic
testset generation. It is not the experiment itself.

LlamaIndex can be considered for dataset loading, evaluation adapters, or
comparison pipelines, but adopting it requires explicit integration work. The
baseline harness should not assume LlamaIndex abstractions unless the project
chooses to expand in that direction.

## Output Tables

Baseline summary:

| Dataset tier | Dataset | Scope | Applies to | Split/subset | Rows | Primary metrics | Status |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| A | BEIR SciFact | config | RAG configs | TBD | TBD | Recall@K, nDCG | planned |
| A | MS MARCO subset | config | RAG configs | TBD | TBD | MRR@10 | planned |
| B | QASPER | KB | `research_papers` | TBD | TBD | answer + evidence | planned |
| B | Open RAG Benchmark | KB | `research_papers` / PDF RAG KB | TBD | TBD | answer + source | planned |
| B | RAGBench | KB or config | selected imported KBs | TBD | TBD | end-to-end RAG | optional |
| D | Failure regression | KB or global | local | local | TBD | failure categories | planned |

Baseline RAG result:

| Dataset | Retrieval score | Answer score | Groundedness | Latency p95 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |
