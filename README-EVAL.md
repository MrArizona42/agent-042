# Evaluation & Benchmarking


## Краткое описание

**Типы задач**

* Chat (QA)
* Summarization
* Code generation

**Метрики**

* Chat: Relevance (1–5), Correctness (1–5), BERTScore, ROUGE-L
* Summarization: Faithfulness (1–5), Coverage (1–5), BERTScore, ROUGE-L
* Code generation: Executable rate, pass@1
* RAG-specific: Recall@k, nDCG@k, Groundedness

**Датасеты**

* Chat: HotpotQA, Natural Questions
* Summarization: ArXiv-summarization
* Code generation: HumanEval
* RAG: MS MARCO, BEIR-SciFact, BEIR-NFCorpus

**Полный список проверок**

* Chat - HotpotQA - Relevance (llm-as-judge)
* Chat - HotpotQA - Correctness (llm-as-judge)
* Chat - HotpotQA - BERTScore (llm-encoder-based)
* Chat - HotpotQA - ROUGE-L (automatic)
* Chat - Natural Questions - Relevance (llm-as-judge)
* Chat - Natural Questions - Correctness (llm-as-judge)
* Chat - Natural Questions - BERTScore (llm-encoder-based)
* Chat - Natural Questions - ROUGE-L (automatic)
* Code generation - HumanEval - Executable rate (sandboxed execution)
* Code generation - HumanEval - pass@1 (sandboxed execution)
* Summarization - ArXiv-summarization - Faithfulness (llm-as-judge)
* Summarization - ArXiv-summarization - Coverage (llm-as-judge)
* Summarization - ArXiv-summarization - BERTScore (llm-encoder-based)
* Summarization - ArXiv-summarization - ROUGE-L (automatic)
* RAG in Chat - HotpotQA - Groundedness (llm-as-judge)
* RAG in Chat - Natural Questions - Groundedness (llm-as-judge)
* RAG in Code generation - HumanEval - Groundedness (llm-as-judge)
* RAG-only, retrieval - MS MARCO - Recall@k (automatic)
* RAG-only, retrieval - MS MARCO - nDCG@k (automatic)
* RAG-only, retrieval - BEIR-SciFact - Recall@k (automatic)
* RAG-only, retrieval - BEIR-SciFact - nDCG@k (automatic)
* RAG-only, retrieval - BEIR-NFCorpus - Recall@k (automatic)
* RAG-only, retrieval - BEIR-NFCorpus - nDCG@k (automatic)

**LLM-as-Judge**

Gemini 2.0 Flash через Google AI Studio API — достаточно сильная модель для нашей системы (текущая реализация базируется на QWEN 3 0.6b), поддерживает структурированный JSON-вывод, и имеет бесплатный лимит (15 RPM, 1M токенов в день).

---

**Архитектура оценивания**

* Один Airflow DAG = один eval-suite (например: `eval_chat_hotpotqa`, `eval_retrieval_beir_scifact`,
  `eval_code_humaneval`), а не отдельный DAG на каждую micro-metric.
* Внутри DAG:
    * шаг подготовки конфигурации run,
    * шаг инференса/получения предсказаний,
    * шаг расчёта метрик,
    * шаг логирования в БД

### Аргументы eval-run

Каждый eval-run должен принимать два аргумента для матричного сравнения конфигураций:

* `rag_aliases: list[str]` — список alias-ов RAG (Qdrant collection aliases).
* `lora_aliases: list[str]` — список alias-ов LoRA (MLflow Model Registry aliases).

По умолчанию:

* `rag_aliases=["champion"]`
* `lora_aliases=["champion"]`

Оба аргумента поддерживают более одного значения, например:

* `rag_aliases=["champion","challenger"]`
* `lora_aliases=["champion","challenger"]`

В таком случае один запуск формирует декартово произведение конфигураций и считает метрики для
каждой пары `(rag_alias, lora_alias)`.

Production inference policy:

* По умолчанию online inference использует `rag_alias="champion"`.
* Другие alias-ы (`challenger` и др.) используются только для экспериментов.


---

**Схемы БД**

Задача - залогировать для каждой проверки (task + dataset + metric) уникальный run_id, значение метрики и полную конфигурацию системы (модель, RAG, LoRA, параметры генерации и т.д.) для последующего анализа и построения дашбордов.

```sql
CREATE TABLE eval_runs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at           TIMESTAMPTZ,
    status                TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed

    task                  TEXT NOT NULL,                     -- chat | summarize | code | retrieval
    dataset_name          TEXT NOT NULL,
    metric_name           TEXT NOT NULL,
    metric_value          DOUBLE PRECISION NOT NULL,

    -- Model
    base_model            TEXT NOT NULL,
    adapter_name          TEXT,
    adapter_version       INTEGER,
    adapter_mlflow_run_id TEXT, --read from MLflow Model Registry at eval start.

    -- RAG
    rag_enabled           BOOLEAN NOT NULL DEFAULT false,
    knowledge_base        TEXT,
    embedding_model       TEXT,
    chunking_strategy     TEXT,
    chunk_size            INTEGER,
    chunk_overlap         INTEGER,
    retrieval_top_k       INTEGER,
    score_threshold       DOUBLE PRECISION,
    qdrant_snapshot_id    TEXT, -- call POST /collections/{name}/snapshots before eval, record the ID.
    dataset_dvc_hash      TEXT, -- dvc status or read from .dvc file at eval start.
    reranking_strategy    TEXT, -- none | cross_encoder | llm

    -- Judge & metrics config
    judge_model           TEXT,
    bert_score_model      TEXT,

    -- Generation params
    temperature           DOUBLE PRECISION,
    max_tokens            INTEGER,

    extra                 JSONB NOT NULL DEFAULT '{}',

    error_message         TEXT
);

CREATE INDEX idx_eval_runs_task ON eval_runs (task);
CREATE INDEX idx_eval_runs_dataset ON eval_runs (dataset_name);
CREATE INDEX idx_eval_runs_adapter ON eval_runs (adapter_name, adapter_version);
CREATE INDEX idx_eval_runs_created ON eval_runs (created_at DESC);
CREATE INDEX idx_eval_runs_base_model ON eval_runs (base_model);
CREATE INDEX idx_eval_runs_extra ON eval_runs USING gin (extra);

```

В JSONB-поле `extra` хранится только дополнительная информация, которая может гибко расширяться без миграций.

## Этапы реализации системы оценивания агентского сервиса

### Этап 1: Базовая LLM

**Что нужно сделать:**

### Этап 2: Базовая LLM + RAG

**Что нужно сделать:**
### Этап 3: Базовая LLM + RAG + LoRA

**Что нужно сделать:**
### Этап 4: Агентский сервис с оркестратором

**Что нужно сделать:**
