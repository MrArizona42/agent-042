"""Evaluation runner — main entry point for benchmarking.

Stages
------
Stage 1 (basic): Base LLM evaluation (no RAG, no LoRA).
Stage 2 (rag):   Base LLM + RAG evaluation (with retrieval-only evals).
Stage 3 (lora):  Base LLM + RAG + LoRA evaluation (full matrix).

Each eval-suite is a unique ``(task, dataset, metric)`` triple.  The runner
accepts a ``--metric`` flag to select exactly **one** metric per invocation.

Usage::

    # Stage 1 -- base model chat eval, single metric
    python -m experiments.eval.eval_scripts.runner \\
        --task chat --dataset hotpotqa --metric rouge_l

    # Stage 2 -- RAG eval, LLM-as-judge metric
    python -m experiments.eval.eval_scripts.runner \\
        --task chat --dataset hotpotqa --metric relevance \\
        --rag_aliases champion,challenger

    # Stage 3 -- full matrix
    python -m experiments.eval.eval_scripts.runner \\
        --task chat --dataset hotpotqa --metric correctness \\
        --rag_aliases champion,challenger \\
        --lora_aliases champion,challenger
"""

from __future__ import annotations

import itertools
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fire
import httpx
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    insert,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# Add src to path so shared/rag modules are importable
# Canonical directory for pre-downloaded datasets (HF Arrow format).
from experiments.eval.eval_scripts.datasets import DATASET_LOCAL, load_dataset_samples
from experiments.eval.eval_scripts.metrics.automatic import (
    compute_bertscore,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_rouge_l,
)
from experiments.eval.eval_scripts.metrics.code_exec import (
    compute_pass_at_1,
    evaluate_humaneval_sample,
)
from experiments.eval.eval_scripts.metrics.llm_judge import judge_batch
from experiments.eval.eval_scripts.retrieval_bench import (
    build_temp_collection,
    delete_temp_collection,
    read_build_config,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore
from shared.config import (
    bootstrap_local_settings_env,
    get_eval_settings,
    get_registry_settings,
    get_settings,
)
from shared.model_registry import AdapterRegistry

logger = logging.getLogger(__name__)

bootstrap_local_settings_env(repo_root=Path(__file__).resolve().parents[3])

# ---------------------------------------------------------------------------
# Eval-suite configuration: (task, dataset) → fixed KB
# ---------------------------------------------------------------------------

_SUITE_KB: dict[tuple[str, str], str | None] = {
    ("chat", "hotpotqa"): "arxiv",
    ("chat", "nq"): "arxiv",
    ("code", "humaneval"): "pytorch_docs",
    ("summarize", "arxiv_summarization"): None,
    ("retrieval", "beir_scifact"): None,  # KB set via --kb flag
    ("retrieval", "msmarco"): None,
    ("retrieval", "beir_nfcorpus"): None,
}

# Valid metrics per task — each metric is a separate eval-suite
_TASK_METRICS: dict[str, list[str]] = {
    "chat": ["relevance", "correctness", "bertscore_f1", "rouge_l"],
    "summarize": ["faithfulness", "coverage", "bertscore_f1", "rouge_l"],
    "code": ["pass_at_1", "executable_rate"],
    "retrieval": ["recall_at_k", "ndcg_at_k", "mrr_at_k"],
}

# LLM-judge metrics (need Gemini API)
_JUDGE_METRICS = {"relevance", "correctness", "faithfulness", "coverage", "groundedness"}

# Automatic metrics (computed locally, no external API needed)
_AUTOMATIC_METRICS = {"bertscore_f1", "rouge_l", "recall_at_k", "ndcg_at_k", "mrr_at_k"}

# Code-execution metrics (sandboxed Docker execution)
_CODE_EXEC_METRICS = {"pass_at_1", "executable_rate"}

# Groundedness added when RAG is enabled for generation tasks
_RAG_GENERATION_TASKS = {"chat", "code"}

# System prompt for HumanEval code generation requests.
# Explicit instruction to output only code prevents models from wrapping the
# answer in prose or markdown, which would otherwise cause false negatives
# during sandboxed execution.  extract_code_from_response() is still applied
# as a fallback for models that ignore instruction-following.
_CODE_EVAL_SYSTEM_PROMPT = (
    "You are a precise Python coding assistant. "
    "When given a function signature and docstring, output ONLY the Python "
    "function body that completes it — no explanation, no markdown fences, "
    "no import statements unless strictly required by the function, and no "
    "repetition of the signature. Indentation must use 4 spaces."
)


def _build_code_eval_messages(prompt: str) -> list[dict[str, str]]:
    """Build the chat messages list for a single HumanEval sample."""
    return [
        {"role": "system", "content": _CODE_EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Gateway API helpers
# ---------------------------------------------------------------------------


def _call_gateway(
    *,
    messages: list[dict[str, str]],
    gateway_url: str,
    model: str | None = None,
    rag_sources: list[dict[str, str]] | None = None,
    temperature: float,
    max_tokens: int,
    internal_api_key: str,
) -> dict[str, Any]:
    """Call the gateway chat completions API."""
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    if rag_sources:
        payload["rag_sources"] = rag_sources

    headers: dict[str, str] = {}
    if internal_api_key:
        headers["X-API-Key"] = internal_api_key

    resp = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LoRA alias resolution via MLflow
# ---------------------------------------------------------------------------


def _resolve_lora_alias(
    lora_alias: str,
    task: str,
) -> dict[str, Any]:
    """Resolve a LoRA alias to an adapter name and version.

    The adapter name is deterministic: ``{model_name}-{alias}`` which matches
    how ``AdapterSyncer`` registers adapters with vLLM.  MLflow is still
    queried for the version and run_id metadata (for logging), but failure
    to reach MLflow only emits a warning.

    Returns:
        Dict with ``adapter_name``, ``adapter_version``, ``adapter_mlflow_run_id``.
        For ``lora_alias="none"``, all values are ``None``.
    """
    if lora_alias == "none":
        return {"adapter_name": None, "adapter_version": None, "adapter_mlflow_run_id": None}

    model_name = f"lora-{task}"
    adapter_name = f"{model_name}-{lora_alias}"

    # Optionally fetch version/run_id from MLflow for logging
    adapter_version = None
    adapter_run_id = None
    try:
        reg_settings = get_registry_settings()
        registry = AdapterRegistry(tracking_uri=reg_settings.mlflow_tracking_uri)
        mv = registry.client.get_model_version_by_alias(model_name, lora_alias)
        adapter_version = int(mv.version)
        adapter_run_id = mv.run_id
    except Exception as e:
        err_str = str(e)
        # If the registered model or alias doesn't exist, the adapter is
        # definitely not loaded in vLLM — fail early instead of sending
        # requests that will all 404.
        if "RESOURCE_DOES_NOT_EXIST" in err_str or "not found" in err_str.lower():
            raise RuntimeError(
                f"LoRA adapter '{adapter_name}' not available: {e}. "
                f"Register model '{model_name}' in MLflow and assign alias "
                f"'{lora_alias}', then run adapter-sync."
            ) from e
        logger.warning("Could not fetch MLflow metadata for '%s': %s", adapter_name, e)

    return {
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "adapter_mlflow_run_id": adapter_run_id,
    }


# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------


def _log_to_db(
    rows: list[dict[str, Any]],
    db_url: str,
    sample_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write eval result rows to ``eval_runs`` and per-sample detail to ``eval_samples``."""
    if not db_url:
        logger.warning("No DB URL configured; skipping database logging")
        return

    try:
        engine = create_engine(db_url)
        meta = MetaData()

        eval_runs = Table(
            "eval_runs",
            meta,
            Column("id", PG_UUID(as_uuid=True), primary_key=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("finished_at", DateTime(timezone=True)),
            Column("status", Text, nullable=False, server_default="running"),
            Column("task", Text, nullable=False),
            Column("dataset_name", Text, nullable=False),
            Column("metric_name", Text, nullable=False),
            Column("metric_value", Float, nullable=False),
            Column("base_model", Text, nullable=False),
            Column("adapter_name", Text),
            Column("adapter_version", Integer),
            Column("adapter_mlflow_run_id", Text),
            Column("lora_alias", Text),
            Column("rag_enabled", Boolean, nullable=False, server_default="false"),
            Column("rag_alias", Text),
            Column("knowledge_base", Text),
            Column("qdrant_collection", Text),
            Column("embedding_model", Text),
            Column("chunking_strategy", Text),
            Column("chunk_size", Integer),
            Column("chunk_overlap", Integer),
            Column("retrieval_top_k", Integer),
            Column("score_threshold", Float),
            Column("qdrant_snapshot_id", Text),
            Column("dataset_dvc_hash", Text),
            Column("reranking_strategy", Text),
            Column("judge_model", Text),
            Column("bert_score_model", Text),
            Column("temperature", Float),
            Column("max_tokens", Integer),
            Column("extra", JSONB, nullable=False, server_default="{}"),
            Column("error_message", Text),
        )

        eval_samples = Table(
            "eval_samples",
            meta,
            Column("id", PG_UUID(as_uuid=True), primary_key=True),
            Column(
                "eval_run_id",
                PG_UUID(as_uuid=True),
                ForeignKey("eval_runs.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("sample_idx", Integer, nullable=False),
            Column("sample_id", Text),
            Column("input", Text),
            Column("output", Text),
            Column("reference", Text),
            Column("detail", JSONB, nullable=False, server_default="{}"),
            UniqueConstraint("eval_run_id", "sample_idx"),
        )

        meta.create_all(engine, tables=[eval_runs, eval_samples], checkfirst=True)

        with engine.begin() as conn:
            for row in rows:
                if "id" not in row:
                    row["id"] = uuid.uuid4()
                conn.execute(insert(eval_runs).values(**row))

            if sample_rows:
                for sr in sample_rows:
                    if "id" not in sr:
                        sr["id"] = uuid.uuid4()
                    conn.execute(insert(eval_samples).values(**sr))

        n_samples = len(sample_rows) if sample_rows else 0
        logger.info("Logged %d eval rows + %d sample rows to database", len(rows), n_samples)
    except Exception as e:
        logger.error("Failed to log to database: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_common_fields(
    *,
    task: str,
    dataset_name: str,
    base_model: str,
    lora_alias: str,
    lora_info: dict[str, Any],
    rag_alias: str,
    rag_enabled: bool,
    kb_name: str | None,
    eval_settings: Any,
    eval_context: dict[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Build the common fields shared by all metric rows in one eval."""
    eval_context = eval_context or {}
    return {
        "id": uuid.uuid4(),
        "created_at": now,
        "status": "running",
        "task": task,
        "dataset_name": dataset_name,
        "base_model": base_model,
        "adapter_name": lora_info.get("adapter_name"),
        "adapter_version": lora_info.get("adapter_version"),
        "adapter_mlflow_run_id": lora_info.get("adapter_mlflow_run_id"),
        "lora_alias": lora_alias,
        "rag_enabled": rag_enabled,
        "rag_alias": rag_alias if rag_alias != "none" else None,
        "knowledge_base": kb_name if rag_enabled else None,
        "judge_model": eval_context.get("judge_model", eval_settings.judge_model),
        "bert_score_model": eval_context.get(
            "bert_score_model",
            eval_settings.bert_score_model,
        ),
        "temperature": eval_context.get("temperature", eval_settings.temperature),
        "max_tokens": eval_context.get("max_tokens", eval_settings.max_tokens),
        "extra": dict(eval_context.get("extra", {})),
    }


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------

# Project root (repo top-level directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_load_dataset_samples = load_dataset_samples  # alias for internal use
_DATASET_LOCAL = DATASET_LOCAL  # backward-compat alias for tests

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_eval(
    *,
    task: str,
    dataset_name: str,
    metric: str,
    kb_name: str | None = None,
    rag_aliases: list[str],
    lora_aliases: list[str],
    k: int = 10,
) -> list[dict[str, Any]]:
    """Run evaluation for a single metric across all (rag_alias, lora_alias) combinations.

    Each call represents one eval-suite = ``(task, dataset, metric)``.
    Internally delegates to :func:`fetch_predictions` (phase 1) and
    :func:`calculate_metrics` (phase 2).

    Returns:
        All metric result rows.
    """
    # Validate metric is valid for this task
    valid_metrics = _TASK_METRICS.get(task, [])
    if metric not in valid_metrics:
        raise ValueError(
            f"Metric '{metric}' is not valid for task '{task}'. Valid metrics: {valid_metrics}"
        )

    prediction_data = fetch_predictions(
        task=task,
        dataset_name=dataset_name,
        kb_name=kb_name,
        rag_aliases=rag_aliases,
        lora_aliases=lora_aliases,
        k=k,
    )

    return calculate_metrics(
        metric=metric,
        prediction_data=prediction_data,
    )


# Two-phase evaluation: fetch predictions, then compute metrics
# ---------------------------------------------------------------------------


def fetch_predictions(
    *,
    task: str,
    dataset_name: str,
    kb_name: str | None = None,
    rag_aliases: list[str],
    lora_aliases: list[str],
    k: int = 10,
) -> dict[str, Any]:
    """Phase 1: Generate predictions for all (rag, lora) combinations.

    Calls the gateway / retrieval system to produce predictions *without*
    computing any metrics.  The returned dict is JSON-serializable and
    contains everything :func:`calculate_metrics` needs.
    """
    eval_settings = get_eval_settings()
    settings = get_settings()
    base_model = settings.default_model

    if task not in _TASK_METRICS:
        raise ValueError(f"Unknown task: {task!r}")

    if kb_name is None:
        kb_name = _SUITE_KB.get((task, dataset_name))

    if task == "summarize":
        rag_aliases = ["none"]
    if task == "retrieval":
        lora_aliases = ["none"]

    bundles: list[dict[str, Any]] = []
    failures: list[tuple[str, str, BaseException]] = []

    for rag_alias, lora_alias in itertools.product(rag_aliases, lora_aliases):
        logger.info(
            "Fetching predictions: task=%s dataset=%s rag=%s lora=%s",
            task,
            dataset_name,
            rag_alias,
            lora_alias,
        )
        try:
            if task == "retrieval":
                bundle = _fetch_retrieval_predictions(
                    dataset_name=dataset_name,
                    rag_alias=rag_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    k=k,
                )
            elif task == "code":
                bundle = _fetch_code_predictions(
                    dataset_name=dataset_name,
                    rag_alias=rag_alias,
                    lora_alias=lora_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                )
            else:
                bundle = _fetch_generation_predictions(
                    task=task,
                    dataset_name=dataset_name,
                    rag_alias=rag_alias,
                    lora_alias=lora_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                )
            bundles.append(bundle)
        except Exception as e:
            logger.error(
                "Prediction fetch failed for rag=%s lora=%s: %s",
                rag_alias,
                lora_alias,
                e,
                exc_info=True,
            )
            failures.append((rag_alias, lora_alias, e))

    if failures:
        failed_pairs = ", ".join(f"rag={r} lora={lo}" for r, lo, _ in failures)
        raise RuntimeError(
            f"{len(failures)} prediction fetch(es) failed ({failed_pairs}): {failures[0][2]}"
        )

    return {
        "task": task,
        "dataset_name": dataset_name,
        "kb_name": kb_name,
        "base_model": base_model,
        "temperature": eval_settings.temperature,
        "max_tokens": eval_settings.max_tokens,
        "judge_model": eval_settings.judge_model,
        "bert_score_model": eval_settings.bert_score_model,
        "k": k,
        "bundles": bundles,
    }


def _fetch_generation_predictions(
    *,
    task: str,
    dataset_name: str,
    rag_alias: str,
    lora_alias: str,
    kb_name: str | None,
    eval_settings: Any,
) -> dict[str, Any]:
    """Fetch predictions for a single (rag, lora) pair (chat/summarize tasks)."""
    lora_info = _resolve_lora_alias(lora_alias, task)
    model_name = lora_info["adapter_name"] if lora_info["adapter_name"] else None

    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True

    samples = _load_dataset_samples(task, dataset_name, limit=eval_settings.sample_limit)
    if not samples:
        raise RuntimeError(f"No samples loaded for {task}/{dataset_name}")

    predictions: list[str] = []
    references: list[str] = []
    judge_samples: list[dict[str, str]] = []
    sample_details: list[dict[str, Any]] = []
    gateway_failures = 0

    for idx, sample in enumerate(samples):
        question = sample["question"]
        reference = sample.get("answer", "")

        messages = [{"role": "user", "content": question}]
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=eval_settings.gateway_url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=eval_settings.temperature,
                max_tokens=eval_settings.max_tokens,
                internal_api_key=eval_settings.internal_api_key,
            )
            answer = response["choices"][0]["message"]["content"]

            rag_context = ""
            if rag_enabled and "rag_context" in response:
                chunks = response.get("rag_context") or []
                rag_context = "\n".join(c.get("content", "") for c in chunks)
        except Exception as e:
            logger.error("Gateway call failed: %s", e)
            answer = ""
            rag_context = ""
            gateway_failures += 1

        predictions.append(answer)
        references.append(reference)
        judge_samples.append(
            {
                "question": question,
                "answer": answer,
                "reference": reference,
                "context": rag_context,
            }
        )
        sample_details.append(
            {
                "sample_idx": idx,
                "sample_id": sample.get("id"),
                "input": question,
                "output": answer,
                "reference": reference,
                "detail": {"rag_context": rag_context} if rag_context else {},
            }
        )

    if gateway_failures == len(samples):
        raise RuntimeError(
            f"All {gateway_failures} gateway calls failed for {task}/{dataset_name}; "
            "check that the gateway service is reachable"
        )

    return {
        "rag_alias": rag_alias,
        "lora_alias": lora_alias,
        "lora_info": lora_info,
        "rag_enabled": rag_enabled,
        "predictions": predictions,
        "references": references,
        "judge_samples": judge_samples,
        "sample_details": sample_details,
    }


def _fetch_code_predictions(
    *,
    dataset_name: str,
    rag_alias: str,
    lora_alias: str,
    kb_name: str | None,
    eval_settings: Any,
) -> dict[str, Any]:
    """Fetch code generation predictions for a single (rag, lora) pair."""
    lora_info = _resolve_lora_alias(lora_alias, "code")
    model_name = lora_info["adapter_name"] if lora_info["adapter_name"] else None

    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True

    samples = _load_dataset_samples("code", dataset_name, limit=eval_settings.sample_limit)
    if not samples:
        raise RuntimeError(f"No samples loaded for code/{dataset_name}")

    exec_results: list[dict[str, Any]] = []
    sample_details: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        test_code = sample.get("test", "")

        messages = _build_code_eval_messages(prompt)
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=eval_settings.gateway_url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=eval_settings.temperature,
                max_tokens=eval_settings.max_tokens,
                internal_api_key=eval_settings.internal_api_key,
            )
            generated = response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Gateway call failed: %s", e)
            generated = ""

        result = evaluate_humaneval_sample(
            prompt=prompt,
            generated_code=generated,
            test_code=test_code,
            timeout=eval_settings.code_exec_timeout,
            mem_limit=eval_settings.code_exec_mem_limit,
            cpus=eval_settings.code_exec_cpus,
        )
        exec_results.append(result)
        sample_details.append(
            {
                "sample_idx": idx,
                "sample_id": sample.get("task_id"),
                "input": prompt,
                "output": generated,
                "reference": test_code,
                "detail": {
                    "passed": result["passed"],
                    "exit_code": result["exit_code"],
                    "stderr": result.get("stderr", ""),
                },
            }
        )

    return {
        "rag_alias": rag_alias,
        "lora_alias": lora_alias,
        "lora_info": lora_info,
        "rag_enabled": rag_enabled,
        "exec_results": exec_results,
        "sample_details": sample_details,
    }


def _fetch_retrieval_predictions(
    *,
    dataset_name: str,
    rag_alias: str,
    kb_name: str | None,
    eval_settings: Any,
    k: int,
) -> dict[str, Any]:
    """Fetch retrieval query results for a single rag_alias."""
    if not kb_name:
        raise ValueError("Retrieval eval requires kb_name")

    settings = get_settings()
    qdrant_host = settings.qdrant_host
    qdrant_port = settings.qdrant_port

    build_config = read_build_config(
        kb_name=kb_name,
        rag_alias=rag_alias,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    if build_config is None:
        raise RuntimeError(f"Cannot read build config for {kb_name}_{rag_alias}")

    samples = _load_dataset_samples("retrieval", dataset_name, limit=eval_settings.sample_limit)
    if not samples:
        raise RuntimeError(f"No samples loaded for retrieval/{dataset_name}")

    # BEIR datasets supply per-query relevant_docs; msmarco-style supply a flat
    # list of corpus items with a "text" field.  Build the corpus accordingly.
    if samples[0].get("relevant_docs") is not None:
        # BEIR: deduplicate across queries
        seen: set[str] = set()
        corpus = []
        for s in samples:
            for doc in s["relevant_docs"]:
                if doc["doc_id"] not in seen:
                    corpus.append(doc)
                    seen.add(doc["doc_id"])
    else:
        corpus = [
            {"doc_id": s.get("doc_id", str(i)), "text": s.get("text", "")}
            for i, s in enumerate(samples)
        ]

    temp_collection = build_temp_collection(
        kb_name=kb_name,
        dataset_name=dataset_name,
        rag_alias=rag_alias,
        corpus=corpus,
        build_config=build_config,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        embeddings_url=settings.embeddings_url,
    )

    try:
        embedding_model = build_config.embedding_model
        emb_service = EmbeddingService(
            model_name=embedding_model,
            embeddings_url=settings.embeddings_url,
        )
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=temp_collection)

        queries = [s for s in samples if s.get("query")]
        query_results: list[dict[str, Any]] = []
        sample_details: list[dict[str, Any]] = []

        for idx, q in enumerate(queries):
            query_emb = emb_service.embed_query(q["query"])
            results = vs.search(query_embedding=query_emb, top_k=k, score_threshold=0.0)
            retrieved_ids = [doc.metadata.get("source", "") for doc in results]
            relevance = q.get("relevance", {})
            query_results.append(
                {
                    "retrieved_ids": retrieved_ids,
                    "relevance": relevance,
                }
            )
            sample_details.append(
                {
                    "sample_idx": idx,
                    "sample_id": q.get("query_id"),
                    "input": q["query"],
                    "output": None,
                    "reference": None,
                    "detail": {
                        "retrieved_ids": retrieved_ids,
                        "relevance": relevance,
                    },
                }
            )
    finally:
        try:
            delete_temp_collection(
                temp_collection,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
            )
        except Exception as e:
            logger.warning("Failed to delete temp collection: %s", e)

    return {
        "rag_alias": rag_alias,
        "lora_alias": "none",
        "lora_info": {
            "adapter_name": None,
            "adapter_version": None,
            "adapter_mlflow_run_id": None,
        },
        "rag_enabled": True,
        "query_results": query_results,
        "sample_details": sample_details,
        "build_config": build_config.to_payload(),
        "temp_collection": temp_collection,
    }


def calculate_metrics(
    *,
    metric: str,
    prediction_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Phase 2: Compute a metric on pre-fetched predictions and log to DB.

    Args:
        metric: Metric to compute.
        prediction_data: Output from :func:`fetch_predictions`.

    Returns:
        Metric result rows suitable for database insertion.
    """
    task = prediction_data["task"]
    dataset_name = prediction_data["dataset_name"]
    kb_name = prediction_data["kb_name"]
    base_model = prediction_data["base_model"]

    valid_metrics = _TASK_METRICS.get(task, [])
    if metric not in valid_metrics:
        raise ValueError(
            f"Metric {metric!r} is not valid for task {task!r}. Valid: {valid_metrics}"
        )

    eval_context = prediction_data.get("eval_context") or {}

    eval_settings = get_eval_settings()

    all_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, BaseException]] = []

    for bundle in prediction_data["bundles"]:
        rag_alias = bundle["rag_alias"]
        lora_alias = bundle["lora_alias"]

        logger.info(
            "Computing metric=%s for rag=%s lora=%s",
            metric,
            rag_alias,
            lora_alias,
        )

        try:
            if task == "retrieval":
                rows = _compute_retrieval_metric(
                    metric=metric,
                    bundle=bundle,
                    dataset_name=dataset_name,
                    base_model=base_model,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    eval_context=eval_context,
                    k=prediction_data["k"],
                )
            elif task == "code":
                rows = _compute_code_metric(
                    metric=metric,
                    bundle=bundle,
                    dataset_name=dataset_name,
                    base_model=base_model,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    eval_context=eval_context,
                )
            else:
                rows = _compute_generation_metric(
                    metric=metric,
                    bundle=bundle,
                    task=task,
                    dataset_name=dataset_name,
                    base_model=base_model,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    eval_context=eval_context,
                )

            # Link per-sample details to the eval_run row(s)
            sample_details = bundle.get("sample_details", [])
            for row in rows:
                run_id = row.setdefault("id", uuid.uuid4())
                for sd in sample_details:
                    all_sample_rows.append(
                        {
                            "eval_run_id": run_id,
                            "sample_idx": sd["sample_idx"],
                            "sample_id": sd.get("sample_id"),
                            "input": sd.get("input"),
                            "output": sd.get("output"),
                            "reference": sd.get("reference"),
                            "detail": sd.get("detail", {}),
                        }
                    )

            all_rows.extend(rows)
        except Exception as e:
            logger.error(
                "Metric computation failed for rag=%s lora=%s: %s",
                rag_alias,
                lora_alias,
                e,
                exc_info=True,
            )
            failures.append((rag_alias, lora_alias, e))

    _log_to_db(all_rows, eval_settings.db_url, sample_rows=all_sample_rows)

    if failures:
        failed_pairs = ", ".join(f"rag={r} lora={lo}" for r, lo, _ in failures)
        raise RuntimeError(
            f"{len(failures)} metric computation(s) failed ({failed_pairs}): {failures[0][2]}"
        )

    logger.info("Metrics complete: %d rows", len(all_rows))
    return all_rows


def _compute_generation_metric(
    *,
    metric: str,
    bundle: dict[str, Any],
    task: str,
    dataset_name: str,
    base_model: str,
    kb_name: str | None,
    eval_settings: Any,
    eval_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute a single metric on pre-fetched generation predictions."""
    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task=task,
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias=bundle["lora_alias"],
        lora_info=bundle["lora_info"],
        rag_alias=bundle["rag_alias"],
        rag_enabled=bundle["rag_enabled"],
        kb_name=kb_name,
        eval_settings=eval_settings,
        eval_context=eval_context,
        now=now,
    )

    predictions = bundle["predictions"]
    references = bundle["references"]
    judge_samples = bundle["judge_samples"]

    rows: list[dict[str, Any]] = []

    if metric in _AUTOMATIC_METRICS:
        metric_value: float | None = None
        if metric == "rouge_l":
            scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
            metric_value = sum(scores) / len(scores) if scores else 0.0
        elif metric.startswith("bertscore"):
            bert = compute_bertscore(
                predictions,
                references,
                model_name=(eval_context or {}).get(
                    "bert_score_model",
                    eval_settings.bert_score_model,
                ),
            )
            metric_value = bert.get(metric)
        if metric_value is not None:
            rows.append({**common, "metric_name": metric, "metric_value": metric_value})
    elif metric in _JUDGE_METRICS:
        google_ai_api_key = (eval_context or {}).get(
            "google_ai_api_key",
            eval_settings.google_ai_api_key,
        )
        judge_model = (eval_context or {}).get("judge_model", eval_settings.judge_model)
        if not google_ai_api_key:
            raise RuntimeError(f"LLM-as-Judge metric {metric!r} requires EVAL_GOOGLE_AI_API_KEY")
        result = judge_batch(
            metric,
            samples=judge_samples,
            api_key=google_ai_api_key,
            model=judge_model,
        )
        rows.append({**common, "metric_name": metric, "metric_value": result[metric]})

    finished = datetime.now(timezone.utc)
    for row in rows:
        row["finished_at"] = finished
        row["status"] = "completed"

    return rows


def _compute_code_metric(
    *,
    metric: str,
    bundle: dict[str, Any],
    dataset_name: str,
    base_model: str,
    kb_name: str | None,
    eval_settings: Any,
    eval_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute a single metric on pre-fetched code execution results."""
    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task="code",
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias=bundle["lora_alias"],
        lora_info=bundle["lora_info"],
        rag_alias=bundle["rag_alias"],
        rag_enabled=bundle["rag_enabled"],
        kb_name=kb_name,
        eval_settings=eval_settings,
        eval_context=eval_context,
        now=now,
    )

    metrics = compute_pass_at_1(bundle["exec_results"])

    rows = []
    if metric in metrics:
        rows.append(
            {
                **common,
                "metric_name": metric,
                "metric_value": metrics[metric],
                "finished_at": now,
                "status": "completed",
            }
        )
    return rows


def _compute_retrieval_metric(
    *,
    metric: str,
    bundle: dict[str, Any],
    dataset_name: str,
    base_model: str,
    kb_name: str | None,
    eval_settings: Any,
    eval_context: dict[str, Any] | None = None,
    k: int,
) -> list[dict[str, Any]]:
    """Compute a single retrieval metric on pre-fetched query results."""
    query_results = bundle["query_results"]
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []

    for qr in query_results:
        retrieved_ids = qr["retrieved_ids"]
        relevance = qr["relevance"]
        relevant_ids = {doc_id for doc_id, rel in relevance.items() if rel > 0}

        recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, k=k))
        ndcg_scores.append(compute_ndcg_at_k(retrieved_ids, relevance, k=k))
        mrr_scores.append(compute_mrr_at_k(retrieved_ids, relevant_ids, k=k))

    if not recall_scores:
        raise RuntimeError(f"No query results for retrieval/{dataset_name}")

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task="retrieval",
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias="none",
        lora_info={
            "adapter_name": None,
            "adapter_version": None,
            "adapter_mlflow_run_id": None,
        },
        rag_alias=bundle["rag_alias"],
        rag_enabled=True,
        kb_name=kb_name,
        eval_settings=eval_settings,
        eval_context=eval_context,
        now=now,
    )

    build_config = bundle.get("build_config", {})
    common.update(
        {
            "qdrant_collection": bundle.get("temp_collection"),
            "embedding_model": build_config.get("embedding_model"),
            "chunking_strategy": build_config.get("chunking_strategy"),
            "chunk_size": build_config.get("chunk_size"),
            "chunk_overlap": build_config.get("chunk_overlap"),
        }
    )

    rows = []
    if metric == "recall_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"recall_at_{k}",
                "metric_value": avg_recall,
                "finished_at": now,
                "status": "completed",
            }
        )
    elif metric == "ndcg_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"ndcg_at_{k}",
                "metric_value": avg_ndcg,
                "finished_at": now,
                "status": "completed",
            }
        )
    elif metric == "mrr_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"mrr_at_{k}",
                "metric_value": avg_mrr,
                "finished_at": now,
                "status": "completed",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    task: str,
    dataset: str,
    metric: str,
    kb: str | None = None,
    rag_aliases: str = "none",
    lora_aliases: str = "none",
    k: int = 10,
) -> None:
    """Run evaluation for a single metric.

    Args:
        task: One of chat, summarize, code, retrieval.
        dataset: Dataset name (e.g. hotpotqa, humaneval).
        metric: Metric to compute (e.g. rouge_l, relevance, pass_at_1, recall_at_k).
        kb: Knowledge base name (required for retrieval evals).
        rag_aliases: Comma-separated RAG alias roles.
        lora_aliases: Comma-separated LoRA alias roles.
        k: Top-K cutoff for retrieval metrics (default: 10).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _valid_tasks = ("chat", "summarize", "code", "retrieval")
    if task not in _valid_tasks:
        raise SystemExit(f"Invalid task '{task}'. Choose from {_valid_tasks}")

    rag_list = [a.strip() for a in rag_aliases.split(",")]
    lora_list = [a.strip() for a in lora_aliases.split(",")]

    rows = run_eval(
        task=task,
        dataset_name=dataset,
        metric=metric,
        kb_name=kb,
        rag_aliases=rag_list,
        lora_aliases=lora_list,
        k=k,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Eval complete: {len(rows)} metric rows")
    print(f"{'=' * 60}")
    for row in rows:
        print(
            f"  {row['task']}/{row['dataset_name']} "
            f"metric={row['metric_name']} "
            f"rag={row.get('rag_alias') or 'none'} "
            f"lora={row.get('lora_alias') or 'none'} "
            f"-> {row['metric_value']:.4f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
