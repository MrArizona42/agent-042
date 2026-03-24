"""Evaluation runner — main entry point for benchmarking.

Stages
------
Stage 1 (basic): Base LLM evaluation (no RAG, no LoRA).
Stage 2 (rag):   Base LLM + RAG evaluation (with retrieval-only evals).
Stage 3 (lora):  Base LLM + RAG + LoRA evaluation (full matrix).

Each eval-suite is a unique ``(task, dataset, metric)`` triple.  The runner
accepts a ``--metric`` flag to select exactly **one** metric per invocation.

Usage::

    # Stage 1 — base model chat eval, single metric
    python -m experiments.scripts.eval.runner \\
        --task chat --dataset hotpotqa --metric rouge_l

    # Stage 2 — RAG eval, LLM-as-judge metric
    python -m experiments.scripts.eval.runner \\
        --task chat --dataset hotpotqa --metric relevance \\
        --rag-aliases champion,challenger

    # Stage 3 — full matrix
    python -m experiments.scripts.eval.runner \\
        --task chat --dataset hotpotqa --metric correctness \\
        --rag-aliases champion,challenger \\
        --lora-aliases champion,challenger
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Add src to path so shared/rag modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from shared.config import get_eval_settings, get_settings

logger = logging.getLogger(__name__)

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
    "retrieval": ["recall_at_10", "ndcg_at_10"],
}

# LLM-judge metrics (need Gemini API)
_JUDGE_METRICS = {"relevance", "correctness", "faithfulness", "coverage", "groundedness"}

# Automatic metrics (computed locally, no external API needed)
_AUTOMATIC_METRICS = {"bertscore_f1", "rouge_l", "recall_at_10", "ndcg_at_10"}

# Code-execution metrics (sandboxed Docker execution)
_CODE_EXEC_METRICS = {"pass_at_1", "executable_rate"}

# Groundedness added when RAG is enabled for generation tasks
_RAG_GENERATION_TASKS = {"chat", "code"}


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
    """Resolve a LoRA alias to an adapter name and version via MLflow.

    Returns:
        Dict with ``adapter_name``, ``adapter_version``, ``adapter_mlflow_run_id``.
        For ``lora_alias="none"``, all values are ``None``.
    """
    if lora_alias == "none":
        return {"adapter_name": None, "adapter_version": None, "adapter_mlflow_run_id": None}

    try:
        from experiments.scripts.train_adapter.registry import AdapterRegistry
        from shared.config import get_registry_settings

        reg_settings = get_registry_settings()
        registry = AdapterRegistry(tracking_uri=reg_settings.mlflow_tracking_uri)

        # Adapter names follow the lora-<task> convention
        model_name = f"lora-{task}"
        mv = registry.client.get_model_version_by_alias(model_name, lora_alias)

        return {
            "adapter_name": model_name,
            "adapter_version": int(mv.version),
            "adapter_mlflow_run_id": mv.run_id,
        }
    except Exception as e:
        logger.warning("Could not resolve LoRA alias '%s': %s", lora_alias, e)
        return {"adapter_name": None, "adapter_version": None, "adapter_mlflow_run_id": None}


# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------


def _log_to_db(rows: list[dict[str, Any]], db_url: str) -> None:
    """Write eval result rows to the ``eval_runs`` table."""
    if not db_url:
        logger.warning("No DB URL configured; skipping database logging")
        return

    try:
        from sqlalchemy import (
            Boolean,
            Column,
            DateTime,
            Float,
            Integer,
            MetaData,
            Table,
            Text,
            create_engine,
            insert,
        )
        from sqlalchemy.dialects.postgresql import JSONB
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID

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

        meta.create_all(engine, tables=[eval_runs], checkfirst=True)

        with engine.begin() as conn:
            for row in rows:
                if "id" not in row:
                    row["id"] = uuid.uuid4()
                conn.execute(insert(eval_runs).values(**row))

        logger.info("Logged %d eval rows to database", len(rows))
    except Exception as e:
        logger.error("Failed to log to database: %s", e)


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


def _evaluate_generation(
    *,
    task: str,
    dataset_name: str,
    metric: str,
    rag_alias: str,
    lora_alias: str,
    kb_name: str | None,
    eval_settings: Any,
    base_model: str,
) -> list[dict[str, Any]]:
    """Run generation eval for a single (rag_alias, lora_alias) pair and a single metric.

    Returns a list of metric result dicts ready for DB insertion.
    """
    from experiments.scripts.eval.metrics.automatic import compute_automatic_metrics
    from experiments.scripts.eval.metrics.llm_judge import judge_batch

    gateway_url = eval_settings.gateway_url
    temperature = eval_settings.temperature
    max_tokens = eval_settings.max_tokens
    sample_limit = eval_settings.sample_limit
    internal_api_key = eval_settings.internal_api_key

    # Resolve LoRA adapter
    lora_info = _resolve_lora_alias(lora_alias, task)
    model_name = lora_info["adapter_name"] if lora_info["adapter_name"] else None

    # Build RAG sources
    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True

    # Load dataset samples (placeholder — datasets loaded from HuggingFace or local)
    samples = _load_dataset_samples(task, dataset_name, limit=sample_limit)
    if not samples:
        logger.warning("No samples loaded for %s/%s", task, dataset_name)
        return []

    # Generate predictions
    predictions: list[str] = []
    references: list[str] = []
    judge_samples: list[dict[str, str]] = []
    gateway_failures = 0

    for sample in samples:
        question = sample["question"]
        reference = sample.get("answer", "")

        messages = [{"role": "user", "content": question}]
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=gateway_url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=temperature,
                max_tokens=max_tokens,
                internal_api_key=internal_api_key,
            )
            answer = response["choices"][0]["message"]["content"]

            # Extract RAG context for groundedness
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

    if gateway_failures == len(samples):
        raise RuntimeError(
            f"All {gateway_failures} gateway calls failed for {task}/{dataset_name}; "
            "check that the gateway service is reachable"
        )

    # Compute the single requested metric
    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task=task,
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias=lora_alias,
        lora_info=lora_info,
        rag_alias=rag_alias,
        rag_enabled=rag_enabled,
        kb_name=kb_name,
        eval_settings=eval_settings,
        now=now,
    )

    rows: list[dict[str, Any]] = []

    try:
        if metric in _AUTOMATIC_METRICS:
            auto_metrics = compute_automatic_metrics(
                predictions,
                references,
                bert_score_model=eval_settings.bert_score_model,
                metric=metric,
            )
            if metric in auto_metrics:
                rows.append(
                    {
                        **common,
                        "metric_name": metric,
                        "metric_value": auto_metrics[metric],
                    }
                )

        elif metric in _JUDGE_METRICS:
            if not eval_settings.google_ai_api_key:
                logger.error(
                    "LLM-as-Judge metric '%s' requires EVAL_GOOGLE_AI_API_KEY",
                    metric,
                )
                return []
            result = judge_batch(
                metric,
                samples=judge_samples,
                api_key=eval_settings.google_ai_api_key,
                model=eval_settings.judge_model,
            )
            rows.append(
                {
                    **common,
                    "metric_name": metric,
                    "metric_value": result[metric],
                }
            )
    except Exception as e:
        logger.error("Metric computation failed: %s", e, exc_info=True)
        finished = datetime.now(timezone.utc)
        for row in rows:
            row["finished_at"] = finished
            row["status"] = "failed"
            row["error_message"] = str(e)
        return rows

    # Mark completed
    finished = datetime.now(timezone.utc)
    for row in rows:
        row["finished_at"] = finished
        row["status"] = "completed"

    return rows


def _evaluate_code(
    *,
    dataset_name: str,
    metric: str,
    rag_alias: str,
    lora_alias: str,
    kb_name: str | None,
    eval_settings: Any,
    base_model: str,
) -> list[dict[str, Any]]:
    """Run HumanEval code generation eval for a single metric."""
    from experiments.scripts.eval.metrics.code_exec import (
        compute_pass_at_1,
        evaluate_humaneval_sample,
    )

    gateway_url = eval_settings.gateway_url
    internal_api_key = eval_settings.internal_api_key
    lora_info = _resolve_lora_alias(lora_alias, "code")
    model_name = lora_info["adapter_name"] if lora_info["adapter_name"] else None

    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True

    samples = _load_dataset_samples("code", dataset_name, limit=eval_settings.sample_limit)
    if not samples:
        return []

    exec_results: list[dict[str, Any]] = []
    for sample in samples:
        prompt = sample["prompt"]
        test_code = sample.get("test", "")

        messages = [{"role": "user", "content": f"Complete this Python function:\n\n{prompt}"}]
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=gateway_url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=eval_settings.temperature,
                max_tokens=eval_settings.max_tokens,
                internal_api_key=internal_api_key,
            )
            generated = response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Gateway call failed: %s", e)
            generated = ""

        result = evaluate_humaneval_sample(
            prompt=prompt,
            generated_code=generated,
            test_code=test_code,
            image=eval_settings.code_exec_image,
            timeout=eval_settings.code_exec_timeout,
            mem_limit=eval_settings.code_exec_mem_limit,
            cpus=eval_settings.code_exec_cpus,
        )
        exec_results.append(result)

    try:
        metrics = compute_pass_at_1(exec_results)
    except Exception as e:
        logger.error("Code metric computation failed: %s", e, exc_info=True)
        now = datetime.now(timezone.utc)
        common = _build_common_fields(
            task="code",
            dataset_name=dataset_name,
            base_model=base_model,
            lora_alias=lora_alias,
            lora_info=lora_info,
            rag_alias=rag_alias,
            rag_enabled=rag_enabled,
            kb_name=kb_name,
            eval_settings=eval_settings,
            now=now,
        )
        return [
            {
                **common,
                "metric_name": metric,
                "metric_value": 0.0,
                "finished_at": now,
                "status": "failed",
                "error_message": str(e),
            }
        ]

    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task="code",
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias=lora_alias,
        lora_info=lora_info,
        rag_alias=rag_alias,
        rag_enabled=rag_enabled,
        kb_name=kb_name,
        eval_settings=eval_settings,
        now=now,
    )

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
    now: datetime,
) -> dict[str, Any]:
    """Build the common fields shared by all metric rows in one eval."""
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
        "judge_model": eval_settings.judge_model,
        "bert_score_model": eval_settings.bert_score_model,
        "temperature": eval_settings.temperature,
        "max_tokens": eval_settings.max_tokens,
        "extra": {},
    }


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------

# Project root (repo top-level directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Canonical directory for pre-downloaded datasets (HF Arrow format).
DATASETS_DIR = _PROJECT_ROOT / "assets" / "datasets"

# Mapping from eval runner dataset_name → local folder under assets/datasets/
# and the split to read.  Datasets are saved via ``save_to_disk`` in the
# prefetch notebook and pulled via DVC.
_DATASET_LOCAL: dict[str, tuple[str, str]] = {
    "hotpotqa": ("hotpotqa", "validation"),
    "nq": ("natural-questions", "validation"),
    "arxiv_summarization": ("arxiv-summarization", "validation"),
    "humaneval": ("humaneval", "train"),  # HumanEval test split saved as "train"
    "beir_scifact": ("beir-scifact", "train"),
    "beir_nfcorpus": ("beir-nfcorpus", "train"),
    "msmarco": ("msmarco", "validation"),
}


def _load_dataset_samples(task: str, dataset_name: str, limit: int) -> list[dict[str, str]]:
    """Load evaluation dataset samples from local Arrow files.

    Datasets must be pre-downloaded to ``assets/datasets/{folder_name}``
    (HuggingFace Arrow format, saved via ``DatasetDict.save_to_disk``).
    Use ``experiments/notebooks/prefetch_assets.ipynb`` or ``dvc pull`` to
    populate the directory.

    Returns:
        List of sample dicts with at least ``question`` and ``answer`` keys
        (or ``prompt`` and ``test`` for code tasks).
    """
    from datasets import load_from_disk

    if dataset_name not in _DATASET_LOCAL:
        logger.warning("Unknown dataset: %s", dataset_name)
        return []

    folder_name, split_name = _DATASET_LOCAL[dataset_name]
    dataset_path = DATASETS_DIR / folder_name

    if not dataset_path.exists():
        logger.error(
            "Dataset directory not found: %s — run prefetch_assets notebook or "
            "'dvc pull' to download datasets",
            dataset_path,
        )
        return []

    ds_dict = load_from_disk(str(dataset_path))

    # -----------------------------------------------------------------------
    # BEIR-style datasets: queries + qrels splits present
    # -----------------------------------------------------------------------
    if task == "retrieval" and "queries" in ds_dict and "qrels" in ds_dict:
        queries_ds = ds_dict["queries"]
        qrels_ds = ds_dict["qrels"]

        # Build relevance map: query_id -> {corpus_id: score}
        relevance_map: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qid = str(row["query_id"])
            cid = str(row["corpus_id"])
            relevance_map.setdefault(qid, {})[cid] = int(row["score"])

        corpus_split = next((s for s in ("corpus", "train") if s in ds_dict), None)
        corpus_ds = ds_dict[corpus_split] if corpus_split else None

        samples: list[dict[str, str]] = []
        for item in queries_ds:
            qid = str(item["_id"])
            if qid not in relevance_map:
                continue  # no judgements for this query
            query_text = item.get("text", "")
            if not query_text:
                continue

            rel = relevance_map[qid]

            # Corpus docs for this query (needed to build temp collection)
            relevant_docs: list[dict] = []
            if corpus_ds is not None:
                relevant_cids = set(rel.keys())
                for doc in corpus_ds:
                    if str(doc["_id"]) in relevant_cids:
                        relevant_docs.append(
                            {"doc_id": str(doc["_id"]), "text": doc.get("text", "")}
                        )

            samples.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "relevance": rel,
                    "relevant_docs": relevant_docs,
                }
            )
            if limit > 0 and len(samples) >= limit:
                break

        logger.info(
            "Loaded %d BEIR queries from %s",
            len(samples),
            dataset_path,
        )
        return samples

    # -----------------------------------------------------------------------
    # Standard single-split path
    # -----------------------------------------------------------------------
    if split_name not in ds_dict:
        logger.error(
            "Split '%s' not found in %s (available: %s)",
            split_name,
            dataset_path,
            list(ds_dict.keys()),
        )
        return []

    ds = ds_dict[split_name]

    samples: list[dict[str, str]] = []
    for item in ds:
        if task == "code":
            samples.append(
                {
                    "prompt": item.get("prompt", ""),
                    "test": item.get("test", ""),
                    "answer": item.get("canonical_solution", ""),
                }
            )
        elif task == "summarize":
            samples.append(
                {
                    "question": item.get("article", "")[:2000],
                    "answer": item.get("abstract", ""),
                }
            )
        elif task == "retrieval":
            # Support both BEIR-style (top-level "text"/"_id") and
            # msmarco-style (nested "passages.passage_text") datasets.
            text = item.get("text", "")
            if not text:
                passages_data = item.get("passages", {})
                passage_texts = passages_data.get("passage_text", [])
                is_selected = passages_data.get("is_selected", [0] * len(passage_texts))
                selected_texts = [t for t, s in zip(passage_texts, is_selected) if s]
                text = (
                    selected_texts[0]
                    if selected_texts
                    else (passage_texts[0] if passage_texts else "")
                )
            if not text:
                continue
            doc_id = str(
                item.get("doc_id") or item.get("_id") or item.get("query_id") or len(samples)
            )
            query = item.get("query", "") or item.get("question", "")
            samples.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "query": query,
                    "relevance": {doc_id: 1} if query else {},
                }
            )
        else:
            # chat / QA
            # NQ stores question as {"text": "...", "tokens": [...]}; unwrap if needed
            question = item.get("question", "")
            if isinstance(question, dict):
                question = question.get("text", "")

            # NQ has no top-level "answer" — extract from annotations.short_answers
            answer = item.get("answer", "")
            if not answer:
                annotations = item.get("annotations", {})

                short_answers_col = (
                    annotations.get("short_answers", []) if isinstance(annotations, dict) else []
                )
                for sa_list in short_answers_col:
                    for sa in sa_list if isinstance(sa_list, list) else []:
                        texts = sa.get("text", []) if isinstance(sa, dict) else []
                        if texts:
                            answer = texts[0]
                            break
                    if answer:
                        break

            samples.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )

        if limit > 0 and len(samples) >= limit:
            break

    logger.info(
        "Loaded %d samples from %s [%s]",
        len(samples),
        dataset_path,
        split_name,
    )
    return samples


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
) -> list[dict[str, Any]]:
    """Run evaluation for a single metric across all (rag_alias, lora_alias) combinations.

    Each call represents one eval-suite = ``(task, dataset, metric)``.
    Computes the Cartesian product of alias lists and runs each pair.

    Returns:
        All metric result rows.
    """
    eval_settings = get_eval_settings()
    settings = get_settings()
    base_model = settings.default_model

    # Validate metric is valid for this task
    valid_metrics = _TASK_METRICS.get(task, [])
    if metric not in valid_metrics:
        raise ValueError(
            f"Metric '{metric}' is not valid for task '{task}'. Valid metrics: {valid_metrics}"
        )

    # Resolve fixed KB for this suite
    if kb_name is None:
        kb_name = _SUITE_KB.get((task, dataset_name))

    # For summarization, RAG is irrelevant
    if task == "summarize":
        rag_aliases = ["none"]

    # For retrieval tasks, LoRA is irrelevant
    if task == "retrieval":
        lora_aliases = ["none"]

    all_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, BaseException]] = []

    for rag_alias, lora_alias in itertools.product(rag_aliases, lora_aliases):
        logger.info(
            "Evaluating: task=%s dataset=%s metric=%s rag=%s lora=%s",
            task,
            dataset_name,
            metric,
            rag_alias,
            lora_alias,
        )

        try:
            if task == "code":
                rows = _evaluate_code(
                    dataset_name=dataset_name,
                    metric=metric,
                    rag_alias=rag_alias,
                    lora_alias=lora_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    base_model=base_model,
                )
            elif task == "retrieval":
                rows = _evaluate_retrieval(
                    dataset_name=dataset_name,
                    metric=metric,
                    rag_alias=rag_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    base_model=base_model,
                )
            else:
                rows = _evaluate_generation(
                    task=task,
                    dataset_name=dataset_name,
                    metric=metric,
                    rag_alias=rag_alias,
                    lora_alias=lora_alias,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
                    base_model=base_model,
                )
            all_rows.extend(rows)
        except Exception as e:
            logger.error(
                "Eval failed for rag=%s lora=%s: %s", rag_alias, lora_alias, e, exc_info=True
            )
            failures.append((rag_alias, lora_alias, e))

    # Log to DB
    _log_to_db(all_rows, eval_settings.db_url)

    if failures:
        failed_pairs = ", ".join(f"rag={r} lora={lo}" for r, lo, _ in failures)
        raise RuntimeError(
            f"{len(failures)} eval combination(s) failed ({failed_pairs}): {failures[0][2]}"
        )

    logger.info("Evaluation complete: %d metric rows", len(all_rows))
    return all_rows


def _evaluate_retrieval(
    *,
    dataset_name: str,
    metric: str,
    rag_alias: str,
    kb_name: str | None,
    eval_settings: Any,
    base_model: str,
) -> list[dict[str, Any]]:
    """Run retrieval-only eval for one rag_alias and a single metric."""
    from experiments.scripts.eval.metrics.automatic import compute_ndcg_at_k, compute_recall_at_k
    from experiments.scripts.eval.retrieval_bench import (
        build_temp_collection,
        delete_temp_collection,
        read_build_config,
    )

    if not kb_name:
        logger.error("Retrieval eval requires --kb argument")
        return []

    settings = get_settings()
    qdrant_host = settings.qdrant_host
    qdrant_port = settings.qdrant_port

    # Read build config from production collection
    build_config = read_build_config(
        kb_name=kb_name,
        rag_alias=rag_alias,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    if build_config is None:
        logger.error("Cannot read build config for %s_%s", kb_name, rag_alias)
        return []

    # Load benchmark corpus and queries
    samples = _load_dataset_samples("retrieval", dataset_name, limit=eval_settings.sample_limit)
    if not samples:
        return []

    corpus = [
        {"doc_id": s.get("doc_id", str(i)), "text": s.get("text", "")}
        for i, s in enumerate(samples)
    ]

    # Build temp collection
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
        from rag.embeddings import EmbeddingService
        from rag.vector_store import QdrantVectorStore

        embedding_model = build_config["embedding_model"]
        emb_service = EmbeddingService(model_name=embedding_model)
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=temp_collection)

        # Run queries
        queries = [s for s in samples if s.get("query")]
        recall_scores: list[float] = []
        ndcg_scores: list[float] = []

        for q in queries:
            query_emb = emb_service.embed_query(q["query"])
            results = vs.search(query_embedding=query_emb, top_k=10, score_threshold=0.0)
            retrieved_ids = [doc.metadata.get("source", "") for doc in results]

            relevance = q.get("relevance", {})
            relevant_ids = {doc_id for doc_id, rel in relevance.items() if rel > 0}

            recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, k=10))
            ndcg_scores.append(compute_ndcg_at_k(retrieved_ids, relevance, k=10))

        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    finally:
        try:
            delete_temp_collection(
                temp_collection,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
            )
        except Exception as e:
            logger.warning("Failed to delete temp collection: %s", e)

    now = datetime.now(timezone.utc)
    common = _build_common_fields(
        task="retrieval",
        dataset_name=dataset_name,
        base_model=base_model,
        lora_alias="none",
        lora_info={"adapter_name": None, "adapter_version": None, "adapter_mlflow_run_id": None},
        rag_alias=rag_alias,
        rag_enabled=True,
        kb_name=kb_name,
        eval_settings=eval_settings,
        now=now,
    )
    common.update(
        {
            "qdrant_collection": temp_collection,
            "embedding_model": build_config.get("embedding_model"),
            "chunking_strategy": build_config.get("chunking_strategy"),
            "chunk_size": build_config.get("chunk_size"),
            "chunk_overlap": build_config.get("chunk_overlap"),
        }
    )

    rows = []
    if metric == "recall_at_10":
        rows.append(
            {
                **common,
                "metric_name": "recall_at_10",
                "metric_value": avg_recall,
                "finished_at": now,
                "status": "completed",
            }
        )
    elif metric == "ndcg_at_10":
        rows.append(
            {
                **common,
                "metric_name": "ndcg_at_10",
                "metric_value": avg_ndcg,
                "finished_at": now,
                "status": "completed",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Two-phase evaluation: fetch predictions, then compute metrics
# ---------------------------------------------------------------------------


def fetch_predictions(
    *,
    task: str,
    dataset_name: str,
    kb_name: str | None = None,
    rag_aliases: list[str],
    lora_aliases: list[str],
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
    gateway_failures = 0

    for sample in samples:
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
    from experiments.scripts.eval.metrics.code_exec import evaluate_humaneval_sample

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
    for sample in samples:
        prompt = sample["prompt"]
        test_code = sample.get("test", "")

        messages = [{"role": "user", "content": f"Complete this Python function:\n\n{prompt}"}]
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
            image=eval_settings.code_exec_image,
            timeout=eval_settings.code_exec_timeout,
            mem_limit=eval_settings.code_exec_mem_limit,
            cpus=eval_settings.code_exec_cpus,
        )
        exec_results.append(result)

    return {
        "rag_alias": rag_alias,
        "lora_alias": lora_alias,
        "lora_info": lora_info,
        "rag_enabled": rag_enabled,
        "exec_results": exec_results,
    }


def _fetch_retrieval_predictions(
    *,
    dataset_name: str,
    rag_alias: str,
    kb_name: str | None,
    eval_settings: Any,
) -> dict[str, Any]:
    """Fetch retrieval query results for a single rag_alias."""
    from experiments.scripts.eval.retrieval_bench import (
        build_temp_collection,
        delete_temp_collection,
        read_build_config,
    )

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
        from rag.embeddings import EmbeddingService
        from rag.vector_store import QdrantVectorStore

        embedding_model = build_config["embedding_model"]
        emb_service = EmbeddingService(model_name=embedding_model)
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=temp_collection)

        queries = [s for s in samples if s.get("query")]
        query_results: list[dict[str, Any]] = []

        for q in queries:
            query_emb = emb_service.embed_query(q["query"])
            results = vs.search(query_embedding=query_emb, top_k=10, score_threshold=0.0)
            retrieved_ids = [doc.metadata.get("source", "") for doc in results]
            relevance = q.get("relevance", {})
            query_results.append(
                {
                    "retrieved_ids": retrieved_ids,
                    "relevance": relevance,
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
        "build_config": {
            "embedding_model": build_config.get("embedding_model"),
            "chunking_strategy": build_config.get("chunking_strategy"),
            "chunk_size": build_config.get("chunk_size"),
            "chunk_overlap": build_config.get("chunk_overlap"),
        },
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

    eval_settings = get_eval_settings()

    all_rows: list[dict[str, Any]] = []
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
                )
            elif task == "code":
                rows = _compute_code_metric(
                    metric=metric,
                    bundle=bundle,
                    dataset_name=dataset_name,
                    base_model=base_model,
                    kb_name=kb_name,
                    eval_settings=eval_settings,
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

    _log_to_db(all_rows, eval_settings.db_url)

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
) -> list[dict[str, Any]]:
    """Compute a single metric on pre-fetched generation predictions."""
    from experiments.scripts.eval.metrics.automatic import compute_automatic_metrics
    from experiments.scripts.eval.metrics.llm_judge import judge_batch

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
        now=now,
    )

    predictions = bundle["predictions"]
    references = bundle["references"]
    judge_samples = bundle["judge_samples"]

    rows: list[dict[str, Any]] = []

    if metric in _AUTOMATIC_METRICS:
        auto = compute_automatic_metrics(
            predictions,
            references,
            bert_score_model=eval_settings.bert_score_model,
            metric=metric,
        )
        if metric in auto:
            rows.append({**common, "metric_name": metric, "metric_value": auto[metric]})
    elif metric in _JUDGE_METRICS:
        if not eval_settings.google_ai_api_key:
            raise RuntimeError(f"LLM-as-Judge metric {metric!r} requires EVAL_GOOGLE_AI_API_KEY")
        result = judge_batch(
            metric,
            samples=judge_samples,
            api_key=eval_settings.google_ai_api_key,
            model=eval_settings.judge_model,
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
) -> list[dict[str, Any]]:
    """Compute a single metric on pre-fetched code execution results."""
    from experiments.scripts.eval.metrics.code_exec import compute_pass_at_1

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
) -> list[dict[str, Any]]:
    """Compute a single retrieval metric on pre-fetched query results."""
    from experiments.scripts.eval.metrics.automatic import compute_ndcg_at_k, compute_recall_at_k

    query_results = bundle["query_results"]
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    for qr in query_results:
        retrieved_ids = qr["retrieved_ids"]
        relevance = qr["relevance"]
        relevant_ids = {doc_id for doc_id, rel in relevance.items() if rel > 0}

        recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, k=10))
        ndcg_scores.append(compute_ndcg_at_k(retrieved_ids, relevance, k=10))

    if not recall_scores:
        raise RuntimeError(f"No query results for retrieval/{dataset_name}")

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

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
    if metric == "recall_at_10":
        rows.append(
            {
                **common,
                "metric_name": "recall_at_10",
                "metric_value": avg_recall,
                "finished_at": now,
                "status": "completed",
            }
        )
    elif metric == "ndcg_at_10":
        rows.append(
            {
                **common,
                "metric_name": "ndcg_at_10",
                "metric_value": avg_ndcg,
                "finished_at": now,
                "status": "completed",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation runner for agent-042",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=["chat", "summarize", "code", "retrieval"])
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. hotpotqa, humaneval)")
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric to compute (e.g. rouge_l, relevance, pass_at_1, recall_at_10)",
    )
    parser.add_argument("--kb", default=None, help="Knowledge base (required for retrieval evals)")
    parser.add_argument(
        "--rag-aliases",
        default="none",
        help="Comma-separated RAG alias roles (default: none)",
    )
    parser.add_argument(
        "--lora-aliases",
        default="none",
        help="Comma-separated LoRA alias roles (default: none)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    rag_aliases = [a.strip() for a in args.rag_aliases.split(",")]
    lora_aliases = [a.strip() for a in args.lora_aliases.split(",")]

    rows = run_eval(
        task=args.task,
        dataset_name=args.dataset,
        metric=args.metric,
        kb_name=args.kb,
        rag_aliases=rag_aliases,
        lora_aliases=lora_aliases,
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
            f"→ {row['metric_value']:.4f}"
        )


if __name__ == "__main__":
    main()
