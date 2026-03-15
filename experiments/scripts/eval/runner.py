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
import json
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
    temperature: float = 0.0,
    max_tokens: int = 512,
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

    resp = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json=payload,
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
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        engine = create_engine(db_url)
        # Import the model lazily to avoid circular imports
        from shared.db.models import Base, EvalRun

        Base.metadata.create_all(engine, tables=[EvalRun.__table__], checkfirst=True)

        with Session(engine) as session:
            for row in rows:
                run = EvalRun(**row)
                session.add(run)
            session.commit()
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

        predictions.append(answer)
        references.append(reference)
        judge_samples.append({
            "question": question,
            "answer": answer,
            "reference": reference,
            "context": rag_context,
        })

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
                predictions, references, bert_score_model=eval_settings.bert_score_model
            )
            if metric in auto_metrics:
                rows.append({
                    **common,
                    "metric_name": metric,
                    "metric_value": auto_metrics[metric],
                })

        elif metric in _JUDGE_METRICS:
            if not eval_settings.google_ai_api_key:
                logger.error(
                    "LLM-as-Judge metric '%s' requires EVAL_GOOGLE_AI_API_KEY", metric,
                )
                return []
            result = judge_batch(
                metric,
                samples=judge_samples,
                api_key=eval_settings.google_ai_api_key,
                model=eval_settings.judge_model,
            )
            rows.append({
                **common,
                "metric_name": metric,
                "metric_value": result[metric],
            })
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
        return [{
            **common,
            "metric_name": metric,
            "metric_value": 0.0,
            "finished_at": now,
            "status": "failed",
            "error_message": str(e),
        }]

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
        rows.append({
            **common,
            "metric_name": metric,
            "metric_value": metrics[metric],
            "finished_at": now,
            "status": "completed",
        })
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


def _load_dataset_samples(
    task: str, dataset_name: str, limit: int = 0
) -> list[dict[str, str]]:
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
    if split_name not in ds_dict:
        logger.error(
            "Split '%s' not found in %s (available: %s)",
            split_name, dataset_path, list(ds_dict.keys()),
        )
        return []

    ds = ds_dict[split_name]

    samples: list[dict[str, str]] = []
    for item in ds:
        if task == "code":
            samples.append({
                "prompt": item.get("prompt", ""),
                "test": item.get("test", ""),
                "answer": item.get("canonical_solution", ""),
            })
        elif task == "summarize":
            samples.append({
                "question": item.get("article", "")[:2000],
                "answer": item.get("abstract", ""),
            })
        else:
            # chat / QA
            samples.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

        if limit > 0 and len(samples) >= limit:
            break

    logger.info(
        "Loaded %d samples from %s [%s]", len(samples), dataset_path, split_name,
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
            f"Metric '{metric}' is not valid for task '{task}'. "
            f"Valid metrics: {valid_metrics}"
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

    for rag_alias, lora_alias in itertools.product(rag_aliases, lora_aliases):
        logger.info(
            "Evaluating: task=%s dataset=%s metric=%s rag=%s lora=%s",
            task, dataset_name, metric, rag_alias, lora_alias,
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

    # Log to DB
    _log_to_db(all_rows, eval_settings.db_url)

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
    )

    try:
        from rag.embeddings import EmbeddingService
        from rag.vector_store import QdrantVectorStore

        embedding_model = build_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        emb_service = EmbeddingService(model_name=embedding_model)
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=temp_collection)

        # Run queries
        queries = [s for s in samples if s.get("query")]
        recall_scores: list[float] = []
        ndcg_scores: list[float] = []

        for q in queries:
            query_emb = emb_service.embed_query(q["query"])
            results = vs.search(query_embedding=query_emb, top_k=10)
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
    common.update({
        "qdrant_collection": temp_collection,
        "embedding_model": build_config.get("embedding_model"),
        "chunking_strategy": build_config.get("chunking_strategy"),
        "chunk_size": build_config.get("chunk_size"),
        "chunk_overlap": build_config.get("chunk_overlap"),
    })

    rows = []
    if metric == "recall_at_10":
        rows.append({
            **common,
            "metric_name": "recall_at_10",
            "metric_value": avg_recall,
            "finished_at": now,
            "status": "completed",
        })
    elif metric == "ndcg_at_10":
        rows.append({
            **common,
            "metric_name": "ndcg_at_10",
            "metric_value": avg_ndcg,
            "finished_at": now,
            "status": "completed",
        })
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
    print(f"\n{'='*60}")
    print(f"Eval complete: {len(rows)} metric rows")
    print(f"{'='*60}")
    for row in rows:
        print(
            f"  {row['task']}/{row['dataset_name']} "
            f"metric={row['metric_name']} "
            f"rag={row.get('rag_alias', 'none')} "
            f"lora={row.get('lora_alias', 'none')} "
            f"→ {row['metric_value']:.4f}"
        )


if __name__ == "__main__":
    main()
