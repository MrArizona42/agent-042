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
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
from rag.reranker import get_reranker
from rag.retriever import Retriever
from rag.sources.materialize import validate_strategy_supported
from rag.sparse_encoder import SparseEncoderService
from rag.vector_store import QdrantVectorStore
from shared.catalog import get_kb_config
from shared.config import (
    JudgeSettings,
    get_settings,
    secret_value,
)
from shared.model_registry import AdapterRegistry

logger = logging.getLogger(__name__)

_GATEWAY_STREAM_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=None,
    write=30.0,
    pool=30.0,
)

# ---------------------------------------------------------------------------
# Eval-suite configuration: (task, dataset) → fixed KB
# ---------------------------------------------------------------------------

_SUITE_KB: dict[tuple[str, str], str | None] = {
    ("chat", "hotpotqa"): None,
    ("chat", "nq"): None,
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

# LLM-judge metrics (need a configured judge backend)
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

_CHAT_EVAL_SYSTEM_PROMPT = (
    "You are answering a benchmark question. "
    "Reply with the shortest correct answer only. "
    "Do not ask for clarification, do not explain, and do not preface the answer. "
    "If the answer is unknown, say 'unknown'."
)


def _build_code_eval_messages(prompt: str) -> list[dict[str, str]]:
    """Build the chat messages list for a single HumanEval sample."""
    return [
        {"role": "system", "content": _CODE_EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_chat_eval_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages list for a single chat benchmark sample."""
    return [
        {"role": "system", "content": _CHAT_EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _progress_log_stride(total: int, *, target_updates: int = 20) -> int:
    """Return a log stride that keeps progress output bounded."""
    if total <= 0:
        return 1
    if target_updates <= 1:
        return total
    return max(1, (total + target_updates - 1) // target_updates)


def _render_progress_bar(completed: int, total: int, *, width: int = 20) -> str:
    """Render a compact ASCII progress bar for task logs."""
    if total <= 0:
        return f"[{'-' * width}]"

    ratio = min(1.0, max(0.0, completed / total))
    filled = min(width, int(ratio * width))
    if completed >= total:
        filled = width
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _log_fetch_progress(
    *,
    phase: str,
    task: str,
    dataset_name: str,
    rag_alias: str,
    lora_alias: str,
    completed: int,
    total: int,
    every: int,
    unit: str = "samples",
    gateway_failures: int = 0,
) -> None:
    """Log bounded per-bundle progress into the Airflow task log."""
    if total <= 0:
        return
    if completed not in (0, total) and completed % every != 0:
        return

    percent = (completed / total) * 100.0
    failure_suffix = f" gateway_failures={gateway_failures}" if gateway_failures else ""
    logger.info(
        "%s progress: task=%s dataset=%s rag=%s lora=%s %s %d/%d %s (%.1f%%)%s",
        phase,
        task,
        dataset_name,
        rag_alias,
        lora_alias,
        _render_progress_bar(completed, total),
        completed,
        total,
        unit,
        percent,
        failure_suffix,
    )


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
    internal_api_key: str,
    max_completion_tokens: int | None = None,
    expect_rag_context: bool = False,
) -> dict[str, Any]:
    """Call the gateway SSE chat API and rebuild the final response shape.

    The gateway owns idle-timeout enforcement for async streams, so the eval
    client disables the HTTP read timeout and only keeps connection/write/pool
    timeouts bounded.
    """
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if model:
        payload["model"] = model
    if rag_sources is not None:
        payload["rag_sources"] = rag_sources
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens

    headers: dict[str, str] = {}
    if internal_api_key:
        headers["X-API-Key"] = internal_api_key

    with httpx.stream(
        "POST",
        f"{gateway_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=_GATEWAY_STREAM_TIMEOUT,
    ) as resp:
        resp.raise_for_status()

        request_id = resp.headers.get("X-Request-Id")
        response_id = f"chatcmpl-{request_id}" if request_id else None
        assistant_fragments: list[str] = []
        finish_reason = "stop"
        usage: dict[str, Any] = {}

        for raw_payload in _iter_sse_payloads(resp.iter_lines()):
            if raw_payload == "[DONE]":
                break

            chunk = json.loads(raw_payload)
            error = chunk.get("error")
            if isinstance(error, dict):
                raise RuntimeError(error.get("message", "Gateway streaming error"))

            if response_id is None and isinstance(chunk.get("id"), str):
                response_id = chunk["id"]

            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content", "")
                if content:
                    assistant_fragments.append(content)
                finish_reason = choices[0].get("finish_reason") or finish_reason

            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage = chunk_usage

    result: dict[str, Any] = {
        "id": response_id or "chatcmpl-eval-stream",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(assistant_fragments),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

    if request_id:
        result["request_id"] = request_id
        preview = _fetch_prompt_preview(
            gateway_url=gateway_url,
            request_id=request_id,
            internal_api_key=internal_api_key,
        )
        if preview is not None:
            prompt_messages = preview.get("prompt_messages")
            if isinstance(prompt_messages, list):
                result["_prompt_messages"] = prompt_messages
            rag_context = preview.get("rag_context")
            if isinstance(rag_context, list):
                result["rag_context"] = rag_context

    if expect_rag_context and "rag_context" not in result:
        raise RuntimeError("Prompt preview response did not include rag_context")

    return result


def _iter_sse_payloads(lines: Iterable[str]) -> Iterable[str]:
    """Yield SSE payload bodies from an HTTP line iterator."""
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


def _fetch_prompt_preview(
    *,
    gateway_url: str,
    request_id: str,
    internal_api_key: str,
) -> dict[str, Any] | None:
    """Fetch prompt preview metadata for a streamed request."""
    headers: dict[str, str] = {}
    if internal_api_key:
        headers["X-API-Key"] = internal_api_key

    resp = httpx.get(
        f"{gateway_url}/v1/chat/prompt-preview/{request_id}",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    preview = resp.json()
    return preview if isinstance(preview, dict) else None


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
        settings = get_settings()
        registry = AdapterRegistry(tracking_uri=settings.platform.mlflow_tracking_uri)
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
            Column("qdrant_alias", Text),
            Column("qdrant_collection", Text),
            Column("rag_manifest_id", Text),
            Column("embedding_model", Text),
            Column("chunking_strategy", Text),
            Column("chunk_size", Integer),
            Column("chunk_overlap", Integer),
            Column("retrieval_top_k", Integer),
            Column("score_threshold", Float),
            Column("qdrant_snapshot_id", Text),
            Column("dataset_dvc_hash", Text),
            Column("reranking_strategy", Text),
            Column("judge_backend", Text),
            Column("judge_model", Text),
            Column("bert_score_model", Text),
            Column("temperature", Float),
            Column("max_tokens", Integer),
            Column("eval_verdict", Text),
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


def _object_attr(obj: Any, name: str, default: Any = None) -> Any:
    values = getattr(obj, "__dict__", None)
    if isinstance(values, dict) and name in values:
        return values[name]
    return default


def _eval_metrics(eval_settings: Any) -> Any:
    return _object_attr(eval_settings, "metrics", eval_settings)


def _eval_sandbox(eval_settings: Any) -> Any:
    return _object_attr(eval_settings, "sandbox", eval_settings)


def _resolve_judge_settings(
    *,
    eval_settings: Any,
    eval_context: dict[str, Any] | None = None,
    platform: Any | None = None,
) -> JudgeSettings:
    resolve = getattr(type(eval_settings), "resolve_judge_settings", None)
    if callable(resolve):
        effective_platform = (
            platform or _object_attr(eval_settings, "platform", None) or get_settings().platform
        )
        base = eval_settings.resolve_judge_settings(effective_platform)
    else:
        base = JudgeSettings(
            backend=_object_attr(eval_settings, "judge_backend", "local_vllm"),
            model=_object_attr(eval_settings, "judge_model", ""),
            base_url=(
                _object_attr(eval_settings, "judge_base_url", "")
                or _object_attr(eval_settings, "vllm_base_url", "")
            ),
            api_key=_object_attr(eval_settings, "judge_api_key", None),
            timeout=float(_object_attr(eval_settings, "judge_timeout", 60.0)),
            request_delay_seconds=float(
                _object_attr(eval_settings, "judge_request_delay_seconds", 0.0)
            ),
        )

    context = eval_context or {}
    backend = context.get("judge_backend", base.backend)
    model = context.get("judge_model", base.model)
    base_url = context.get("judge_base_url", base.base_url)
    api_key = context.get("judge_api_key", base.api_key)
    timeout = float(context.get("judge_timeout", base.timeout))
    request_delay_seconds = float(
        context.get("judge_request_delay_seconds", base.request_delay_seconds)
    )

    if backend == "local_vllm":
        if not base_url:
            base_url = (
                _object_attr(eval_settings, "vllm_base_url", "")
                or _object_attr(_object_attr(eval_settings, "platform", None), "vllm_base_url", "")
                or get_settings().platform.vllm_base_url
            )
    elif backend == "openai_compatible":
        if not model:
            raise RuntimeError("LLM-as-Judge backend 'openai_compatible' requires judge_model")
        if not base_url:
            raise RuntimeError("LLM-as-Judge backend 'openai_compatible' requires judge_base_url")
    else:
        raise RuntimeError(f"Unsupported judge backend: {backend!r}")

    if not model:
        raise RuntimeError(f"LLM-as-Judge backend {backend!r} could not resolve a model")
    if not base_url:
        raise RuntimeError(f"LLM-as-Judge backend {backend!r} could not resolve a base URL")

    return JudgeSettings(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key.strip() or None if isinstance(api_key, str) else api_key,
        timeout=timeout,
        request_delay_seconds=request_delay_seconds,
    )


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
    metrics = _eval_metrics(eval_settings)
    judge_settings = _resolve_judge_settings(
        eval_settings=eval_settings,
        eval_context=eval_context,
    )
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
        "judge_backend": judge_settings.backend,
        "judge_model": judge_settings.model,
        "bert_score_model": eval_context.get(
            "bert_score_model",
            metrics.bert_score_model,
        ),
        "dataset_dvc_hash": eval_context.get("dataset_dvc_hash") or _dataset_dvc_hash(dataset_name),
        "temperature": eval_context.get("temperature", metrics.temperature),
        "max_tokens": eval_context.get("max_tokens"),
        "extra": dict(eval_context.get("extra", {})),
    }


def _rag_observability_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract resolved RAG alias/collection/manifest metadata from prompt chunks."""
    if not chunks:
        return {}

    qdrant_aliases: set[str] = set()
    collections: set[str] = set()
    manifest_ids: set[str] = set()
    capabilities: set[str] = set()
    scores: list[float] = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        qdrant_alias = metadata.get("qdrant_alias")
        collection_name = metadata.get("collection_name")
        manifest_id = metadata.get("manifest_id")
        capability = metadata.get("retrieval_capability")
        if qdrant_alias:
            qdrant_aliases.add(str(qdrant_alias))
        if collection_name:
            collections.add(str(collection_name))
        if manifest_id:
            manifest_ids.add(str(manifest_id))
        if capability:
            capabilities.add(str(capability))
        score = chunk.get("score")
        if isinstance(score, int | float):
            scores.append(float(score))

    return {
        "qdrant_alias": next(iter(sorted(qdrant_aliases)), None),
        "qdrant_collection": next(iter(sorted(collections)), None),
        "rag_manifest_id": next(iter(sorted(manifest_ids)), None),
        "retrieval_capability": next(iter(sorted(capabilities)), None),
        "hit_count": len(chunks),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_avg": sum(scores) / len(scores) if scores else None,
        "qdrant_aliases": sorted(qdrant_aliases),
        "qdrant_collections": sorted(collections),
        "rag_manifest_ids": sorted(manifest_ids),
    }


def _merge_rag_observability(row: dict[str, Any], bundle: dict[str, Any]) -> None:
    """Copy bundle-level RAG observability fields into a DB row."""
    observability = bundle.get("rag_observability")
    if not isinstance(observability, dict):
        return

    for source_key, row_key in (
        ("qdrant_alias", "qdrant_alias"),
        ("qdrant_collection", "qdrant_collection"),
        ("rag_manifest_id", "rag_manifest_id"),
    ):
        value = observability.get(source_key)
        if value and not row.get(row_key):
            row[row_key] = value

    extra = dict(row.get("extra") or {})
    rag_extra = dict(extra.get("rag") or {})
    for key, value in observability.items():
        if value is not None:
            rag_extra[key] = value
    if rag_extra:
        extra["rag"] = rag_extra
    row["extra"] = extra


def _metric_verdict(
    *,
    metric_name: str,
    metric_value: float,
    eval_context: dict[str, Any] | None,
) -> str:
    """Return pass/warn/fail when thresholds are configured; otherwise unscored."""
    thresholds = (eval_context or {}).get("thresholds")
    if not isinstance(thresholds, dict):
        return "unscored"
    threshold = thresholds.get(metric_name)
    if not isinstance(threshold, dict):
        return "unscored"

    higher_is_better = bool(threshold.get("higher_is_better", True))
    pass_value = threshold.get("pass")
    warn_value = threshold.get("warn")
    if not isinstance(pass_value, int | float) or not isinstance(warn_value, int | float):
        return "unscored"

    if higher_is_better:
        if metric_value >= float(pass_value):
            return "pass"
        if metric_value >= float(warn_value):
            return "warn"
        return "fail"

    if metric_value <= float(pass_value):
        return "pass"
    if metric_value <= float(warn_value):
        return "warn"
    return "fail"


def _finalize_metric_rows(
    rows: list[dict[str, Any]],
    *,
    finished_at: datetime,
    bundle: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach standard completion and observability fields to metric rows."""
    for row in rows:
        row["finished_at"] = finished_at
        row["status"] = "completed"
        _merge_rag_observability(row, bundle)
        row["eval_verdict"] = _metric_verdict(
            metric_name=str(row["metric_name"]),
            metric_value=float(row["metric_value"]),
            eval_context=eval_context,
        )
    return rows


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------

# Project root (repo top-level directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_load_dataset_samples = load_dataset_samples  # alias for internal use
_DATASET_LOCAL = DATASET_LOCAL  # backward-compat alias for tests


def _dvc_pointer_hash(pointer_path: Path) -> str | None:
    """Read the first md5 value from a DVC pointer file."""
    if not pointer_path.exists():
        return None
    text = pointer_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*-?\s*md5:\s*([A-Za-z0-9_.-]+)\s*$", text)
    return match.group(1) if match else None


def _dataset_dvc_hash(dataset_name: str) -> str | None:
    """Return the DVC pointer hash for an eval dataset if available."""
    dataset_info = _DATASET_LOCAL.get(dataset_name)
    if dataset_info is None:
        return None
    dataset_dir = dataset_info[0]
    return _dvc_pointer_hash(_PROJECT_ROOT / "assets" / "datasets" / f"{dataset_dir}.dvc")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_eval(
    *,
    task: str,
    dataset_name: str,
    metric: str,
    kb_name: str | None = None,
    use_auto_rag: bool = False,
    rag_aliases: list[str],
    lora_aliases: list[str],
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
        use_auto_rag=use_auto_rag,
        rag_aliases=rag_aliases,
        lora_aliases=lora_aliases,
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
    use_auto_rag: bool = False,
    rag_aliases: list[str],
    lora_aliases: list[str],
) -> dict[str, Any]:
    """Phase 1: Generate predictions for all (rag, lora) combinations.

    Calls the gateway / retrieval system to produce predictions *without*
    computing any metrics.  The returned dict is JSON-serializable and
    contains everything :func:`calculate_metrics` needs.
    """
    settings = get_settings()
    eval_settings = settings.eval
    metrics = eval_settings.metrics
    base_model = settings.vllm.model
    judge_settings = _resolve_judge_settings(
        eval_settings=eval_settings,
        platform=settings.platform,
    )

    if task not in _TASK_METRICS:
        raise ValueError(f"Unknown task: {task!r}")

    if task == "retrieval" and kb_name is None:
        raise ValueError("Retrieval eval requires kb_name")

    if use_auto_rag:
        if task == "retrieval":
            raise ValueError("Retrieval eval does not support use_auto_rag")
        kb_name = None
    elif kb_name is None:
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
        "temperature": metrics.temperature,
        "judge_backend": judge_settings.backend,
        "judge_model": judge_settings.model,
        "bert_score_model": metrics.bert_score_model,
        "eval_context": {
            "temperature": metrics.temperature,
            "judge_backend": judge_settings.backend,
            "judge_model": judge_settings.model,
            "judge_base_url": judge_settings.base_url,
            "judge_timeout": judge_settings.timeout,
            "judge_request_delay_seconds": judge_settings.request_delay_seconds,
            "bert_score_model": metrics.bert_score_model,
            "max_tokens": metrics.max_completion_tokens,
        },
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
    settings = get_settings()
    metrics = _eval_metrics(eval_settings)

    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True
    elif task == "chat":
        # Explicitly disable gateway auto-selection for chat benchmarks.
        rag_sources = []

    samples = _load_dataset_samples(task, dataset_name)
    if not samples:
        raise RuntimeError(f"No samples loaded for {task}/{dataset_name}")
    total_samples = len(samples)
    progress_every = _progress_log_stride(total_samples)

    predictions: list[str] = []
    references: list[str] = []
    judge_samples: list[dict[str, str]] = []
    sample_details: list[dict[str, Any]] = []
    gateway_failures = 0
    rag_context_chunks_seen: list[dict[str, Any]] = []

    _log_fetch_progress(
        phase="generation",
        task=task,
        dataset_name=dataset_name,
        rag_alias=rag_alias,
        lora_alias=lora_alias,
        completed=0,
        total=total_samples,
        every=progress_every,
    )

    for idx, sample in enumerate(samples):
        question = sample["question"]
        reference = sample.get("answer", "")

        if task == "chat":
            messages = _build_chat_eval_messages(question)
        else:
            messages = [{"role": "user", "content": question}]
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=settings.gateway.url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=metrics.temperature,
                internal_api_key=secret_value(settings.auth.internal_api_key) or "",
                max_completion_tokens=metrics.max_completion_tokens,
                expect_rag_context=rag_enabled,
            )
            answer = response["choices"][0]["message"]["content"]

            rag_context = ""
            if rag_enabled and "rag_context" in response:
                chunks = response.get("rag_context") or []
                if isinstance(chunks, list):
                    rag_context_chunks_seen.extend(
                        chunk for chunk in chunks if isinstance(chunk, dict)
                    )
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
        _log_fetch_progress(
            phase="generation",
            task=task,
            dataset_name=dataset_name,
            rag_alias=rag_alias,
            lora_alias=lora_alias,
            completed=idx + 1,
            total=total_samples,
            every=progress_every,
            gateway_failures=gateway_failures,
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
        "rag_observability": _rag_observability_from_chunks(rag_context_chunks_seen),
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
    settings = get_settings()
    metrics = _eval_metrics(eval_settings)
    sandbox = _eval_sandbox(eval_settings)

    rag_sources = None
    rag_enabled = False
    if rag_alias != "none" and kb_name:
        rag_sources = [{"knowledge_base": kb_name, "alias": rag_alias}]
        rag_enabled = True

    samples = _load_dataset_samples("code", dataset_name)
    if not samples:
        raise RuntimeError(f"No samples loaded for code/{dataset_name}")
    total_samples = len(samples)
    progress_every = _progress_log_stride(total_samples)

    exec_results: list[dict[str, Any]] = []
    sample_details: list[dict[str, Any]] = []
    gateway_failures = 0
    rag_context_chunks_seen: list[dict[str, Any]] = []

    _log_fetch_progress(
        phase="code",
        task="code",
        dataset_name=dataset_name,
        rag_alias=rag_alias,
        lora_alias=lora_alias,
        completed=0,
        total=total_samples,
        every=progress_every,
    )

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        test_code = sample.get("test", "")

        messages = _build_code_eval_messages(prompt)
        try:
            response = _call_gateway(
                messages=messages,
                gateway_url=settings.gateway.url,
                model=model_name,
                rag_sources=rag_sources,
                temperature=metrics.temperature,
                internal_api_key=secret_value(settings.auth.internal_api_key) or "",
                max_completion_tokens=metrics.max_completion_tokens,
                expect_rag_context=rag_enabled,
            )
            generated = response["choices"][0]["message"]["content"]
            if rag_enabled and isinstance(response.get("rag_context"), list):
                rag_context_chunks_seen.extend(
                    chunk for chunk in response["rag_context"] if isinstance(chunk, dict)
                )
        except Exception as e:
            logger.error("Gateway call failed: %s", e)
            generated = ""
            gateway_failures += 1

        result = evaluate_humaneval_sample(
            prompt=prompt,
            generated_code=generated,
            test_code=test_code,
            entry_point=sample.get("entry_point"),
            timeout=sandbox.code_exec_timeout,
            mem_limit=sandbox.code_exec_mem_limit,
            cpus=sandbox.code_exec_cpus,
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
        _log_fetch_progress(
            phase="code",
            task="code",
            dataset_name=dataset_name,
            rag_alias=rag_alias,
            lora_alias=lora_alias,
            completed=idx + 1,
            total=total_samples,
            every=progress_every,
            gateway_failures=gateway_failures,
        )

    return {
        "rag_alias": rag_alias,
        "lora_alias": lora_alias,
        "lora_info": lora_info,
        "rag_enabled": rag_enabled,
        "exec_results": exec_results,
        "sample_details": sample_details,
        "rag_observability": _rag_observability_from_chunks(rag_context_chunks_seen),
    }


def _fetch_retrieval_predictions(
    *,
    dataset_name: str,
    rag_alias: str,
    kb_name: str | None,
    eval_settings: Any,
) -> dict[str, Any]:
    """Fetch retrieval query results for a single rag_alias."""
    if not kb_name:
        raise ValueError("Retrieval eval requires kb_name")

    settings = get_settings()
    qdrant_host = settings.platform.qdrant_host
    qdrant_port = settings.platform.qdrant_port

    kb_config = get_kb_config(kb_name)
    if kb_config is None:
        raise RuntimeError(f"KB '{kb_name}' not found in the catalog")
    alias_config = kb_config.aliases.get(rag_alias)
    if alias_config is None:
        raise RuntimeError(f"Alias '{rag_alias}' not found for KB '{kb_name}'")

    build_config = read_build_config(
        kb_name=kb_name,
        rag_alias=rag_alias,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    if build_config is None:
        raise RuntimeError(f"Cannot read build config for {kb_name}_{rag_alias}")

    validate_strategy_supported(
        retrieval_strategy=alias_config.retrieval_strategy,
        retrieval_capability=build_config.retrieval_capability,
    )
    if alias_config.retrieval_strategy == "hybrid" and (
        build_config.sparse_encoder != settings.rag.sparse_encoder_model
    ):
        raise ValueError(
            f"Runtime sparse encoder '{settings.rag.sparse_encoder_model}' does not match "
            f"collection sparse encoder '{build_config.sparse_encoder}'"
        )

    samples = _load_dataset_samples("retrieval", dataset_name)
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
        embeddings_url=settings.platform.embeddings_url,
    )

    emb_service = EmbeddingService(
        model_name=build_config.embedding_model,
        embeddings_url=settings.platform.embeddings_url,
    )
    sparse_encoder = None
    if alias_config.retrieval_strategy in {"hybrid", "sparse"}:
        sparse_encoder = SparseEncoderService(embeddings_url=settings.platform.embeddings_url)
    reranker = get_reranker(alias_config.reranker) if alias_config.reranker else None

    try:
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=temp_collection)
        retriever = Retriever(
            embedding_service=emb_service,
            vector_store=vs,
            reranker=reranker,
            sparse_encoder_service=sparse_encoder,
            reranker_multiplier=alias_config.reranker_multiplier,
        )

        queries = [s for s in samples if s.get("query")]
        total_queries = len(queries)
        progress_every = _progress_log_stride(total_queries)
        query_results: list[dict[str, Any]] = []
        sample_details: list[dict[str, Any]] = []

        _log_fetch_progress(
            phase="retrieval",
            task="retrieval",
            dataset_name=dataset_name,
            rag_alias=rag_alias,
            lora_alias="none",
            completed=0,
            total=total_queries,
            every=progress_every,
            unit="queries",
        )

        for idx, q in enumerate(queries):
            results = retriever.retrieve(
                query=q["query"],
                top_k=alias_config.top_k,
                score_threshold=alias_config.score_threshold,
                strategy=alias_config.retrieval_strategy,
            )
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
            _log_fetch_progress(
                phase="retrieval",
                task="retrieval",
                dataset_name=dataset_name,
                rag_alias=rag_alias,
                lora_alias="none",
                completed=idx + 1,
                total=total_queries,
                every=progress_every,
                unit="queries",
            )
    finally:
        emb_service.close()
        if sparse_encoder is not None:
            sparse_encoder.close()
        if reranker is not None:
            reranker.close()
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
        "retrieval_top_k": alias_config.top_k,
        "score_threshold": alias_config.score_threshold,
        "build_config": build_config.to_payload(),
        "rag_observability": {
            "qdrant_alias": build_config.qdrant_alias,
            "qdrant_collection": build_config.collection_name,
            "rag_manifest_id": build_config.manifest_id,
            "retrieval_capability": build_config.retrieval_capability.value,
            "hit_count": sum(len(result["retrieved_ids"]) for result in query_results),
            "temp_collection": temp_collection,
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

    eval_context = prediction_data.get("eval_context") or {}

    eval_settings = get_settings().eval

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
            metrics = _eval_metrics(eval_settings)
            bert = compute_bertscore(
                predictions,
                references,
                model_name=(eval_context or {}).get(
                    "bert_score_model",
                    metrics.bert_score_model,
                ),
            )
            metric_value = bert.get(metric)
        if metric_value is not None:
            rows.append({**common, "metric_name": metric, "metric_value": metric_value})
    elif metric in _JUDGE_METRICS:
        judge_settings = _resolve_judge_settings(
            eval_settings=eval_settings,
            eval_context=eval_context,
        )
        result = judge_batch(
            metric,
            samples=judge_samples,
            judge_settings=judge_settings,
        )
        rows.append({**common, "metric_name": metric, "metric_value": result[metric]})

    return _finalize_metric_rows(
        rows,
        finished_at=datetime.now(timezone.utc),
        bundle=bundle,
        eval_context=eval_context,
    )


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
            }
        )
    return _finalize_metric_rows(
        rows,
        finished_at=now,
        bundle=bundle,
        eval_context=eval_context,
    )


def _compute_retrieval_metric(
    *,
    metric: str,
    bundle: dict[str, Any],
    dataset_name: str,
    base_model: str,
    kb_name: str | None,
    eval_settings: Any,
    eval_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute a single retrieval metric on pre-fetched query results."""
    query_results = bundle["query_results"]
    retrieval_top_k = bundle.get("retrieval_top_k")
    if not isinstance(retrieval_top_k, int) or retrieval_top_k <= 0:
        raise RuntimeError("Retrieval bundle is missing a valid retrieval_top_k")

    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []

    for qr in query_results:
        retrieved_ids = qr["retrieved_ids"]
        relevance = qr["relevance"]
        relevant_ids = {doc_id for doc_id, rel in relevance.items() if rel > 0}

        recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, k=retrieval_top_k))
        ndcg_scores.append(compute_ndcg_at_k(retrieved_ids, relevance, k=retrieval_top_k))
        mrr_scores.append(compute_mrr_at_k(retrieved_ids, relevant_ids, k=retrieval_top_k))

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
            "qdrant_collection": (
                build_config.get("collection_name") or bundle.get("temp_collection")
            ),
            "qdrant_alias": build_config.get("qdrant_alias"),
            "rag_manifest_id": build_config.get("manifest_id"),
            "embedding_model": build_config.get("embedding_model"),
            "chunking_strategy": build_config.get("chunking_strategy"),
            "chunk_size": build_config.get("chunk_size"),
            "chunk_overlap": build_config.get("chunk_overlap"),
            "retrieval_top_k": retrieval_top_k,
            "score_threshold": bundle.get("score_threshold"),
        }
    )

    rows = []
    if metric == "recall_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"recall_at_{retrieval_top_k}",
                "metric_value": avg_recall,
            }
        )
    elif metric == "ndcg_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"ndcg_at_{retrieval_top_k}",
                "metric_value": avg_ndcg,
            }
        )
    elif metric == "mrr_at_k":
        rows.append(
            {
                **common,
                "metric_name": f"mrr_at_{retrieval_top_k}",
                "metric_value": avg_mrr,
            }
        )
    return _finalize_metric_rows(
        rows,
        finished_at=now,
        bundle=bundle,
        eval_context=eval_context,
    )


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
) -> None:
    """Run evaluation for a single metric.

    Args:
        task: One of chat, summarize, code, retrieval.
        dataset: Dataset name (e.g. hotpotqa, humaneval).
        metric: Metric to compute (e.g. rouge_l, relevance, pass_at_1, recall_at_k).
        kb: Knowledge base name (required for retrieval evals).
        rag_aliases: Comma-separated RAG alias roles.
        lora_aliases: Comma-separated LoRA alias roles.
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
