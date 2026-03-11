"""Hydra-driven evaluation runner — entry point.

Usage examples::

    # Regression eval for chat task (defaults)
    python experiments/scripts/eval/run_eval.py

    # Full eval for summarization with specific adapter
    python experiments/scripts/eval/run_eval.py \\
        eval/task=summarize tier=full dataset.max_examples=null \\
        adapter.name=lora-summarization adapter.version=3

    # Code eval (no judge, no RAG)
    python experiments/scripts/eval/run_eval.py \\
        eval/task=code eval/rag=no_rag

Execution flow (Phase 1 skeleton — automatic metrics only):

1. Hydra resolves config (composition + CLI overrides).
2. Build ``EvalConfig`` from resolved OmegaConf.
3. Create ``eval_runs`` row in Postgres (status='running').
4. Load dataset (subsample to ``max_examples`` with ``seed`` if set).
5. For each example:
   a. Build prompt (system + optional context + user input).
   b. Call vLLM for generation.
   c. Compute automatic metrics (ROUGE-L, BERTScore).
   d. Write ``eval_examples`` row.
6. Compute aggregate metrics → write ``eval_metrics`` rows.
7. Update ``eval_runs``: ``status='completed'``, ``finished_at=now()``.
8. Print summary to stdout.
"""

from __future__ import annotations

import logging
import re
import statistics
import uuid
from datetime import datetime, timezone

import hydra
from eval.config import (
    AdapterConfig,
    DatasetConfig,
    EvalConfig,
    GenerationConfig,
    JudgeConfig,
    MetricsConfig,
    RAGConfig,
)
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_url(url: str) -> str:
    """Convert an async DB URL to a synchronous one.

    The gateway stores ``postgresql+asyncpg://...`` but the eval runner
    uses a sync SQLAlchemy engine, so we need ``postgresql+psycopg2://...``.
    """
    return re.sub(r"postgresql\+asyncpg", "postgresql+psycopg2", url)


def build_eval_config(cfg: DictConfig) -> EvalConfig:
    """Build an :class:`EvalConfig` from the resolved Hydra config."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)

    adapter_raw = raw.get("adapter", {}) or {}
    rag_raw = raw.get("rag", {}) or {}
    judge_raw = raw.get("judge", {}) or {}
    gen_raw = raw.get("generation", {}) or {}
    dataset_raw = raw.get("dataset", {}) or {}
    metrics_raw = raw.get("metrics", {}) or {}

    model_raw = raw.get("model", {}) or {}

    return EvalConfig(
        base_model=model_raw.get("base_model", ""),
        vllm_base_url=model_raw.get("vllm_base_url", "http://localhost:8000"),
        adapter=AdapterConfig(**adapter_raw),
        rag=RAGConfig(
            **{k: v for k, v in rag_raw.items() if k in RAGConfig.model_fields}
        ),
        task=raw.get("task", "chat"),
        tier=raw.get("tier", "regression"),
        dataset=DatasetConfig(
            **{k: v for k, v in dataset_raw.items() if k in DatasetConfig.model_fields}
        ),
        task_metrics=raw.get("task_metrics", []),
        judge=JudgeConfig(
            **{k: v for k, v in judge_raw.items() if k in JudgeConfig.model_fields}
        ),
        generation=GenerationConfig(**gen_raw),
        metrics=MetricsConfig(
            **{k: v for k, v in metrics_raw.items() if k in MetricsConfig.model_fields}
        ),
        db_url=raw.get("db_url", ""),
    )


# ---------------------------------------------------------------------------
# Database helpers (sync SQLAlchemy)
# ---------------------------------------------------------------------------


def _get_sync_engine(db_url: str):
    """Create a sync SQLAlchemy engine for writing eval results."""
    from sqlalchemy import create_engine

    return create_engine(_make_sync_url(db_url), echo=False, pool_size=2)


def _create_eval_run(session, eval_config: EvalConfig) -> uuid.UUID:
    """Insert a new ``eval_runs`` row and return its id."""
    from shared.db.models import EvalRun

    run = EvalRun(
        status="running",
        tier=eval_config.tier,
        task=eval_config.task,
        config=eval_config.model_dump(mode="json"),
        base_model=eval_config.base_model,
        adapter_name=eval_config.adapter.name,
        adapter_version=eval_config.adapter.version,
        dataset_name=eval_config.dataset.name,
        dataset_split=eval_config.dataset.split,
        knowledge_base=eval_config.rag.knowledge_base if eval_config.rag.enabled else None,
    )
    session.add(run)
    session.flush()
    return run.id


def _save_example(session, run_id: uuid.UUID, idx: int, example: dict, scores: dict):
    """Insert a single ``eval_examples`` row."""
    from shared.db.models import EvalExample

    row = EvalExample(
        run_id=run_id,
        example_index=idx,
        input_text=example["input"],
        reference_text=example.get("reference"),
        generated_text=example.get("generated", ""),
        rouge_l=scores.get("rouge_l"),
        bert_score=scores.get("bert_score"),
    )
    session.add(row)


def _save_aggregate_metrics(session, run_id: uuid.UUID, per_example_scores: list[dict]):
    """Compute and save aggregate metrics from per-example scores."""
    from shared.db.models import EvalMetric

    aggregated: dict[str, list[float]] = {}
    for scores in per_example_scores:
        for key, val in scores.items():
            if val is not None:
                aggregated.setdefault(key, []).append(val)

    for metric_name, values in aggregated.items():
        if values:
            session.add(
                EvalMetric(
                    run_id=run_id,
                    metric_name=f"{metric_name}_mean",
                    value=statistics.mean(values),
                )
            )


def _finish_run(session, run_id: uuid.UUID, status: str, error: str | None = None):
    """Update the eval run status and finished_at timestamp."""
    from shared.db.models import EvalRun

    run = session.get(EvalRun, run_id)
    if run:
        run.status = status
        run.finished_at = datetime.now(timezone.utc)
        if error:
            run.error_message = error


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------


def _call_vllm(
    base_url: str,
    prompt: str,
    model: str,
    generation: GenerationConfig,
) -> str:
    """Send a chat completion request to the vLLM OpenAI-compatible API."""
    import httpx

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(eval_config: EvalConfig) -> None:
    """Execute the evaluation pipeline.

    Phase 1 supports automatic metrics (ROUGE-L, BERTScore) for the
    chat and summarization tasks.  Judge scoring and code sandbox
    execution will be added in later phases.
    """
    from eval.metrics import compute_auto_metrics
    from sqlalchemy.orm import Session

    logger.info(
        "Starting eval — task=%s, tier=%s",
        eval_config.task,
        eval_config.tier,
    )
    logger.info("Config: %s", eval_config.model_dump(mode="json"))

    if not eval_config.db_url:
        raise RuntimeError(
            "db_url is not set.  Export GATEWAY_AGENT042_DB_URL "
            "or pass db_url=postgresql://... on the CLI."
        )

    engine = _get_sync_engine(eval_config.db_url)

    # Determine model name for vLLM requests
    model_name = eval_config.adapter.name or eval_config.base_model

    # ------------------------------------------------------------------
    # Dataset loading (placeholder — Phase 1 loads a trivial list)
    # Real implementation will load from HuggingFace datasets or disk.
    # ------------------------------------------------------------------
    dataset = _load_dataset(eval_config)

    per_example_scores: list[dict] = []
    run_id: uuid.UUID | None = None

    with Session(engine) as session:
        try:
            run_id = _create_eval_run(session, eval_config)
            session.commit()
            logger.info("Created eval run %s", run_id)

            for idx, example in enumerate(dataset):
                # Generate
                generated = _call_vllm(
                    eval_config.vllm_base_url,
                    example["input"],
                    model_name,
                    eval_config.generation,
                )
                example["generated"] = generated

                # Compute automatic metrics
                scores_obj = compute_auto_metrics(
                    reference=example.get("reference", ""),
                    generated=generated,
                    task_metrics=eval_config.task_metrics,
                    bert_score_model=eval_config.metrics.bert_score_model,
                )
                scores = {
                    "rouge_l": scores_obj.rouge_l,
                    "bert_score": scores_obj.bert_score,
                }
                per_example_scores.append(scores)

                _save_example(session, run_id, idx, example, scores)
                if (idx + 1) % 10 == 0:
                    session.commit()
                    logger.info("Processed %d / %d examples", idx + 1, len(dataset))

            session.commit()

            # Aggregate metrics
            _save_aggregate_metrics(session, run_id, per_example_scores)
            _finish_run(session, run_id, status="completed")
            session.commit()

            logger.info("Eval run %s completed successfully", run_id)
            _print_summary(run_id, per_example_scores)

        except Exception:
            logger.exception("Eval run failed")
            if run_id:
                _finish_run(session, run_id, status="failed", error="See logs for details")
                session.commit()
            raise


def _load_dataset(eval_config: EvalConfig) -> list[dict]:
    """Load and subsample the evaluation dataset.

    Returns a list of dicts with at least ``input`` and optionally
    ``reference`` keys.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    dataset_path = f"assets/datasets/{eval_config.dataset.name}"
    logger.info("Loading dataset from %s (split=%s)", dataset_path, eval_config.dataset.split)

    ds = load_from_disk(dataset_path)

    # Handle DatasetDict (multiple splits)
    if hasattr(ds, "keys"):
        if eval_config.dataset.split in ds:
            ds = ds[eval_config.dataset.split]
        else:
            available = list(ds.keys())
            raise ValueError(
                f"Split '{eval_config.dataset.split}' not found. "
                f"Available splits: {available}"
            )

    # Subsample if max_examples is set
    if eval_config.dataset.max_examples is not None:
        ds = ds.shuffle(seed=eval_config.dataset.seed)
        ds = ds.select(range(min(eval_config.dataset.max_examples, len(ds))))

    # Map dataset columns to standard keys
    examples = []
    for row in ds:
        example = _map_row_to_example(row, eval_config.task)
        examples.append(example)

    logger.info("Loaded %d examples", len(examples))
    return examples


def _map_row_to_example(row: dict, task: str) -> dict:
    """Map dataset-specific column names to standard ``input``/``reference``."""
    if task == "chat":
        # HotpotQA has "question" and "answer"
        return {
            "input": row.get("question", row.get("input", "")),
            "reference": row.get("answer", row.get("reference", "")),
        }
    elif task == "summarize":
        # ArXiv-summarization has "article" and "abstract"
        return {
            "input": row.get("article", row.get("input", "")),
            "reference": row.get("abstract", row.get("reference", "")),
        }
    else:
        return {
            "input": row.get("input", row.get("prompt", "")),
            "reference": row.get("reference", row.get("canonical_solution", "")),
        }


def _print_summary(run_id: uuid.UUID, per_example_scores: list[dict]) -> None:
    """Print a human-readable summary of the eval results."""
    print(f"\n{'=' * 60}")
    print(f"  Eval Run: {run_id}")
    print(f"{'=' * 60}")

    aggregated: dict[str, list[float]] = {}
    for scores in per_example_scores:
        for key, val in scores.items():
            if val is not None:
                aggregated.setdefault(key, []).append(val)

    for metric_name, values in sorted(aggregated.items()):
        mean = statistics.mean(values)
        print(f"  {metric_name:20s}: {mean:.4f}  (n={len(values)})")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../conf", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_config = build_eval_config(cfg)
    run_evaluation(eval_config)


if __name__ == "__main__":
    main()
