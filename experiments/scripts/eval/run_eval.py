"""Hydra-driven evaluation runner — Phase 1 skeleton.

Usage
-----
    python experiments/scripts/eval/run_eval.py                         # defaults
    python experiments/scripts/eval/run_eval.py eval/task=summarize     # override task
    python experiments/scripts/eval/run_eval.py tier=full dataset.max_examples=null

The runner:
1. Resolves the full EvalConfig from Hydra.
2. Creates an ``eval_runs`` row in PostgreSQL (status='running').
3. Iterates over dataset examples, calling vLLM for generation and
   computing automatic metrics.
4. Writes per-example rows and aggregate metrics to PostgreSQL.
5. Updates the run to ``status='completed'``.
"""

from __future__ import annotations

import logging
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure the project ``src/`` and ``experiments/scripts/`` are importable
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "experiments" / "scripts"))

from eval.config import EvalConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_eval_config(cfg: DictConfig) -> EvalConfig:
    """Construct an :class:`EvalConfig` from the resolved Hydra config."""
    rag = cfg.get("rag", {})
    judge = cfg.get("judge", {})
    metrics = cfg.get("metrics", {})

    return EvalConfig(
        # Model
        base_model=cfg.model.base_model,
        vllm_base_url=cfg.model.vllm_base_url,
        # Adapter
        adapter_name=cfg.adapter.get("name"),
        adapter_version=cfg.adapter.get("version"),
        # RAG
        rag_enabled=rag.get("enabled", True),
        knowledge_base=rag.get("knowledge_base"),
        embedding_model=rag.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        chunking_strategy=rag.get("chunking_strategy", "fixed_token"),
        chunk_size=rag.get("chunk_size", 512),
        chunk_overlap=rag.get("chunk_overlap", 50),
        retrieval_top_k=rag.get("retrieval_top_k", 5),
        score_threshold=rag.get("score_threshold", 0.35),
        reranking_strategy=rag.get("reranking_strategy"),
        # Eval
        dataset_name=cfg.dataset.name,
        dataset_split=cfg.dataset.split,
        task=cfg.task,
        tier=cfg.tier,
        max_examples=cfg.dataset.get("max_examples"),
        seed=cfg.dataset.get("seed", 42),
        # Judge
        judge_enabled=judge.get("enabled", False),
        judge_model=judge.get("model") if judge.get("enabled", False) else None,
        bert_score_model=metrics.get("bert_score_model", "roberta-large"),
        # Generation
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        max_tokens=cfg.generation.max_tokens,
    )


# ---------------------------------------------------------------------------
# Database helpers (sync wrappers around async SQLAlchemy)
# ---------------------------------------------------------------------------


def _get_sync_db_url(async_url: str) -> str:
    """Convert an ``asyncpg`` URL to a synchronous ``psycopg2`` URL.

    The eval runner operates synchronously (Hydra + simple loop) so we use
    the sync SQLAlchemy engine.
    """
    return async_url.replace("postgresql+asyncpg://", "postgresql://")


def _create_engine(db_url: str):
    from sqlalchemy import create_engine

    return create_engine(db_url, echo=False)


def _create_tables(engine):
    """Ensure all tables exist (safe ``CREATE TABLE IF NOT EXISTS``)."""
    from shared.db.models import Base

    Base.metadata.create_all(engine)


def _insert_eval_run(engine, eval_config: EvalConfig) -> uuid.UUID:
    """Insert a new ``eval_runs`` row and return its UUID."""
    from sqlalchemy.orm import Session

    from shared.db.models import EvalRun

    run_id = uuid.uuid4()
    with Session(engine) as session:
        run = EvalRun(
            id=run_id,
            status="running",
            tier=eval_config.tier,
            task=eval_config.task,
            config=eval_config.model_dump(),
            base_model=eval_config.base_model,
            adapter_name=eval_config.adapter_name,
            adapter_version=eval_config.adapter_version,
            dataset_name=eval_config.dataset_name,
            dataset_split=eval_config.dataset_split,
            knowledge_base=eval_config.knowledge_base,
        )
        session.add(run)
        session.commit()
    logger.info("Created eval run %s", run_id)
    return run_id


def _finish_eval_run(engine, run_id: uuid.UUID, *, status: str, error: str | None = None):
    """Update the eval run status and ``finished_at`` timestamp."""
    from sqlalchemy.orm import Session

    from shared.db.models import EvalRun

    with Session(engine) as session:
        run = session.get(EvalRun, run_id)
        if run is None:
            logger.error("Run %s not found", run_id)
            return
        run.status = status
        run.finished_at = datetime.now(timezone.utc)
        if error:
            run.error_message = error
        session.commit()


def _insert_example(engine, run_id: uuid.UUID, example: dict):
    """Insert one ``eval_examples`` row."""
    from sqlalchemy.orm import Session

    from shared.db.models import EvalExample

    with Session(engine) as session:
        ex = EvalExample(run_id=run_id, **example)
        session.add(ex)
        session.commit()


def _insert_metrics(engine, run_id: uuid.UUID, metrics: dict[str, float]):
    """Insert aggregate ``eval_metrics`` rows."""
    from sqlalchemy.orm import Session

    from shared.db.models import EvalMetric

    with Session(engine) as session:
        for name, value in metrics.items():
            session.add(EvalMetric(run_id=run_id, metric_name=name, value=value))
        session.commit()


# ---------------------------------------------------------------------------
# vLLM client (minimal)
# ---------------------------------------------------------------------------


def _call_vllm(
    base_url: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Send a chat-completion request to vLLM and return the assistant text."""
    import httpx

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(cfg: DictConfig, eval_config: EvalConfig) -> None:
    """Execute the full evaluation pipeline.

    Phase 1 supports:
    - Chat task with automatic metrics (ROUGE-L).
    - Summarize task with automatic metrics (ROUGE-L).
    - BERTScore is supported but requires ``bert-score`` installed.
    """
    from eval.metrics import compute_rouge_l

    db_url = OmegaConf.to_container(cfg, resolve=True).get("db_url", "")
    if not db_url:
        logger.error("db_url is not set — cannot persist results")
        raise RuntimeError(
            "db_url must be set. Export GATEWAY_AGENT042_DB_URL or pass db_url=... via CLI."
        )

    sync_url = _get_sync_db_url(db_url)
    engine = _create_engine(sync_url)
    _create_tables(engine)

    run_id = _insert_eval_run(engine, eval_config)

    # ── Load dataset ──
    # Phase 1 expects a HuggingFace dataset on disk or loadable by name.
    try:
        from datasets import load_dataset, load_from_disk

        dataset_path = Path(f"assets/datasets/{eval_config.dataset_name}")
        if dataset_path.exists():
            ds = load_from_disk(str(dataset_path))
            if isinstance(ds, dict) or hasattr(ds, "keys"):
                ds = ds[eval_config.dataset_split]
        else:
            ds = load_dataset(eval_config.dataset_name, split=eval_config.dataset_split)

        # Subsample if requested
        if eval_config.max_examples and len(ds) > eval_config.max_examples:
            ds = ds.shuffle(seed=eval_config.seed).select(range(eval_config.max_examples))

        logger.info(
            "Loaded %d examples from %s/%s",
            len(ds), eval_config.dataset_name, eval_config.dataset_split,
        )
    except Exception as exc:
        _finish_eval_run(engine, run_id, status="failed", error=str(exc))
        raise

    # Determine model name for vLLM requests
    model_name = eval_config.adapter_name or eval_config.base_model

    # ── Per-example loop ──
    rouge_scores: list[float] = []
    try:
        for idx, row in enumerate(ds):
            # Build prompt (task-dependent)
            if eval_config.task == "chat":
                input_text = row.get("question", row.get("input", ""))
                reference_text = row.get("answer", row.get("reference", ""))
            elif eval_config.task == "summarize":
                input_text = row.get("article", row.get("input", ""))
                reference_text = row.get("abstract", row.get("reference", ""))
            else:
                input_text = row.get("prompt", row.get("input", ""))
                reference_text = row.get("canonical_solution", row.get("reference", ""))

            # Call vLLM
            generated_text = _call_vllm(
                eval_config.vllm_base_url,
                model_name,
                input_text,
                temperature=eval_config.temperature,
                top_p=eval_config.top_p,
                max_tokens=eval_config.max_tokens,
            )

            # Compute automatic metrics
            rl = compute_rouge_l(reference_text, generated_text) if reference_text else None
            if rl is not None:
                rouge_scores.append(rl)

            # Persist example
            _insert_example(engine, run_id, {
                "example_index": idx,
                "input_text": input_text,
                "reference_text": reference_text,
                "generated_text": generated_text,
                "rouge_l": rl,
            })

            if (idx + 1) % 10 == 0:
                logger.info("Processed %d / %d examples", idx + 1, len(ds))

    except Exception as exc:
        _finish_eval_run(engine, run_id, status="failed", error=str(exc))
        raise

    # ── Aggregate metrics ──
    agg: dict[str, float] = {}
    if rouge_scores:
        agg["rouge_l_mean"] = statistics.mean(rouge_scores)
        agg["rouge_l_median"] = statistics.median(rouge_scores)

    _insert_metrics(engine, run_id, agg)
    _finish_eval_run(engine, run_id, status="completed")

    # ── Summary ──
    logger.info("Eval run %s completed", run_id)
    for name, value in agg.items():
        logger.info("  %s = %.4f", name, value)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../conf", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for the evaluation runner."""
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    eval_config = build_eval_config(cfg)
    logger.info("EvalConfig: %s", eval_config.model_dump_json(indent=2))
    run_evaluation(cfg, eval_config)


if __name__ == "__main__":
    main()
