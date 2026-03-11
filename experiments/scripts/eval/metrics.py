"""Automatic metrics computation — ROUGE-L and BERTScore.

These are the non-judge metrics used across chat and summarization tasks.
The BERTScore model is pinned via ``metrics.bert_score_model`` in the
Hydra config so that scores remain comparable across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AutoMetricScores:
    """Scores computed for a single example."""

    rouge_l: float | None = None
    bert_score: float | None = None


def compute_rouge_l(reference: str, generated: str) -> float:
    """Compute ROUGE-L F1 between *reference* and *generated*.

    Lazy-imports ``rouge_score`` so that the module can be imported
    without installing the heavy dependency (useful for tests that
    only exercise config / model code).
    """
    from rouge_score import rouge_scorer  # type: ignore[import-untyped]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores["rougeL"].fmeasure


def compute_bert_score(
    references: list[str],
    predictions: list[str],
    model_type: str = "roberta-large",
) -> list[float]:
    """Compute BERTScore F1 for a batch of (reference, prediction) pairs.

    Parameters
    ----------
    references:
        Gold texts.
    predictions:
        Model-generated texts.
    model_type:
        HuggingFace model name used by the ``bert_score`` library.
        Must be fixed across runs for comparability.

    Returns
    -------
    List of F1 floats, one per example.
    """
    import bert_score  # type: ignore[import-untyped]

    logger.info(
        "Computing BERTScore with model=%s for %d examples",
        model_type,
        len(references),
    )
    _P, _R, F1 = bert_score.score(
        predictions,
        references,
        model_type=model_type,
        verbose=False,
    )
    return F1.tolist()


def compute_auto_metrics(
    reference: str,
    generated: str,
    task_metrics: list[str],
    bert_score_model: str = "roberta-large",
) -> AutoMetricScores:
    """Compute all applicable automatic metrics for a single example.

    Only metrics listed in *task_metrics* are computed.  BERTScore is
    computed per-example here for simplicity; for large batches the
    caller should use :func:`compute_bert_score` directly.
    """
    scores = AutoMetricScores()

    if "rouge_l" in task_metrics and reference:
        scores.rouge_l = compute_rouge_l(reference, generated)

    if "bert_score" in task_metrics and reference:
        f1_list = compute_bert_score([reference], [generated], model_type=bert_score_model)
        scores.bert_score = f1_list[0] if f1_list else None

    return scores
