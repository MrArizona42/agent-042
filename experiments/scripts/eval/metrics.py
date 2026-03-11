"""Automatic evaluation metrics: ROUGE-L and BERTScore.

Phase 1 implements these two automatic metrics.  LLM-as-judge scoring
is deferred to Phase 2.
"""

from __future__ import annotations

import logging

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F-measure between *reference* and *hypothesis*.

    Returns a float in [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

_bert_scorer = None


def _get_bert_scorer(model_type: str = "roberta-large"):
    """Lazily create and cache a BERTScorer instance.

    The scorer downloads the model on first use (~1.4 GB for roberta-large).
    """
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer

        logger.info("Loading BERTScorer with model_type=%s", model_type)
        _bert_scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True)
    return _bert_scorer


def compute_bert_score(
    references: list[str],
    hypotheses: list[str],
    model_type: str = "roberta-large",
) -> list[float]:
    """Compute BERTScore F1 for parallel lists of *references* and *hypotheses*.

    Returns a list of float F1 scores, one per pair.
    """
    scorer = _get_bert_scorer(model_type)
    _P, _R, F1 = scorer.score(hypotheses, references)
    return F1.tolist()
