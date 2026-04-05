"""Automatic evaluation metrics: BERTScore, ROUGE-L, Recall@k, nDCG@k.

These metrics are computed locally without any LLM calls.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between a single prediction and reference.

    Uses longest common subsequence (LCS) based scoring.
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1].lower() == ref_tokens[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_name: str,
) -> dict[str, float]:
    """Compute BERTScore (precision, recall, F1) averaged over pairs.

    Requires the ``bert-score`` package.

    Returns:
        Dict with keys ``bertscore_precision``, ``bertscore_recall``,
        ``bertscore_f1``.
    """
    import threading

    import torch
    from bert_score import BERTScorer

    # Suppress HuggingFace's auto-conversion thread exceptions.
    # The transformers library spawns a daemon thread that tries to create
    # a safetensors conversion PR on the Hub, which fails for unauthenticated
    # users or models without proper safetensors support. This thread is
    # non-fatal and we silence its OSError to avoid noisy logs.
    _original_excepthook = threading.excepthook

    def _suppress_safetensors_conversion_error(args):
        if args.thread and "auto_conversion" in args.thread.name:
            # Silently ignore safetensors conversion thread errors
            return
        _original_excepthook(args)

    threading.excepthook = _suppress_safetensors_conversion_error

    scorer = BERTScorer(model_type=model_name, use_fast_tokenizer=False)

    # Workaround: some models (e.g. DeBERTa) report a huge model_max_length
    # (~10^30) that overflows the Rust tokenizer's usize in
    # enable_truncation().  Cap it to the model's actual positional limit.
    max_pos = getattr(scorer._model.config, "max_position_embeddings", None)
    if max_pos and scorer._tokenizer.model_max_length > max_pos:
        scorer._tokenizer.model_max_length = max_pos

    try:
        with torch.no_grad():
            P, R, F1 = scorer.score(predictions, references)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    finally:
        threading.excepthook = _original_excepthook
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at position *k*."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_labels: dict[str, float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at *k*.

    Args:
        retrieved_ids: Ordered list of document IDs returned by retrieval.
        relevance_labels: Gold relevance dict ``{doc_id: relevance_score}``.
        k: Cutoff position.
    """
    # Deduplicate retrieved_ids while preserving rank order so that multiple
    # chunks from the same source document are counted only once.  Without this
    # a single highly-relevant doc split into N chunks would inflate DCG by
    # contributing its relevance score at N positions.
    seen_ids: set[str] = set()
    deduped: list[str] = [x for x in retrieved_ids if not (x in seen_ids or seen_ids.add(x))]
    rels = [relevance_labels.get(doc_id, 0.0) for doc_id in deduped[:k]]
    dcg = _dcg(rels, k)
    ideal_rels = sorted(relevance_labels.values(), reverse=True)[:k]
    idcg = _dcg(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall at *k*: fraction of relevant docs found in top-k results.

    Args:
        retrieved_ids: Ordered list of returned document IDs.
        relevant_ids: Set of gold relevant document IDs.
        k: Cutoff position.
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / len(relevant_ids)
