"""Dataset loading for the evaluation pipeline.

Loads pre-downloaded HuggingFace Arrow datasets from
``assets/datasets/{folder_name}``.  Use ``experiments/misc_ops/prefetch_assets.ipynb``
or ``dvc pull`` to populate the directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root (repo top-level directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Canonical directory for pre-downloaded datasets (HF Arrow format).
DATASETS_DIR = _PROJECT_ROOT / "assets" / "datasets"

# Mapping from eval runner dataset_name -> local folder under assets/datasets/
# and the split to read.  Datasets are saved via ``save_to_disk`` in the
# prefetch notebook and pulled via DVC.
DATASET_LOCAL: dict[str, tuple[str, str]] = {
    "hotpotqa": ("hotpotqa", "validation"),
    "nq": ("natural-questions", "validation"),
    "arxiv_summarization": ("arxiv-summarization", "validation"),
    "humaneval": ("humaneval", "train"),  # HumanEval test split saved as "train"
}


def _extract_text_from_nq_span(item: dict, start_token: int, end_token: int) -> str:
    """Best-effort token-span to text conversion for Natural Questions."""
    if start_token < 0 or end_token <= start_token:
        return ""

    document = item.get("document")
    if not isinstance(document, dict):
        return ""
    tokens_obj = document.get("tokens")
    if not isinstance(tokens_obj, dict):
        return ""
    tokens = tokens_obj.get("token")
    if not isinstance(tokens, list):
        return ""

    safe_start = max(0, start_token)
    safe_end = min(len(tokens), end_token)
    if safe_start >= safe_end:
        return ""

    span_tokens = [str(t).strip() for t in tokens[safe_start:safe_end] if str(t).strip()]
    return " ".join(span_tokens).strip()


def _extract_nq_answer(item: dict) -> str:
    """Extract a short answer from an NQ sample across known schema variants."""
    answer = item.get("answer", "")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    annotations = item.get("annotations", {})
    if not isinstance(annotations, dict):
        return ""
    short_answers_col = annotations.get("short_answers", [])

    # Some NQ exports use list[dict], while others may use list[list[dict]].
    if isinstance(short_answers_col, list):
        flattened: list[dict] = []
        for entry in short_answers_col:
            if isinstance(entry, dict):
                flattened.append(entry)
            elif isinstance(entry, list):
                flattened.extend(x for x in entry if isinstance(x, dict))
    else:
        flattened = []

    for sa in flattened:
        texts = sa.get("text", []) if isinstance(sa, dict) else []
        if isinstance(texts, list):
            for txt in texts:
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
        elif isinstance(texts, str) and texts.strip():
            return texts.strip()

        starts = sa.get("start_token", []) if isinstance(sa, dict) else []
        ends = sa.get("end_token", []) if isinstance(sa, dict) else []
        if not isinstance(starts, list):
            starts = [starts]
        if not isinstance(ends, list):
            ends = [ends]
        for s, e in zip(starts, ends):
            if isinstance(s, int) and isinstance(e, int):
                span_text = _extract_text_from_nq_span(item, s, e)
                if span_text:
                    return span_text

    return ""


def load_dataset_samples(task: str, dataset_name: str) -> list[dict[str, str]]:
    """Load evaluation dataset samples from local Arrow files.

    Datasets must be pre-downloaded to ``assets/datasets/{folder_name}``
    (HuggingFace Arrow format, saved via ``DatasetDict.save_to_disk``).
    Use ``experiments/misc_ops/prefetch_assets.ipynb`` or ``dvc pull`` to
    populate the directory.

    Returns:
        List of sample dicts with at least ``question`` and ``answer`` keys
        (or ``prompt`` and ``test`` for code tasks).
    """
    if dataset_name not in DATASET_LOCAL:
        logger.warning("Unknown dataset: %s", dataset_name)
        return []

    folder_name, split_name = DATASET_LOCAL[dataset_name]
    dataset_path = DATASETS_DIR / folder_name

    if not dataset_path.exists():
        logger.error(
            "Dataset directory not found: %s — run prefetch_assets notebook or "
            "'dvc pull' to download datasets",
            dataset_path,
        )
        return []

    from datasets import load_from_disk

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

        # Build the full corpus once — all documents, not just judged-relevant ones.
        # Filtering to only relevant docs would make the task trivially easy (no
        # hard negatives) and inflate Recall/nDCG scores.
        full_corpus: list[dict] = []
        if corpus_ds is not None:
            full_corpus = [
                {"doc_id": str(doc["_id"]), "text": doc.get("text", "")} for doc in corpus_ds
            ]
            logger.info("Loaded full BEIR corpus: %d docs from %s", len(full_corpus), dataset_path)

        samples: list[dict[str, str]] = []
        for item in queries_ds:
            qid = str(item["_id"])
            if qid not in relevance_map:
                continue  # no judgements for this query
            query_text = item.get("text", "")
            if not query_text:
                continue

            rel = relevance_map[qid]

            samples.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "relevance": rel,
                    # All queries share the same full corpus list object —
                    # no memory duplication; runner.py deduplication handles it.
                    "relevant_docs": full_corpus,
                }
            )
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

    # -----------------------------------------------------------------------
    # MSMARCO retrieval: build the full corpus from ALL rows first so the
    # distractor pool is always the full split.  Mirroring the BEIR approach.
    # -----------------------------------------------------------------------
    if task == "retrieval":
        full_corpus: list[dict] = []
        all_query_samples: list[dict] = []
        seen_corpus_ids: set[str] = set()

        for item in ds:
            passages_data = item.get("passages", {})
            passage_texts = passages_data.get("passage_text", [])
            is_selected = passages_data.get("is_selected", [0] * len(passage_texts))

            if not passage_texts:
                continue

            query_id = str(
                item.get("query_id")
                or item.get("_id")
                or item.get("doc_id")
                or len(all_query_samples)
            )
            query = item.get("query", "") or item.get("question", "")

            passage_docs = [
                {"doc_id": f"{query_id}:{i}", "text": t} for i, t in enumerate(passage_texts) if t
            ]
            if not passage_docs:
                continue

            # Accumulate all passages into the shared corpus.
            for doc in passage_docs:
                if doc["doc_id"] not in seen_corpus_ids:
                    full_corpus.append(doc)
                    seen_corpus_ids.add(doc["doc_id"])

            relevance = {
                f"{query_id}:{i}": 1
                for i, s in enumerate(is_selected)
                if s and i < len(passage_texts) and passage_texts[i]
            }
            if not relevance:
                relevance = {passage_docs[0]["doc_id"]: 1}

            all_query_samples.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "relevance": relevance,
                }
            )

        logger.info(
            "Loaded %d MSMARCO queries with shared corpus of %d passages from %s",
            len(all_query_samples),
            len(full_corpus),
            dataset_path,
        )
        for s in all_query_samples:
            s["relevant_docs"] = full_corpus
        return all_query_samples

    samples: list[dict[str, str]] = []
    for item in ds:
        if task == "code":
            samples.append(
                {
                    "prompt": item.get("prompt", ""),
                    "test": item.get("test", ""),
                    "answer": item.get("canonical_solution", ""),
                    "entry_point": item.get("entry_point", ""),
                    "task_id": item.get("task_id", ""),
                }
            )
        elif task == "summarize":
            samples.append(
                {
                    "question": item.get("article", "")[:2000],
                    "answer": item.get("abstract", ""),
                }
            )
        else:
            # chat / QA
            # NQ stores question as {"text": "...", "tokens": [...]}; unwrap if needed
            question = item.get("question", "")
            if isinstance(question, dict):
                question = question.get("text", "")

            # NQ has no reliable top-level "answer" field across exports.
            answer = _extract_nq_answer(item)

            samples.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )

    logger.info(
        "Loaded %d samples from %s [%s]",
        len(samples),
        dataset_path,
        split_name,
    )
    return samples
