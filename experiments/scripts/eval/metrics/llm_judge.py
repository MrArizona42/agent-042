"""LLM-as-Judge evaluation via Google Gemini API.

Uses Gemini 2.0 Flash through Google AI Studio for structured scoring.
Supports Relevance, Correctness, Faithfulness, Coverage, and Groundedness.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Gemini free tier: 15 RPM
_RPM_DELAY = 4.5  # seconds between calls to stay under 15 RPM

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = """\
You are an expert evaluator. Rate the relevance of the answer to the question.
Score from 1 (completely irrelevant) to 5 (perfectly relevant).

Question: {question}
Answer: {answer}
Reference: {reference}

Respond with JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

_CORRECTNESS_PROMPT = """\
You are an expert evaluator. Rate the factual correctness of the answer compared to the reference.
Score from 1 (completely wrong) to 5 (fully correct).

Question: {question}
Answer: {answer}
Reference: {reference}

Respond with JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

_FAITHFULNESS_PROMPT = """\
You are an expert evaluator. Rate how faithful the summary is to the source document.
Score from 1 (contains fabricated information) to 5 (fully faithful to source).

Source: {reference}
Summary: {answer}

Respond with JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

_COVERAGE_PROMPT = """\
You are an expert evaluator. Rate how well the summary covers the key points of the source.
Score from 1 (misses all key points) to 5 (covers all key points).

Source: {reference}
Summary: {answer}

Respond with JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

_GROUNDEDNESS_PROMPT = """\
You are an expert evaluator. Rate whether the answer is grounded in (supported by) the provided context.
Score from 1 (not grounded at all) to 5 (fully grounded in context).

Context: {context}
Question: {question}
Answer: {answer}

Respond with JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

_METRIC_PROMPTS = {
    "relevance": _RELEVANCE_PROMPT,
    "correctness": _CORRECTNESS_PROMPT,
    "faithfulness": _FAITHFULNESS_PROMPT,
    "coverage": _COVERAGE_PROMPT,
    "groundedness": _GROUNDEDNESS_PROMPT,
}


# ---------------------------------------------------------------------------
# Gemini API client
# ---------------------------------------------------------------------------


def _call_gemini(
    prompt: str,
    *,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """Call Gemini API and parse JSON response.

    Returns:
        Parsed JSON dict from the model response.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    resp = httpx.post(url, json=payload, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_single(
    metric: str,
    *,
    question: str = "",
    answer: str,
    reference: str = "",
    context: str = "",
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """Score a single (question, answer) pair on the given metric.

    Args:
        metric: One of ``relevance``, ``correctness``, ``faithfulness``,
            ``coverage``, ``groundedness``.
        question: The original question (not used for faithfulness/coverage).
        answer: Model-generated answer or summary.
        reference: Gold reference answer or source document.
        context: Retrieved RAG context (only for groundedness).
        api_key: Google AI Studio API key.
        model: Gemini model name.

    Returns:
        ``{"score": int, "reason": str}``
    """
    template = _METRIC_PROMPTS.get(metric)
    if template is None:
        raise ValueError(f"Unknown judge metric: {metric}")

    prompt = template.format(
        question=question,
        answer=answer,
        reference=reference,
        context=context,
    )

    try:
        result = _call_gemini(prompt, api_key=api_key, model=model)
        return {"score": int(result.get("score", 0)), "reason": result.get("reason", "")}
    except Exception as e:
        logger.error("Gemini judge call failed: %s", e)
        return {"score": 0, "reason": f"error: {e}"}


def judge_batch(
    metric: str,
    *,
    samples: list[dict[str, str]],
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict[str, float]:
    """Score a batch of samples and return the average.

    Each sample dict should have keys matching the ``judge_single`` kwargs:
    ``question``, ``answer``, ``reference``, ``context``.

    Returns:
        ``{"<metric>": avg_score}``
    """
    scores: list[int] = []
    for i, sample in enumerate(samples):
        result = judge_single(
            metric,
            question=sample.get("question", ""),
            answer=sample.get("answer", ""),
            reference=sample.get("reference", ""),
            context=sample.get("context", ""),
            api_key=api_key,
            model=model,
        )
        scores.append(result["score"])
        if (i + 1) < len(samples):
            time.sleep(_RPM_DELAY)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {metric: avg}
