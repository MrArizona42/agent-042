"""LLM-as-Judge evaluation via a generic OpenAI-compatible chat backend.

Supports Relevance, Correctness, Faithfulness, Coverage, and Groundedness.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app_config.runtime import JudgeSettings

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation model. "
    "Return exactly one JSON object with keys 'score' and 'reason'."
)

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
You are an expert evaluator. Rate whether the answer is grounded in \
(supported by) the provided context.
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
# Judge transport helpers
# ---------------------------------------------------------------------------


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _coerce_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments)
    if isinstance(content, dict):
        return json.dumps(content)
    raise TypeError(f"Unsupported judge response content type: {type(content)!r}")


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            break
        else:
            raise ValueError(f"Judge response did not contain JSON: {text!r}") from None

    if not isinstance(parsed, dict):
        raise ValueError(f"Judge response JSON must be an object, got: {type(parsed)!r}")
    return parsed


def _call_openai_compatible(prompt: str, *, judge_settings: JudgeSettings) -> dict[str, Any]:
    """Call an OpenAI-compatible chat backend and parse a JSON judge response."""
    payload = {
        "model": judge_settings.model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if judge_settings.api_key:
        headers["Authorization"] = f"Bearer {judge_settings.api_key}"

    resp = httpx.post(
        _chat_completions_url(judge_settings.base_url),
        json=payload,
        headers=headers,
        timeout=judge_settings.timeout,
    )
    resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Judge backend returned no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Judge backend returned an invalid message payload")

    text = _coerce_response_text(message.get("content", ""))
    return _extract_json_object(text)


def _call_judge_model(prompt: str, *, judge_settings: JudgeSettings) -> dict[str, Any]:
    if judge_settings.backend in {"local_vllm", "openai_compatible"}:
        return _call_openai_compatible(prompt, judge_settings=judge_settings)
    raise ValueError(f"Unsupported judge backend: {judge_settings.backend}")


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
    judge_settings: JudgeSettings,
) -> dict[str, Any]:
    """Score a single (question, answer) pair on the given metric.

    Args:
        metric: One of ``relevance``, ``correctness``, ``faithfulness``,
            ``coverage``, ``groundedness``.
        question: The original question (not used for faithfulness/coverage).
        answer: Model-generated answer or summary.
        reference: Gold reference answer or source document.
        context: Retrieved RAG context (only for groundedness).
        judge_settings: Resolved LLM-as-judge transport and model config.

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
        result = _call_judge_model(prompt, judge_settings=judge_settings)
        return {"score": int(result.get("score", 0)), "reason": result.get("reason", "")}
    except Exception as e:
        logger.error("Judge model call failed: %s", e)
        return {"score": 0, "reason": f"error: {e}"}


def judge_batch(
    metric: str,
    *,
    samples: list[dict[str, str]],
    judge_settings: JudgeSettings,
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
            judge_settings=judge_settings,
        )
        scores.append(result["score"])
        if (i + 1) < len(samples) and judge_settings.request_delay_seconds > 0:
            time.sleep(judge_settings.request_delay_seconds)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {metric: avg}
