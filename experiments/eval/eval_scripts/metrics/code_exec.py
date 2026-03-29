"""Sandboxed code execution for HumanEval evaluation.

Executes generated Python code in the ``code-sandbox`` sidecar container,
which is always running alongside the airflow-worker and provides Docker-level
isolation (read-only filesystem, tmpfs /tmp, no internet access, CPU/memory
limits) without requiring kernel user-namespace support on the host.

The URL of the sandbox is read from the ``CODE_SANDBOX_URL`` environment
variable (default: ``http://code-sandbox:8200``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Resolved at import time so tests can override via environment.
_SANDBOX_URL: str = os.environ.get("CODE_SANDBOX_URL", "http://code-sandbox:8200").rstrip("/")

# Matches the first ```python / ```py / ``` fenced block in an LLM response.
_FENCE_RE = re.compile(r"```(?:python3?|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Line-start tokens that strongly indicate Python code rather than prose.
_CODE_START_TOKENS = (
    "return ",
    "if ",
    "for ",
    "while ",
    "try:",
    "raise ",
    "yield ",
    "with ",
    "assert ",
    "pass",
    "#",
    "    ",
    "\t",
)


def extract_code_from_response(response: str, prompt: str) -> str:
    """Extract the function body from a raw LLM response for HumanEval evaluation.

    LLMs often wrap their answer in markdown fences, re-declare the function
    signature, or add explanatory prose.  This function normalises all common
    patterns so that only the function *body* (the fragment that follows the
    signature already present in *prompt*) is returned.

    Priority order:

    1. **Fenced block** — the first ```python / ``` block is extracted and
       used as the candidate.
    2. **Re-declared signature** — if the candidate contains a ``def`` line
       that matches the function name in *prompt*, the signature line and any
       immediately following docstring are stripped (they are already provided
       by *prompt*).
    3. **Prose prefix** — if the candidate starts with a non-code sentence
       (no leading indent, not a recognised Python keyword), the extractor
       skips forward to the first indented / keyword line.
    4. **Pass-through** — if none of the above applies the candidate is
       returned as-is (the model likely produced a clean indented body).
    """
    # ── 1. Prefer the first fenced code block ─────────────────────────────
    fence_match = _FENCE_RE.search(response)
    candidate: str = fence_match.group(1).rstrip() if fence_match else response.strip()

    # ── 2. Strip a re-declared function signature ──────────────────────────
    # The prompt already contains the `def` line and docstring; including them
    # again produces a duplicate definition that shadows the intended one.
    func_name_match = re.search(r"^def\s+(\w+)", prompt, re.MULTILINE)
    if func_name_match:
        func_name = re.escape(func_name_match.group(1))
        # Match `def name(...)` with an optional return-type annotation.
        redecl = re.search(
            rf"(?m)^def\s+{func_name}\s*\([^)]*\)\s*(?:->\s*[\w\[\], |.]+\s*)?:",
            candidate,
        )
        if redecl:
            after_def = candidate[redecl.end() :]
            stripped = after_def.lstrip("\n")
            # Drop an optional docstring (single- or triple-quoted).
            doc_match = re.match(
                r'\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
                stripped,
            )
            candidate = stripped[doc_match.end() :] if doc_match else stripped

    # ── 3. Skip a prose prefix (non-indented, non-code leading text) ───────
    lines = candidate.lstrip("\n").split("\n")
    first_nonempty = next((ln for ln in lines if ln.strip()), "")
    if first_nonempty and not first_nonempty.startswith(_CODE_START_TOKENS):
        for i, line in enumerate(lines):
            if line.startswith(("    ", "\t")) or (
                line.strip()
                and any(
                    line.lstrip().startswith(t)
                    for t in (
                        "return ",
                        "if ",
                        "for ",
                        "while ",
                        "try:",
                        "raise ",
                        "yield ",
                        "with ",
                        "assert ",
                        "pass",
                        "#",
                    )
                )
            ):
                candidate = "\n".join(lines[i:])
                break

    return candidate.strip("\n")


def _run_in_sandbox(
    code: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    """Send *code* to the code-sandbox sidecar and return the execution result.

    Raises ``RuntimeError`` if the sidecar is unreachable or returns an
    unexpected HTTP status, so the Airflow task (and therefore the DAG) is
    marked as failed rather than silently recording a zero pass@1.

    Args:
        code: Python source to execute.
        timeout: Wall-clock timeout in seconds passed to the sandbox.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    url = f"{_SANDBOX_URL}/execute"
    payload = json.dumps({"code": code, "timeout": timeout}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Add a small margin so the HTTP connection itself doesn't outlive the
    # sandbox's own timeout.
    http_timeout = timeout + 10
    try:
        with urllib.request.urlopen(req, timeout=http_timeout) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"code-sandbox unreachable at {url}: {exc.reason}") from exc

    exit_code: int = result["exit_code"]
    return {
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def evaluate_humaneval_sample(
    prompt: str,
    generated_code: str,
    test_code: str,
    *,
    timeout: int,
    mem_limit: str,  # noqa: ARG001 – kept for call-site compatibility
    cpus: float,  # noqa: ARG001 – kept for call-site compatibility
) -> dict[str, Any]:
    """Evaluate a single HumanEval sample.

    Combines the function prompt, generated completion, and test assertions
    into one script, then executes it in the code-sandbox sidecar container.

    Args:
        prompt: The function signature / docstring from HumanEval.
        generated_code: Model-generated function body.
        test_code: Assertion-based test code from the dataset.
        timeout: Execution timeout in seconds.
        mem_limit: Unused; kept for call-site compatibility.
        cpus: Unused; kept for call-site compatibility.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    body = extract_code_from_response(generated_code, prompt)
    full_code = f"{prompt}\n{body}\n\n{test_code}\n"
    return _run_in_sandbox(full_code, timeout=timeout)


def compute_pass_at_1(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute pass@1 and executable rate from a list of execution results.

    Args:
        results: List of dicts from ``evaluate_humaneval_sample``.

    Returns:
        ``{"pass_at_1": float, "executable_rate": float}``
    """
    if not results:
        return {"pass_at_1": 0.0, "executable_rate": 0.0}

    passed = sum(1 for r in results if r["passed"])
    executable = sum(1 for r in results if r["exit_code"] == 0)

    return {
        "pass_at_1": passed / len(results),
        "executable_rate": executable / len(results),
    }
