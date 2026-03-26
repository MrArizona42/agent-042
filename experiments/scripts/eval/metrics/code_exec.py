"""Sandboxed code execution for HumanEval evaluation.

Runs generated Python code in a child ``python3`` subprocess with a CPU-time
rlimit (``RLIMIT_CPU``) and a wall-clock timeout.  No Docker socket and no
SUID sandbox binary are required.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEM_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3}

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


def _parse_mem_limit(mem_limit: str) -> int:
    """Convert a Docker-style memory string (e.g. ``'512m'``) to bytes."""
    m = re.fullmatch(r"(\d+)([kmg])?", mem_limit.strip().lower())
    if not m:
        raise ValueError(f"Unparseable mem_limit: {mem_limit!r}")
    value, unit = int(m.group(1)), m.group(2) or "b"
    return value * _MEM_UNITS.get(unit, 1)


def _run_in_container(
    code: str,
    *,
    timeout: int,
    mem_limit: str,  # noqa: ARG001 – kept for call-site compatibility
    cpus: float,
) -> dict[str, Any]:
    """Execute *code* in a resource-limited subprocess.

    Runs ``python3`` in a child process.  A ``RLIMIT_CPU`` rlimit is applied
    inside the child via ``preexec_fn`` and a wall-clock ``timeout`` is
    enforced by the parent so runaway tasks cannot stall the worker
    indefinitely.  No additional sandboxing binary is required.

    Note:
        ``mem_limit`` is accepted for config compatibility but is not applied
        as ``RLIMIT_AS``.  On 64-bit hosts the virtual-address-space limit
        interferes with Python's own startup allocations (shared-library
        mappings) and produces spurious ``MemoryError`` failures unrelated to
        the evaluated code.  The wall-clock ``timeout`` remains the primary
        runaway-task guard.

    Args:
        code: Python source to execute.
        timeout: Wall-clock timeout in seconds (also used as CPU-time cap).
        mem_limit: Ignored; kept for backward-compatible call sites.
        cpus: CPU share; multiplied by *timeout* to derive the rlimit-cpu cap.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    cpu_seconds = max(1, int(timeout * cpus))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        host_path = f.name

    def _set_limits() -> None:
        """CPU-time rlimit applied inside the child process before exec."""
        try:
            import resource as _r  # Linux / macOS only

            _r.setrlimit(_r.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass  # Best-effort; non-Linux or unprivileged container

    try:
        proc = subprocess.run(
            ["python3", host_path],
            capture_output=True,
            timeout=timeout,
            preexec_fn=_set_limits,
        )
        exit_code = proc.returncode
        return {
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "stdout": proc.stdout.decode(errors="replace").strip(),
            "stderr": proc.stderr.decode(errors="replace").strip(),
        }
    except subprocess.TimeoutExpired:
        logger.warning("Code execution timed out after %ds", timeout)
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        logger.error("Code execution error: %s", e)
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": str(e)}
    finally:
        Path(host_path).unlink(missing_ok=True)


def evaluate_humaneval_sample(
    prompt: str,
    generated_code: str,
    test_code: str,
    *,
    timeout: int,
    mem_limit: str,
    cpus: float,
) -> dict[str, Any]:
    """Evaluate a single HumanEval sample.

    Combines the function prompt, generated completion, and test assertions
    into one script, then executes it in a Firejail sandbox.

    Args:
        prompt: The function signature / docstring from HumanEval.
        generated_code: Model-generated function body.
        test_code: Assertion-based test code from the dataset.
        timeout: Execution timeout in seconds.
        mem_limit: Memory limit string (e.g. ``'512m'``).
        cpus: CPU share for rlimit-cpu calculation.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    body = extract_code_from_response(generated_code, prompt)
    full_code = f"{prompt}\n{body}\n\n{test_code}\n"
    return _run_in_container(full_code, timeout=timeout, mem_limit=mem_limit, cpus=cpus)


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
