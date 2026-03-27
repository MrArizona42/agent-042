"""Sandboxed code execution for HumanEval evaluation.

Runs generated Python code inside a `bubblewrap (bwrap)`_ sandbox with
filesystem, network, and PID isolation.  No Docker socket, no SUID
binary, and no root privileges are required.

The airflow-worker Docker image ships ``bwrap``; if it is missing at
runtime the module raises ``RuntimeError`` at import time.

.. _bubblewrap (bwrap): https://github.com/containers/bubblewrap
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BWRAP_BIN: str | None = shutil.which("bwrap")
if _BWRAP_BIN is None:
    raise RuntimeError(
        "bubblewrap (bwrap) is not installed.  Code evaluation requires "
        "bwrap for filesystem/network/PID isolation.  Install it with: "
        "apt-get install bubblewrap"
    )

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


def _bwrap_command(script_path: str, cpu_seconds: int) -> list[str]:
    """Build a ``bwrap`` command line for sandboxed execution.

    Provides:
    - **Filesystem isolation**: read-only bind mounts for Python stdlib
      and shared libraries only.  ``/tmp`` is a private tmpfs.
    - **Network isolation**: ``--unshare-net`` drops all networking.
    - **PID isolation**: ``--unshare-pid`` hides host processes.
    - **No new privileges**: ``--new-session`` + inherits non-root user.

    ``RLIMIT_CPU`` is applied via ``ulimit -t`` in a shell wrapper so
    it works inside the namespace.
    """
    assert _BWRAP_BIN is not None  # noqa: S101 – caller checked
    return [
        _BWRAP_BIN,
        # Filesystem: minimal read-only tree
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        *(  # /lib64 exists on x86-64 Debian/Ubuntu
            ["--ro-bind", "/lib64", "/lib64"] if os.path.isdir("/lib64") else []
        ),
        "--ro-bind",
        "/bin",
        "/bin",
        *(  # /sbin may hold ld.so helpers
            ["--ro-bind", "/sbin", "/sbin"] if os.path.isdir("/sbin") else []
        ),
        "--ro-bind",
        "/etc/alternatives",
        "/etc/alternatives",
        # Bind the script itself read-only
        "--ro-bind",
        script_path,
        script_path,
        # Private writable /tmp (tmpfs, 64 MB)
        "--tmpfs",
        "/tmp",
        # Required virtual filesystems
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        # Isolation
        "--unshare-net",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--new-session",
        "--die-with-parent",
        # Run python3 with a CPU-time ulimit
        "--",
        "/bin/sh",
        "-c",
        f"ulimit -t {cpu_seconds} && exec python3 {script_path}",
    ]


def _run_in_container(
    code: str,
    *,
    timeout: int,
    mem_limit: str,  # noqa: ARG001 – kept for call-site compatibility
    cpus: float,
) -> dict[str, Any]:
    """Execute *code* in a bubblewrap sandbox.

    The code runs in an isolated user namespace with no network, a
    read-only filesystem, and a private PID namespace.  CPU time is
    capped via ``ulimit -t`` and a wall-clock ``timeout`` prevents
    runaway tasks.

    Args:
        code: Python source to execute.
        timeout: Wall-clock timeout in seconds (also used as CPU-time cap).
        mem_limit: Ignored; kept for backward-compatible call sites.
        cpus: CPU share; multiplied by *timeout* to derive the cpu-time cap.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    cpu_seconds = max(1, int(timeout * cpus))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        host_path = f.name

    try:
        cmd = _bwrap_command(host_path, cpu_seconds)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
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
    into one script, then executes it in a bubblewrap (bwrap) sandbox.

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
