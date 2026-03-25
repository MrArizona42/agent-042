"""Sandboxed code execution for HumanEval evaluation.

Runs generated Python code under Firejail with strict resource limits and no
network access.  Firejail must be installed in the worker image (see
infra/docker/airflow-worker/Dockerfile).  No Docker socket is required.
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
    mem_limit: str,
    cpus: float,
) -> dict[str, Any]:
    """Execute *code* under Firejail with resource and network isolation.

    Args:
        code: Python source to execute.
        timeout: Wall-clock timeout in seconds (also used as CPU-time cap).
        mem_limit: Memory limit string (e.g. ``'512m'``).
        cpus: CPU share; multiplied by *timeout* to derive the rlimit-cpu cap.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    mem_bytes = _parse_mem_limit(mem_limit)
    cpu_seconds = max(1, int(timeout * cpus))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        host_path = f.name

    cmd = [
        "firejail",
        "--quiet",
        "--net=none",  # no network access inside the jail
        "--noroot",  # prevent privilege escalation inside the jail
        f"--rlimit-as={mem_bytes}",  # virtual-memory cap
        f"--rlimit-cpu={cpu_seconds}",  # CPU-time cap
        "--",
        "python3",
        host_path,
    ]

    try:
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
        logger.warning("Firejail execution timed out after %ds", timeout)
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": "timeout"}
    except FileNotFoundError:
        logger.error("firejail binary not found; is it installed in the worker image?")
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": "firejail not found"}
    except Exception as e:
        logger.error("Firejail execution error: %s", e)
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
    full_code = f"{prompt}\n{generated_code}\n\n{test_code}\n"
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
