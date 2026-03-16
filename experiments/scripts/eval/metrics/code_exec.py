"""Sandboxed code execution for HumanEval evaluation.

Runs generated Python code in ephemeral Docker containers with strict
resource limits and no network access.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _run_in_container(
    code: str,
    *,
    image: str = "python:3.11-slim",
    timeout: int = 30,
    mem_limit: str = "512m",
    cpus: float = 1.0,
) -> dict[str, Any]:
    """Execute *code* in an ephemeral Docker container.

    Args:
        code: Python source to execute.
        image: Docker image name.
        timeout: Maximum execution time in seconds.
        mem_limit: Memory limit string for Docker.
        cpus: Number of CPUs.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    try:
        import docker
    except ImportError:
        logger.warning("docker package not installed; marking sample as failed")
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": "docker not installed"}

    client = docker.from_env()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        host_path = f.name

    container = None
    try:
        container = client.containers.run(
            image,
            command=["python", "/code/solution.py"],
            volumes={host_path: {"bind": "/code/solution.py", "mode": "ro"}},
            network_mode="none",
            mem_limit=mem_limit,
            nano_cpus=int(cpus * 1e9),
            detach=True,
            remove=False,
        )

        result = container.wait(timeout=timeout)
        exit_code = result.get("StatusCode", -1)
        stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")

        return {
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
    except Exception as e:
        logger.error("Container execution error: %s", e)
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": str(e)}
    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass
        Path(host_path).unlink(missing_ok=True)


def evaluate_humaneval_sample(
    prompt: str,
    generated_code: str,
    test_code: str,
    *,
    image: str = "python:3.11-slim",
    timeout: int = 30,
) -> dict[str, Any]:
    """Evaluate a single HumanEval sample.

    Combines the function prompt, generated completion, and test assertions
    into one script, then executes it in a sandboxed container.

    Args:
        prompt: The function signature / docstring from HumanEval.
        generated_code: Model-generated function body.
        test_code: Assertion-based test code from the dataset.
        image: Docker image for execution.
        timeout: Execution timeout in seconds.

    Returns:
        ``{"passed": bool, "exit_code": int, "stdout": str, "stderr": str}``
    """
    full_code = f"{prompt}\n{generated_code}\n\n{test_code}\n"
    return _run_in_container(full_code, image=image, timeout=timeout)


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
