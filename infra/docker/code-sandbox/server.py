"""Minimal code execution HTTP server for the code-sandbox sidecar.

Accepts POST /execute with JSON body::

    {"code": "<python source>", "timeout": <seconds int>}

Returns JSON::

    {"exit_code": int, "stdout": str, "stderr": str}

Isolation is provided entirely by Docker: the container is started with
``read_only: true``, a tmpfs ``/tmp``, no host-network access, and CPU/memory
limits.  The subprocess runs plain ``python3`` with no extra sandboxing inside
the container.

``/health`` returns 200 for Docker healthchecks.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_PORT = int(os.environ.get("SANDBOX_PORT", "8200"))
_MAX_CODE_BYTES = 256 * 1024  # 256 KB — guard against absurd payloads


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"}, status=200)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/execute":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        if length > _MAX_CODE_BYTES:
            self.send_error(413, "Payload too large")
            return

        try:
            body = json.loads(self.rfile.read(length))
            code: str = str(body["code"])
            timeout: int = int(body.get("timeout", 30))
        except Exception:
            self.send_error(400, "Bad request")
            return

        result = _execute(code, timeout)
        self._send_json(result)

    def _send_json(self, data: dict, *, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # suppress access log
        pass


def _execute(code: str, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=False) as f:
        f.write(code)
        script = f.name

    try:
        proc = subprocess.run(
            ["python3", script],
            capture_output=True,
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout.decode(errors="replace").strip(),
            "stderr": proc.stderr.decode(errors="replace").strip(),
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "timeout"}
    except Exception as exc:
        logger.error("Execution error: %s", exc)
        return {"exit_code": -1, "stdout": "", "stderr": str(exc)}
    finally:
        Path(script).unlink(missing_ok=True)


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", _PORT), _Handler)  # noqa: S104
    logger.warning("code-sandbox listening on :%d", _PORT)
    server.serve_forever()
