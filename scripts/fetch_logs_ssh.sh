#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run dump_docker_logs.sh on a remote server via SSH, then scp the
# resulting logs into the local artifacts/infra/logs/ directory.
#
# Usage:
#   bash scripts/fetch_logs_ssh.sh [user@]host [remote_project_root] [service ...]
#
#   user@host             — SSH target (required)
#   remote_project_root   — absolute path on the remote (required)
#   service ...           — optional list of services to dump (default: all)
#
# Examples:
#   bash scripts/fetch_logs_ssh.sh deploy@10.0.0.1 /home/anton-m/agent-042
#   bash scripts/fetch_logs_ssh.sh deploy@10.0.0.1 /home/anton-m/agent-042 gateway ui
#
# Requires: ssh, scp (both available via OpenSSH / Git Bash)
#
# Example
# bash scripts/fetch_logs_ssh.sh my_server /home/anton-m/Git/agent-042
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_LOG_DIR="$PROJECT_ROOT/artifacts/infra/logs"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [user@]host remote_project_root [service ...]"
    exit 1
fi

SSH_HOST="$1"
REMOTE_ROOT="$2"
shift 2
SERVICES=("$@")
REMOTE_SCRIPT="$REMOTE_ROOT/scripts/dump_docker_logs.sh"
REMOTE_ENV_FILE="${REMOTE_ENV_FILE:-$REMOTE_ROOT/.env}"
REMOTE_LOG_PATH="$REMOTE_ROOT/artifacts/infra/logs"

# ── Sudo password (interactive prompt) ───────────────────────────────
read -r -s -p "sudo password for $SSH_HOST: " SUDO_PASS
echo

# ── Step 1: run dump_docker_logs.sh on the remote ────────────────────
remote_parts=("COMPOSE_ENV_FILE=$(printf '%q' "$REMOTE_ENV_FILE")" "sudo" "-S" "bash" "$(printf '%q' "$REMOTE_SCRIPT")")
for svc in "${SERVICES[@]}"; do
    remote_parts+=("$(printf '%q' "$svc")")
done
REMOTE_CMD="${remote_parts[*]}"

echo "==> Running dump_docker_logs.sh on $SSH_HOST ..."
echo "$SUDO_PASS" | ssh "$SSH_HOST" "$REMOTE_CMD"

# ── Step 2: scp logs back to local ───────────────────────────────────
mkdir -p "$LOCAL_LOG_DIR"

echo "==> Fetching logs from $SSH_HOST:$REMOTE_LOG_PATH ..."
scp -r "$SSH_HOST:$REMOTE_LOG_PATH/." "$LOCAL_LOG_DIR/"

echo "==> Done. Logs saved to $LOCAL_LOG_DIR/"
