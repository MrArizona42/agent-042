#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run dump_docker_logs.sh on a remote server via SSH, then scp the
# resulting logs into the local artifacts/infra/compose_logs/ directory.
#
# Usage:
#   bash scripts/fetch_logs_ssh.sh [user@]host [service ...]
#
#   user@host             — SSH target (required)
#   service ...           — optional list of services to dump (default: all)
#
# Examples:
#   bash scripts/fetch_logs_ssh.sh my_server
#   bash scripts/fetch_logs_ssh.sh my_server gateway ui
#
# Requires: ssh, scp (both available via OpenSSH / Git Bash)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_LOG_DIR="$PROJECT_ROOT/artifacts/infra/compose_logs"

SSH_HOST="$1"
shift 1
SERVICES=("$@")
REMOTE_SCRIPT="/home/anton-m/agent-042/current/scripts/dump_docker_logs.sh"
REMOTE_LOG_PATH="/home/anton-m/agent-042/artifacts/infra/compose_logs"

# ── Sudo password (interactive prompt) ───────────────────────────────
read -r -s -p "sudo password for $SSH_HOST: " SUDO_PASS
echo

# ── Step 1: run dump_docker_logs.sh on the remote ────────────────────
remote_parts=("sudo" "-S" "bash" "$(printf '%q' "$REMOTE_SCRIPT")")
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
