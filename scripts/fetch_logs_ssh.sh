#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run dump_docker_logs.sh on a remote server via SSH, then scp the
# resulting logs into the local infra/compose/logs/ directory.
#
# Usage:
#   bash scripts/fetch_logs_ssh.sh [user@]host [remote_project_root] [service ...]
#
#   user@host             — SSH target (required)
#   remote_project_root   — absolute path on the remote (required)
#   service ...           — optional list of services to dump (default: all)
#
# Examples:
#   bash scripts/fetch_logs_ssh.sh deploy@10.0.0.1 /srv/agent-042
#   bash scripts/fetch_logs_ssh.sh deploy@10.0.0.1 /srv/agent-042 gateway ui
#
# Requires: ssh, scp (both available via OpenSSH / Git Bash)
#
# Example
# bash scripts/fetch_logs_ssh.sh my_server /home/anton-m/Git/agent-042
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_LOG_DIR="$PROJECT_ROOT/infra/compose"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [user@]host remote_project_root [service ...]"
    exit 1
fi

SSH_HOST="$1"
REMOTE_ROOT="$2"
shift 2
SERVICES=("$@")

# ── Sudo password (interactive prompt) ───────────────────────────────
read -r -s -p "sudo password for $SSH_HOST: " SUDO_PASS
echo

# ── Step 1: run dump_docker_logs.sh on the remote ────────────────────
REMOTE_SCRIPT="$REMOTE_ROOT/scripts/dump_docker_logs.sh"
REMOTE_CMD="sudo -S bash $REMOTE_SCRIPT ${SERVICES[*]+"${SERVICES[*]}"}"

echo "==> Running dump_docker_logs.sh on $SSH_HOST ..."
echo "$SUDO_PASS" | ssh "$SSH_HOST" "$REMOTE_CMD"

# ── Step 2: scp logs back to local ───────────────────────────────────
REMOTE_LOG_PATH="$REMOTE_ROOT/infra/compose/logs/"

mkdir -p "$LOCAL_LOG_DIR"

echo "==> Fetching logs from $SSH_HOST:$REMOTE_LOG_PATH ..."
scp -r "$SSH_HOST:$REMOTE_LOG_PATH" "$LOCAL_LOG_DIR/"

echo "==> Done. Logs saved to $LOCAL_LOG_DIR/"
