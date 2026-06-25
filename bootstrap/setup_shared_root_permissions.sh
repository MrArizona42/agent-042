#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sudo bash bootstrap/setup_shared_root_permissions.sh [options]

Prepare Phase 2 shared-root permissions for /home/anton-m/agent-042.

The helper creates the shared roots, discovers the effective container UIDs
for Airflow and Jupyter, and applies setgid plus ACLs so the runtime can use
external bind mounts without restoring airflow-prepare-dirs.

Options:
    --server-root PATH       Target shared root (default: /home/anton-m/agent-042)
  --deploy-user USER       Host login that should own the shared roots
                           (default: SUDO_USER)
  --group NAME             Host group for shared roots (default: agent042)
  --env-file PATH          Compose env file (default: <repo>/.env)
  --compose-file PATH      Compose file (default: <repo>/infra/compose/docker-compose.yaml)
  --dvc-config-source PATH Source .dvc/config.local to copy if target is missing
                           (default: <repo>/.dvc/config.local)
  --airflow-uid UID        Override detected UID for airflow-worker
  --airflow-gpu-uid UID    Override detected UID for airflow-worker-gpu
  --jupyter-uid UID        Override detected UID for jupyter
  --skip-usermod           Do not add the deploy user to the shared host group
  --help                   Show this message

Examples:
  sudo bash bootstrap/setup_shared_root_permissions.sh --deploy-user anton
    sudo bash bootstrap/setup_shared_root_permissions.sh --deploy-user anton --server-root /home/anton-m/agent-042
  sudo bash bootstrap/setup_shared_root_permissions.sh --deploy-user anton --airflow-uid 50000 --jupyter-uid 1000
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SERVER_ROOT="/home/anton-m/agent-042"
DEPLOY_USER="${SUDO_USER:-}"
GROUP_NAME="agent042"
ENV_FILE="$REPO_ROOT/.env"
COMPOSE_FILE="$REPO_ROOT/infra/compose/docker-compose.yaml"
DVC_CONFIG_SOURCE="$REPO_ROOT/.dvc/config.local"

AIRFLOW_UID_OVERRIDE=""
AIRFLOW_GPU_UID_OVERRIDE=""
JUPYTER_UID_OVERRIDE=""
SKIP_USERMOD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-root)
            SERVER_ROOT="$2"
            shift 2
            ;;
        --deploy-user)
            DEPLOY_USER="$2"
            shift 2
            ;;
        --group)
            GROUP_NAME="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --compose-file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --dvc-config-source)
            DVC_CONFIG_SOURCE="$2"
            shift 2
            ;;
        --airflow-uid)
            AIRFLOW_UID_OVERRIDE="$2"
            shift 2
            ;;
        --airflow-gpu-uid)
            AIRFLOW_GPU_UID_OVERRIDE="$2"
            shift 2
            ;;
        --jupyter-uid)
            JUPYTER_UID_OVERRIDE="$2"
            shift 2
            ;;
        --skip-usermod)
            SKIP_USERMOD=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_root() {
    if [[ "$EUID" -ne 0 ]]; then
        echo "Run this helper as root (for example via sudo)." >&2
        exit 1
    fi
}

require_cmd() {
    local name="$1"

    if ! command -v "$name" >/dev/null 2>&1; then
        echo "Required command not found: $name" >&2
        exit 1
    fi
}

require_compose() {
    if ! docker compose version >/dev/null 2>&1; then
        echo "docker compose is required but not available." >&2
        exit 1
    fi
}

compose() {
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

ensure_user_exists() {
    if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
        echo "Deploy user does not exist: $DEPLOY_USER" >&2
        exit 1
    fi
}

maybe_add_user_to_group() {
    local current_groups

    if [[ "$SKIP_USERMOD" -eq 1 ]]; then
        return
    fi

    current_groups="$(id -nG "$DEPLOY_USER")"
    if [[ " $current_groups " == *" $GROUP_NAME "* ]]; then
        return
    fi

    echo "Adding $DEPLOY_USER to host group $GROUP_NAME"
    usermod -aG "$GROUP_NAME" "$DEPLOY_USER"
    USER_ADDED_TO_GROUP=1
}

detect_uid() {
    local service="$1"
    local override="$2"
    local detected_uid=""

    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
        return 0
    fi

    if detected_uid="$(compose exec -T "$service" id -u 2>/dev/null | tr -d '\r')" && [[ -n "$detected_uid" ]]; then
        printf '%s\n' "$detected_uid"
        return 0
    fi

    if detected_uid="$(compose run --rm --no-deps -T --entrypoint id "$service" -u 2>/dev/null | tr -d '\r')" && [[ -n "$detected_uid" ]]; then
        printf '%s\n' "$detected_uid"
        return 0
    fi

    return 1
}

append_unique() {
    local value="$1"
    local existing

    if [[ -z "$value" ]]; then
        return
    fi

    for existing in "${RUNTIME_UIDS[@]:-}"; do
        if [[ "$existing" == "$value" ]]; then
            return
        fi
    done

    RUNTIME_UIDS+=("$value")
}

grant_rw_acl() {
    local target="$1"
    local acl_spec="u:${DEPLOY_USER}:rwx,g:${GROUP_NAME}:rwx"
    local uid

    for uid in "${RUNTIME_UIDS[@]}"; do
        acl_spec+=",u:${uid}:rwx"
    done

    setfacl -R -m "$acl_spec" "$target"
    setfacl -R -d -m "$acl_spec" "$target"
}

grant_read_acl() {
    local target="$1"
    shift
    local uid

    for uid in "$@"; do
        [[ -n "$uid" ]] || continue
        setfacl -m "u:${uid}:r" "$target"
    done
}

require_root
require_cmd docker
require_cmd find
require_cmd id
require_cmd install
require_cmd setfacl
require_compose

if [[ -z "$DEPLOY_USER" ]]; then
    echo "--deploy-user is required when SUDO_USER is not set." >&2
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Compose env file not found: $ENV_FILE" >&2
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
fi

ensure_user_exists

USER_ADDED_TO_GROUP=0
RUNTIME_UIDS=()

echo "Ensuring shared host group exists: $GROUP_NAME"
groupadd --force "$GROUP_NAME"
maybe_add_user_to_group

echo "Creating shared root skeleton under $SERVER_ROOT"
install -d -o "$DEPLOY_USER" -g "$GROUP_NAME" -m 2775 \
    "$SERVER_ROOT" \
    "$SERVER_ROOT/assets" \
    "$SERVER_ROOT/assets/models" \
    "$SERVER_ROOT/assets/adapters" \
    "$SERVER_ROOT/assets/datasets" \
    "$SERVER_ROOT/assets/rag_data" \
    "$SERVER_ROOT/artifacts" \
    "$SERVER_ROOT/artifacts/training" \
    "$SERVER_ROOT/.dvc"

echo "Detecting runtime UIDs"
AIRFLOW_UID="$(detect_uid airflow-worker "$AIRFLOW_UID_OVERRIDE")"
append_unique "$AIRFLOW_UID"

if AIRFLOW_GPU_UID="$(detect_uid airflow-worker-gpu "$AIRFLOW_GPU_UID_OVERRIDE" 2>/dev/null)"; then
    append_unique "$AIRFLOW_GPU_UID"
else
    AIRFLOW_GPU_UID=""
fi

JUPYTER_UID="$(detect_uid jupyter "$JUPYTER_UID_OVERRIDE")"
append_unique "$JUPYTER_UID"

echo "  airflow-worker uid: $AIRFLOW_UID"
if [[ -n "$AIRFLOW_GPU_UID" ]]; then
    echo "  airflow-worker-gpu uid: $AIRFLOW_GPU_UID"
fi
echo "  jupyter uid: $JUPYTER_UID"

echo "Applying shared-root group ownership and directory modes"
chgrp -R "$GROUP_NAME" "$SERVER_ROOT/assets" "$SERVER_ROOT/artifacts" "$SERVER_ROOT/.dvc"
find "$SERVER_ROOT/assets" "$SERVER_ROOT/artifacts" -type d -exec chmod 2775 {} +
chmod 2775 "$SERVER_ROOT/.dvc"

echo "Applying ACLs to writable shared roots"
grant_rw_acl "$SERVER_ROOT/assets"
grant_rw_acl "$SERVER_ROOT/artifacts"

TARGET_DVC_CONFIG="$SERVER_ROOT/.dvc/config.local"
if [[ -f "$TARGET_DVC_CONFIG" ]]; then
    echo "Keeping existing DVC local config: $TARGET_DVC_CONFIG"
    chown "$DEPLOY_USER:$GROUP_NAME" "$TARGET_DVC_CONFIG"
    chmod 640 "$TARGET_DVC_CONFIG"
elif [[ -f "$DVC_CONFIG_SOURCE" ]]; then
    echo "Copying DVC local config to shared root"
    install -o "$DEPLOY_USER" -g "$GROUP_NAME" -m 640 "$DVC_CONFIG_SOURCE" "$TARGET_DVC_CONFIG"
else
    echo "Skipping missing DVC local config source: $DVC_CONFIG_SOURCE"
fi

if [[ -f "$TARGET_DVC_CONFIG" ]]; then
    echo "Granting read ACLs on DVC local config to Airflow workers"
    grant_read_acl "$TARGET_DVC_CONFIG" "$AIRFLOW_UID" "$AIRFLOW_GPU_UID"
fi

echo
echo "Shared-root permission bootstrap complete."
echo "Writable roots:"
echo "  $SERVER_ROOT/assets"
echo "  $SERVER_ROOT/artifacts"
echo "Read-only DVC config:"
echo "  $TARGET_DVC_CONFIG"

if [[ "$USER_ADDED_TO_GROUP" -eq 1 ]]; then
    echo
    echo "Note: $DEPLOY_USER was added to $GROUP_NAME. Re-login or run 'newgrp $GROUP_NAME'"
    echo "before relying on the new host group membership."
fi
