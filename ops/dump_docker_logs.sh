#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Dump current Docker Compose logs into artifacts/infra/compose_logs/<service>.log
#
# Usage:
#   bash ops/dump_docker_logs.sh            # all services
#   bash ops/dump_docker_logs.sh gateway ui  # specific services
#
# Uses the canonical env file by default:
# - repo-root `.env` for local checkouts
# - release-root `.env` when run from current/ or releases/<sha>/
# Override with COMPOSE_ENV_FILE=/absolute/path/to/.env if needed.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

default_env_file() {
    local repo_env="$REPO_ROOT/.env"
    local repo_name parent_name release_root_env

    if [[ -f "$repo_env" ]]; then
        printf '%s\n' "$repo_env"
        return
    fi

    repo_name="$(basename "$REPO_ROOT")"
    parent_name="$(basename "$(dirname "$REPO_ROOT")")"

    if [[ "$repo_name" == "current" ]]; then
        release_root_env="$(cd "$REPO_ROOT/.." && pwd)/.env"
        if [[ -f "$release_root_env" ]]; then
            printf '%s\n' "$release_root_env"
            return
        fi
    elif [[ "$parent_name" == "releases" ]]; then
        release_root_env="$(cd "$REPO_ROOT/../.." && pwd)/.env"
        if [[ -f "$release_root_env" ]]; then
            printf '%s\n' "$release_root_env"
            return
        fi
    fi

    printf '%s\n' "$repo_env"
}

ENV_FILE="${COMPOSE_ENV_FILE:-$(default_env_file)}"

[[ -f "$ENV_FILE" ]] || {
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
}

read_env_value() {
    local name="$1"
    awk -F= -v key="$name" '
        $0 !~ /^[[:space:]]*#/ && $1 == key {
            value = substr($0, index($0, "=") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "$ENV_FILE"
}

compose_project_name="$(read_env_value COMPOSE_PROJECT_NAME)"
project_root="$(read_env_value PROJECT_ROOT)"
compose_file="$project_root/infra/compose/docker-compose.yaml"

[[ -n "$compose_project_name" ]] || {
    echo "COMPOSE_PROJECT_NAME is missing in $ENV_FILE" >&2
    exit 1
}
[[ -n "$project_root" ]] || {
    echo "PROJECT_ROOT is missing in $ENV_FILE" >&2
    exit 1
}
[[ -f "$compose_file" ]] || {
    echo "Compose file not found: $compose_file" >&2
    exit 1
}

LOG_DIR="$project_root/artifacts/infra/compose_logs"
compose_args=(--env-file "$ENV_FILE" --project-name "$compose_project_name" -f "$compose_file")

mkdir -p "$LOG_DIR"

docker_compose() {
    docker compose "${compose_args[@]}" "$@"
}

dump_inventory() {
    local compose_status="$LOG_DIR/compose_status.txt"
    local compose_status_err="$LOG_DIR/compose_status.err.log"
    local docker_containers="$LOG_DIR/docker_containers.txt"

    if docker_compose ps -a > "$compose_status" 2> "$compose_status_err"; then
        rm -f "$compose_status_err"
    else
        echo "Failed to dump compose status; stderr saved to $compose_status_err" >&2
    fi

    docker ps -a \
        --filter "label=com.docker.compose.project=$compose_project_name" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.ID}}' \
        > "$docker_containers"
}

dump_container_logs_by_label() {
    local service="$1"
    local -a containers=()
    local container name

    mapfile -t containers < <(
        docker ps -aq \
            --filter "label=com.docker.compose.project=$compose_project_name" \
            --filter "label=com.docker.compose.service=$service"
    )

    [[ ${#containers[@]} -gt 0 ]] || return 1

    for container in "${containers[@]}"; do
        name="$(docker inspect --format '{{.Name}}' "$container" 2>/dev/null | sed 's#^/##')"
        if [[ ${#containers[@]} -gt 1 ]]; then
            printf '===== %s (%s) =====\n' "${name:-$container}" "$container"
        fi
        docker logs --timestamps "$container"
    done
}

dump_service_logs() {
    local service="$1"
    local outfile="$LOG_DIR/${service}.log"
    local errfile="$LOG_DIR/${service}.err.log"
    local fallback_file="$outfile.fallback"

    echo "Dumping $service -> $outfile"

    if docker_compose logs --no-color --no-log-prefix "$service" > "$outfile" 2> "$errfile"; then
        if [[ ! -s "$outfile" ]]; then
            if dump_container_logs_by_label "$service" > "$fallback_file" 2>> "$errfile" && [[ -s "$fallback_file" ]]; then
                mv "$fallback_file" "$outfile"
                echo "  used docker logs fallback for $service"
            else
                rm -f "$fallback_file"
            fi
        fi
    else
        echo "  docker compose logs failed for $service; trying docker logs fallback" >&2
        if ! dump_container_logs_by_label "$service" > "$outfile" 2>> "$errfile"; then
            echo "  no logs for $service"
        fi
    fi

    if [[ -s "$errfile" ]]; then
        echo "  stderr saved to $errfile" >&2
    else
        rm -f "$errfile"
    fi
}

dump_inventory

# Resolve services: args or all running services
if [[ $# -gt 0 ]]; then
    services=("$@")
else
    mapfile -t services < <(
        docker_compose ps -a --format '{{.Service}}'
    )
fi

if [[ ${#services[@]} -eq 0 ]]; then
    echo "No running services found."
    exit 0
fi

for svc in "${services[@]}"; do
    dump_service_logs "$svc"
done

echo "Done. Logs saved to $LOG_DIR/"
