#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash ops/deploy_release.sh \
    --release-root /home/anton-m/agent-042 \
    --release-dir /home/anton-m/agent-042/releases/<sha> \
    --env-file /home/anton-m/agent-042/.env \
    --image-tag <branch-slug>-<sha12> \
    [--source-branch <branch>] \
    [--source-sha <sha>] \
    [--keep-releases 5] \
    [--compose-project-name agent-042] \
    [--smoke-timeout-seconds 180]
EOF
}

log() {
    echo "==> $*"
}

fail() {
    echo "error: $*" >&2
    exit 1
}

trim_quotes() {
    local value="$1"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s\n' "$value"
}

read_env_value() {
    local key="$1"
    local file="$2"
    local value

    value="$(awk -F= -v key="$key" '
        $0 ~ "^[[:space:]]*" key "=" {
            print substr($0, index($0, "=") + 1)
            exit
        }
    ' "$file")"

    trim_quotes "$value"
}

upsert_env_value() {
    local key="$1"
    local value="$2"
    local file="$3"
    local tmp_file

    tmp_file="$(mktemp)"
    awk -v key="$key" -v value="$value" '
        BEGIN {
            updated = 0
        }
        $0 ~ "^[[:space:]]*" key "=" && updated == 0 {
            print key "=" value
            updated = 1
            next
        }
        {
            print
        }
        END {
            if (updated == 0) {
                print key "=" value
            }
        }
    ' "$file" > "$tmp_file"

    mv "$tmp_file" "$file"
}

compose_file_for() {
    local project_root="$1"
    printf '%s/infra/compose/docker-compose.yaml\n' "$project_root"
}

compose() {
    local project_root="$1"
    local image_tag="$2"
    shift 2

    local compose_file
    compose_file="$(compose_file_for "$project_root")"

    [[ -f "$compose_file" ]] || fail "Compose file not found: $compose_file"

    COMPOSE_PROJECT_NAME="$compose_project_name" \
    PROJECT_ROOT="$project_root" \
    SHARED_ROOT="$release_root" \
    IMAGE_TAG="$image_tag" \
    docker compose \
        --project-name "$compose_project_name" \
        --env-file "$env_file" \
        -f "$compose_file" \
        "$@"
}

service_is_running() {
    local project_root="$1"
    local image_tag="$2"
    local service="$3"

    compose "$project_root" "$image_tag" ps --services --status running | grep -Fxq "$service"
}

wait_for_running_service() {
    local project_root="$1"
    local image_tag="$2"
    local service="$3"

    local deadline
    deadline=$((SECONDS + smoke_timeout_seconds))

    until service_is_running "$project_root" "$image_tag" "$service"; do
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 5
    done
}

wait_for_health_command() {
    local project_root="$1"
    local image_tag="$2"
    local service="$3"
    local health_command="$4"

    local deadline
    deadline=$((SECONDS + smoke_timeout_seconds))

    until compose "$project_root" "$image_tag" exec -T "$service" sh -lc "$health_command" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 5
    done
}

current_release_target() {
    if [[ -L "$current_link" ]]; then
        readlink -f "$current_link"
        return 0
    fi

    if [[ -f "$env_file" ]]; then
        local project_root_from_env
        project_root_from_env="$(read_env_value PROJECT_ROOT "$env_file")"
        if [[ -n "$project_root_from_env" && -d "$project_root_from_env" ]]; then
            printf '%s\n' "$project_root_from_env"
            return 0
        fi
    fi

    return 1
}

current_image_tag() {
    local metadata_file
    metadata_file="$1/.deploy-meta.env"

    if [[ -f "$metadata_file" ]]; then
        read_env_value IMAGE_TAG "$metadata_file"
        return 0
    fi

    local running_gateway_image
    running_gateway_image="$(docker ps \
        --filter "label=com.docker.compose.project=$compose_project_name" \
        --filter 'label=com.docker.compose.service=gateway' \
        --format '{{.Image}}' | head -n 1)"
    if [[ -n "$running_gateway_image" ]]; then
        printf '%s\n' "${running_gateway_image##*:}"
        return 0
    fi

    if [[ -f "$env_file" ]]; then
        local image_tag_from_env
        image_tag_from_env="$(read_env_value IMAGE_TAG "$env_file")"
        if [[ -n "$image_tag_from_env" ]]; then
            printf '%s\n' "$image_tag_from_env"
            return 0
        fi
    fi

    return 1
}

record_release_metadata() {
    cat > "$release_metadata_file" <<EOF
SOURCE_BRANCH=$source_branch
SOURCE_SHA=$source_sha
IMAGE_TAG=$image_tag
RELEASE_DIR=$release_dir
COMPOSE_PROJECT_NAME=$compose_project_name
DEPLOYED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
}

show_compose_status() {
    local project_root="$1"
    local image_tag="$2"
    compose "$project_root" "$image_tag" ps || true
}

show_service_diagnostics() {
    local project_root="$1"
    local image_tag="$2"
    local service="$3"

    local container_id
    container_id="$(compose "$project_root" "$image_tag" ps -q "$service" 2>/dev/null | head -n 1 || true)"

    echo
    echo "--- service: $service ---"
    if [[ -z "$container_id" ]]; then
        echo "container: not found"
        return 0
    fi

    docker inspect \
        --format 'container={{.Name}} image={{.Config.Image}} status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} error={{.State.Error}}' \
        "$container_id" || true

    echo "health log:"
    docker inspect \
        --format '{{if .State.Health}}{{range .State.Health.Log}}[{{.Start}}] exit={{.ExitCode}} output={{printf "%q" .Output}}{{println}}{{end}}{{else}}(no Docker healthcheck configured){{end}}' \
        "$container_id" || true
}

show_failure_diagnostics() {
    local project_root="$1"
    local image_tag="$2"
    local phase="$3"

    echo
    echo "==> Failure diagnostics ($phase)"
    echo "release: $project_root"
    echo "image tag: $image_tag"

    echo
    echo "==> Compose status"
    compose "$project_root" "$image_tag" ps -a || true

    local -a inspected_services=(
        gateway
        ui
        embeddings
        reranker
        vllm
        vllm-adapter-sync
        qdrant
        rabbitmq
        redis
        postgres
        mlflow
        prometheus
        grafana
        flower
        airflow-init
        airflow-webserver
        airflow-dag-processor
        airflow-scheduler
        airflow-worker
        airflow-worker-gpu
        celery-worker
        code-sandbox
        jupyter
        redisinsight
    )

    echo
    echo "==> Container state and health"
    local service
    for service in "${inspected_services[@]}"; do
        show_service_diagnostics "$project_root" "$image_tag" "$service"
    done

    echo
    echo "==> Recent logs for likely blockers"
    local -a logged_services=(
        gateway
        ui
        embeddings
        reranker
        vllm
        vllm-adapter-sync
        qdrant
        rabbitmq
        postgres
        mlflow
        flower
        airflow-init
        airflow-webserver
    )

    for service in "${logged_services[@]}"; do
        echo
        echo "--- logs: $service ---"
        compose "$project_root" "$image_tag" logs --no-color --timestamps --tail=160 "$service" || true
    done
}

apply_db_migrations() {
    local project_root="$1"

    local migrations_script="$project_root/bootstrap/apply_agent042_db_migrations.sh"
    [[ -f "$migrations_script" ]] || fail "Migrations script not found: $migrations_script"

    COMPOSE_FILE="$(compose_file_for "$project_root")" \
    ENV_FILE="$env_file" \
    COMPOSE_PROJECT_NAME="$compose_project_name" \
        bash "$migrations_script"
}

smoke_check() {
    local project_root="$1"
    local image_tag="$2"

    local -a required_running_services=(
        postgres
        mlflow
        vllm
        qdrant
        rabbitmq
        redis
        embeddings
        reranker
        celery-worker
        gateway
        ui
        airflow-webserver
        airflow-dag-processor
        airflow-scheduler
        airflow-worker
        airflow-worker-gpu
        code-sandbox
        jupyter
        prometheus
        grafana
        flower
        redisinsight
    )

    local service
    for service in "${required_running_services[@]}"; do
        wait_for_running_service "$project_root" "$image_tag" "$service" || {
            show_compose_status "$project_root" "$image_tag"
            echo "error: timed out waiting for service '$service' to enter running state" >&2
            return 1
        }
    done

    wait_for_health_command "$project_root" "$image_tag" gateway 'curl -fsS http://localhost:9000/health' || {
        show_compose_status "$project_root" "$image_tag"
        echo "error: gateway health check failed" >&2
        return 1
    }
    wait_for_health_command "$project_root" "$image_tag" embeddings 'curl -fsS http://localhost:8100/health' || {
        show_compose_status "$project_root" "$image_tag"
        echo "error: embeddings health check failed" >&2
        return 1
    }
    wait_for_health_command "$project_root" "$image_tag" reranker 'curl -fsS http://localhost:8101/health' || {
        show_compose_status "$project_root" "$image_tag"
        echo "error: reranker health check failed" >&2
        return 1
    }
}

promote_current_release() {
    mkdir -p "$release_root"

    if [[ -e "$current_link" && ! -L "$current_link" ]]; then
        fail "Current path exists but is not a symlink: $current_link"
    fi

    ln -sfn "$release_dir" "$current_link"
}

prune_old_releases() {
    local keep_count="$1"
    [[ -d "$releases_dir" ]] || return 0

    local -a stale_releases=()
    mapfile -t stale_releases < <(
        find "$releases_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' |
            sort -nr |
            awk -v keep_count="$keep_count" 'NR > keep_count {sub(/^[^ ]+ /, ""); print}'
    )

    if [[ ${#stale_releases[@]} -eq 0 ]]; then
        return 0
    fi

    local stale_release
    for stale_release in "${stale_releases[@]}"; do
        if [[ "$stale_release" == "$release_dir" ]]; then
            continue
        fi
        log "Removing old release $stale_release"
        rm -rf "$stale_release"
    done
}

rollback_to_previous_release() {
    if [[ -z "$previous_release" || -z "$previous_image_tag" ]]; then
        echo "warning: rollback skipped because no previous release target could be resolved" >&2
        return 1
    fi

    log "Rolling back to $previous_release with tag $previous_image_tag"
    compose "$previous_release" "$previous_image_tag" up -d --remove-orphans
    smoke_check "$previous_release" "$previous_image_tag"

    if [[ -L "$current_link" || ! -e "$current_link" ]]; then
        ln -sfn "$previous_release" "$current_link"
    fi

    return 0
}

release_root=""
release_dir=""
env_file=""
image_tag=""
source_branch=""
source_sha=""
keep_releases="5"
compose_project_name=""
smoke_timeout_seconds="180"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-root)
            release_root="$2"
            shift 2
            ;;
        --release-dir)
            release_dir="$2"
            shift 2
            ;;
        --env-file)
            env_file="$2"
            shift 2
            ;;
        --image-tag)
            image_tag="$2"
            shift 2
            ;;
        --source-branch)
            source_branch="$2"
            shift 2
            ;;
        --source-sha)
            source_sha="$2"
            shift 2
            ;;
        --keep-releases)
            keep_releases="$2"
            shift 2
            ;;
        --compose-project-name)
            compose_project_name="$2"
            shift 2
            ;;
        --smoke-timeout-seconds)
            smoke_timeout_seconds="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$release_root" ]] || fail "--release-root is required"
[[ -n "$release_dir" ]] || fail "--release-dir is required"
[[ -n "$env_file" ]] || fail "--env-file is required"
[[ -n "$image_tag" ]] || fail "--image-tag is required"
[[ -f "$env_file" ]] || fail "Env file not found: $env_file"
[[ -d "$release_dir" ]] || fail "Release directory not found: $release_dir"
[[ "$keep_releases" =~ ^[0-9]+$ ]] || fail "--keep-releases must be an integer"
[[ "$smoke_timeout_seconds" =~ ^[0-9]+$ ]] || fail "--smoke-timeout-seconds must be an integer"

if [[ -z "$compose_project_name" ]]; then
    compose_project_name="$(basename "$release_root")"
fi
[[ -n "$compose_project_name" ]] || fail "Could not derive compose project name from release root"

current_link="$release_root/current"
releases_dir="$release_root/releases"
release_metadata_file="$release_dir/.deploy-meta.env"
canonical_project_root="$release_root/current"

mkdir -p "$releases_dir"

previous_release="$(current_release_target || true)"
previous_image_tag=""
if [[ -n "$previous_release" ]]; then
    previous_image_tag="$(current_image_tag "$previous_release" || true)"
fi

record_release_metadata

log "Validating compose configuration for $release_dir"
compose "$release_dir" "$image_tag" config -q

log "Pulling images for tag $image_tag"
compose "$release_dir" "$image_tag" pull

log "Starting release from $release_dir"
if ! compose "$release_dir" "$image_tag" up -d --remove-orphans; then
    show_failure_diagnostics "$release_dir" "$image_tag" "docker compose up"
    rollback_to_previous_release || true
    fail "Deployment failed during docker compose up"
fi

log "Applying agent042 DB migrations"
if ! apply_db_migrations "$release_dir"; then
    show_failure_diagnostics "$release_dir" "$image_tag" "db migrations"
    rollback_to_previous_release || true
    fail "Deployment failed applying DB migrations"
fi

log "Running smoke checks"
if ! smoke_check "$release_dir" "$image_tag"; then
    show_failure_diagnostics "$release_dir" "$image_tag" "smoke checks"
    rollback_to_previous_release || true
    fail "Deployment failed smoke checks"
fi

log "Promoting current symlink"
promote_current_release

log "Updating canonical PROJECT_ROOT in $env_file"
upsert_env_value PROJECT_ROOT "$canonical_project_root" "$env_file"

log "Persisting shared root in $env_file"
upsert_env_value SHARED_ROOT "$release_root" "$env_file"

log "Persisting active IMAGE_TAG in $env_file"
upsert_env_value IMAGE_TAG "$image_tag" "$env_file"

log "Persisting COMPOSE_PROJECT_NAME in $env_file"
upsert_env_value COMPOSE_PROJECT_NAME "$compose_project_name" "$env_file"

log "Pruning old releases"
prune_old_releases "$keep_releases"

log "Deployment finished successfully"
show_compose_status "$release_dir" "$image_tag"
