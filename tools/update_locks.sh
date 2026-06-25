#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Regenerate all Docker-service lock files from pyproject.toml extras.
#
# Usage:
#   ops/update_locks.sh              # update everything
#   ops/update_locks.sh gateway ui   # update only selected services
#   ops/update_locks.sh --list       # list available services
#   ops/update_locks.sh --dry-run    # print commands without running
#
# Must be executed from the repository root.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colour helpers (disabled when piped) ─────────────────────────────
if [[ -t 1 ]]; then
  GREEN=$'\033[32m' YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
  GREEN="" YELLOW="" RED="" RESET=""
fi

# ── Service definitions ──────────────────────────────────────────────
# Format:  name|selectors|python_version|output_file|extra_flags
#
# selectors may contain any mix of:
#   --extra <name>
#   --group <name>
#
# extra_flags may contain:
#   --constraint <file>
#   --extra-index-url <url> --index-strategy <strategy>

AIRFLOW_CONSTRAINTS="infra/docker/airflow/airflow-core-constraints.txt"
TORCH_CPU="--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"

SERVICES=(
  "gateway|--extra gateway|3.12|infra/docker/gateway/requirements-gateway.lock|"
  "ui|--extra ui|3.12|infra/docker/ui/requirements-ui.lock|"
  "celery|--extra worker|3.12|infra/docker/celery/requirements-celery.lock|"
  "mlflow|--extra mlflow|3.12|infra/docker/mlflow/requirements-mlflow.lock|"
  "adapter-sync|--extra mlflow|3.12|infra/docker/adapter-sync/requirements-adapter-sync.lock|"
  "embeddings|--extra embeddings|3.12|infra/docker/embeddings/requirements-embeddings.lock|"
  "reranker|--extra reranker|3.12|infra/docker/reranker/requirements-reranker.lock|"
  "jupyter|--extra training --extra rag --group dev --extra mlflow|3.13|infra/docker/jupyter/requirements-jupyter.lock|"
  "airflow|--extra airflow|3.12|infra/docker/airflow/requirements.lock|--constraint ${AIRFLOW_CONSTRAINTS}"
  "airflow-worker|--extra airflow-worker|3.12|infra/docker/airflow-worker/requirements-airflow-worker.lock|--constraint ${AIRFLOW_CONSTRAINTS} ${TORCH_CPU}"
  "airflow-worker-gpu|--extra airflow-worker-gpu|3.12|infra/docker/airflow-worker-gpu/requirements-airflow-worker-gpu.lock|--constraint ${AIRFLOW_CONSTRAINTS}"
)

# ── Helpers ──────────────────────────────────────────────────────────
list_services() {
  printf "\nAvailable services:\n"
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name selectors pyver outfile flags <<< "$entry"
    printf "  %-16s → %s\n" "$name" "$outfile"
  done
  echo
}

usage() {
  sed -n '2,/^# ─.*─$/{ /^# ─.*─$/d; s/^# //; p }' "$0"
}

compile_service() {
  local name="$1" selectors="$2" pyver="$3" outfile="$4" flags="$5"

  local cmd="uv --no-config pip compile pyproject.toml ${selectors} --python-version ${pyver} --python-platform linux"
  [[ -n "$flags" ]] && cmd+=" ${flags}"
  cmd+=" -o ${outfile}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "${YELLOW}[dry-run]${RESET} $cmd"
    return 0
  fi

  printf "${GREEN}[%s]${RESET} %s\n" "$name" "$cmd"
  eval "$cmd"
}

# ── Parse arguments ──────────────────────────────────────────────────
DRY_RUN=0
REQUESTED=()

for arg in "$@"; do
  case "$arg" in
    --help|-h)   usage; exit 0 ;;
    --list|-l)   list_services; exit 0 ;;
    --dry-run)   DRY_RUN=1 ;;
    -*)          echo "${RED}Unknown flag: $arg${RESET}"; usage; exit 1 ;;
    *)           REQUESTED+=("$arg") ;;
  esac
done

# ── Preflight checks ────────────────────────────────────────────────
if [[ ! -f pyproject.toml ]]; then
  echo "${RED}Error: pyproject.toml not found. Run this script from the repository root.${RESET}" >&2
  exit 1
fi

if ! command -v uv &>/dev/null; then
  echo "${RED}Error: 'uv' is not installed. Install it from https://docs.astral.sh/uv/${RESET}" >&2
  exit 1
fi

# ── Resolve which services to build ─────────────────────────────────
if [[ ${#REQUESTED[@]} -gt 0 ]]; then
  # Validate requested names
  for req in "${REQUESTED[@]}"; do
    found=0
    for entry in "${SERVICES[@]}"; do
      IFS='|' read -r name _ _ _ _ <<< "$entry"
      [[ "$name" == "$req" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
      echo "${RED}Unknown service: $req${RESET}"
      list_services
      exit 1
    fi
  done
fi

# ── Step 1: update root uv.lock ─────────────────────────────────────
should_run_root() {
  [[ ${#REQUESTED[@]} -eq 0 ]]  # only when updating everything
}

if should_run_root; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "${YELLOW}[dry-run]${RESET} uv lock"
  else
    printf "${GREEN}[root]${RESET} uv lock\n"
    uv lock
  fi
fi

# ── Step 2: compile Docker-service lock files ───────────────────────
failed=0
for entry in "${SERVICES[@]}"; do
  IFS='|' read -r name selectors pyver outfile flags <<< "$entry"

  # Skip if specific services were requested and this isn't one of them
  if [[ ${#REQUESTED[@]} -gt 0 ]]; then
    skip=1
    for req in "${REQUESTED[@]}"; do
      [[ "$name" == "$req" ]] && skip=0 && break
    done
    [[ $skip -eq 1 ]] && continue
  fi

  if ! compile_service "$name" "$selectors" "$pyver" "$outfile" "$flags"; then
    echo "${RED}  ✗ Failed: $name${RESET}" >&2
    failed=$((failed + 1))
  fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo
if [[ $failed -gt 0 ]]; then
  echo "${RED}Done with $failed failure(s).${RESET}"
  exit 1
else
  echo "${GREEN}All lock files updated successfully.${RESET}"
fi
