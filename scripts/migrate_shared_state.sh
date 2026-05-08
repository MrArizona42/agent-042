#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/migrate_shared_state.sh SOURCE_ROOT TARGET_ROOT

Copies the Phase 2 shared-state directories from an existing checkout-backed
deployment tree into the external server roots expected by the new contract.

Examples:
    bash scripts/migrate_shared_state.sh /home/anton-m/agent-042/current /home/anton-m/agent-042
    bash scripts/migrate_shared_state.sh "$PWD" /home/anton-m/agent-042
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 1
fi

SOURCE_ROOT="$(cd "$1" && pwd)"
TARGET_ROOT="$2"

copy_tree() {
    local source_dir="$1"
    local target_dir="$2"

    mkdir -p "$target_dir"
    if [[ ! -d "$source_dir" ]]; then
        echo "Skipping missing directory: $source_dir"
        return
    fi

    echo "Syncing $source_dir -> $target_dir"
    cp -a "$source_dir"/. "$target_dir"/
}

copy_file_if_missing() {
    local source_file="$1"
    local target_file="$2"

    if [[ ! -f "$source_file" ]]; then
        echo "Skipping missing file: $source_file"
        return
    fi

    mkdir -p "$(dirname "$target_file")"
    if [[ -f "$target_file" ]]; then
        echo "Keeping existing file: $target_file"
        return
    fi

    echo "Copying $source_file -> $target_file"
    cp -a "$source_file" "$target_file"
}

echo "Preparing target root: $TARGET_ROOT"
mkdir -p "$TARGET_ROOT/assets" "$TARGET_ROOT/artifacts" "$TARGET_ROOT/.dvc"

copy_tree "$SOURCE_ROOT/assets/models" "$TARGET_ROOT/assets/models"
copy_tree "$SOURCE_ROOT/assets/adapters" "$TARGET_ROOT/assets/adapters"
copy_tree "$SOURCE_ROOT/assets/datasets" "$TARGET_ROOT/assets/datasets"
copy_tree "$SOURCE_ROOT/assets/rag_data" "$TARGET_ROOT/assets/rag_data"
copy_tree "$SOURCE_ROOT/artifacts/training" "$TARGET_ROOT/artifacts/training"
copy_file_if_missing "$SOURCE_ROOT/.dvc/config.local" "$TARGET_ROOT/.dvc/config.local"

echo "Phase 2 shared-state migration complete."
