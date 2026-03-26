"""RAG collection management CLI (Fire).

Provides commands for promoting aliases, listing aliases and their
target collections, and inspecting collection metadata.

Usage::

    python -m scripts.manage_rag promote \
        --kb pytorch_docs --from_alias challenger --to_alias champion
    python -m scripts.manage_rag list
    python -m scripts.manage_rag inspect --kb pytorch_docs --alias champion
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

# Add src to path so shared/rag modules are importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.vector_store import QdrantVectorStore
from shared.config import get_settings

# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def _resolve_qdrant(qdrant_host, qdrant_port):
    """Fill *None* values from shared settings."""
    s = get_settings()
    return qdrant_host or s.qdrant_host, qdrant_port or s.qdrant_port


def cmd_list(qdrant_host: str | None = None, qdrant_port: int | None = None) -> None:
    """List all Qdrant aliases and the collections they point to."""
    host, port = _resolve_qdrant(qdrant_host, qdrant_port)
    vs = QdrantVectorStore(host=host, port=port, collection_name="_dummy")
    aliases = vs.client.get_aliases().aliases

    if not aliases:
        print("No aliases found.")
        return

    print(f"{'Alias':<40} {'Collection':<50}")
    print("-" * 90)
    for a in sorted(aliases, key=lambda x: x.alias_name):
        print(f"{a.alias_name:<40} {a.collection_name:<50}")


def cmd_inspect(
    kb: str,
    alias: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> None:
    """Inspect the _meta point of a resolved alias."""
    host, port = _resolve_qdrant(qdrant_host, qdrant_port)
    qdrant_alias = f"{kb}_{alias}"
    vs = QdrantVectorStore(
        host=host,
        port=port,
        collection_name=qdrant_alias,
    )

    if not vs.collection_exists():
        print(f"Error: alias '{qdrant_alias}' does not resolve to any collection.")
        sys.exit(1)

    meta = vs.read_meta()
    if meta is None:
        print(f"No _meta point found in collection behind '{qdrant_alias}'.")
    else:
        print(json.dumps(meta, indent=2, default=str))


def cmd_promote(
    kb: str,
    from_alias: str,
    to_alias: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> None:
    """Re-point *to_alias* to the collection currently behind *from_alias*."""
    host, port = _resolve_qdrant(qdrant_host, qdrant_port)
    src_alias = f"{kb}_{from_alias}"
    dst_alias = f"{kb}_{to_alias}"

    vs = QdrantVectorStore(
        host=host,
        port=port,
        collection_name=src_alias,
    )

    src_collection = vs.resolve_alias(src_alias)
    if src_collection is None:
        print(f"Error: source alias '{src_alias}' does not resolve.")
        sys.exit(1)

    print(f"Source: {src_alias} -> {src_collection}")
    print(f"Target: {dst_alias} -> {src_collection}")

    vs.update_alias(dst_alias, src_collection)
    print(f"Promoted: '{dst_alias}' now points to '{src_collection}'")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


def main() -> None:
    fire.Fire(
        {
            "list": cmd_list,
            "inspect": cmd_inspect,
            "promote": cmd_promote,
        }
    )


if __name__ == "__main__":
    main()
