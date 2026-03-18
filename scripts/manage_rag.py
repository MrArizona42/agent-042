"""RAG collection management CLI.

Provides commands for promoting aliases, listing aliases and their
target collections, and inspecting collection metadata.

Usage::

    # Promote challenger → champion
    python -m scripts.manage_rag promote --kb pytorch_docs --from challenger --to champion

    # List all aliases and their target collections
    python -m scripts.manage_rag list

    # Inspect collection metadata
    python -m scripts.manage_rag inspect --kb pytorch_docs --alias champion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path so shared/rag modules are importable when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.vector_store import QdrantVectorStore

# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> None:
    """List all Qdrant aliases and the collections they point to."""
    vs = QdrantVectorStore(host=args.qdrant_host, port=args.qdrant_port, collection_name="_dummy")
    aliases = vs.client.get_aliases().aliases

    if not aliases:
        print("No aliases found.")
        return

    print(f"{'Alias':<40} {'Collection':<50}")
    print("-" * 90)
    for a in sorted(aliases, key=lambda x: x.alias_name):
        print(f"{a.alias_name:<40} {a.collection_name:<50}")


def _cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect the _meta point of a resolved alias."""
    qdrant_alias = f"{args.kb}_{args.alias}"
    vs = QdrantVectorStore(
        host=args.qdrant_host, port=args.qdrant_port, collection_name=qdrant_alias,
    )

    if not vs.collection_exists():
        print(f"Error: alias '{qdrant_alias}' does not resolve to any collection.")
        sys.exit(1)

    meta = vs.read_meta()
    if meta is None:
        print(f"No _meta point found in collection behind '{qdrant_alias}'.")
    else:
        print(json.dumps(meta, indent=2, default=str))


def _cmd_promote(args: argparse.Namespace) -> None:
    """Re-point *--to* alias to the collection currently behind *--from* alias."""
    src_alias = f"{args.kb}_{args.from_alias}"
    dst_alias = f"{args.kb}_{args.to_alias}"

    vs = QdrantVectorStore(
        host=args.qdrant_host, port=args.qdrant_port, collection_name=src_alias,
    )

    src_collection = vs.resolve_alias(src_alias)
    if src_collection is None:
        print(f"Error: source alias '{src_alias}' does not resolve.")
        sys.exit(1)

    print(f"Source: {src_alias} → {src_collection}")
    print(f"Target: {dst_alias} → {src_collection}")

    vs.update_alias(dst_alias, src_collection)
    print(f"✅ Promoted: '{dst_alias}' now points to '{src_collection}'")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage RAG Qdrant aliases and collections",
    )
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")

    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all aliases")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect _meta of a collection")
    p_inspect.add_argument("--kb", required=True, help="Knowledge base name")
    p_inspect.add_argument("--alias", default="champion", help="Alias role")

    # promote
    p_promote = sub.add_parser("promote", help="Promote one alias to another")
    p_promote.add_argument("--kb", required=True, help="Knowledge base name")
    p_promote.add_argument("--from", dest="from_alias", required=True, help="Source alias role")
    p_promote.add_argument("--to", dest="to_alias", required=True, help="Target alias role")

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "promote":
        _cmd_promote(args)


if __name__ == "__main__":
    main()
