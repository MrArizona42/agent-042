"""Production RAG operations.

This package contains the production-safe create, update, inspect, and alias
management workflows used by notebooks and Airflow DAGs.
"""

from rag.ops.aliases import assign_alias_to_collection, detach_alias, promote_alias
from rag.ops.create import create_arxiv_collection, create_pytorch_docs_collection
from rag.ops.inspect import inspect_alias, inspect_collection, list_alias_mappings
from rag.ops.meta import (
    BuildConfig,
    CollectionMeta,
    ImplementationInfo,
    build_collection_meta,
    read_build_config_for_alias,
    read_collection_meta,
    write_collection_meta,
)
from rag.ops.update import update_arxiv_collection, update_pytorch_docs_collection

__all__ = [
    "BuildConfig",
    "CollectionMeta",
    "ImplementationInfo",
    "assign_alias_to_collection",
    "build_collection_meta",
    "create_arxiv_collection",
    "create_pytorch_docs_collection",
    "detach_alias",
    "inspect_alias",
    "inspect_collection",
    "list_alias_mappings",
    "promote_alias",
    "read_build_config_for_alias",
    "read_collection_meta",
    "update_arxiv_collection",
    "update_pytorch_docs_collection",
    "write_collection_meta",
]
