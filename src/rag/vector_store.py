"""Vector store abstraction for Qdrant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    Filter,
    PointStruct,
    VectorParams,
)

from shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with content and metadata."""

    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QdrantVectorStore:
    """Qdrant vector database client for storing and retrieving documents."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "documents",
    ):
        """Initialize Qdrant client.

        Args:
            host: Qdrant server host (uses config default if None)
            port: Qdrant server port (uses config default if None)
            collection_name: Name of the collection to use
        """
        settings = get_settings()
        host = host if host is not None else settings.qdrant_host
        port = port if port is not None else settings.qdrant_port

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_collection(self, dimension: int, force_recreate: bool = False):
        """Create a collection for storing vectors.

        Args:
            dimension: Dimension of embedding vectors
            force_recreate: If True, delete existing collection and create new one
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and force_recreate:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating collection: {self.collection_name} with dimension: {dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Add documents with their embeddings to the collection.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document (auto-generated if None)
        """
        if not documents:
            return

        if metadatas is None:
            metadatas = [{} for _ in documents]

        if ids is None:
            # Generate sequential integer IDs starting from current collection size
            # Qdrant requires positive integers (>0) or UUIDs as IDs
            collection_info = self.client.get_collection(self.collection_name)
            start_id = collection_info.points_count
            # Ensure IDs start from 1, not 0
            ids = [start_id + i + 1 for i in range(len(documents))]

        points = []
        for doc_id, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            payload = {
                "content": doc,
                **metadata,
            }
            points.append(
                PointStruct(
                    id=doc_id,  # Must be int > 0 or UUID string
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Added {len(points)} documents to {self.collection_name}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Optional metadata filters

        Returns:
            List of Document objects with content, metadata, and similarity scores
        """
        # search_result = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=query_embedding,
        #     limit=top_k,
        #     score_threshold=score_threshold,
        #     query_filter=filter_dict,
        # )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=Filter(**filter_dict) if filter_dict else None,
        )

        documents = []
        for hit in search_result.points:
            payload = hit.payload
            content = payload.pop("content", "")
            documents.append(
                Document(
                    content=content,
                    metadata=payload,
                    score=hit.score,
                )
            )

        logger.info(f"Found {len(documents)} documents with scores above {score_threshold}")
        return documents

    def collection_exists(self) -> bool:
        """Check if collection exists (supports aliases)."""
        return self.client.collection_exists(self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.collection_exists():
            return {"exists": False}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            # vectors_count doesn't exist in newer Qdrant versions
            # points_count is the number of vectors/documents
        }

    # ------------------------------------------------------------------
    # Alias helpers (used by index-build scripts for zero-downtime swaps)
    # ------------------------------------------------------------------

    def resolve_alias(self, alias_name: str) -> Optional[str]:
        """Resolve an alias to its target collection name.

        Returns:
            The collection name the alias points to, or None if the alias
            does not exist.
        """
        for alias in self.client.get_aliases().aliases:
            if alias.alias_name == alias_name:
                return alias.collection_name
        return None

    def update_alias(self, alias_name: str, collection_name: str) -> None:
        """Create or atomically update an alias to point to a collection.

        If the alias already exists it is re-pointed in a single atomic
        operation, so live queries see either the old *or* the new collection
        — never an empty gap.
        """
        self.client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            ]
        )
        logger.info(f"Alias '{alias_name}' now points to collection '{collection_name}'")

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by explicit name."""
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
