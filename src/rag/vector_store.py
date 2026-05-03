"""Vector store abstraction for Qdrant."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    NamedSparseVector,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

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
        host: str,
        port: int,
        collection_name: str,
    ):
        """Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_collection(
        self,
        dimension: int,
        retrieval_capability: str = "dense",
        force_recreate: bool = False,
    ):
        """Create a collection for storing vectors.

        Args:
            dimension: Dimension of embedding vectors
            retrieval_capability: ``"dense"`` or ``"hybrid"``; controls whether
                a sparse vector index is added alongside the dense index.
            force_recreate: If True, delete existing collection and create new one
        """
        # Use collection_exists() which correctly handles aliases
        exists = self.collection_exists()

        if exists and force_recreate:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating collection: {self.collection_name} with dimension: {dimension}")
            sparse_vectors_config = None
            if retrieval_capability == "hybrid":
                sparse_vectors_config = {"sparse": SparseVectorParams(index=SparseIndexParams())}
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=dimension, distance=Distance.COSINE)},
                sparse_vectors_config=sparse_vectors_config,
            )
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        sparse_vectors: Optional[List[SparseVector]] = None,
    ):
        """Add documents with their embeddings to the collection.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document (auto-generated if None)
            sparse_vectors: Optional sparse vectors for hybrid collections.
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
        for i, (doc_id, doc, embedding, metadata) in enumerate(
            zip(ids, documents, embeddings, metadatas)
        ):
            payload = {
                "content": doc,
                **metadata,
            }
            vec: Any = {"dense": embedding}
            if sparse_vectors is not None:
                vec["sparse"] = sparse_vectors[i]
            points.append(
                PointStruct(
                    id=doc_id,  # Must be int > 0 or UUID string
                    vector=vec,
                    payload=payload,
                )
            )

        batch_size = 500
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        logger.info(f"Added {len(points)} documents to {self.collection_name}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        filter_dict: Optional[Dict[str, Any]] = None,
        strategy: str = "dense",
        sparse_query: Optional[SparseVector] = None,
    ) -> List[Document]:
        """Search for similar documents.

        Automatically excludes collection metadata sentinel points
        (``type=collection_meta``).

        Args:
            query_embedding: Dense query vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            filter_dict: Optional metadata filters.
            strategy: ``"dense"`` (default) or ``"hybrid"``.
            sparse_query: Sparse query vector; required when ``strategy="hybrid"``.

        Returns:
            List of Document objects with content, metadata, and similarity scores
        """
        # Exclude metadata sentinel points from search results
        meta_exclusion = FieldCondition(
            key="type",
            match=MatchValue(value="collection_meta"),
        )

        if filter_dict:
            qf = Filter(**filter_dict)
            if qf.must_not is None:
                qf.must_not = [meta_exclusion]
            else:
                qf.must_not.append(meta_exclusion)
        else:
            qf = Filter(must_not=[meta_exclusion])

        if strategy == "hybrid":
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(query=query_embedding, using="dense", limit=top_k),
                    Prefetch(
                        query=NamedSparseVector(
                            name="sparse",
                            vector=sparse_query,  # type: ignore[arg-type]
                        ),
                        using="sparse",
                        limit=top_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.DBSF),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=qf,
            )
        else:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=qf,
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

    @staticmethod
    def _extract_vector_size(collection_info: Any) -> Optional[int]:
        """Extract the dense vector size from a Qdrant collection info object."""
        params = getattr(
            getattr(getattr(collection_info, "config", None), "params", None), "vectors", None
        )
        if params is None:
            return None

        if isinstance(params, dict):
            for vector_params in params.values():
                size = getattr(vector_params, "size", None)
                if isinstance(size, int):
                    return size
        return None

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.collection_exists():
            return {"exists": False}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "vector_size": self._extract_vector_size(info),
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

    def delete_alias(self, alias_name: str) -> None:
        """Delete an alias by name."""
        self.client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=alias_name),
                )
            ]
        )
        logger.info(f"Deleted alias: {alias_name}")

    def list_aliases(self) -> List[Dict[str, str]]:
        """List all aliases visible to the client."""
        return [
            {
                "alias_name": alias.alias_name,
                "collection_name": alias.collection_name,
            }
            for alias in self.client.get_aliases().aliases
        ]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by explicit name."""
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    # ------------------------------------------------------------------
    # Collection metadata helpers
    # ------------------------------------------------------------------

    # Deterministic UUID for the metadata sentinel point.
    # Qdrant requires point IDs to be integers or valid UUID strings.
    # We use UUID5 with a fixed namespace so the ID is stable across runs.
    _META_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")
    _META_ID = str(uuid.uuid5(_META_NS, "_meta"))

    def write_meta(self, payload: Dict[str, Any], dimension: int) -> None:
        """Write a metadata sentinel point to the collection.

        The sentinel uses a deterministic UUID as its ID and a zero-filled
        dummy vector so that it does not influence similarity search (the
        ``search()`` method automatically excludes points with
        ``type=collection_meta``).

        Args:
            payload: Arbitrary metadata dict (build_config, kb_name, etc.).
            dimension: Embedding vector dimension (needed for the dummy vector).
        """
        meta_payload = {"type": "collection_meta", **payload}
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self._META_ID,
                    vector=[0.0] * dimension,
                    payload=meta_payload,
                )
            ],
        )
        logger.info(f"Wrote _meta point to '{self.collection_name}'")

    def read_meta(self) -> Optional[Dict[str, Any]]:
        """Read the metadata sentinel point from the collection.

        Returns:
            The payload dict of the ``_meta`` point, or ``None`` if it
            does not exist.
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._META_ID],
                with_payload=True,
            )
            if points:
                return points[0].payload
        except (
            KeyError,
            ValueError,
            RuntimeError,
        ):
            # Collection may not exist or _meta point may be absent
            logger.debug(f"No _meta point in '{self.collection_name}'", exc_info=True)
        return None
