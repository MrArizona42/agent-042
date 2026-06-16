"""Adapters between project RAG contracts and LlamaIndex core objects."""

from __future__ import annotations

from llama_index.core import Document
from llama_index.core.schema import TextNode

from rag.contracts.models import Chunk, ExtractedDocument


def extracted_document_to_llama_document(document: ExtractedDocument) -> Document:
    """Convert an extracted document to a LlamaIndex document."""
    return Document(
        text=document.text,
        doc_id=document.id,
        metadata={
            "document_id": document.id,
            "source_document_id": document.source_document_id,
            "extraction_method": document.extraction_method,
            "extraction_warnings": list(document.extraction_warnings),
            **document.metadata,
        },
    )


def chunk_to_text_node(chunk: Chunk) -> TextNode:
    """Convert a project chunk to a LlamaIndex text node."""
    return TextNode(
        id_=chunk.id,
        text=chunk.text,
        metadata={
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "source_document_id": chunk.source_document_id,
            "section_title": chunk.section_title,
            "ordinal": chunk.ordinal,
            "token_count": chunk.token_count,
            **chunk.metadata,
        },
    )


def extracted_documents_to_llama_documents(
    documents: list[ExtractedDocument],
) -> list[Document]:
    """Convert extracted documents to LlamaIndex documents."""
    return [extracted_document_to_llama_document(document) for document in documents]


def chunks_to_text_nodes(chunks: list[Chunk]) -> list[TextNode]:
    """Convert project chunks to LlamaIndex text nodes."""
    return [chunk_to_text_node(chunk) for chunk in chunks]
