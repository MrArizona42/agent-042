"""Compatibility exports for LlamaIndex contract adapters."""

from rag.contracts.llamaindex import (
    chunk_to_text_node,
    chunks_to_text_nodes,
    extracted_document_to_llama_document,
    extracted_documents_to_llama_documents,
)

__all__ = [
    "chunk_to_text_node",
    "chunks_to_text_nodes",
    "extracted_document_to_llama_document",
    "extracted_documents_to_llama_documents",
]
