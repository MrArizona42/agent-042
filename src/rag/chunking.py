"""Document chunking strategies for RAG."""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import get_settings


class BaseChunker:
    """Base class for document chunking."""

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        raise NotImplementedError


class FixedTokenChunker(BaseChunker):
    """Simple fixed-size chunking with overlap.

    Good baseline for most documents.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target size of each chunk in characters (uses config default if None)
            chunk_overlap: Number of overlapping characters between chunks
            (uses config default if None)
        """
        settings = get_settings()
        chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        return self.splitter.split_text(text)


class CodeChunker(BaseChunker):
    """Chunking strategy optimized for code.

    Tries to keep functions and classes together.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize code chunker.

        Args:
            chunk_size: Target size of each chunk (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.code_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.code_chunk_overlap
        )

    def chunk(self, text: str) -> List[str]:
        """Split code into chunks, preserving logical boundaries."""
        # Try to split by functions/classes first
        patterns = [
            r"\n(?=def\s+\w+)",  # Python functions
            r"\n(?=class\s+\w+)",  # Python classes
            r"\n(?=async\s+def)",  # Async functions
            r"\n\n",  # Double newlines
            r"\n",  # Single newlines
        ]

        chunks = [text]
        for pattern in patterns:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    # Split by pattern
                    parts = re.split(pattern, chunk)
                    # Rejoin with separator
                    splits = []
                    current = ""
                    for part in parts:
                        if len(current) + len(part) <= self.chunk_size:
                            current += part
                        else:
                            if current:
                                splits.append(current)
                            current = part
                    if current:
                        splits.append(current)
                    new_chunks.extend(splits if splits else [chunk])
            chunks = new_chunks

        # Final cleanup: remove empty chunks and strip
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks


class SectionAwareChunker(BaseChunker):
    """Chunking that respects document sections (for papers/docs).

    Tries to split by markdown headers and sections.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize section-aware chunker.

        Args:
            chunk_size: Target chunk size (uses config default if None)
            chunk_overlap: Overlap size (uses config default if None)
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.section_chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.section_chunk_overlap
        )

    def chunk(self, text: str) -> List[str]:
        """Split text by sections while respecting size limits."""
        # Split by markdown headers
        sections = re.split(r"\n(?=#{1,6}\s)", text)

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Section too large, use recursive splitting
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks.extend(splitter.split_text(section))

        return [c.strip() for c in chunks if c.strip()]


def get_chunker(task: str = "chat", **kwargs) -> BaseChunker:
    """Factory function to get appropriate chunker for task.

    Args:
        task: Task type (chat, code, summarize)
        **kwargs: Additional arguments passed to chunker

    Returns:
        Appropriate chunker instance
    """
    if task == "code":
        return CodeChunker(**kwargs)
    elif task == "summarize":
        return SectionAwareChunker(**kwargs)
    else:
        return FixedTokenChunker(**kwargs)
