"""Sparse vector encoder (BM25 / SPLADE). Not yet implemented."""

from __future__ import annotations

from typing import Any


class SparseEncoderService:
    """Sparse vector encoder (BM25 / SPLADE)."""

    def encode(self, texts: list[str]) -> list[Any]:
        raise NotImplementedError
