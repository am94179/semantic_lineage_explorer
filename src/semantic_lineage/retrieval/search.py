"""Application-facing semantic search function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_lineage.retrieval.qdrant_store import QdrantSemanticStore


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    document_id: str
    document_type: str
    content: str
    payload: dict[str, Any]
    score: float


def semantic_search(
    store: QdrantSemanticStore,
    query: str,
    *,
    limit: int = 5,
    document_types: list[str] | None = None,
    domains: list[str] | None = None,
    asset_ids: list[str] | None = None,
) -> list[SemanticSearchResult]:
    """Return typed application results from the Qdrant semantic index."""
    return [
        SemanticSearchResult(
            document_id=result.document_id,
            document_type=result.document_type,
            content=result.content,
            payload=result.payload,
            score=result.score,
        )
        for result in store.search(
            query,
            limit=limit,
            document_types=document_types,
            domains=domains,
            asset_ids=asset_ids,
        )
    ]
