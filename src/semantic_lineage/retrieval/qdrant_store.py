"""Qdrant persistence and reconciliation for derived semantic catalog documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from semantic_lineage.catalog.documents import SemanticDocument
from semantic_lineage.retrieval.embeddings import EmbeddingProvider


class QdrantServiceError(RuntimeError):
    """Raised when the configured Qdrant service or collection cannot be used."""


@dataclass(frozen=True, slots=True)
class IndexingResult:
    collection_name: str
    documents_indexed: int
    documents_deleted: int
    collection_point_count: int


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    collection_name: str
    vector_dimensions: int
    point_count: int
    qdrant_version: str


@dataclass(frozen=True, slots=True)
class StoredSearchResult:
    document_id: str
    document_type: str
    content: str
    payload: dict[str, Any]
    score: float


class QdrantSemanticStore:
    """Indexes canonical documents and removes stale points from its named collection only."""

    def __init__(
        self,
        client: QdrantClient,
        embedding_provider: EmbeddingProvider,
        collection_name: str,
    ) -> None:
        self._client = client
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name

    def ensure_collection(self) -> None:
        """Create the collection once; existing collections are left intact for idempotent upserts."""
        try:
            if not self._client.collection_exists(self._collection_name):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self._embedding_provider.dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise QdrantServiceError(
                "could not create or inspect Qdrant collection "
                f"'{self._collection_name}'. Start Qdrant with `docker compose up -d`."
            ) from exc

    def index_documents(
        self, documents: list[SemanticDocument], batch_size: int = 64
    ) -> IndexingResult:
        """Upsert desired documents and delete only stale IDs from this collection."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.ensure_collection()
        desired_ids = {_point_id(document.id) for document in documents}
        existing_ids = self.point_ids()

        if documents:
            vectors = self._embedding_provider.embed_documents(
                [document.content for document in documents]
            )
            if len(vectors) != len(documents):
                raise ValueError(
                    "embedding provider returned a different number of vectors than documents"
                )
            points = [
                qmodels.PointStruct(
                    id=_point_id(document.id),
                    vector=vector,
                    payload={
                        **document.payload,
                        "content": document.content,
                        "document_id": document.id,
                        "document_type": document.document_type,
                    },
                )
                for document, vector in zip(documents, vectors, strict=True)
            ]
            try:
                for start in range(0, len(points), batch_size):
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=points[start : start + batch_size],
                        wait=True,
                    )
            except Exception as exc:
                raise QdrantServiceError(
                    f"could not index documents into Qdrant collection '{self._collection_name}'."
                ) from exc

        stale_ids = existing_ids - desired_ids
        self._delete_point_ids(stale_ids, batch_size)
        return IndexingResult(
            collection_name=self._collection_name,
            documents_indexed=len(documents),
            documents_deleted=len(stale_ids),
            collection_point_count=self.point_count(),
        )

    def point_ids(self, batch_size: int = 256) -> set[str]:
        """Return all point IDs in the named collection using paginated Qdrant scrolling."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        point_ids: set[str] = set()
        offset: int | str | None = None
        try:
            while True:
                records, offset = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                point_ids.update(str(record.id) for record in records)
                if offset is None:
                    return point_ids
        except Exception as exc:
            raise QdrantServiceError(
                f"could not list points in Qdrant collection '{self._collection_name}'."
            ) from exc

    def _delete_point_ids(self, point_ids: set[str], batch_size: int) -> None:
        """Delete explicit stale IDs only; never delete a collection or use a broad filter."""
        if not point_ids:
            return
        try:
            sorted_ids = sorted(point_ids)
            for start in range(0, len(sorted_ids), batch_size):
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=qmodels.PointIdsList(
                        points=sorted_ids[start : start + batch_size]
                    ),
                    wait=True,
                )
        except Exception as exc:
            raise QdrantServiceError(
                f"could not delete stale documents from Qdrant collection '{self._collection_name}'."
            ) from exc

    def collection_summary(self) -> CollectionSummary:
        """Inspect collection compatibility before an evaluation sends embedding API requests."""
        try:
            if not self._client.collection_exists(self._collection_name):
                raise QdrantServiceError(
                    f"Qdrant collection '{self._collection_name}' does not exist. "
                    "Index the catalog into this collection before evaluation."
                )
            collection = self._client.get_collection(self._collection_name)
            vectors = collection.config.params.vectors
            if not isinstance(vectors, qmodels.VectorParams):
                raise QdrantServiceError(
                    f"Qdrant collection '{self._collection_name}' does not use a single vector configuration."
                )
            version = str(self._client.info().version)
            point_count = int(self._client.count(self._collection_name, exact=True).count)
            return CollectionSummary(
                collection_name=self._collection_name,
                vector_dimensions=vectors.size,
                point_count=point_count,
                qdrant_version=version,
            )
        except QdrantServiceError:
            raise
        except Exception as exc:
            raise QdrantServiceError(
                f"could not inspect Qdrant collection '{self._collection_name}'. "
                "Confirm Qdrant is running and reachable."
            ) from exc

    def point_count(self) -> int:
        """Return the current collection point count for validation and E2E checks."""
        return self.collection_summary().point_count

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        document_types: list[str] | None = None,
        domains: list[str] | None = None,
        asset_ids: list[str] | None = None,
    ) -> list[StoredSearchResult]:
        """Search semantic documents with optional filterable catalog payload fields."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_filter = _build_filter(document_types, domains, asset_ids)
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=self._embedding_provider.embed_query(query),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise QdrantServiceError(
                "could not search Qdrant collection "
                f"'{self._collection_name}'. Index the catalog and confirm Qdrant is running."
            ) from exc
        results: list[StoredSearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            results.append(
                StoredSearchResult(
                    document_id=str(payload.pop("document_id")),
                    document_type=str(payload.pop("document_type")),
                    content=str(payload.pop("content")),
                    payload=payload,
                    score=float(point.score),
                )
            )
        return results


def create_qdrant_client(url: str) -> QdrantClient:
    """Create the normal networked Qdrant client from a configured local service URL."""
    return QdrantClient(url=url)


def assert_qdrant_ready(client: QdrantClient) -> None:
    """Fail with an actionable message when the configured Qdrant service is unavailable."""
    try:
        client.get_collections()
    except Exception as exc:
        raise QdrantServiceError(
            "Qdrant is unavailable. Start it with `docker compose up -d` and verify "
            "SEMANTIC_LINEAGE_QDRANT_URL."
        ) from exc


def _point_id(document_id: str) -> str:
    """Return the stable Qdrant UUID for one canonical semantic document ID."""
    return str(uuid5(NAMESPACE_URL, document_id))


def _build_filter(
    document_types: list[str] | None,
    domains: list[str] | None,
    asset_ids: list[str] | None,
) -> qmodels.Filter | None:
    must: list[qmodels.FieldCondition] = []
    for key, values in (
        ("document_type", document_types),
        ("domain", domains),
        ("asset_id", asset_ids),
    ):
        if values:
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=values)))
    return qmodels.Filter(must=must) if must else None
