"""Reproducible, retrieval-only benchmark loading, metrics, and run provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from semantic_lineage.retrieval.search import SemanticSearchResult


class EvaluationLoadError(ValueError):
    """Raised when a retrieval benchmark file is not a valid evaluation contract."""


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    id: str
    query: str
    relevant_document_ids: list[str]
    description: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    query: str
    relevant_document_ids: list[str]
    retrieved_document_ids: list[str]
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    case_count: int
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float
    cases: list[CaseResult]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report with stable field names."""
        return {
            "case_count": self.case_count,
            "metrics": {
                "recall_at_5": self.recall_at_5,
                "recall_at_10": self.recall_at_10,
                "mean_reciprocal_rank": self.mean_reciprocal_rank,
            },
            "cases": [asdict(case) for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    run_type: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    catalog_path: str
    catalog_sha256: str
    catalog_version: str
    benchmark_path: str
    benchmark_sha256: str
    collection: str
    document_count: int
    qdrant_version: str
    qdrant_url: str
    started_at_utc: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """Load a small, reviewable YAML retrieval benchmark."""
    source = Path(path)
    try:
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationLoadError(f"could not load evaluation file '{source}': {exc}") from exc
    if not isinstance(parsed, dict) or parsed.get("version") != 1:
        raise EvaluationLoadError("evaluation file must contain version: 1")
    raw_cases = parsed.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationLoadError("evaluation file must contain at least one case")
    cases: list[RetrievalCase] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise EvaluationLoadError(f"case {index} must be a mapping")
        case_id = raw_case.get("id")
        query = raw_case.get("query")
        relevant_ids = raw_case.get("relevant_document_ids")
        description = raw_case.get("description", "")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluationLoadError(f"case {index} has an invalid id")
        if case_id in case_ids:
            raise EvaluationLoadError(f"duplicate evaluation case id: {case_id}")
        if not isinstance(query, str) or not query.strip():
            raise EvaluationLoadError(f"case '{case_id}' has an invalid query")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or not all(isinstance(item, str) and item.strip() for item in relevant_ids)
        ):
            raise EvaluationLoadError(f"case '{case_id}' needs non-empty relevant_document_ids")
        if not isinstance(description, str):
            raise EvaluationLoadError(f"case '{case_id}' has an invalid description")
        case_ids.add(case_id)
        cases.append(RetrievalCase(case_id, query, relevant_ids, description))
    return cases


def evaluate_retrieval(
    cases: list[RetrievalCase], search: Callable[[str, int], list[SemanticSearchResult]]
) -> RetrievalEvaluationReport:
    """Calculate macro recall@5, recall@10, and MRR from retrieval results only."""
    if not cases:
        raise ValueError("at least one evaluation case is required")
    results = [_evaluate_case(case, search(case.query, 10)) for case in cases]
    case_count = len(results)
    return RetrievalEvaluationReport(
        case_count=case_count,
        recall_at_5=sum(result.recall_at_5 for result in results) / case_count,
        recall_at_10=sum(result.recall_at_10 for result in results) / case_count,
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / case_count,
        cases=results,
    )


def build_evaluation_payload(
    report: RetrievalEvaluationReport, provenance: EvaluationProvenance
) -> dict[str, object]:
    """Combine stable retrieval metrics and reproducibility metadata for JSON output."""
    return {"provenance": provenance.as_dict(), **report.as_dict()}


def file_sha256(path: str | Path) -> str:
    """Return a stable content hash for an evaluation input file."""
    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationLoadError(f"could not hash evaluation input '{source}': {exc}") from exc


def utc_timestamp() -> str:
    """Return the current UTC timestamp in an unambiguous JSON-safe format."""
    return datetime.now(UTC).isoformat()


def _evaluate_case(case: RetrievalCase, results: list[SemanticSearchResult]) -> CaseResult:
    retrieved_ids = [result.document_id for result in results]
    relevant = set(case.relevant_document_ids)
    return CaseResult(
        case_id=case.id,
        query=case.query,
        relevant_document_ids=case.relevant_document_ids,
        retrieved_document_ids=retrieved_ids,
        recall_at_5=_recall_at_k(retrieved_ids, relevant, 5),
        recall_at_10=_recall_at_k(retrieved_ids, relevant, 10),
        reciprocal_rank=_reciprocal_rank(retrieved_ids, relevant),
    )


def _recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def _reciprocal_rank(retrieved_ids: list[str], relevant: set[str]) -> float:
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def validate_collection_compatibility(
    vector_dimensions: int,
    point_count: int,
    provider_dimensions: int,
    expected_document_count: int,
) -> None:
    """Reject a missing re-index or embedding-model mismatch before benchmark queries run."""
    if vector_dimensions != provider_dimensions:
        raise EvaluationLoadError(
            "Qdrant collection vector dimensions "
            f"({vector_dimensions}) do not match the active embedding provider ({provider_dimensions}). "
            "Re-index into a dedicated collection with the active embedding model."
        )
    if point_count != expected_document_count:
        raise EvaluationLoadError(
            f"Qdrant collection contains {point_count} documents, but the current catalog generates "
            f"{expected_document_count}. Re-index this collection before evaluation."
        )
