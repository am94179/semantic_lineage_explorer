from __future__ import annotations

from pathlib import Path

import pytest

from semantic_lineage.evaluation import (
    EvaluationLoadError,
    RetrievalCase,
    evaluate_retrieval,
    load_retrieval_cases,
)
from semantic_lineage.retrieval.search import SemanticSearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def result(document_id: str) -> SemanticSearchResult:
    return SemanticSearchResult(document_id, "asset", "", {}, 1.0)


def test_metric_calculation_matches_hand_computed_fixture() -> None:
    cases = [
        RetrievalCase("one", "first", ["a"], ""),
        RetrievalCase("two", "second", ["b", "c"], ""),
        RetrievalCase("three", "third", ["d"], ""),
    ]
    rankings = {
        "first": [result("a")],
        "second": [result("x"), result("b")],
        "third": [result("x"), result("y")],
    }

    report = evaluate_retrieval(cases, lambda query, limit: rankings[query][:limit])

    assert report.recall_at_5 == pytest.approx(0.5)
    assert report.recall_at_10 == pytest.approx(0.5)
    assert report.mean_reciprocal_rank == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.cases[1].recall_at_5 == 0.5


def test_ci_benchmark_is_small_and_valid() -> None:
    cases = load_retrieval_cases(PROJECT_ROOT / "evaluation" / "retrieval_cases_ci.yaml")

    assert len(cases) == 3
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.relevant_document_ids for case in cases)


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "version: 1\ncases:\n  - id: repeat\n    query: one\n    relevant_document_ids: [asset:a]\n  - id: repeat\n    query: two\n    relevant_document_ids: [asset:b]\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationLoadError, match="duplicate"):
        load_retrieval_cases(path)


def test_evaluation_payload_records_reproducible_provenance(tmp_path: Path) -> None:
    from semantic_lineage.evaluation import (
        EvaluationProvenance,
        build_evaluation_payload,
        file_sha256,
    )

    catalog = tmp_path / "catalog.yaml"
    benchmark = tmp_path / "cases.yaml"
    catalog.write_text("assets: []\n", encoding="utf-8")
    benchmark.write_text("version: 1\ncases: []\n", encoding="utf-8")
    report = evaluate_retrieval(
        [RetrievalCase("one", "first", ["a"], "")], lambda query, limit: [result("a")]
    )
    provenance = EvaluationProvenance(
        run_type="real_embedding_baseline",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        catalog_path=str(catalog),
        catalog_sha256=file_sha256(catalog),
        catalog_version="unversioned",
        benchmark_path=str(benchmark),
        benchmark_sha256=file_sha256(benchmark),
        collection="semantic_catalog_openai_baseline",
        document_count=47,
        qdrant_version="1.19.1",
        qdrant_url="http://localhost:6333",
        started_at_utc="2026-09-23T00:00:00+00:00",
        elapsed_seconds=0.125,
    )

    payload = build_evaluation_payload(report, provenance)

    assert payload["provenance"] == {
        "run_type": "real_embedding_baseline",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "catalog_path": str(catalog),
        "catalog_sha256": file_sha256(catalog),
        "catalog_version": "unversioned",
        "benchmark_path": str(benchmark),
        "benchmark_sha256": file_sha256(benchmark),
        "collection": "semantic_catalog_openai_baseline",
        "document_count": 47,
        "qdrant_version": "1.19.1",
        "qdrant_url": "http://localhost:6333",
        "started_at_utc": "2026-09-23T00:00:00+00:00",
        "elapsed_seconds": 0.125,
    }
    assert payload["metrics"]["recall_at_5"] == 1.0


@pytest.mark.parametrize(
    ("vector_dimensions", "point_count", "message"),
    [
        (64, 47, "vector dimensions"),
        (1536, 46, "current catalog generates"),
    ],
)
def test_collection_compatibility_rejects_mismatched_index(
    vector_dimensions: int, point_count: int, message: str
) -> None:
    from semantic_lineage.evaluation import validate_collection_compatibility

    with pytest.raises(EvaluationLoadError, match=message):
        validate_collection_compatibility(vector_dimensions, point_count, 1536, 47)
