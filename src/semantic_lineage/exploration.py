"""Deterministic semantic discovery and lineage responses built on catalog evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from semantic_lineage.catalog.lineage import get_upstream_lineage
from semantic_lineage.models import Catalog, DataAsset, LineageEdge, Metric, Relationship
from semantic_lineage.retrieval.qdrant_store import QdrantSemanticStore
from semantic_lineage.retrieval.search import SemanticSearchResult, semantic_search


class ExplorationError(ValueError):
    """Raised when a requested catalog target cannot be resolved deterministically."""


@dataclass(frozen=True, slots=True)
class AssetSummary:
    id: str
    qualified_name: str
    description: str
    grain: str


@dataclass(frozen=True, slots=True)
class DiscoveryResponse:
    question: str
    assets: list[AssetSummary]
    relationships: list[Relationship]
    evidence: list[SemanticSearchResult]


@dataclass(frozen=True, slots=True)
class LineageResponse:
    request: str
    target_column_id: str
    target_metric: Metric | None
    upstream_edges: list[LineageEdge]
    assets: list[AssetSummary]
    relationships: list[Relationship]
    evidence: list[SemanticSearchResult]


class CatalogExplorer:
    """Combines semantic retrieval with explicit catalog and lineage facts."""

    def __init__(self, catalog: Catalog, store: QdrantSemanticStore) -> None:
        self._catalog = catalog
        self._store = store
        self._assets_by_id = {asset.id: asset for asset in catalog.assets}
        self._columns_by_id = {column.id: column for column in catalog.columns}
        self._metrics_by_id = {metric.id: metric for metric in catalog.metrics}

    def retrieve(self, question: str, limit: int = 8) -> list[SemanticSearchResult]:
        """Retrieve semantic evidence for an orchestration layer."""
        return semantic_search(self._store, question, limit=limit)

    def discover(self, question: str, limit: int = 8) -> DiscoveryResponse:
        """Retrieve catalog evidence and present the associated assets and relationships."""
        evidence = self.retrieve(question, limit)
        return self.discover_from_evidence(question, evidence)

    def discover_from_evidence(
        self, question: str, evidence: list[SemanticSearchResult]
    ) -> DiscoveryResponse:
        """Present catalog facts from orchestration-supplied retrieval evidence."""
        asset_ids = self._asset_ids_from_evidence(evidence)
        return DiscoveryResponse(
            question=question,
            assets=self._summarize_assets(asset_ids),
            relationships=self._relationships_for_assets(asset_ids),
            evidence=evidence,
        )

    def lineage(self, target_or_question: str, search_limit: int = 8) -> LineageResponse:
        """Resolve a target and traverse only explicit upstream lineage edges."""
        evidence = self.retrieve(target_or_question, search_limit)
        return self.lineage_from_evidence(target_or_question, evidence)

    def lineage_from_evidence(
        self, target_or_question: str, evidence: list[SemanticSearchResult]
    ) -> LineageResponse:
        """Resolve and traverse lineage from orchestration-supplied retrieval evidence."""
        target_column_id, target_metric = self._resolve_lineage_target(target_or_question, evidence)
        upstream_edges = get_upstream_lineage(self._catalog, target_column_id)
        asset_ids = self._asset_ids_from_lineage(target_column_id, upstream_edges)
        return LineageResponse(
            request=target_or_question,
            target_column_id=target_column_id,
            target_metric=target_metric,
            upstream_edges=upstream_edges,
            assets=self._summarize_assets(asset_ids),
            relationships=self._relationships_for_assets(asset_ids),
            evidence=evidence,
        )

    def asset_ids_from_evidence(self, evidence: list[SemanticSearchResult]) -> list[str]:
        """Return catalog asset IDs referenced by retrieved semantic documents."""
        return self._asset_ids_from_evidence(evidence)

    def _resolve_lineage_target(
        self, target_or_question: str, evidence: list[SemanticSearchResult]
    ) -> tuple[str, Metric | None]:
        if target_or_question in self._metrics_by_id:
            metric = self._metrics_by_id[target_or_question]
            return metric.output_column_id, metric
        if target_or_question in self._columns_by_id:
            return target_or_question, self._metric_for_column(target_or_question)
        metric = _metric_matching_question(target_or_question, self._catalog.metrics)
        if metric is not None:
            return metric.output_column_id, metric
        if target_or_question in self._assets_by_id:
            metric = self._metric_for_asset(target_or_question)
            if metric is not None:
                return metric.output_column_id, metric
        for result in evidence:
            metric_id = result.payload.get("metric_id")
            if isinstance(metric_id, str) and metric_id in self._metrics_by_id:
                metric = self._metrics_by_id[metric_id]
                return metric.output_column_id, metric
        for result in evidence:
            column_id = _column_id_from_result(result)
            if column_id in self._columns_by_id:
                return column_id, self._metric_for_column(column_id)
        for result in evidence:
            asset_id = _first_asset_id(result.payload)
            if asset_id:
                metric = self._metric_for_asset(asset_id)
                if metric is not None:
                    return metric.output_column_id, metric
        raise ExplorationError(
            "could not resolve a lineage target. Use a metric ID, column ID, or derived asset ID."
        )

    def _metric_for_column(self, column_id: str) -> Metric | None:
        return next(
            (metric for metric in self._catalog.metrics if metric.output_column_id == column_id),
            None,
        )

    def _metric_for_asset(self, asset_id: str) -> Metric | None:
        return next(
            (
                metric
                for metric in self._catalog.metrics
                if self._columns_by_id[metric.output_column_id].asset_id == asset_id
            ),
            None,
        )

    def _asset_ids_from_evidence(self, evidence: list[SemanticSearchResult]) -> list[str]:
        asset_ids: list[str] = []
        for result in evidence:
            for asset_id in _asset_ids_from_payload(result.payload):
                if asset_id in self._assets_by_id and asset_id not in asset_ids:
                    asset_ids.append(asset_id)
            for column_id in _column_ids_from_payload(result.payload):
                column = self._columns_by_id.get(column_id)
                if column is not None and column.asset_id not in asset_ids:
                    asset_ids.append(column.asset_id)
        return asset_ids

    def _asset_ids_from_lineage(
        self, target_column_id: str, upstream_edges: list[LineageEdge]
    ) -> list[str]:
        column_ids = [target_column_id]
        for edge in upstream_edges:
            column_ids.extend((edge.source_column_id, edge.target_column_id))
        asset_ids: list[str] = []
        for column_id in column_ids:
            asset_id = self._columns_by_id[column_id].asset_id
            if asset_id not in asset_ids:
                asset_ids.append(asset_id)
        return asset_ids

    def _summarize_assets(self, asset_ids: list[str]) -> list[AssetSummary]:
        return [self._asset_summary(self._assets_by_id[asset_id]) for asset_id in asset_ids]

    def _relationships_for_assets(self, asset_ids: list[str]) -> list[Relationship]:
        asset_id_set = set(asset_ids)
        return [
            relationship
            for relationship in self._catalog.relationships
            if relationship.source_asset_id in asset_id_set
            or relationship.target_asset_id in asset_id_set
        ]

    @staticmethod
    def _asset_summary(asset: DataAsset) -> AssetSummary:
        return AssetSummary(
            id=asset.id,
            qualified_name=f"{asset.schema_name}.{asset.name}",
            description=asset.description,
            grain=asset.grain,
        )


def _metric_matching_question(question: str, metrics: list[Metric]) -> Metric | None:
    """Prefer a strong direct metric-name match over imperfect vector ranking."""
    question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    best_metric: Metric | None = None
    best_score = 0
    for metric in metrics:
        metric_terms = set(re.findall(r"[a-z0-9]+", f"{metric.name} {metric.id}".lower()))
        score = len(question_terms & metric_terms)
        if score > best_score:
            best_metric, best_score = metric, score
    return best_metric if best_score >= 3 else None


def _asset_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    asset_ids: list[str] = []
    for key in ("asset_id", "source_asset_id", "target_asset_id"):
        value = payload.get(key)
        if isinstance(value, str):
            asset_ids.append(value)
    for key in ("input_asset_ids", "source_asset_ids"):
        inputs = payload.get(key)
        if isinstance(inputs, list):
            asset_ids.extend(value for value in inputs if isinstance(value, str))
    return asset_ids


def _column_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    column_ids: list[str] = []
    for key in ("source_column_id", "target_column_id", "output_column_id"):
        value = payload.get(key)
        if isinstance(value, str):
            column_ids.append(value)
    return column_ids


def _first_asset_id(payload: dict[str, Any]) -> str | None:
    asset_ids = _asset_ids_from_payload(payload)
    return asset_ids[0] if asset_ids else None


def _column_id_from_result(result: SemanticSearchResult) -> str | None:
    if result.document_type == "column" and result.document_id.startswith("column:"):
        return result.document_id.removeprefix("column:")
    return None
