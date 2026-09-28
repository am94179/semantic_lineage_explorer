"""Small, typed LangGraph workflow for discovery, lineage, and safe analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from semantic_lineage.agents.planner import Intent, Planner
from semantic_lineage.exploration import CatalogExplorer, ExplorationError
from semantic_lineage.models import Catalog
from semantic_lineage.presentation.responses import (
    render_discovery_response,
    render_lineage_response,
)
from semantic_lineage.retrieval.search import SemanticSearchResult
from semantic_lineage.warehouse.duckdb_client import QueryResult, execute_validated_query
from semantic_lineage.warehouse.schema import TableSchema, catalog_table_names, inspect_schema
from semantic_lineage.warehouse.sql_safety import (
    SQLSafetyError,
    ValidatedSQL,
    validate_read_only_sql,
)


class WorkflowState(TypedDict, total=False):
    question: str
    intent: Intent
    intent_rationale: str
    evidence: list[SemanticSearchResult]
    retrieved_document_ids: list[str]
    candidate_asset_ids: list[str]
    schema_context: list[TableSchema]
    generated_sql: str
    validated_sql: ValidatedSQL
    query_result: QueryResult
    answer: str
    lineage_context: str
    errors: list[str]


@dataclass(frozen=True, slots=True)
class WorkflowResponse:
    intent: Intent
    answer: str
    retrieved_document_ids: list[str]
    candidate_asset_ids: list[str]
    sql: str | None = None
    result: QueryResult | None = None
    lineage_context: str | None = None


class SemanticDataWorkflow:
    """Coordinates tested functions; the model never gets a database connection or tool access."""

    def __init__(
        self,
        catalog: Catalog,
        explorer: CatalogExplorer,
        planner: Planner,
        database_path: str | Path,
        *,
        retrieval_limit: int = 8,
        max_rows: int = 100,
        query_timeout_seconds: float = 10.0,
    ) -> None:
        self._catalog = catalog
        self._explorer = explorer
        self._planner = planner
        self._database_path = Path(database_path)
        self._retrieval_limit = retrieval_limit
        self._max_rows = max_rows
        self._query_timeout_seconds = query_timeout_seconds
        self._graph = self._build_graph()

    def ask(self, question: str) -> WorkflowResponse:
        """Run one question through the graph and return only structured audit data."""
        state = self._graph.invoke({"question": question, "errors": []})
        if state.get("errors"):
            raise RuntimeError("workflow failed: " + "; ".join(state["errors"]))
        return WorkflowResponse(
            intent=state["intent"],
            answer=state["answer"],
            retrieved_document_ids=state["retrieved_document_ids"],
            candidate_asset_ids=state["candidate_asset_ids"],
            sql=state.get("validated_sql").sql if state.get("validated_sql") else None,
            result=state.get("query_result"),
            lineage_context=state.get("lineage_context"),
        )

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("semantic_retrieve", self._semantic_retrieve)
        graph.add_node("discovery_response", self._discovery_response)
        graph.add_node("lineage_response", self._lineage_response)
        graph.add_node("inspect_schema", self._inspect_schema)
        graph.add_node("generate_sql", self._generate_sql)
        graph.add_node("validate_sql", self._validate_sql)
        graph.add_node("execute_duckdb", self._execute_duckdb)
        graph.add_node("synthesize_analysis_response", self._synthesize_analysis_response)
        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "semantic_retrieve")
        graph.add_conditional_edges(
            "semantic_retrieve",
            lambda state: state["intent"],
            {
                "discovery": "discovery_response",
                "lineage": "lineage_response",
                "analysis": "inspect_schema",
            },
        )
        graph.add_edge("discovery_response", END)
        graph.add_edge("lineage_response", END)
        graph.add_edge("inspect_schema", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_edge("validate_sql", "execute_duckdb")
        graph.add_edge("execute_duckdb", "synthesize_analysis_response")
        graph.add_edge("synthesize_analysis_response", END)
        return graph.compile()

    def _classify_intent(self, state: WorkflowState) -> dict[str, object]:
        decision = self._planner.classify_intent(
            state["question"], "Intent is selected before retrieval."
        )
        return {"intent": decision.intent, "intent_rationale": decision.rationale}

    def _semantic_retrieve(self, state: WorkflowState) -> dict[str, object]:
        evidence = self._explorer.retrieve(state["question"], self._retrieval_limit)
        asset_ids = self._explorer.asset_ids_from_evidence(evidence)
        if state["intent"] == "analysis":
            asset_ids = self._analysis_asset_ids(asset_ids)
        return {
            "evidence": evidence,
            "retrieved_document_ids": [result.document_id for result in evidence],
            "candidate_asset_ids": asset_ids,
        }

    def _discovery_response(self, state: WorkflowState) -> dict[str, object]:
        response = self._explorer.discover_from_evidence(state["question"], state["evidence"])
        return {"answer": render_discovery_response(response)}

    def _lineage_response(self, state: WorkflowState) -> dict[str, object]:
        response = self._explorer.lineage_from_evidence(state["question"], state["evidence"])
        return {
            "answer": render_lineage_response(response),
            "lineage_context": render_lineage_response(response),
        }

    def _inspect_schema(self, state: WorkflowState) -> dict[str, object]:
        schemas = inspect_schema(self._database_path, self._catalog, state["candidate_asset_ids"])
        return {"schema_context": schemas}

    def _generate_sql(self, state: WorkflowState) -> dict[str, object]:
        plan = self._planner.generate_sql(
            state["question"], state["schema_context"], _evidence_context(state["evidence"])
        )
        return {"generated_sql": plan.sql}

    def _validate_sql(self, state: WorkflowState) -> dict[str, object]:
        allowed_tables = catalog_table_names(self._catalog, state["candidate_asset_ids"])
        allowed_columns = {
            table.qualified_name: {column.name for column in table.columns}
            for table in state["schema_context"]
        }
        try:
            validated_sql = validate_read_only_sql(
                state["generated_sql"],
                allowed_tables,
                allowed_columns=allowed_columns,
                max_rows=self._max_rows,
            )
        except SQLSafetyError as exc:
            raise RuntimeError(f"SQL validation failed: {exc}") from exc
        return {"validated_sql": validated_sql}

    def _execute_duckdb(self, state: WorkflowState) -> dict[str, object]:
        return {
            "query_result": execute_validated_query(
                self._database_path,
                state["validated_sql"],
                timeout_seconds=self._query_timeout_seconds,
            )
        }

    def _synthesize_analysis_response(self, state: WorkflowState) -> dict[str, object]:
        try:
            lineage = self._explorer.lineage_from_evidence(
                "customer_yearly_net_revenue", state["evidence"]
            )
            lineage_context = render_lineage_response(lineage)
        except ExplorationError:
            lineage_context = (
                "No explicit metric lineage was resolved from retrieved catalog evidence."
            )
        answer = self._planner.synthesize_analysis(
            state["question"], state["query_result"], state["validated_sql"].sql, lineage_context
        )
        return {"answer": answer, "lineage_context": lineage_context}

    def _analysis_asset_ids(self, retrieved_asset_ids: list[str]) -> list[str]:
        """Ensure the small revenue path is available while retaining retrieval-selected context."""
        required = {"analytics.customer_revenue_yearly", "raw.customer"}
        metric_sources = {
            source
            for metric in self._catalog.metrics
            if metric.id == "customer_yearly_net_revenue"
            for source in metric.source_asset_ids
        }
        selected = required | metric_sources | set(retrieved_asset_ids)
        return [asset.id for asset in self._catalog.assets if asset.id in selected]


def _evidence_context(evidence: list[SemanticSearchResult]) -> str:
    return "\n".join(f"{item.document_id}: {item.content}" for item in evidence)
