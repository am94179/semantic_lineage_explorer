"""Structured LLM planning boundary plus a deterministic local demonstration planner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import openai
from openai import OpenAI

from semantic_lineage.config import Settings
from semantic_lineage.warehouse.duckdb_client import QueryResult
from semantic_lineage.warehouse.schema import TableSchema

Intent = Literal["discovery", "lineage", "analysis"]


class PlannerError(RuntimeError):
    """Raised when the hosted planner cannot return a usable structured response."""


@dataclass(frozen=True, slots=True)
class IntentDecision:
    intent: Intent
    rationale: str


@dataclass(frozen=True, slots=True)
class SQLPlan:
    sql: str
    explanation: str


class Planner(Protocol):
    def classify_intent(self, question: str, retrieval_context: str) -> IntentDecision: ...

    def generate_sql(
        self, question: str, schemas: list[TableSchema], retrieval_context: str
    ) -> SQLPlan: ...

    def synthesize_analysis(
        self, question: str, result: QueryResult, sql: str, lineage_context: str
    ) -> str: ...


class OpenAIPlanner:
    """Uses Responses structured output; all data access remains outside the model."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError(
                "SEMANTIC_LINEAGE_LLM_API_KEY (or SEMANTIC_LINEAGE_EMBEDDING_API_KEY) is required"
            )
        self._client = OpenAI(api_key=settings.llm_api_key)
        self._model = settings.llm_model

    def classify_intent(self, question: str, retrieval_context: str) -> IntentDecision:
        result = self._structured(
            "intent_decision",
            {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": ["discovery", "lineage", "analysis"]},
                    "rationale": {"type": "string"},
                },
                "required": ["intent", "rationale"],
                "additionalProperties": False,
            },
            "Classify this data question. Discovery finds data assets; lineage explains recorded provenance; analysis needs numeric DuckDB results. Return JSON only.\n"
            f"Question: {question}\nRetrieved catalog context:\n{retrieval_context}",
        )
        intent = self._required_string(result, "intent", "intent decision")
        if intent not in {"discovery", "lineage", "analysis"}:
            raise PlannerError(
                "OpenAI returned an unsupported intent. Retry the request or check the planner schema."
            )
        return IntentDecision(
            intent=cast(Intent, intent),
            rationale=self._required_string(result, "rationale", "intent decision"),
        )

    def generate_sql(
        self, question: str, schemas: list[TableSchema], retrieval_context: str
    ) -> SQLPlan:
        schema_context = _schema_context(schemas)
        result = self._structured(
            "sql_plan",
            {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["sql", "explanation"],
                "additionalProperties": False,
            },
            "Write one DuckDB SELECT query only. Use only schema-qualified tables and columns from the supplied schema. Never use file, network, extension, DDL, DML, PRAGMA, or multiple statements. Return JSON only.\n"
            f"Question: {question}\nSchema:\n{schema_context}\nCatalog context:\n{retrieval_context}",
        )
        return SQLPlan(
            sql=self._required_string(result, "sql", "SQL plan"),
            explanation=self._required_string(result, "explanation", "SQL plan"),
        )

    def synthesize_analysis(
        self, question: str, result: QueryResult, sql: str, lineage_context: str
    ) -> str:
        rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
        output = self._structured(
            "analysis_answer",
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "Explain the provided query result without inventing facts. Mention that results are from DuckDB. Return JSON only.\n"
            f"Question: {question}\nRows: {json.dumps(rows, default=str)}\nLineage: {lineage_context}\nSQL: {sql}",
        )
        return self._required_string(output, "answer", "analysis answer")

    def _structured(self, name: str, schema: dict[str, object], prompt: str) -> dict[str, str]:
        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except openai.APIConnectionError as exc:
            raise PlannerError(
                "Could not connect to the OpenAI API. Check network access and retry."
            ) from exc
        except openai.RateLimitError as exc:
            raise PlannerError("OpenAI rate limit reached. Wait briefly and retry.") from exc
        except openai.APIError as exc:
            raise PlannerError(
                "OpenAI request failed. Check the API key, model name, and account access."
            ) from exc

        status = getattr(response, "status", None)
        if status and status != "completed":
            raise PlannerError(
                f"OpenAI response did not complete ({status}). Retry the request or reduce its context."
            )

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise PlannerError(
                "OpenAI returned no structured output. The response may have been refused or incomplete."
            )
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise PlannerError(
                "OpenAI returned invalid structured output. Retry the request."
            ) from exc
        if not isinstance(result, dict):
            raise PlannerError("OpenAI returned structured output that is not a JSON object.")
        return result

    @staticmethod
    def _required_string(result: dict[str, str], field: str, response_name: str) -> str:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PlannerError(
                f"OpenAI returned an invalid {response_name}: required field '{field}' is missing or empty."
            )
        return value


class DeterministicPlanner:
    """Local, non-LLM planner for tests and the documented offline demonstration."""

    def classify_intent(self, question: str, retrieval_context: str) -> IntentDecision:
        lowered = question.lower()
        if any(term in lowered for term in ("lineage", "where does", "contribute", "come from")):
            return IntentDecision("lineage", "question asks for recorded provenance")
        if any(
            term in lowered for term in ("top", "most", "how many", "average", "total", "revenue")
        ):
            return IntentDecision("analysis", "question asks for data-derived values")
        return IntentDecision("discovery", "question asks to locate relevant data")

    def generate_sql(
        self, question: str, schemas: list[TableSchema], retrieval_context: str
    ) -> SQLPlan:
        year_match = re.search(r"\b(19|20)\d{2}\b", question)
        year = year_match.group(0) if year_match else "2001"
        limit_match = re.search(r"\btop\s+(\d+)\b", question.lower())
        limit = min(int(limit_match.group(1)), 100) if limit_match else 10
        return SQLPlan(
            sql=(
                "SELECT c.c_customer_id, c.c_first_name, c.c_last_name, r.total_net_paid "
                "FROM analytics.customer_revenue_yearly AS r "
                "JOIN raw.customer AS c ON r.customer_sk = c.c_customer_sk "
                f"WHERE r.calendar_year = {year} ORDER BY r.total_net_paid DESC LIMIT {limit}"
            ),
            explanation="Ranks customers by the documented yearly net-paid revenue metric.",
        )

    def synthesize_analysis(
        self, question: str, result: QueryResult, sql: str, lineage_context: str
    ) -> str:
        if not result.rows:
            return "DuckDB returned no rows for this question."
        rendered_rows = "; ".join(f"{row[1]} {row[2]} ({row[0]}): {row[3]}" for row in result.rows)
        return f"DuckDB returned {len(result.rows)} row(s): {rendered_rows}."


def _schema_context(schemas: list[TableSchema]) -> str:
    return "\n".join(
        f"{table.qualified_name}({', '.join(column.name for column in table.columns)})"
        for table in schemas
    )
