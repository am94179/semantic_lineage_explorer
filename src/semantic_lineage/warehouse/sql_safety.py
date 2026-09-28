"""Parse and constrain LLM-produced SQL before DuckDB receives it."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope


class SQLSafetyError(ValueError):
    """Raised when SQL is not a bounded, allowed read-only query."""


@dataclass(frozen=True, slots=True)
class ValidatedSQL:
    sql: str
    referenced_tables: list[str]
    row_limit: int


_DENIED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_parquet",
    "read_text",
    "read_blob",
    "query_table",
    "httpfs",
    "load_extension",
    "install_extension",
}


def validate_read_only_sql(
    sql: str,
    allowed_tables: list[str],
    *,
    allowed_columns: Mapping[str, Set[str]],
    max_rows: int = 100,
) -> ValidatedSQL:
    """Allow one bounded SELECT over inspected catalog tables and columns only."""
    if max_rows < 1:
        raise SQLSafetyError("max_rows must be at least one")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.ParseError as exc:
        raise SQLSafetyError(f"SQL could not be parsed: {exc}") from exc
    if len(statements) != 1:
        raise SQLSafetyError("exactly one SQL statement is required")
    expression = statements[0]
    if not isinstance(expression, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLSafetyError("only SELECT queries are allowed")
    if expression.find(exp.Into):
        raise SQLSafetyError("SELECT INTO is not allowed")
    with_expression = expression.args.get("with_")
    if with_expression and with_expression.args.get("recursive"):
        raise SQLSafetyError("recursive CTEs are not allowed")

    normalized_columns = {
        table.lower(): {column.lower() for column in columns}
        for table, columns in allowed_columns.items()
    }
    allowed = {table.lower() for table in allowed_tables}
    if not allowed:
        raise SQLSafetyError("at least one approved table is required")
    if not allowed.issubset(normalized_columns):
        missing = sorted(allowed - normalized_columns.keys())
        raise SQLSafetyError(
            "inspected schema is missing column metadata for approved table(s): "
            + ", ".join(missing)
        )

    referenced_tables = _validate_sources_and_columns(expression, allowed, normalized_columns)
    if not referenced_tables:
        raise SQLSafetyError("query must reference at least one approved table")
    for function in expression.find_all(exp.Func):
        if function.name.lower() in _DENIED_FUNCTIONS:
            raise SQLSafetyError(f"external data function is not allowed: {function.name}")
    _validate_bounds(expression, max_rows)
    return ValidatedSQL(
        sql=expression.sql(dialect="duckdb"),
        referenced_tables=referenced_tables,
        row_limit=max_rows,
    )


def _validate_sources_and_columns(
    expression: exp.Expression,
    allowed_tables: set[str],
    allowed_columns: Mapping[str, Set[str]],
) -> list[str]:
    referenced_tables: list[str] = []
    for scope in traverse_scope(expression):
        sources = _scope_sources(scope, allowed_tables, referenced_tables)
        _validate_scope_columns(scope, sources, allowed_columns)
    return referenced_tables


def _scope_sources(
    scope: Scope, allowed_tables: set[str], referenced_tables: list[str]
) -> dict[str, tuple[str, str | set[str]]]:
    sources: dict[str, tuple[str, str | set[str]]] = {}
    for alias, (source_expression, source) in scope.selected_sources.items():
        alias_key = alias.lower()
        if isinstance(source, Scope):
            sources[alias_key] = (
                "derived",
                {name.lower() for name in source.expression.named_selects},
            )
            continue
        if not isinstance(source_expression, exp.Table) or not isinstance(
            source_expression.this, exp.Identifier
        ):
            raise SQLSafetyError("table-valued functions and non-table sources are not allowed")
        if not source_expression.db or source_expression.catalog:
            raise SQLSafetyError("all physical table references must be schema-qualified")
        qualified_name = f"{source_expression.db}.{source_expression.name}".lower()
        if qualified_name not in allowed_tables:
            raise SQLSafetyError(f"table is not allowed: {qualified_name}")
        sources[alias_key] = ("table", qualified_name)
        table_name_key = source_expression.name.lower()
        sources.setdefault(table_name_key, ("table", qualified_name))
        if qualified_name not in referenced_tables:
            referenced_tables.append(qualified_name)
    return sources


def _validate_scope_columns(
    scope: Scope,
    sources: Mapping[str, tuple[str, str | set[str]]],
    allowed_columns: Mapping[str, Set[str]],
) -> None:
    select_aliases = {name.lower() for name in scope.expression.named_selects}
    for column in scope.columns:
        column_name = column.name.lower()
        if column_name == "*":
            continue
        qualifier = column.table.lower()
        if qualifier:
            source = sources.get(qualifier)
            if source is None:
                raise SQLSafetyError(f"column qualifier is not a selected source: {column.table}")
            _validate_column_against_source(column_name, source, column.sql(), allowed_columns)
            continue
        if _is_order_alias(column, select_aliases):
            continue
        candidates: list[tuple[str, str | set[str]]] = []
        for source in sources.values():
            if _source_has_column(column_name, source, allowed_columns) and not any(
                source == candidate for candidate in candidates
            ):
                candidates.append(source)
        if not candidates:
            raise SQLSafetyError(f"column is not present in the inspected schema: {column.name}")
        if len(candidates) > 1:
            raise SQLSafetyError(
                f"unqualified column is ambiguous across selected sources: {column.name}"
            )


def _validate_column_against_source(
    column_name: str,
    source: tuple[str, str | set[str]],
    rendered_column: str,
    allowed_columns: Mapping[str, Set[str]],
) -> None:
    if not _source_has_column(column_name, source, allowed_columns):
        source_kind, source_value = source
        if source_kind == "table":
            raise SQLSafetyError(
                f"column is not present in inspected schema: {source_value}.{column_name}"
            )
        raise SQLSafetyError(f"column is not projected by derived source: {rendered_column}")


def _source_has_column(
    column_name: str,
    source: tuple[str, str | set[str]],
    allowed_columns: Mapping[str, Set[str]],
) -> bool:
    source_kind, source_value = source
    if source_kind == "table":
        assert isinstance(source_value, str)
        return column_name in allowed_columns[source_value]
    assert isinstance(source_value, set)
    return column_name in source_value


def _is_order_alias(column: exp.Column, select_aliases: set[str]) -> bool:
    return (
        not column.table
        and column.name.lower() in select_aliases
        and column.find_ancestor(exp.Order) is not None
    )


def _validate_bounds(expression: exp.Expression, max_rows: int) -> None:
    limit = expression.args.get("limit")
    if limit is None:
        expression.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        limit_expression = limit.expression
        _validate_non_negative_literal(limit_expression, "LIMIT")
        assert isinstance(limit_expression, exp.Literal)
        if int(limit_expression.this) > max_rows:
            expression.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    offset = expression.args.get("offset")
    if offset is not None:
        _validate_non_negative_literal(offset.expression, "OFFSET")


def _validate_non_negative_literal(expression: exp.Expression | None, name: str) -> None:
    if (
        isinstance(expression, exp.Neg)
        and isinstance(expression.this, exp.Literal)
        and expression.this.is_int
    ):
        raise SQLSafetyError(f"{name} must not be negative")
    if not isinstance(expression, exp.Literal) or not expression.is_int:
        raise SQLSafetyError(f"{name} must be an integer literal")
    if int(expression.this) < 0:
        raise SQLSafetyError(f"{name} must not be negative")
