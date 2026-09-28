"""Read-only, time-bounded execution of SQL accepted by the safety boundary."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import duckdb

from semantic_lineage.warehouse.sql_safety import ValidatedSQL


class QueryExecutionError(RuntimeError):
    """Raised when DuckDB cannot execute an approved query."""


class QueryTimeoutError(QueryExecutionError):
    """Raised when an approved query exceeds its wall-clock execution deadline."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_limit: int


def execute_validated_query(
    database_path: str | Path, query: ValidatedSQL, *, timeout_seconds: float = 10.0
) -> QueryResult:
    """Run an approved query in a process that can be terminated at its deadline.

    A subprocess provides a real wall-clock boundary for a local embedded database: if DuckDB
    does not complete by the deadline, its process is terminated rather than merely abandoning a
    result fetch in the caller.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_execute_query_worker,
        args=(str(database_path), query.sql, query.row_limit, sender),
        daemon=True,
    )
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout_seconds):
            _terminate_process(process)
            raise QueryTimeoutError(
                f"DuckDB query exceeded the {timeout_seconds:g}-second execution limit. "
                "Refine the question and retry."
            )
        try:
            status, payload = receiver.recv()
        except EOFError as exc:
            raise QueryExecutionError(
                "DuckDB query worker exited without returning a result. Retry the request."
            ) from exc
        process.join()
        if status == "error":
            raise QueryExecutionError(f"DuckDB query failed: {payload}")
        columns, rows = payload
        return QueryResult(columns=columns, rows=rows, row_limit=query.row_limit)
    finally:
        sender.close()
        receiver.close()
        if process.is_alive():
            _terminate_process(process)
        elif process.pid is not None:
            process.join()
            process.close()


def _execute_query_worker(database_path: str, sql: str, row_limit: int, sender: Connection) -> None:
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(database_path, read_only=True)
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        sender.send(("ok", (columns, cursor.fetchmany(row_limit))))
    except duckdb.Error as exc:
        sender.send(("error", str(exc)))
    except Exception as exc:  # Defensive boundary for worker startup and result serialization.
        sender.send(("error", f"unexpected worker failure: {exc}"))
    finally:
        if connection is not None:
            connection.close()
        sender.close()


def _terminate_process(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()
