"""Run the SQL the model writes on the clean database: read-only, one SELECT, a bounded result."""

import json
import re
import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from src.rag.config import SQL_MAX_BYTES, SQL_MAX_SECONDS

READ_STATEMENT = re.compile(r"\s*(SELECT|WITH)\b", re.IGNORECASE)
PROGRESS_STEPS = 1000
ENTITY_NAME_TABLES = (
    "abilities",
    "datasheets",
    "datasheets_abilities",
    "datasheets_models",
    "datasheets_wargear",
    "detachment_abilities",
    "detachments",
    "enhancements",
    "factions",
    "source",
    "stratagems",
)


@dataclass(frozen=True)
class SqlResult:
    """The result of one query; an error is returned in `error`, never raised."""

    columns: list[str]
    rows: list[tuple]
    truncated: bool  # rows omitted by the row or serialized-byte limit
    error: str | None


def load_entity_names(db_path: Path) -> list[str]:
    """Load distinct names from the database's explicit entity-name fields.

    Descriptions and other free-text fields are deliberately excluded so that words in rules or
    prose do not become protected from spelling correction.

    Args:
        db_path: The clean datasheet database.

    Returns:
        Non-empty entity names in table order, with duplicates removed.
    """
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    names: dict[str, None] = {}
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for table in ENTITY_NAME_TABLES:
            for (name,) in connection.execute(
                f"SELECT DISTINCT name FROM \"{table}\" WHERE name IS NOT NULL AND name != ''"
            ):
                names[name] = None
    return list(names)


class SqlRetriever:
    """Runs single `SELECT` (or `WITH ... SELECT`) statements on a database opened read-only."""

    def __init__(
        self,
        db_path: Path,
        max_rows: int,
        max_seconds: float = SQL_MAX_SECONDS,
        max_bytes: int = SQL_MAX_BYTES,
    ):
        """Create a retriever on one database.

        Args:
            db_path: The clean database.
            max_rows: The most rows a result returns; the rest is reported by `truncated`.
            max_seconds: Query execution/lock-wait budget.
            max_bytes: Maximum serialized result bytes and SQLite value/row allocation.
        """
        if max_rows < 1 or max_seconds <= 0 or max_bytes < 1:
            raise ValueError("SQL limits must be positive")
        self._uri = f"{db_path.resolve().as_uri()}?mode=ro"
        self._max_rows = max_rows
        self._max_seconds = max_seconds
        self._max_bytes = max_bytes

    def run(self, sql: str, parameters: Sequence[str] = ()) -> SqlResult:
        """Run one query.

        The connection is read-only, so no statement can change the data, whatever it is;
        statements that do not start with SELECT or WITH are refused before running, and SQLite
        refuses several statements in one call. Errors come back in the result, so the model can
        correct its query.

        Args:
            sql: One SQLite statement.
            parameters: Bound text values for placeholders in the statement.

        Returns:
            The columns and at most `max_rows` rows, or the error.
        """
        if not READ_STATEMENT.match(sql):
            return SqlResult(
                [], [], False, "only one SELECT (or WITH ... SELECT) statement is allowed"
            )
        try:
            with closing(
                sqlite3.connect(self._uri, uri=True, timeout=self._max_seconds)
            ) as connection:
                deadline = time.monotonic() + self._max_seconds
                connection.set_progress_handler(
                    lambda: time.monotonic() >= deadline, PROGRESS_STEPS
                )
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self._max_bytes)
                cursor = connection.execute(sql, parameters)
                columns = [description[0] for description in cursor.description or []]
                rows = cursor.fetchmany(self._max_rows + 1)
        except sqlite3.Error as error:
            return SqlResult([], [], False, f"{type(error).__name__}: {error}")
        truncated = len(rows) > self._max_rows
        rows = rows[: self._max_rows]
        while (
            len(
                json.dumps(
                    {"columns": columns, "rows": rows, "truncated": truncated, "error": None},
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            )
            > self._max_bytes
        ):
            if not rows:
                return SqlResult([], [], False, "query result columns exceed the byte limit")
            rows.pop()
            truncated = True
        if not rows and truncated:
            return SqlResult(
                [], [], False, "query result exceeds the byte limit; select fewer/smaller columns"
            )
        return SqlResult(columns, rows, truncated, None)
