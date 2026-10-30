import hashlib
import sqlite3
from contextlib import closing

import pytest

from src.rag.retriever.sql_retriever import ENTITY_NAME_TABLES, SqlRetriever, load_entity_names

TOUGHNESS = (
    "SELECT d.name, m.t FROM datasheets d "
    "JOIN datasheets_models m ON m.DatasheetId = d.Id ORDER BY d.name"
)


def test_entity_names_come_only_from_explicit_name_fields(tmp_path):
    path = tmp_path / "entities.sqlite"
    with closing(sqlite3.connect(path)) as database:
        for table in ENTITY_NAME_TABLES:
            database.execute(f'CREATE TABLE "{table}" (name TEXT, description TEXT)')
            database.execute(
                f'INSERT INTO "{table}" VALUES (?, ?)',
                (f"{table} name", f"{table} prose must not be protected"),
            )
        database.execute('CREATE TABLE "notes" (name TEXT, description TEXT)')
        database.execute("INSERT INTO notes VALUES ('ignored name', 'ignored prose')")
        database.execute("INSERT INTO datasheets VALUES ('shared name', '')")
        database.execute("INSERT INTO datasheets_models VALUES ('shared name', '')")
        database.commit()

    names = load_entity_names(path)

    assert names.count("shared name") == 1
    assert all(f"{table} name" in names for table in ENTITY_NAME_TABLES)
    assert "ignored name" not in names
    assert not any("prose" in name for name in names)


def test_a_select_returns_columns_and_rows(tiny_db):
    result = SqlRetriever(tiny_db, max_rows=10).run(TOUGHNESS)
    assert result.error is None
    assert result.columns == ["name", "t"]
    assert result.rows == [("Heavy Intercessor Squad", "6"), ("Intercessor Squad", "4")]
    assert not result.truncated


def test_a_with_query_is_allowed(tiny_db):
    result = SqlRetriever(tiny_db, 10).run(
        "WITH sm AS (SELECT name FROM datasheets) SELECT * FROM sm"
    )
    assert result.error is None and len(result.rows) == 2


def test_rows_beyond_the_limit_are_cut_and_reported(tiny_db):
    result = SqlRetriever(tiny_db, max_rows=1).run(TOUGHNESS)
    assert result.rows == [("Heavy Intercessor Squad", "6")]
    assert result.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE datasheets",
        "UPDATE datasheets SET name = 'x'",
        "INSERT INTO datasheets VALUES ('3', 'x', 'SM')",
        "PRAGMA writable_schema = 1",
    ],
)
def test_statements_other_than_select_are_refused(tiny_db, sql):
    result = SqlRetriever(tiny_db, 10).run(sql)
    assert result.error and "SELECT" in result.error
    assert result.rows == []


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (SELECT 1) DELETE FROM datasheets",  # passes the first check
        "SELECT 1; DROP TABLE datasheets",  # two statements
    ],
)
def test_writes_are_impossible_even_past_the_check(tiny_db, sql):
    before = hashlib.sha256(tiny_db.read_bytes()).hexdigest()
    result = SqlRetriever(tiny_db, 10).run(sql)
    assert result.error
    assert hashlib.sha256(tiny_db.read_bytes()).hexdigest() == before


def test_an_error_is_returned_not_raised(tiny_db):
    result = SqlRetriever(tiny_db, 10).run("SELECT toughness FROM datasheets")
    assert result.error and "toughness" in result.error
    assert (result.columns, result.rows, result.truncated) == ([], [], False)


def test_a_path_with_spaces_is_opened(tmp_path, tiny_db):
    folder = tmp_path / "a folder with spaces"
    folder.mkdir()
    moved = folder / "tiny db.sqlite"
    moved.write_bytes(tiny_db.read_bytes())
    assert SqlRetriever(moved, 10).run("SELECT count(*) FROM datasheets").rows == [(2,)]


def test_unbounded_recursive_aggregate_is_interrupted_and_next_query_works(tiny_db):
    retriever = SqlRetriever(tiny_db, 10, max_seconds=0.01)
    result = retriever.run(
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"
    )
    assert result.error and "interrupted" in result.error
    assert retriever.run("SELECT count(*) FROM datasheets").rows == [(2,)]


def test_oversized_values_are_refused_before_returning_them(tiny_db):
    result = SqlRetriever(tiny_db, 10, max_bytes=4096).run("SELECT hex(zeroblob(10000))")
    assert result.error and result.rows == []


def test_result_bytes_are_bounded_without_silently_dropping_truncation(tiny_db):
    result = SqlRetriever(tiny_db, 50, max_bytes=4096).run(
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<20) "
        "SELECT hex(zeroblob(200)) AS value FROM n"
    )
    assert result.error is None
    assert 0 < len(result.rows) < 20 and result.truncated
