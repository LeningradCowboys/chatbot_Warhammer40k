import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

# Detailed assertion messages in the helper modules, as in the test files.
pytest.register_assert_rewrite("tests.fakes", "tests.gemini_check")

# RAGAS sends usage analytics unless told not to; it reads this once, before its first event.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# UTF-8 text decoded as Latin-1, as in the raw database: "T’au Empire".
CORRUPTED_TAU = "T’au Empire".encode().decode("latin-1")


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    """A small database shaped like `wahadb.sqlite`: every column TEXT, no keys, one corrupted
    faction name and one stratagem whose card labels are glued to the sentence before."""
    path = tmp_path / "tiny.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.executescript(
            """
            CREATE TABLE factions (Id TEXT, name TEXT);
            CREATE TABLE datasheets (Id TEXT, name TEXT, factionId TEXT);
            CREATE TABLE datasheets_models (DatasheetId TEXT, line TEXT, name TEXT, t TEXT);
            CREATE TABLE datasheets_keywords (DatasheetId TEXT, keyword TEXT);
            CREATE TABLE stratagems (id TEXT, name TEXT, description TEXT);
            INSERT INTO datasheets VALUES ('001', 'Intercessor Squad', 'SM'),
                                          ('002', 'Heavy Intercessor Squad', 'SM');
            INSERT INTO datasheets_models VALUES ('001', '1', 'Intercessor', '4'),
                                                 ('002', '1', 'Heavy Intercessor', '6');
            INSERT INTO datasheets_keywords VALUES ('001', 'Battleline'), ('002', 'Infantry');
            """
        )
        db.execute(
            "INSERT INTO stratagems VALUES ('s1', 'GRENADE', ?)",
            ("WHEN: Your Shooting phase.TARGET: One unit.EFFECT: Roll six D6.",),
        )
        db.execute(
            "INSERT INTO factions VALUES ('TAU', ?), ('SM', 'Space Marines')", (CORRUPTED_TAU,)
        )
        db.commit()
    return path
