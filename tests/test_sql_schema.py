"""The schema note given to the model must agree with the data (plan §2.3, R33).

Each example runs and returns what its title promises, checked against facts confirmed apart from
the example; the facts the note states hold on the clean database. The tests on the real database
are slow: they need `src.scripts.prepare_db` to have run.
"""

import re
import sqlite3
from collections import defaultdict
from contextlib import closing

import pytest

from src.rag.config import CLEAN_DB, SQL_SCHEMA

slow = pytest.mark.slow


def _examples_by_number() -> dict[int, list[str]]:
    """The SQL blocks of the note's Examples section, under their example number."""
    section = SQL_SCHEMA.read_text(encoding="utf-8").split("## Examples", 1)[1]
    pieces = re.split(r"^(\d+)\. ", section, flags=re.MULTILINE)
    return {
        int(number): re.findall(r"```sql\n(.*?)```", body, re.DOTALL)
        for number, body in zip(pieces[1::2], pieces[2::2], strict=True)
    }


EXAMPLES = _examples_by_number()
ALL_SQL = [(number, sql) for number, blocks in EXAMPLES.items() for sql in blocks]


def _rows(db: sqlite3.Connection, sql: str) -> list[dict]:
    cursor = db.execute(sql)
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


@pytest.fixture(scope="module")
def db():
    if not CLEAN_DB.exists():
        pytest.skip("wahadb_clean.sqlite not built: run src.scripts.prepare_db first")
    with closing(sqlite3.connect(f"{CLEAN_DB.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        yield connection


def _one(db, sql: str):
    return db.execute(sql).fetchone()


def test_the_note_has_ten_numbered_examples():
    assert sorted(EXAMPLES) == list(range(1, 11))
    assert all(EXAMPLES.values())


def test_example_6_keeps_an_unknown_game_mode_as_null():
    """A stratagem whose detachment is not in `detachments` must not look like the standard game."""
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(
            """
            CREATE TABLE datasheets (Id TEXT, factionId TEXT);
            CREATE TABLE datasheets_stratagems (DatasheetId TEXT, stratagemId TEXT);
            CREATE TABLE stratagems (id TEXT, name TEXT, FactionId TEXT, type TEXT, detachment TEXT,
                                     detachmentId TEXT, cpCost TEXT, phase TEXT);
            CREATE TABLE detachments (Id TEXT, type TEXT);
            INSERT INTO datasheets VALUES ('000001157', 'SM');
            INSERT INTO datasheets_stratagems VALUES ('000001157', 's1'), ('000001157', 's2');
            INSERT INTO stratagems VALUES ('s1', 'KNOWN', 'SM', 't', 'd', 'D1', '1', 'Any'),
                                          ('s2', 'MISSING', 'SM', 't', 'd', 'D2', '1', 'Any');
            INSERT INTO detachments VALUES ('D1', '');
            """
        )
        modes = {row["name"]: row["game_mode"] for row in _rows(db, EXAMPLES[6][0])}
    assert modes == {"KNOWN": "", "MISSING": None}


@slow
@pytest.mark.parametrize(("number", "sql"), ALL_SQL, ids=[f"example-{n}" for n, _ in ALL_SQL])
def test_every_example_runs_and_finds_rows(db, number, sql):
    assert db.execute(sql).fetchall()


@slow
def test_example_1_gives_the_intercessor_profile(db):
    rows = _rows(db, EXAMPLES[1][0])
    assert {row["name"] for row in rows} == {"Intercessor Squad"}
    assert any(row["t"] == "4" and row["faction"] == "Space Marines" for row in rows)


@slow
def test_example_2_gives_t4_with_the_known_exceptions_by_profile(db):
    rows = _rows(db, EXAMPLES[2][0])
    by_unit = {row["name"]: row for row in rows}
    assert len(rows) == len(by_unit) == 3
    assert by_unit["Heavy Intercessor Squad"]["profiles"] == "Heavy Intercessor Squad: T6"
    profiles = by_unit["Decimus Kill Team"]["profiles"]
    assert "GRAVIS VETERAN: T6" in profiles
    assert "KILL TEAM SERGEANT, DEATHWATCH VETERAN: T4" in profiles
    heavy = by_unit["Heavy Intercessor Squad"]
    assert (heavy["source"], heavy["edition"]) == (
        "Space Marines",
        "11",
    )
    assert (by_unit["Decimus Kill Team"]["source"], by_unit["Decimus Kill Team"]["edition"]) == (
        "Deathwatch",
        "10",
    )


@slow
def test_category_example_keeps_both_observed_wardens_profiles_together(db):
    sql = EXAMPLES[2][0].replace("keyword = 'Battleline'", "keyword = 'Infantry'")
    sql = sql.replace("d.factionId = 'SM'", "d.factionId = 'SM' AND d.name = 'Wardens of Ultramar'")
    [wardens] = [row for row in _rows(db, sql) if row["name"] == "Wardens of Ultramar"]
    assert "Ancient Gadriel, Veteran Sergeant Metaurus: T4" in wardens["profiles"]
    assert (
        "Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha: T3" in wardens["profiles"]
    )


@slow
def test_category_distribution_counts_full_profiles_and_units_per_publication(db):
    # Independent row-level enumeration checks that aggregation is not over the limited examples.
    sql = EXAMPLES[2][1].replace("keyword = 'Battleline'", "keyword = 'Infantry'")
    actual = _rows(db, sql)
    expected = defaultdict(set)
    units = db.execute(
        "SELECT d.Id, s.Id, s.name, s.edition FROM datasheets d "
        "LEFT JOIN source s ON s.Id = d.sourceId WHERE d.factionId = 'SM'"
    ).fetchall()
    for unit_id, source_id, source, edition in units:
        if source and "(Warhammer Legends)" in source:
            continue
        keywords = {
            r[0]
            for r in db.execute(
                "SELECT keyword FROM datasheets_keywords WHERE DatasheetId = ?", (unit_id,)
            )
        }
        if "Infantry" not in keywords:
            continue
        for line, toughness in db.execute(
            "SELECT line, t FROM datasheets_models WHERE DatasheetId = ?", (unit_id,)
        ):
            expected[source_id, source, edition, toughness].add((unit_id, line))
    assert len(actual) == len(expected)
    assert sum(row["model_profiles"] for row in actual) > 50
    for row in actual:
        profiles = expected[row["source_id"], row["source"], row["edition"], row["t"]]
        assert row["model_profiles"] == len(profiles)
        assert row["datasheets"] == len({unit for unit, _ in profiles})


@slow
def test_example_3_gives_the_unit_weapons(db):
    names = {row["name"] for row in _rows(db, EXAMPLES[3][0])}
    assert {"Bolt rifle", "Astartes chainsword"} <= names


@slow
def test_example_4_gives_every_ability_once_with_its_text(db):
    rows = _rows(db, EXAMPLES[4][0])
    linked = _one(db, "SELECT COUNT(*) FROM datasheets_abilities WHERE DatasheetId = '000000070'")
    assert len(rows) == linked[0]
    assert all(row["ability_text"] for row in rows)


@slow
def test_example_5_gives_the_grenade_text_and_only_units_with_the_keyword(db):
    stratagem, units = EXAMPLES[5]
    rows = _rows(db, stratagem)
    assert [row["name"] for row in rows] == ["GRENADE"]
    assert rows[0]["description"].startswith("WHEN: Your Shooting phase.\nTARGET:")

    names = {row["name"] for row in _rows(db, units)}
    keywords = "SELECT keyword FROM datasheets_keywords WHERE DatasheetId = '{}'"
    assert ("Grenades",) in db.execute(keywords.format("000001157")).fetchall()  # Intercessor
    assert ("Grenades",) not in db.execute(keywords.format("000000065")).fetchall()  # Land Raider
    assert "Intercessor Squad" in names
    assert "Land Raider" not in names


@slow
def test_example_6_gives_the_unit_faction_stratagems_with_their_game_mode(db):
    rows = _rows(db, EXAMPLES[6][0])
    assert {row["faction"] for row in rows} == {"SM"}  # Intercessor Squad's faction
    boarding = [row for row in rows if row["detachment"] == "Boarding Strike"]
    assert boarding and all(row["game_mode"] == "Boarding Actions" for row in boarding)
    assert any(row["game_mode"] == "" for row in rows)


@slow
def test_example_7_gives_each_keyword_once(db):
    rows = db.execute(EXAMPLES[7][0]).fetchall()
    assert len(rows) == len(set(rows))
    assert ("Adeptus Astartes", "", "true") in rows
    assert ("Battleline", "", "false") in rows


@slow
def test_example_8_gives_the_faction_detachments(db):
    detachments = {row["detachment"] for row in _rows(db, EXAMPLES[8][0])}
    assert {"Gladius Task Force", "Anvil Siege Force"} <= detachments


@slow
def test_example_9_gives_the_units_the_captain_can_lead(db):
    known_link = (
        "SELECT 1 FROM datasheets_leader WHERE LeaderId = '000000073' AND attachedId = '{}'"
    )
    assert _one(db, known_link.format("000001157"))  # Captain -> Intercessor Squad
    assert _one(db, known_link.format("000000065")) is None  # not a Land Raider
    names = {row["name"] for row in _rows(db, EXAMPLES[9][0])}
    assert "Intercessor Squad" in names
    assert "Land Raider" not in names


@slow
def test_example_10_gives_each_points_row_once_without_headers(db):
    rows = db.execute(EXAMPLES[10][0]).fetchall()
    assert len(rows) == len(set(rows))
    assert all(cost for _, cost in rows)
    assert ("5 models", "80") in rows


@slow
def test_the_abilities_pattern_gives_one_row_per_ability_and_every_text(db):
    rows = db.execute(
        """
        SELECT COALESCE(a.description, da.description, '') AS ability_text
        FROM datasheets_abilities da
        LEFT JOIN (SELECT DISTINCT Id, name, description FROM abilities) a ON a.Id = da.abilityId
        """
    ).fetchall()
    assert len(rows) == _one(db, "SELECT COUNT(*) FROM datasheets_abilities")[0] == 6985
    assert all(text for (text,) in rows)
    # The trap: joining on Id alone multiplies the rows.
    assert _one(
        db,
        "SELECT COUNT(*) FROM datasheets_abilities da LEFT JOIN abilities a ON a.Id = da.abilityId",
    ) == (9345,)


@slow
def test_the_stated_counts_hold(db):
    assert _one(db, "SELECT COUNT(*), COUNT(DISTINCT name) FROM datasheets") == (1680, 1291)
    assert _one(db, "SELECT COUNT(*), COUNT(DISTINCT Id) FROM abilities") == (91, 71)
    assert _one(
        db,
        "SELECT COUNT(*), COUNT(DISTINCT DatasheetId || '|' || keyword || '|' || model) "
        "FROM datasheets_keywords",
    ) == (16358, 12172)
    assert _one(db, "SELECT COUNT(*) FROM detachments WHERE type = 'Boarding Actions'") == (59,)
    assert _one(db, "SELECT COUNT(*) FROM detachments") == (327,)


@slow
def test_legends_and_forge_world_are_told_by_the_source_name(db):
    # edition '0' holds both, so the edition cannot exclude Legends alone
    kinds = dict(
        db.execute(
            """
            SELECT CASE WHEN s.name LIKE '%(Warhammer Legends)%' THEN 'legends'
                        WHEN s.name LIKE '%(Forge World)%' THEN 'forge world' END, COUNT(*)
            FROM datasheets d JOIN source s ON s.Id = d.sourceId
            WHERE s.edition = '0' GROUP BY 1
            """
        ).fetchall()
    )
    assert kinds == {"legends": 529, "forge world": 47}
    assert _one(db, "SELECT COUNT(*) FROM datasheets WHERE sourceId = ''") == (2,)


@slow
def test_core_stratagems_come_in_two_forms(db):
    grenade = db.execute("SELECT type FROM stratagems WHERE name = 'GRENADE'").fetchall()
    assert grenade == [("Core – Wargear Stratagem",)]
    linked = dict(
        db.execute(
            """
            SELECT s.type, COUNT(ds.stratagemId) FROM stratagems s
            LEFT JOIN datasheets_stratagems ds ON ds.stratagemId = s.id
            WHERE s.name = 'COMMAND RE-ROLL' AND s.FactionId = '' GROUP BY s.type
            """
        ).fetchall()
    )
    assert linked["Core Stratagem"] > 0
    assert linked["Core – Battle Tactic Stratagem"] == 0


@slow
def test_stratagems_without_a_known_detachment_are_the_six_army_rules(db):
    rows = db.execute(
        """
        SELECT s.FactionId, s.detachment FROM stratagems s
        LEFT JOIN detachments dt ON dt.Id = s.detachmentId
        WHERE s.FactionId != '' AND dt.Id IS NULL
        """
    ).fetchall()
    assert rows == [("AE", "Army Rules")] * 6


@slow
def test_card_labels_start_their_lines(db):
    (description,) = _one(db, "SELECT description FROM stratagems WHERE name = 'GRENADE'")
    assert description.startswith("WHEN: Your Shooting phase.\nTARGET: ")
    assert "\nEFFECT: " in description
