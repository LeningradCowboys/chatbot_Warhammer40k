import hashlib
import sqlite3
import sys
from contextlib import closing

import pytest

from src.rag.storage.clean_db import clean_database, repair_text, separate_card_labels
from src.scripts import prepare_db
from tests.conftest import CORRUPTED_TAU


def test_repair_text_undoes_utf8_read_as_latin1():
    assert repair_text(CORRUPTED_TAU) == "T’au Empire"
    assert repair_text("Core â\x80\x93 Wargear Stratagem") == "Core – Wargear Stratagem"


@pytest.mark.parametrize("text", ["T’au Empire", "café", "plain ASCII"])
def test_repair_text_leaves_correct_text_unchanged(text):
    assert repair_text(text) == text


@pytest.mark.parametrize(
    ("glued", "separated"),
    [
        (
            "WHEN: Your Shooting phase.TARGET: One unit.EFFECT: Roll six D6.",
            "WHEN: Your Shooting phase.\nTARGET: One unit.\nEFFECT: Roll six D6.",
        ),
        ("this phaseEFFECT: Set up", "this phase\nEFFECT: Set up"),  # no full stop
        ("this phase. TARGET: One unit", "this phase.\nTARGET: One unit"),  # a space
        ("rules.RESTRICTIONS: Once per battle.", "rules.\nRESTRICTIONS: Once per battle."),
    ],
)
def test_card_labels_start_a_line(glued, separated):
    assert separate_card_labels(glued) == separated


@pytest.mark.parametrize(
    "text",
    [
        "WHEN: Your Shooting phase.",  # a label at the start
        "phase.\nTARGET: One unit.",  # already on its own line
        "Select one TARGETS: or target: in text",  # not a card label
    ],
)
def test_text_without_glued_labels_is_unchanged(text):
    assert separate_card_labels(text) == text


def test_clean_database_repairs_a_copy_and_leaves_the_source(tiny_db, tmp_path):
    before = hashlib.sha256(tiny_db.read_bytes()).hexdigest()
    target = tmp_path / "clean.sqlite"
    target.write_text("an old copy", encoding="utf-8")

    report = clean_database(tiny_db, target)

    assert hashlib.sha256(tiny_db.read_bytes()).hexdigest() == before
    assert (report.re_decoded, report.card_labels) == (1, 1)
    with closing(sqlite3.connect(target)) as clean:
        assert clean.execute("SELECT name FROM factions WHERE Id = 'TAU'").fetchone() == (
            "T’au Empire",
        )
        assert clean.execute("SELECT description FROM stratagems").fetchone()[0] == (
            "WHEN: Your Shooting phase.\nTARGET: One unit.\nEFFECT: Roll six D6."
        )
        assert clean.execute("SELECT count(*) FROM datasheets").fetchone() == (2,)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_source_as_target_is_refused_before_anything_is_deleted(tiny_db):
    before = _sha256(tiny_db)
    other_spelling = tiny_db.parent / ".." / tiny_db.parent.name / tiny_db.name
    for target in (tiny_db, other_spelling):
        with pytest.raises(ValueError, match="source database itself"):
            clean_database(tiny_db, target)
        assert tiny_db.exists()
        assert _sha256(tiny_db) == before


def _run_script(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_db", *args])
    prepare_db.main()


def test_the_script_writes_the_clean_copy(monkeypatch, tiny_db, tmp_path):
    output = tmp_path / "out" / "clean.sqlite"
    _run_script(monkeypatch, str(tiny_db), "--output", str(output))
    assert output.is_file()
