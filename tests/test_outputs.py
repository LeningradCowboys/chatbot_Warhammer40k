"""The output check of every script (src/scripts/outputs.py, plan R36).

A refused command must change no file and start no work: the parser, the chunker, the cleaning
and the embedder are replaced by `pytest.fail`, so any call fails the test. The raw folder is a
temporary one; the real `data/raw/` is never written.
"""

import hashlib
import os
import sys
from pathlib import Path

import pytest
from docling_core.types.doc import TableData

from src.rag.chunker.markdown_chunker import Chunk, chunks_to_jsonl
from src.scripts import build_index, chunk, outputs, parse, prepare_db
from src.scripts.outputs import output_problem
from tests.test_docling_parser import at, make_document, table_cell


@pytest.fixture
def raw_dir(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "raw"
    folder.mkdir()
    monkeypatch.setattr(outputs, "RAW_DIR", folder)
    return folder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hard_link(target: Path, link: Path) -> None:
    try:
        os.link(target, link)
    except OSError as error:
        pytest.skip(f"hard links are not available here: {error}")


def _run(monkeypatch, module, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()


# --- the shared check ------------------------------------------------------------------------


def test_new_and_derived_outputs_are_allowed(tmp_path, raw_dir):
    source = tmp_path / "input.md"
    source.write_text("x", encoding="utf-8")
    derived = tmp_path / "out.jsonl"
    derived.write_text("old", encoding="utf-8")
    assert output_problem(tmp_path / "new" / "out.jsonl", [source]) is None
    assert output_problem(derived, [source]) is None


def test_the_input_is_refused_under_any_spelling(tmp_path, raw_dir):
    source = tmp_path / "input.md"
    source.write_text("x", encoding="utf-8")
    for output in (source, tmp_path / "sub" / ".." / "input.md"):
        assert "would be overwritten" in output_problem(output, [source])


def test_a_hard_link_to_the_input_with_another_suffix_is_refused(tmp_path, raw_dir):
    source = tmp_path / "Core Rules.pdf"
    source.write_bytes(b"%PDF")
    alias = tmp_path / "core_rules_p001_t1.csv"
    _hard_link(source, alias)
    assert "would be overwritten" in output_problem(alias, [source])


def test_anything_in_the_raw_folder_is_refused(tmp_path, raw_dir):
    for output in (raw_dir, raw_dir / "new" / "x.npz", tmp_path / "elsewhere" / ".." / "raw" / "x"):
        assert "holds the original inputs" in output_problem(output, [])


# --- parse.py --------------------------------------------------------------------------------


def _pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "Core Rules.pdf"
    pdf.write_bytes(b"%PDF-1.7 fixed")
    return pdf


def test_parse_refuses_a_folder_in_raw_before_parsing(monkeypatch, tmp_path, raw_dir):
    pdf = _pdf(tmp_path)
    monkeypatch.setattr(parse, "parse_document", pytest.fail)
    for folder in ("--output-dir", "--tables-dir"):
        other = "--tables-dir" if folder == "--output-dir" else "--output-dir"
        with pytest.raises(SystemExit):
            _run(monkeypatch, parse, str(pdf), folder, str(raw_dir), other, str(tmp_path / "ok"))
    assert list(raw_dir.iterdir()) == []
    assert not (tmp_path / "ok").exists()


def test_parse_refuses_a_markdown_path_that_is_the_input_before_parsing(
    monkeypatch, tmp_path, raw_dir
):
    pdf = _pdf(tmp_path)
    before = _sha256(pdf)
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    _hard_link(pdf, output_dir / "core_rules.md")
    monkeypatch.setattr(parse, "parse_document", pytest.fail)
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            parse,
            str(pdf),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tmp_path / "t"),
        )
    assert _sha256(pdf) == before


def test_parse_checks_the_table_files_after_parsing_and_before_any_write(
    monkeypatch, tmp_path, raw_dir
):
    document = make_document(1)
    cells = [table_cell("DICE RESULT", row=0, col=0), table_cell("4+", row=1, col=0)]
    document.add_table(
        data=TableData(num_rows=2, num_cols=1, table_cells=cells), prov=at(1, 80, 100, 500, 200)
    )
    pdf = _pdf(tmp_path)
    before = _sha256(pdf)
    output_dir, tables_dir = tmp_path / "processed", tmp_path / "tables"
    output_dir.mkdir()
    tables_dir.mkdir()
    old_markdown = output_dir / "core_rules.md"
    old_markdown.write_text("an earlier parse\n", encoding="utf-8")
    _hard_link(pdf, tables_dir / "core_rules_p001_t1.csv")  # where the first table would go
    monkeypatch.setattr(parse, "parse_document", lambda path: document)

    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            parse,
            str(pdf),
            "--output-dir",
            str(output_dir),
            "--tables-dir",
            str(tables_dir),
        )

    assert _sha256(pdf) == before
    assert old_markdown.read_text(encoding="utf-8") == "an earlier parse\n"  # not rewritten first
    assert not (output_dir / "core_rules.json").exists()


# --- chunk.py, prepare_db.py, build_index.py ---------------------------------------------------


def test_chunk_refuses_its_input_and_the_raw_folder_before_chunking(monkeypatch, tmp_path, raw_dir):
    markdown = tmp_path / "core_rules.md"
    markdown.write_text("<!-- page: 1 -->\n## GRENADE\nRoll six D6.\n", encoding="utf-8")
    before = _sha256(markdown)
    monkeypatch.setattr(chunk, "chunk_markdown", pytest.fail)
    for output in (markdown, raw_dir / "chunks.jsonl"):
        with pytest.raises(SystemExit):
            _run(monkeypatch, chunk, str(markdown), "--output", str(output))
    assert _sha256(markdown) == before
    assert list(raw_dir.iterdir()) == []


def test_prepare_db_refuses_its_input_and_the_raw_folder_before_cleaning(
    monkeypatch, tiny_db, tmp_path, raw_dir
):
    before = _sha256(tiny_db)
    monkeypatch.setattr(prepare_db, "clean_database", pytest.fail)
    for output in (tiny_db, raw_dir / "new" / "clean.sqlite"):
        with pytest.raises(SystemExit):
            _run(monkeypatch, prepare_db, str(tiny_db), "--output", str(output))
    assert _sha256(tiny_db) == before
    assert list(raw_dir.iterdir()) == []


def test_build_index_refuses_its_input_and_the_raw_folder_before_embedding(
    monkeypatch, tmp_path, raw_dir
):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        chunks_to_jsonl([Chunk("core_rules-0000", "Section: A\nB", [1], "A", "text", [])]),
        encoding="utf-8",
    )
    before = _sha256(chunks_file)
    monkeypatch.setattr(build_index, "Settings", pytest.fail)
    monkeypatch.setattr(build_index, "GeminiEmbedder", pytest.fail)
    for output in (chunks_file, raw_dir / "rules_embeddings.npz"):
        with pytest.raises(SystemExit):
            _run(monkeypatch, build_index, str(chunks_file), "--output", str(output))
    assert _sha256(chunks_file) == before
    assert list(raw_dir.iterdir()) == []
