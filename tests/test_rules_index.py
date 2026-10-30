import sys
from dataclasses import replace

import numpy as np
import pytest

from src.rag.chunker.markdown_chunker import Chunk, chunks_to_jsonl
from src.rag.config import EMBEDDING_DIM
from src.rag.storage import rules_index
from src.rag.storage.rules_index import load_embeddings, save_embeddings
from src.scripts import build_index
from tests.fakes import FakeEmbedder

MODEL = "gemini-embedding-001"


def _chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(f"core_rules-{i:04d}", text, [i + 1], f"S{i}", "text", [])
        for i, text in enumerate(texts)
    ]


CHUNKS = _chunks("Section: GRENADE\nRoll six D6.", "Section: CHARGE\nRoll 2D6.")


def _vectors(n: int) -> np.ndarray:
    return FakeEmbedder().embed([str(i) for i in range(n)], "RETRIEVAL_DOCUMENT")


def test_saved_vectors_load_back(tmp_path):
    path = tmp_path / "index.npz"
    vectors = _vectors(2)
    save_embeddings(path, vectors, CHUNKS, MODEL)
    loaded = load_embeddings(path, CHUNKS, MODEL)
    np.testing.assert_array_equal(loaded, vectors)
    assert loaded.dtype == np.float32


def test_the_file_keeps_the_name_it_was_given(tmp_path):
    path = tmp_path / "index.bin"
    save_embeddings(path, _vectors(2), CHUNKS, MODEL)
    assert path.is_file() and not (tmp_path / "index.bin.npz").exists()


@pytest.mark.parametrize(
    ("chunks", "model"),
    [
        (
            _chunks("Section: GRENADE\nRoll six D6!", "Section: CHARGE\nRoll 2D6."),
            MODEL,
        ),  # one text
        (_chunks("Section: CHARGE\nRoll 2D6.", "Section: GRENADE\nRoll six D6."), MODEL),  # re-cut
        (CHUNKS, "another-embedding-model"),
    ],
    ids=["a-text-changed", "re-cut-with-the-same-ids", "another-model"],
)
def test_an_index_of_other_chunks_or_another_model_is_refused(tmp_path, chunks, model):
    path = tmp_path / "index.npz"
    save_embeddings(path, _vectors(2), CHUNKS, MODEL)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in CHUNKS]
    with pytest.raises(ValueError, match="re-run src.scripts.build_index"):
        load_embeddings(path, chunks, model)


def test_an_index_of_another_dimension_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "index.npz"
    save_embeddings(path, _vectors(2), CHUNKS, MODEL)
    monkeypatch.setattr(rules_index, "EMBEDDING_DIM", 3072)
    with pytest.raises(ValueError, match="re-run"):
        load_embeddings(path, CHUNKS, MODEL)


def test_picture_references_are_not_part_of_the_fingerprint(tmp_path):
    path = tmp_path / "index.npz"
    save_embeddings(path, _vectors(2), CHUNKS, MODEL)
    with_pictures = [replace(chunk, pictures=["#/pictures/3"]) for chunk in CHUNKS]
    assert load_embeddings(path, with_pictures, MODEL).shape == (2, EMBEDDING_DIM)


def test_vectors_that_do_not_match_the_chunks_are_not_saved(tmp_path):
    with pytest.raises(ValueError, match="for 2 chunks"):
        save_embeddings(tmp_path / "index.npz", _vectors(3), CHUNKS, MODEL)


# --- the script, with the fake embedder ------------------------------------------------------


def _run_script(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["build_index", *args])
    build_index.main()


def test_the_script_embeds_every_chunk_and_saves_a_loadable_index(monkeypatch, tmp_path):
    embedder = FakeEmbedder()
    monkeypatch.setattr(build_index, "GeminiEmbedder", lambda settings, client_args: embedder)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(chunks_to_jsonl(CHUNKS), encoding="utf-8")
    output = tmp_path / "index" / "rules_embeddings.npz"

    _run_script(monkeypatch, str(chunks_file), "--output", str(output))

    assert embedder.calls == [([c.text for c in CHUNKS], "RETRIEVAL_DOCUMENT")]
    assert load_embeddings(output, CHUNKS, embedder.model).shape == (2, EMBEDDING_DIM)


def test_the_script_stops_without_a_key_before_embedding(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_index, "GeminiEmbedder", pytest.fail)
    monkeypatch.setenv("GOOGLE_API_KEY", " ")  # comes before the developer's .env
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(chunks_to_jsonl(CHUNKS), encoding="utf-8")
    with pytest.raises(SystemExit):
        _run_script(monkeypatch, str(chunks_file), "--output", str(tmp_path / "x.npz"))
    assert "GOOGLE_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "x.npz").exists()


def test_the_script_needs_the_chunks(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_script(monkeypatch, str(tmp_path / "missing.jsonl"))
    assert "Run src.scripts.chunk first" in capsys.readouterr().err
