"""The real rules retriever: each reference query finds its supporting section and passage.

Online: each query is embedded by Gemini (one request each, on the embedding model's quota). Needs
the clean database and the index; skipped when either is missing or without a key.
"""

import pytest
from pydantic import ValidationError

from src.rag.chunker.markdown_chunker import jsonl_to_chunks
from src.rag.config import CLEAN_DB, RULES_CHUNKS, RULES_EMBEDDINGS, TOP_K, Settings, key_not_set
from src.rag.llm.gemini import GeminiEmbedder
from src.rag.retriever.rules_retriever import RulesRetriever
from src.rag.retriever.sql_retriever import load_entity_names
from src.rag.storage.rules_index import load_embeddings

pytestmark = pytest.mark.online


@pytest.fixture(scope="module")
def retriever() -> RulesRetriever:
    try:
        settings = Settings()
    except ValidationError as error:
        if key_not_set(error):
            pytest.skip("GOOGLE_API_KEY is not set")
        raise
    if not RULES_EMBEDDINGS.exists():
        pytest.skip("rules_embeddings.npz not built: run src.scripts.build_index first")
    if not CLEAN_DB.exists():
        pytest.skip("wahadb_clean.sqlite not built: run src.scripts.prepare_db first")
    chunks = jsonl_to_chunks(RULES_CHUNKS.read_text(encoding="utf-8"))
    embedder = GeminiEmbedder(settings)
    return RulesRetriever(
        chunks,
        load_embeddings(RULES_EMBEDDINGS, chunks, embedder.model),
        embedder,
        load_entity_names(CLEAN_DB),
    )


# The queries the model would write for the rules reference questions (plan §7, lab2 §5.3).
@pytest.mark.parametrize(
    ("query", "expected_section", "required_text"),
    [
        ("Grenade stratagem", "GRENADE", "Roll six D6"),
        ("use a Stratagem command points", "STRATAGEMS", "Command points"),
        ("scoring points victory points", "CAPTURE AND CONTROL", "maximum of 3VP"),
    ],
)
def test_a_reference_question_finds_the_supporting_passage(
    retriever, query, expected_section, required_text
):
    assert any(
        result.chunk.section == expected_section and required_text in result.chunk.text
        for result in retriever.search(query, TOP_K)
    )
