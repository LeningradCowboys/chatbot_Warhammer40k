import numpy as np
import pytest

from src.rag.chunker.markdown_chunker import Chunk
from src.rag.retriever.rules_retriever import RRF_K, RulesRetriever, tokens
from tests.fakes import FakeEmbedder

# "roll" is in 2 of the 5 chunks: a word in half the chunks or more gets no weight from BM25.
TEXTS = [
    "Section: GRENADE\nRoll six D6: for each 4+, the enemy unit suffers 1 mortal wound.",
    "Section: SMOKESCREEN\nUntil the end of the phase, your unit has the Benefit of Cover.",
    "Section: CHARGE\nRoll 2D6: the result is the maximum distance of the charge move.",
    "Section: FIGHT\nPile in, make attacks, then consolidate.",
    "Section: COMMAND\nEach player gains 1 CP.",
]
CHUNKS = [
    Chunk(f"core_rules-{i:04d}", text, [40 + i], text.split("\n")[0][9:], "text", [])
    for i, text in enumerate(TEXTS)
]
VECTORS = np.eye(len(TEXTS), dtype=np.float32)  # chunk i points along axis i


class QueryEmbedder:
    """Embeds every query as the same given vector, so the vector ranking is chosen by the test."""

    model = "fixed"

    def __init__(self, query_vector: list[float]):
        self.query_vector = np.array(query_vector, dtype=np.float32)

    def embed(self, texts: list[str], task_type: str) -> np.ndarray:
        return np.tile(self.query_vector, (len(texts), 1))


def _ids(results) -> list[str]:
    return [result.chunk.chunk_id for result in results]


def test_tokens_are_lower_cased_words():
    assert tokens("Roll six D6: [BLAST] Grenades!") == ["roll", "six", "d6", "blast", "grenades"]


def test_an_exact_keyword_lifts_its_chunk_above_the_vector_favourite():
    # The vectors favour FIGHT, but only GRENADE shares a word with the query.
    retriever = RulesRetriever(CHUNKS, VECTORS, QueryEmbedder([0.1, 0.0, 0.0, 1.0, 0.0]), [])
    results = retriever.search("grenade", top_k=5)
    assert _ids(results)[0] == "core_rules-0000"
    assert results[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))


@pytest.mark.parametrize(
    ("misspelling", "correct"),
    [("Movment", "movement"), ("attaks", "attacks"), ("eligable", "eligible")],
)
def test_an_unknown_misspelling_adds_its_close_corpus_word_to_bm25(misspelling, correct):
    texts = [f"Section: TARGET\nThe rule uses the word {correct}.", "alpha", "beta", "gamma"]
    chunks = [
        Chunk(f"chunk-{index}", text, [index + 1], "TARGET", "text", [])
        for index, text in enumerate(texts)
    ]
    vectors = np.eye(len(chunks), dtype=np.float32)
    retriever = RulesRetriever(
        chunks,
        vectors,
        QueryEmbedder([0, 0, 0, 1]),
        ["Movement Protocol", "Rapid Attacks"],
    )

    assert _ids(retriever.search(misspelling, top_k=2))[0] == "chunk-0"


def test_exact_technical_terms_are_not_expanded():
    retriever = RulesRetriever(CHUNKS, VECTORS, QueryEmbedder([1, 0, 0, 0, 0]), [])

    assert retriever._keyword_tokens("Charge attacks") == ["charge", "attacks"]


@pytest.mark.parametrize(
    "query",
    [
        "Marne Calgar",
        "Leman Russ",
        "Marne",
        "marne calgar",
        "Marne of Calgar",
        "What can Marne do?",
    ],
)
def test_known_entity_words_do_not_add_unrelated_corpus_words(query):
    texts = ["Space marine", "Keep the battlefield clean", "Unrelated passage"]
    chunks = [
        Chunk(f"chunk-{index}", text, [index + 1], "TEST", "text", [])
        for index, text in enumerate(texts)
    ]
    vectors = np.eye(len(chunks), dtype=np.float32)
    retriever = RulesRetriever(
        chunks,
        vectors,
        QueryEmbedder([0, 0, 1]),
        ["Marneus Calgar in Armour of Antilochus", "Leman Russ Battle Tank"],
    )

    assert _ids(retriever.search(query, top_k=3)) == ["chunk-2", "chunk-0", "chunk-1"]
    assert retriever._keyword_tokens(query) == tokens(query)


def test_fusion_adds_the_reciprocal_ranks_of_both_rankings():
    # Vector ranking: CHARGE, SMOKESCREEN, FIGHT, GRENADE, COMMAND. Keyword ranking: only CHARGE and
    # GRENADE share "roll"; "charge" puts CHARGE first.
    retriever = RulesRetriever(CHUNKS, VECTORS, QueryEmbedder([0.0, 0.5, 1.0, 0.2, 0.0]), [])
    results = retriever.search("roll charge", top_k=5)
    expected = {
        "core_rules-0002": 1 / (RRF_K + 1) + 1 / (RRF_K + 1),
        "core_rules-0001": 1 / (RRF_K + 2),
        "core_rules-0003": 1 / (RRF_K + 3),
        "core_rules-0000": 1 / (RRF_K + 4) + 1 / (RRF_K + 2),
        "core_rules-0004": 1 / (RRF_K + 5),
    }
    assert {r.chunk.chunk_id: r.score for r in results} == pytest.approx(expected)
    assert _ids(results) == sorted(expected, key=expected.get, reverse=True)


def test_a_query_sharing_no_word_returns_exactly_the_vector_ranking():
    retriever = RulesRetriever(CHUNKS, VECTORS, QueryEmbedder([0.3, 0.9, 0.1, 0.5, 0.0]), [])
    results = retriever.search("überhaupt nichts", top_k=5)
    assert _ids(results) == [
        "core_rules-0001",
        "core_rules-0003",
        "core_rules-0000",
        "core_rules-0002",
        "core_rules-0004",
    ]
    assert [r.score for r in results] == pytest.approx([1 / (RRF_K + r) for r in range(1, 6)])


def test_a_word_in_exactly_half_the_chunks_gives_no_keyword_rank():
    # BM25 weighs a word by its rarity: in 2 of 4 chunks, "roll" weighs 0, so GRENADE and CHARGE,
    # which share only that word with the query, get no keyword rank and keep their vector scores.
    chunks, vectors = CHUNKS[:4], VECTORS[:4, :4]
    retriever = RulesRetriever(chunks, vectors, QueryEmbedder([0.1, 0.4, 0.3, 0.2]), [])
    results = retriever.search("roll", top_k=4)
    assert _ids(results) == [
        "core_rules-0001",
        "core_rules-0002",
        "core_rules-0003",
        "core_rules-0000",
    ]
    assert [r.score for r in results] == pytest.approx([1 / (RRF_K + r) for r in range(1, 5)])


def test_search_returns_top_k_and_embeds_the_query_as_a_query():
    embedder = FakeEmbedder()
    vectors = embedder.embed(TEXTS, "RETRIEVAL_DOCUMENT")
    results = RulesRetriever(CHUNKS, vectors, embedder, []).search("charge", top_k=2)
    assert len(results) == 2
    assert embedder.calls[-1] == (["charge"], "RETRIEVAL_QUERY")


def test_a_chunk_comes_back_whole_with_its_pictures():
    with_picture = [
        *CHUNKS[:3],
        Chunk("core_rules-0003", TEXTS[3], [43], "FIGHT", "text", ["#/pictures/9"]),
        CHUNKS[4],
    ]
    retriever = RulesRetriever(with_picture, VECTORS, QueryEmbedder([0.0, 0.0, 0.0, 1.0, 0.0]), [])
    assert retriever.search("fight", top_k=1)[0].chunk == with_picture[3]


def test_vectors_that_do_not_match_the_chunks_are_refused():
    with pytest.raises(ValueError, match="3 vectors for 5 chunks"):
        RulesRetriever(CHUNKS, VECTORS[:3], QueryEmbedder([1.0, 0.0, 0.0, 0.0, 0.0]), [])


def test_exact_sections_are_returned_in_document_order_without_embedding():
    embedder = FakeEmbedder()
    retriever = RulesRetriever(CHUNKS, VECTORS, embedder, [])
    found = retriever.sections([" fight ", "grenade"])
    assert _ids(found) == ["core_rules-0000", "core_rules-0003"]
    assert [item.score for item in found] == [0.0, 0.0]
    assert embedder.calls == []
