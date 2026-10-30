"""Search the rule chunks: embeddings and BM25 keyword scores, fused by Reciprocal Rank Fusion.

The retriever works on chunks and vectors already in memory; it reads no file (plan §6.6).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import get_close_matches

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.chunker.markdown_chunker import Chunk
from src.rag.llm.gemini import Embedder

RRF_K = 60  # the usual constant of Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank)
WORD = re.compile(r"\w+")
FUZZY_MIN_LENGTH = 4
FUZZY_CUTOFF = 0.8


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk found by a search, with its fused score."""

    chunk: Chunk
    score: float


def tokens(text: str) -> list[str]:
    """The lower-cased words of a text, as BM25 counts them."""
    return WORD.findall(text.lower())


class RulesRetriever:
    """Hybrid search over the rule chunks."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder: Embedder,
        entity_names: Iterable[str],
    ):
        """Prepare the search.

        Args:
            chunks: The rule chunks.
            vectors: Their normalised embeddings, row i for chunk i (`load_embeddings`).
            embedder: Embeds the queries, with the model that built `vectors`.
            entity_names: Names from explicit entity-name fields in the datasheet database.
        """
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        documents = [tokens(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(documents)
        self._vocabulary = frozenset(token for document in documents for token in document)
        self._entity_words = frozenset(word for name in entity_names for word in tokens(name))

    def _is_entity_word(self, word: str) -> bool:
        """Whether `word` is a known entity token or a shortened prefix of one."""
        return word in self._entity_words or any(
            entity_word.startswith(word) for entity_word in self._entity_words
        )

    def _keyword_tokens(self, query: str) -> list[str]:
        """Return query words plus close corpus spellings, preserving known entity words."""
        query_tokens = tokens(query)
        expanded = list(query_tokens)
        vocabulary = sorted(self._vocabulary)
        for token in query_tokens:
            if (
                len(token) < FUZZY_MIN_LENGTH
                or token in self._vocabulary
                or self._is_entity_word(token)
            ):
                continue
            match = get_close_matches(token, vocabulary, n=1, cutoff=FUZZY_CUTOFF)
            if match:
                expanded.append(match[0])
        return expanded

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Find the chunks that best match a query.

        1. Every chunk is ranked by the cosine of its vector with the query's (a dot product: the
           vectors are normalised).
        2. The chunks that share a word with the query are ranked by BM25. A chunk sharing none
           has score 0 and gets no BM25 rank: its place among the zeros would only be tie order.
        3. Each chunk scores the sum of 1 / (RRF_K + rank) over the rankings that hold it.

        Args:
            query: The search text, written by the model in the rulebook's terms.
            top_k: How many chunks to return.

        Returns:
            At most `top_k` chunks, best first.
        """
        query_vector = self._embedder.embed([query], "RETRIEVAL_QUERY")[0]
        by_vector = np.argsort(-(self._vectors @ query_vector), kind="stable")
        bm25 = self._bm25.get_scores(self._keyword_tokens(query))
        by_keyword = [int(i) for i in np.argsort(-bm25, kind="stable") if bm25[i] > 0]

        fused = np.zeros(len(self._chunks))
        for ranking in (by_vector, by_keyword):
            for rank, index in enumerate(ranking, start=1):
                fused[index] += 1 / (RRF_K + rank)
        best = np.argsort(-fused, kind="stable")[:top_k]
        return [RetrievedChunk(self._chunks[i], float(fused[i])) for i in best]

    def sections(self, names: list[str]) -> list[RetrievedChunk]:
        """Return chunks whose structural section label exactly matches one of `names`.

        Args:
            names: Section labels from the parsed rulebook.

        Returns:
            Matching chunks in document order, without a relevance score.
        """
        wanted = {name.strip().upper() for name in names}
        return [
            RetrievedChunk(chunk, 0.0)
            for chunk in self._chunks
            if chunk.section.strip().upper() in wanted
        ]
