"""Offline tests of the Gemini embedder: a mock transport stands in for the API, nothing is sent."""

import json
import re
import threading
import time

import httpx
import numpy as np
import pytest
from google.genai import errors

from src.rag.config import EMBEDDING_DIM, Settings
from src.rag.llm.gemini import (
    EMBED_BATCH_SIZE,
    EMBED_TEXTS_PER_MINUTE,
    Embedder,
    GeminiEmbedder,
    TextRateLimit,
)
from tests.fakes import FakeEmbedder

SETTINGS = Settings(google_api_key="test", _env_file=None)


def _values(text: str) -> list[float]:
    """The fake server's embedding: not normalised, and different for texts of different length."""
    return [float(len(text)), 1.0] + [0.0] * (EMBEDDING_DIM - 2)


class FakeClock:
    """A monotonic clock that moves only when the code sleeps or the test says so."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeServer:
    """Answers `batchEmbedContents` like the Gemini API, from a list of HTTP statuses to play.

    `replace` gives the values to answer instead for some texts, by their position (from 1) in
    everything sent so far.
    """

    def __init__(
        self,
        statuses: list[int] | None = None,
        drop: int = 0,
        replace: dict[int, list[float]] | None = None,
        refusal_delay: float = 0.0,
    ):
        self.statuses = statuses or []
        self.drop = drop  # embeddings to leave out of each reply
        self.replace = replace or {}
        self.requests: list[list[dict]] = []
        self.sent = 0  # texts received so far
        self.clock: FakeClock | None = None  # when set, the time of each request is kept
        self.times: list[float] = []
        # seconds the clock moves after a refusal, standing in for the SDK's wait before retrying
        self.refusal_delay = refusal_delay

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(":batchEmbedContents")
        status = self.statuses.pop(0) if self.statuses else 200
        entries = json.loads(request.read())["requests"]
        self.requests.append(entries)
        if self.clock is not None:
            self.times.append(self.clock())
        if status != 200:
            if self.clock is not None:
                self.clock.now += self.refusal_delay
            return httpx.Response(status, json={"error": {"code": status, "message": "refused"}})
        texts = [entry["content"]["parts"][0]["text"] for entry in entries]
        embeddings = [
            {"values": self.replace.get(self.sent + offset + 1, _values(text))}
            for offset, text in enumerate(texts[: len(texts) - self.drop])
        ]
        self.sent += len(texts)
        return httpx.Response(200, content=json.dumps({"embeddings": embeddings}))


def _embedder(server: FakeServer, clock: FakeClock | None = None) -> GeminiEmbedder:
    clock = clock or FakeClock()
    server.clock = clock
    limit = TextRateLimit(EMBED_TEXTS_PER_MINUTE, clock=clock, sleep=clock.sleep)
    return GeminiEmbedder(SETTINGS, {"transport": httpx.MockTransport(server)}, limit)


def test_texts_are_sent_in_batches_with_the_task_and_dimension():
    server = FakeServer()
    texts = [f"text {'x' * i}" for i in range(2 * EMBED_BATCH_SIZE + 50)]

    vectors = _embedder(server).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(batch) for batch in server.requests] == [100, 100, 50]
    entries = [entry for batch in server.requests for entry in batch]
    assert {(e["taskType"], e["outputDimensionality"]) for e in entries} == {
        ("RETRIEVAL_DOCUMENT", EMBEDDING_DIM)
    }
    assert [e["content"]["parts"][0]["text"] for e in entries] == texts
    assert vectors.shape == (len(texts), EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_vectors_are_normalised_and_in_the_order_of_the_texts():
    texts = ["a", "a much longer text"]
    vectors = _embedder(FakeServer()).embed(texts, "RETRIEVAL_QUERY")

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    for text, vector in zip(texts, vectors, strict=True):
        expected = np.array(_values(text)) / np.linalg.norm(_values(text))
        np.testing.assert_allclose(vector, expected, rtol=1e-6)


def test_a_rate_limit_is_retried():
    server = FakeServer(statuses=[429])
    vectors = _embedder(server).embed(["a"], "RETRIEVAL_QUERY")
    assert len(server.requests) == 2
    assert vectors.shape == (1, EMBEDDING_DIM)


def test_another_error_is_raised_without_retry():
    server = FakeServer(statuses=[400])
    with pytest.raises(errors.ClientError):
        _embedder(server).embed(["a"], "RETRIEVAL_QUERY")
    assert len(server.requests) == 1


def test_a_reply_missing_embeddings_is_an_error():
    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        _embedder(FakeServer(drop=1)).embed(["a", "b"], "RETRIEVAL_DOCUMENT")


def test_no_text_sends_no_request():
    server = FakeServer()
    assert _embedder(server).embed([], "RETRIEVAL_DOCUMENT").shape == (0, EMBEDDING_DIM)
    assert server.requests == []


def test_the_fake_embedder_is_deterministic_and_normalised():
    embedder: Embedder = FakeEmbedder()
    first = embedder.embed(["grenade", "stratagem"], "RETRIEVAL_DOCUMENT")
    again = embedder.embed(["stratagem"], "RETRIEVAL_QUERY")

    np.testing.assert_array_equal(first[1], again[0])
    assert not np.allclose(first[0], first[1])
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)
    assert embedder.calls == [
        (["grenade", "stratagem"], "RETRIEVAL_DOCUMENT"),
        (["stratagem"], "RETRIEVAL_QUERY"),
    ]


ZERO = [0.0] * EMBEDDING_DIM


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        (ZERO, "length 0.0"),
        ([float("nan")] + [1.0] * (EMBEDDING_DIM - 1), "non-finite values"),
        ([float("inf")] + [1.0] * (EMBEDDING_DIM - 1), "non-finite values"),
        ([1.0] * (EMBEDDING_DIM - 1), f"{EMBEDDING_DIM - 1} dimensions instead of {EMBEDDING_DIM}"),
    ],
    ids=["zero", "nan", "inf", "dimension"],
)
def test_an_embedding_that_cannot_be_normalised_is_an_error(values, problem):
    server = FakeServer(replace={2: values})
    with pytest.raises(
        RuntimeError, match=rf"^text 2 \(1-based\): .*{re.escape(problem)}"
    ) as error:
        _embedder(server).embed(["first", "secret text"], "RETRIEVAL_DOCUMENT")
    assert "secret" not in str(error.value)


def test_an_invalid_second_batch_stops_before_the_third():
    server = FakeServer(replace={EMBED_BATCH_SIZE + 37: ZERO})
    texts = [f"text {i}" for i in range(2 * EMBED_BATCH_SIZE + 50)]
    with pytest.raises(RuntimeError, match=r"^text 137 \(1-based\)") as error:
        _embedder(server).embed(texts, "RETRIEVAL_DOCUMENT")
    assert "text 136" not in str(error.value)  # the message names the position, not the text
    assert len(server.requests) == 2


# --- the rate limit: texts per minute, across calls ----------------------------------------------


def test_batches_over_the_limit_wait_for_the_minute_to_pass():
    server, clock = FakeServer(), FakeClock()
    texts = [f"text {i}" for i in range(255)]
    _embedder(server, clock).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(batch) for batch in server.requests] == [100, 100, 55]
    assert server.times == [0.0, 60.0, 120.0]
    assert clock.sleeps == [60.0, 60.0]


def test_single_queries_in_a_row_do_not_wait():
    server, clock = FakeServer(), FakeClock()
    embedder = _embedder(server, clock)
    for i in range(10):
        embedder.embed([f"query {i}"], "RETRIEVAL_QUERY")
        clock.now += 1.0  # the time a request and its answer take
    assert len(server.requests) == 10
    assert clock.sleeps == []


def test_two_calls_that_pass_the_limit_together_wait():
    server, clock = FakeServer(), FakeClock()
    embedder = _embedder(server, clock)
    embedder.embed([f"a {i}" for i in range(60)], "RETRIEVAL_DOCUMENT")
    clock.now = 10.0
    embedder.embed([f"b {i}" for i in range(60)], "RETRIEVAL_DOCUMENT")

    assert clock.sleeps == [50.0]  # until the first 60 texts are a minute old
    assert server.times == [0.0, 60.0]


def test_threads_sharing_the_rate_limit_send_one_batch_at_a_time():
    clock = FakeClock()
    limit = TextRateLimit(EMBED_TEXTS_PER_MINUTE, clock=clock, sleep=clock.sleep)

    def send() -> None:
        with limit.sending(60):
            time.sleep(0.01)  # the request, while the other thread tries to send

    threads = [threading.Thread(target=send) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert clock.sleeps == [60.0]  # the second batch waited for the first to be a minute old


def test_after_a_minute_the_limit_is_free_again():
    server, clock = FakeServer(), FakeClock()
    embedder = _embedder(server, clock)
    embedder.embed([f"a {i}" for i in range(100)], "RETRIEVAL_DOCUMENT")
    clock.now = 61.0
    embedder.embed(["query"], "RETRIEVAL_QUERY")
    assert clock.sleeps == []


def test_a_batch_retried_after_a_refusal_counts_from_its_success():
    server, clock = FakeServer(statuses=[429], refusal_delay=30.0), FakeClock()
    texts = [f"text {i}" for i in range(200)]
    _embedder(server, clock).embed(texts, "RETRIEVAL_DOCUMENT")

    assert [len(batch) for batch in server.requests] == [100, 100, 100]
    assert server.times == [0.0, 30.0, 90.0]  # the second batch a minute after the first succeeded
    assert clock.sleeps == [60.0]


def test_no_text_neither_waits_nor_sends():
    server, clock = FakeServer(), FakeClock()
    embedder = _embedder(server, clock)
    embedder.embed([f"a {i}" for i in range(100)], "RETRIEVAL_DOCUMENT")
    assert embedder.embed([], "RETRIEVAL_QUERY").shape == (0, EMBEDDING_DIM)
    assert (clock.sleeps, len(server.requests)) == ([], 1)
