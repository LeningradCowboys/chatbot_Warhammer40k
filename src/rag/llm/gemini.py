"""The one module that builds Gemini objects; every other module receives them as parameters."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import httpx
import numpy as np
from google import genai
from google.genai import errors, types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from src.rag.config import EMBEDDING_DIM, Settings

# Attempts per request, the first one included. The SDK waits about 1, 2, 4, 8, 16 s between them,
# about 30 s in all: a short burst of rate limiting (HTTP 429) can pass, but getting past a limit
# per minute is not guaranteed, and a spent daily limit fails.
MAX_ATTEMPTS = 6
EMBED_BATCH_SIZE = 100  # texts per embedding request, the API's limit
# Observed on 2026-10-15 for this project's key and gemini-embedding-001 on the free tier: at most
# 100 texts embedded per minute, each text of a batch counting. Not a fixed limit of the API: the
# limits depend on the project and its tier.
EMBED_TEXTS_PER_MINUTE = 100
# Observed on 2026-10-12 for this project's key and gemini-2.5-flash on the free tier: 5 requests
# per minute (and 20 per day). A run that counts its requests spaces them by this much.
MIN_SECONDS_BETWEEN_CHAT_REQUESTS = 12.0
REQUEST_ERRORS = (
    errors.APIError,
    ChatGoogleGenerativeAIError,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


class TextRateLimit:
    """Spaces the texts sent so that no minute holds more than `limit` of them.

    It remembers what was sent across calls, so a batch that would take the last minute's texts
    over the limit waits until enough of them are older than a minute; a text sent alone rarely
    waits. Only the texts sent through this object are counted. Threads sharing it (RAGAS embeds
    from several at once) send one batch at a time.
    """

    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()  # (time counted, texts), oldest first
        self._lock = threading.Lock()

    @contextmanager
    def sending(self, count: int) -> Iterator[None]:
        """Wait until `count` more texts fit in the last minute, then send them inside the block.

        The texts are counted at the time the block ends, whether it succeeds or fails: a batch
        retried after a refusal may only have gone through at its last attempt, so counting from
        the first would free the limit too early.
        """
        with self._lock:
            while True:
                now = self._clock()
                while self._sent and self._sent[0][0] <= now - self.window:
                    self._sent.popleft()
                if not self._sent or sum(n for _, n in self._sent) + count <= self.limit:
                    break
                wait = self._sent[0][0] + self.window - now
                logger.info(
                    "waiting %.0f s: at most %d texts per minute are embedded", wait, self.limit
                )
                self._sleep(wait)
            try:
                yield
            finally:
                self._sent.append((self._clock(), count))


class Embedder(Protocol):
    """Turns texts into vectors: one L2-normalised row per text, in order."""

    model: str  # the embedding model's name, part of the index fingerprint

    def embed(self, texts: list[str], task_type: str) -> np.ndarray: ...


class GeminiEmbedder:
    """An `Embedder` over Gemini's embedding model, through the `google-genai` SDK."""

    def __init__(
        self,
        settings: Settings,
        http_client_args: dict[str, Any] | None = None,
        rate_limit: TextRateLimit | None = None,
    ):
        """Create the embedder.

        Args:
            settings: The runtime settings: API key and embedding model name.
            http_client_args: Extra arguments for the underlying `httpx` client, such as a test
                transport.
            rate_limit: Spaces the batches; by default `EMBED_TEXTS_PER_MINUTE` on the real clock.
        """
        self.model = settings.embedding_model
        self._rate_limit = rate_limit or TextRateLimit(EMBED_TEXTS_PER_MINUTE)
        self._client = genai.Client(
            api_key=settings.google_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=MAX_ATTEMPTS),
                client_args=http_client_args,
            ),
        )

    def embed(self, texts: list[str], task_type: str) -> np.ndarray:
        """Embed texts in batches of at most `EMBED_BATCH_SIZE`, each spaced by the rate limit.

        Args:
            texts: The texts, in order.
            task_type: `RETRIEVAL_DOCUMENT` for the chunks, `RETRIEVAL_QUERY` for a query,
                `SEMANTIC_SIMILARITY` for texts compared with each other (RAGAS).

        Returns:
            A float32 array of shape (len(texts), EMBEDDING_DIM), each row of length 1, so that
            a dot product is a cosine.

        Raises:
            RuntimeError: A reply does not hold one valid embedding per text sent. Each batch is
                checked as soon as it comes back, so no later batch is sent after an invalid one.
        """
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM)
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            with self._rate_limit.sending(len(batch)):
                response = self._client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(f"{len(embeddings)} embeddings for {len(batch)} texts")
            for offset, embedding in enumerate(embeddings):
                rows.append(_valid_vector(embedding.values or [], start + offset + 1))
        if not rows:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = np.stack(rows)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _valid_vector(values: list[float], position: int) -> np.ndarray:
    """The embedding of one text as a float32 vector, or an error naming the text's position.

    An embedding that cannot be normalised (wrong dimension, a value that is not finite, length 0)
    is an invalid reply: it is reported, never patched.

    Args:
        values: The values the API returned.
        position: The text's position in the whole input list, counting from 1.
    """
    if len(values) != EMBEDDING_DIM:
        raise RuntimeError(
            f"text {position} (1-based): {len(values)} dimensions instead of {EMBEDDING_DIM}"
        )
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise RuntimeError(f"text {position} (1-based): its embedding has non-finite values")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError(f"text {position} (1-based): its embedding has length {norm}")
    return vector


def create_chat_model(
    settings: Settings, http_client_args: dict[str, Any] | None = None
) -> ChatGoogleGenerativeAI:
    """Create the chat model the graph binds its tools to.

    Args:
        settings: The runtime settings: API key, model name and sampling temperature.
        http_client_args: Extra arguments for the underlying `httpx` client, such as a transport
            that records the traffic (the online compatibility check uses it).

    Returns:
        LangChain's Gemini chat model.
    """
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.google_api_key,
        max_retries=MAX_ATTEMPTS,
        client_args=http_client_args,
    )
