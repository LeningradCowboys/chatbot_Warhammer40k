"""RAGAS's LLM and embeddings over this project's Gemini models (plan §6.9, R49).

RAGAS calls them from its event loop, several at a time. Each call runs the model's synchronous
method in a worker thread: the synchronous `httpx` client is the one that carries a run's
`AttemptCounter` (`llm/budget.py`), so every request RAGAS makes goes through the same retries,
spacing and request budget as the chat's. RAGAS's own retries are turned off: it would repeat a
whole call on any error, 400 and `BudgetExhausted` included, around the SDK's retries.
"""

import asyncio
import dataclasses

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

from src.rag.llm.gemini import Embedder

# RAGAS compares texts of the same kind with each other (questions, statements, chunks), not a
# query with documents.
SIMILARITY_TASK = "SEMANTIC_SIMILARITY"


def _single_attempt(run_config: RunConfig) -> RunConfig:
    """A copy of `run_config` with one attempt per call: the SDK retries the requests, under the
    run's budget. The caller's configuration is left as it is."""
    return dataclasses.replace(run_config, max_retries=1)


class GeminiRagasLLM(BaseRagasLLM):
    """RAGAS's LLM over a LangChain chat model: one model call per completion RAGAS asks for.

    Args:
        model: The chat model, such as `create_chat_model(settings, {"transport": counter})`.
    """

    def __init__(self, model: BaseChatModel):
        super().__init__()
        self.model = model
        self.set_run_config(RunConfig())

    def set_run_config(self, run_config: RunConfig) -> None:
        """Keep RAGAS's configuration (each metric sets its run's), with one attempt per call."""
        super().set_run_config(_single_attempt(run_config))

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Ask the model `n` times, retaining the model factory's sampling configuration.

        RAGAS's temperature argument is not forwarded: Gemini 3.6 and newer models do not use
        the legacy sampling parameter, and older configurable models already receive their
        temperature from the project's single model factory. RAGAS's callbacks are not passed
        on because its runs are not traced.
        """
        messages = prompt.to_messages()
        replies = [self.model.invoke(messages, stop=stop) for _ in range(n)]
        return LLMResult(generations=[[ChatGeneration(message=reply) for reply in replies]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """`generate_text` in a worker thread."""
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        """Whether every completion ended normally: Gemini's finish reason `STOP`, not a reply
        cut by the token limit or blocked."""
        return all(
            generation.message.response_metadata.get("finish_reason") == "STOP"
            for generation in response.generations[0]
        )


class EmbedderRagasEmbeddings(BaseRagasEmbeddings):
    """RAGAS's embeddings over an `Embedder`, every text embedded for similarity.

    Args:
        embedder: The embedder, such as `GeminiEmbedder(settings, {"transport": counter})`.
    """

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder
        self.set_run_config(RunConfig())

    def set_run_config(self, run_config: RunConfig) -> None:
        """Keep RAGAS's configuration (each metric sets its run's), with one attempt per call."""
        super().set_run_config(_single_attempt(run_config))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one call of the embedder, which batches and spaces them."""
        return self.embedder.embed(texts, SIMILARITY_TASK).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed one text, as the documents are: RAGAS compares it with texts of its kind."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """`embed_documents` in a worker thread."""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """`embed_query` in a worker thread."""
        return await asyncio.to_thread(self.embed_query, text)
