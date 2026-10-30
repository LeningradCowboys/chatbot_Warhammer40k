"""Fakes for Gemini and the rules retriever that calls it: no network in unit tests (AGENTS.md)."""

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    message_chunk_to_message,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SkipValidation

from src.rag.chunker.markdown_chunker import Chunk
from src.rag.config import EMBEDDING_DIM
from src.rag.llm.budget import BudgetExhausted
from src.rag.retriever.rules_retriever import RetrievedChunk


class FakeEmbedder:
    """An `Embedder` whose vector for a text is fixed by the text's SHA-256.

    The same text always gets the same vector; tests that need a given ranking build their vectors
    by hand. Every call is recorded as (texts, task type).
    """

    model = "fake-embedding"

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], task_type: str) -> np.ndarray:
        self.calls.append((list(texts), task_type))
        rows = [
            np.random.default_rng(
                int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
            ).standard_normal(self.dim)
            for text in texts
        ]
        vectors = np.array(rows, dtype=np.float32).reshape(len(texts), self.dim)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@dataclass(frozen=True)
class FakeCall:
    """What one call of the fake chat model received."""

    messages: list[BaseMessage]
    tool_choice: str | dict[str, Any] | None
    options: dict[str, Any] = field(default_factory=dict)  # stop and the keyword arguments given


class FakeChatModel(BaseChatModel):
    """Returns scripted replies, one per call, and records every call it receives.

    A reply is a message, or the chunks a streamed reply arrives in; streamed, a message is one
    chunk. `bind_tools` returns a copy that shares the replies and the record, so the calls of every
    binding (tools allowed or disabled) land in one list, in order.
    """

    # not validated: pydantic would try to build a message from a list of chunks
    replies: list[SkipValidation[AIMessage | list[AIMessageChunk]]]
    calls: list[FakeCall] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-chat-model", "temperature": 0.0}

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "FakeChatModel":
        return self.model_copy(update={"tool_choice": tool_choice})

    def _next_reply(
        self, messages: list[BaseMessage], options: dict[str, Any]
    ) -> AIMessage | list[AIMessageChunk]:
        call = FakeCall(messages=list(messages), tool_choice=self.tool_choice, options=options)
        self.calls.append(call)
        return self.replies.pop(0)

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        reply = self._next_reply(messages, {"stop": stop, **kwargs})
        if isinstance(reply, list):
            reply = message_chunk_to_message(sum(reply[1:], reply[0]))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._next_reply(messages, {"stop": stop, **kwargs})
        for chunk in reply if isinstance(reply, list) else [_as_chunk(reply)]:
            yield ChatGenerationChunk(message=chunk)


SCHEMA_INTRO = "JSON Schema:\n"  # what precedes the output schema in a RAGAS prompt
# What `RagasChatModel` answers, by the title of the output schema a RAGAS prompt asks for, from the
# prompt's input: the test set's generation (graph, personas, scenarios, questions), then the
# metrics. Each reply is valid for its schema and as plain as it can be.
RAGAS_REPLIES: dict[str, Callable[[dict], dict]] = {
    "StringIO": lambda data: {"text": "A summary of the rule."},
    "QuestionPotentialOutput": lambda data: {"score": 5},
    "ThemesAndConcepts": lambda data: {"output": ["dice", "rolls"]},
    "NEROutput": lambda data: {"entities": ["D6", "D3"]},
    "Persona": lambda data: {"name": "New player", "role_description": "Learns the rules."},
    "PersonaThemesMapping": lambda data: {
        "mapping": {persona["name"]: data["themes"] for persona in data["personas"]}
    },
    "ConceptCombinations": lambda data: {
        "combinations": [[concepts[0] for concepts in data["lists_of_concepts"]]]
    },
    "GeneratedQueryAnswer": lambda data: {
        "query": f"How do the rules use {data.get('term') or ', '.join(data['themes'])}?",
        "answer": "As the rulebook says.",
    },
    "StatementGeneratorOutput": lambda data: {"statements": [data["answer"]]},
    "ClaimDecompositionOutput": lambda data: {"claims": [data["response"]]},
    "NLIStatementOutput": lambda data: {
        "statements": [
            {"statement": statement, "reason": "In the context.", "verdict": 1}
            for statement in data["statements"]
        ]
    },
    "ResponseRelevanceOutput": lambda data: {"question": data["response"], "noncommittal": 0},
    "Verification": lambda data: {"reason": "Useful for the answer.", "verdict": 1},
    "ContextRecallClassifications": lambda data: {
        "classifications": [
            {"statement": data["answer"], "reason": "In the context.", "attributed": 1}
        ]
    },
}


def ragas_prompt(prompt: str) -> tuple[str, dict]:
    """The title of the output schema a RAGAS prompt asks for, and the prompt's input.

    A RAGAS prompt states the JSON schema of its output, then ends with its input as JSON.
    """
    decoder = json.JSONDecoder()
    title = decoder.raw_decode(prompt, prompt.index(SCHEMA_INTRO) + len(SCHEMA_INTRO))[0]["title"]
    data = decoder.raw_decode(prompt, prompt.rindex("input: ") + len("input: "))[0]
    return title, data


def ragas_reply(prompt: str) -> tuple[str, str]:
    """The title of the output schema a RAGAS prompt asks for, and the reply of `RAGAS_REPLIES` to
    its input, as JSON text."""
    title, data = ragas_prompt(prompt)
    return title, json.dumps(RAGAS_REPLIES[title](data))


class RagasChatModel(BaseChatModel):
    """Answers RAGAS's prompts by the output schema each asks for (`RAGAS_REPLIES`).

    `replies` replaces the replies of some schemas, for a test that needs other verdicts. The
    schema title and the input of every call are recorded. After `limit` calls, it raises
    `BudgetExhausted` as the request budget would, before answering.
    """

    replies: dict[str, Callable[[dict], dict]] = Field(default_factory=dict)
    titles: list[str] = Field(default_factory=list)
    inputs: list[dict] = Field(default_factory=list)
    limit: int | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-ragas-model"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if self.limit is not None and len(self.titles) >= self.limit:
            raise BudgetExhausted(f"request budget of {self.limit} reached: the run is incomplete")
        title, data = ragas_prompt(messages[-1].text)
        self.titles.append(title)
        self.inputs.append(data)
        text = json.dumps(self.replies.get(title, RAGAS_REPLIES[title])(data))
        reply = AIMessage(text, response_metadata={"finish_reason": "STOP"})
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _as_chunk(message: AIMessage) -> AIMessageChunk:
    """A message as the single chunk of a streamed reply."""
    return AIMessageChunk(
        content=message.content,
        additional_kwargs=message.additional_kwargs,
        response_metadata=message.response_metadata,
        tool_call_chunks=[
            tool_call_chunk(
                name=call["name"], args=json.dumps(call["args"]), id=call["id"], index=i
            )
            for i, call in enumerate(message.tool_calls)
        ],
        id=message.id,
    )


def tool_call(name: str, args: dict, call_id: str) -> dict:
    """A tool call as a model message carries it."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


GRENADE = Chunk(
    "core_rules-0150",
    "Section: GRENADE\nRoll six D6.",
    [34, 35],
    "GRENADE",
    "text",
    ["#/pictures/7"],
)
COVER = Chunk("core_rules-0151", "Section: COVER\n+1 to saves.", [44], "COVER", "text", [])


class FakeRules:
    """Returns the same chunks for every query, GRENADE then COVER unless others are given, and
    records the queries."""

    def __init__(
        self,
        found: list[RetrievedChunk] | None = None,
        related: list[RetrievedChunk] | None = None,
    ):
        self.queries: list[tuple[str, int]] = []
        self.found = found or [RetrievedChunk(GRENADE, 0.032), RetrievedChunk(COVER, 0.016)]
        self.related = related or []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.queries.append((query, top_k))
        return list(self.found)

    def sections(self, names: list[str]) -> list[RetrievedChunk]:
        """Return exact structural sections from the fake's fixed chunks."""
        wanted = {name.strip().upper() for name in names}
        return [
            item
            for item in [*self.found, *self.related]
            if item.chunk.section.strip().upper() in wanted
        ]
