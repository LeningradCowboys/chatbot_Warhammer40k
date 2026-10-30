"""Tools of the compatibility check of the chat model (plan §8, R26, R32).

The online test (`test_gemini_online.py`) uses them against the real API; `test_gemini_check.py`
tests them offline, so that no real request is spent on debugging them. They record the HTTP
traffic of the chat model (the raw body of every generate request and of its reply, never the
headers, so the API key is not kept), compare the thought signatures of each reply with what the
next request sends back, and build the graph of plan §6.8 reduced to what the check needs.
"""

import base64
import binascii
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# The chat's own final instruction, so that the check tests the text the chat sends (R41).
from src.rag.chatbot.prompts import FINAL_INSTRUCTION
from src.rag.llm.budget import RequestBudget

# One online session spends at most the daily count observed for the chat model (20), at the pace
# of `MIN_SECONDS_BETWEEN_CHAT_REQUESTS`.
REQUEST_BUDGET = 20

SYSTEM_PROMPT = (
    "You answer questions about a fictional game. You know nothing about its units: "
    "always use the tools to look up values and to compute, never guess or compute yourself."
)
# The prompt of the `final` call: the agent's asks for tools, which this call cannot use (R40).
FINAL_PROMPT = (
    "You answer questions about a fictional game. The tool budget for this question is spent: "
    "no tool can be called any more. Answer now, in text, from the tool results above only. "
    "If they are not enough, give what they show and say what is missing; never guess a value."
)
TOUGHNESS = {"Zorvak Sentinel": 7, "Quillon Brute": 9, "Mirethan Scout": 4}


@tool
def unit_toughness(name: str) -> int:
    """Return the toughness of a unit of the fictional game, by its exact name."""
    return TOUGHNESS[name]


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def unit_lookup(name: str) -> int:
    """Return the toughness of a unit of the fictional game from the unit database."""
    raise ConnectionError("unit database offline")


@dataclass
class Exchange:
    """One attempt of a generate request and its reply, as sent and received."""

    request: dict
    streamed: bool
    status: int  # the reply's HTTP status: a refused attempt (429) is no turn of the conversation
    encoding: str  # the reply's Content-Encoding
    reply: bytearray = field(default_factory=bytearray)

    @property
    def succeeded(self) -> bool:
        return self.status == 200

    def reply_payloads(self) -> list[dict]:
        """The reply's JSON objects: one, or one per server-sent event of a streamed reply."""
        raw = bytes(self.reply)
        if self.encoding == "gzip":
            raw = gzip.decompress(raw)
        elif self.encoding not in ("", "identity"):
            raise ValueError(f"unexpected Content-Encoding {self.encoding!r}")
        text = raw.decode("utf-8")
        if not self.streamed:
            return [json.loads(text)]
        return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]

    def reply_parts(self) -> list[dict]:
        """The parts of the reply, in order; a streamed reply is the concatenation of its events."""
        return [
            part
            for payload in self.reply_payloads()
            for candidate in payload.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        ]


class _Tee(httpx.SyncByteStream):
    """Passes a response body through while keeping a copy of it."""

    def __init__(self, stream: httpx.SyncByteStream, copy: bytearray):
        self._stream = stream
        self._copy = copy

    def __iter__(self) -> Iterator[bytes]:
        for piece in self._stream:
            self._copy.extend(piece)
            yield piece

    def close(self) -> None:
        self._stream.close()


class RecordingTransport(httpx.BaseTransport):
    """An httpx transport that records every attempt of a generate request, and its reply.

    Args:
        budget: Counted before each generate request is sent; `None` counts nothing.
        inner: The transport that sends the requests; the network by default.
    """

    def __init__(
        self, budget: RequestBudget | None = None, inner: httpx.BaseTransport | None = None
    ):
        self._budget = budget
        self._inner = inner or httpx.HTTPTransport()
        self.exchanges: list[Exchange] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.lower()
        if "generatecontent" not in path:
            return self._inner.handle_request(request)
        if self._budget is not None:
            self._budget.spend()
        response = self._inner.handle_request(request)
        exchange = Exchange(
            request=json.loads(request.read()),
            streamed="streamgeneratecontent" in path,
            status=response.status_code,
            encoding=response.headers.get("content-encoding", ""),
        )
        self.exchanges.append(exchange)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_Tee(response.stream, exchange.reply),
            extensions=response.extensions,
            request=request,
        )

    def close(self) -> None:
        self._inner.close()


# What the adapter sends for a Gemini 3 part that lost its signature: valid base64, not a signature.
PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"
# Where the adapter keeps signatures in a stored message (langchain-google-genai 4.4.1): a map from
# tool call ID to signature in `additional_kwargs`, and, in a list of content blocks, a block's
# `signature` or `thought_signature`, or the `signature` in its `extras`.
SIGNATURE_MAP_KEY = "__gemini_function_call_thought_signatures__"
BLOCK_SIGNATURE_KEYS = ("signature", "thought_signature")


def signature_bytes(value: str | bytes) -> bytes:
    """A thought signature as its bytes, whichever base64 alphabet carried it.

    Gemini returns standard base64 (`+`, `/`); the SDK sends the URL-safe form (`-`, `_`) back. Both
    stand for the same bytes, so signatures are compared decoded. The placeholder, and text that is
    not base64, fail the check.
    """
    if isinstance(value, bytes):
        return value
    assert value != PLACEHOLDER_SIGNATURE, "a placeholder stands where a signature was expected"
    try:
        return base64.b64decode(value.replace("-", "+").replace("_", "/"), validate=True)
    except (binascii.Error, ValueError) as error:  # ValueError: characters that are not ASCII
        raise AssertionError(f"a signature is not base64: {type(error).__name__}") from error


def _part_name(part: dict) -> tuple:
    """What a part is, without its signature: a given function call, a text, or another part."""
    call = part.get("functionCall")
    if call is not None:
        return ("call", call["name"], json.dumps(call.get("args", {}), sort_keys=True))
    if set(part) <= {"text", "thought", "thoughtSignature"}:
        return ("thought" if part.get("thought") else "text", part.get("text", ""))
    return ("other", json.dumps({k: v for k, v in part.items() if k != "thoughtSignature"}))


def turn_parts(parts: list[dict], streamed: bool) -> list[tuple[tuple, list[bytes]]]:
    """Every part of one model turn, unsigned ones included, each with its signatures in order.

    Signatures are compared as bytes (`signature_bytes`), in this check only: what is sent to the
    model is never changed. A reply that was not streamed keeps its parts as they are: a signature
    moved to another part, even to an identical text, changes the list. The text of a streamed reply
    comes as pieces spread over its events, and consecutive text pieces are merged into one part
    named by the joined text. With the current recording and merging, adjacent text parts of a
    streamed reply cannot be told apart reliably, so a signature moved between them is not detected.
    A request turn is grouped the same way as the reply it sends back.
    """
    grouped: list[tuple[tuple, list[bytes]]] = []
    for part in parts:
        name = _part_name(part)
        signature = part.get("thoughtSignature")
        signatures = [signature_bytes(signature)] if signature is not None else []
        if streamed and grouped and name[0] in ("text", "thought") and grouped[-1][0][0] == name[0]:
            previous, earlier = grouped.pop()
            grouped.append(((name[0], previous[1] + name[1]), earlier + signatures))
        else:
            grouped.append((name, signatures))
    return grouped


def turns(exchanges: list[Exchange]) -> list[Exchange]:
    """The attempts that got a reply: the turns of the conversation, refused attempts left out."""
    return [exchange for exchange in exchanges if exchange.succeeded]


def assert_signatures_sent_back(exchanges: list[Exchange]) -> None:
    """Each request sends back, model turn by model turn, the signatures of every earlier reply.

    Only successful attempts are turns; a refused attempt and its retry are one turn. The comparison
    starts from the raw replies, so a signature lost or replaced when the adapter first converts a
    reply is caught. The last reply has no next request: it is checked only by
    `assert_signatures_stored`.
    """
    answered = turns(exchanges)
    replies = [turn_parts(exchange.reply_parts(), exchange.streamed) for exchange in answered]
    assert any(signatures for reply in replies for _, signatures in reply), (
        "no reply carried a thought signature"
    )
    for index, exchange in enumerate(answered):
        model_turns = [c for c in exchange.request["contents"] if c.get("role") == "model"]
        assert len(model_turns) == index, f"request {index} does not send back every earlier reply"
        sent = [
            turn_parts(content.get("parts", []), answered[k].streamed)
            for k, content in enumerate(model_turns)
        ]
        assert sent == replies[:index], f"request {index} does not send back the earlier signatures"


def stored_signatures(message: BaseMessage) -> list[bytes]:
    """The signatures a stored message holds, as bytes, read only where the adapter keeps them.

    A field named `signature` anywhere else (the response metadata, a tool call's arguments,
    another key of `additional_kwargs`) is not a stored signature.
    """
    values: list[str | bytes] = list(message.additional_kwargs.get(SIGNATURE_MAP_KEY, {}).values())
    blocks = message.content if isinstance(message.content, list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        values.extend(block[key] for key in BLOCK_SIGNATURE_KEYS if key in block)
        extras = block.get("extras")
        if isinstance(extras, dict) and "signature" in extras:
            values.append(extras["signature"])
    return [signature_bytes(value) for value in values]


def assert_signatures_stored(exchange: Exchange, message: BaseMessage) -> None:
    """Every signature of a raw reply is in a signature field of the message stored for it.

    This proves that the signatures were kept, not that they would be sent back with their parts.
    """
    stored = Counter(stored_signatures(message))
    for part in exchange.reply_parts():
        if "thoughtSignature" in part:
            signature = signature_bytes(part["thoughtSignature"])
            assert stored[signature] > 0, "a signature of the reply is not stored in the message"
            stored[signature] -= 1


def _describe(parts: list[tuple[tuple, list[bytes]]]) -> list[dict]:
    """A turn's parts for the report: kind, call name, text length, signature sizes and hashes."""
    described = []
    for position, (name, signatures) in enumerate(parts):
        entry: dict = {"position": position, "kind": name[0]}
        if name[0] == "call":
            entry["call"] = name[1]
        elif name[0] in ("text", "thought"):
            entry["text_length"] = len(name[1])
        entry["signatures"] = [
            {"bytes": len(signature), "sha256": hashlib.sha256(signature).hexdigest()}
            for signature in signatures
        ]
        described.append(entry)
    return described


def signature_report(exchanges: list[Exchange]) -> dict:
    """A record of the comparison to review after a run: no key, no signature, no text content.

    For each attempt, its status. For each answered attempt (a turn), its reply's parts (kind, call
    name, text length, each signature's decoded length and SHA-256), then the model turns its
    request sends back, described the same way, and whether each matches the reply it stands for.
    A reply, a request or a turn sent back that cannot be read is recorded where it happened, by the
    kind of error only (its class name: its message could quote the data), and the rest of the
    report is still built.
    """
    statuses = [{"status": e.status, "streamed": e.streamed} for e in exchanges]
    answered = [(attempt, e) for attempt, e in enumerate(exchanges) if e.succeeded]
    replies: list[list[tuple[tuple, list[bytes]]] | None] = []
    reply_errors: list[str | None] = []
    for _, exchange in answered:
        try:
            replies.append(turn_parts(exchange.reply_parts(), exchange.streamed))
            reply_errors.append(None)
        except Exception as error:
            replies.append(None)
            reply_errors.append(type(error).__name__)

    report_turns = []
    for turn, (attempt, exchange) in enumerate(answered):
        entry: dict = {"turn": turn, "attempt": attempt}
        reply = replies[turn]
        entry["reply_parts"] = _describe(reply) if reply is not None else None
        if reply_errors[turn]:
            entry["reply_error"] = reply_errors[turn]
        try:
            model_turns = [c for c in exchange.request["contents"] if c.get("role") == "model"]
        except Exception as error:
            entry["request_error"] = type(error).__name__
            model_turns = []
        sent = []
        for k, content in enumerate(model_turns):
            try:
                parts = turn_parts(content.get("parts", []), answered[k][1].streamed)
                matches = k < turn and replies[k] is not None and parts == replies[k]
                sent.append({"turn": k, "parts": _describe(parts), "matches": matches})
            except Exception as error:
                sent.append({"turn": k, "error": type(error).__name__})
        entry["request_sends_back"] = sent
        report_turns.append(entry)
    return {"attempts": statuses, "turns": report_turns}


def write_signature_report(exchanges: list[Exchange], path: Path) -> None:
    """Save `signature_report` as JSON, so a run can be reviewed after it, failed or not.

    An error that the report cannot place (outside a reply, a request or a turn sent back) is a
    report-level error, kept by its class name with the attempts' statuses. An error while writing
    the file is raised: no report is claimed when none was saved.
    """
    try:
        report = signature_report(exchanges)
    except Exception as error:
        report = {
            "attempts": [{"status": e.status, "streamed": e.streamed} for e in exchanges],
            "report_error": type(error).__name__,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")


def function_responses(request: dict) -> list[dict]:
    """The function responses a request sends to the model."""
    return [
        part["functionResponse"]
        for content in request["contents"]
        for part in content.get("parts", [])
        if "functionResponse" in part
    ]


class State(MessagesState):
    tool_rounds: int


def build_graph(model, tools: list, max_tool_rounds: int):
    """The graph of plan §6.8, reduced to what the check needs.

    `final` sends the final prompt, the messages and `FINAL_INSTRUCTION`, and stores only the reply:
    whatever the model returns, an empty reply included, and the graph ends either way.
    """
    agent_model = model.bind_tools(tools)
    final_model = model.bind_tools(tools, tool_choice="none")
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent(state: State) -> dict:
        return {
            "messages": [agent_model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]
        }

    def run_tools(state: State) -> dict:
        result = tool_node.invoke(state)
        return {"messages": result["messages"], "tool_rounds": state["tool_rounds"] + 1}

    def final(state: State) -> dict:
        request = [SystemMessage(FINAL_PROMPT), *state["messages"], HumanMessage(FINAL_INSTRUCTION)]
        return {"messages": [final_model.invoke(request)]}

    def after_tools(state: State) -> str:
        return "agent" if state["tool_rounds"] < max_tool_rounds else "final"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("final", final)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "final": "final"})
    graph.add_edge("final", END)
    return graph.compile(checkpointer=InMemorySaver())


def thread_config(thread: str) -> dict:
    """The run configuration of one conversation thread."""
    return {"configurable": {"thread_id": thread}, "recursion_limit": 20}


def ask(graph, thread: str, question: str) -> dict:
    """Run one question on a thread; return the final state."""
    return graph.invoke(
        {"messages": [HumanMessage(question)], "tool_rounds": 0}, thread_config(thread)
    )


def tool_rounds_of(messages: list[BaseMessage]) -> int:
    """The number of model messages that asked for tools."""
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
