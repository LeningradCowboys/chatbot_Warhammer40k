"""Compatibility check of the chat model on the real Gemini API (plan §8, R26, R32).

Run only on purpose, on its own: `uv run pytest tests/test_gemini_online.py -m online -x -v`. One
session sends at most `REQUEST_BUDGET` generate requests, retries included, spaced for the observed
per-minute limit; reaching the budget fails with "the run is incomplete". Each test saves
its signature report (no key, no signature, no text) to `data/checks/`, failed or not.
"""

import json
import time
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from src.rag.config import ONLINE_CHECK_DIR, Settings, key_not_set
from src.rag.llm.budget import RequestBudget
from src.rag.llm.gemini import MIN_SECONDS_BETWEEN_CHAT_REQUESTS, create_chat_model
from tests.gemini_check import (
    FINAL_INSTRUCTION,
    REQUEST_BUDGET,
    RecordingTransport,
    ask,
    assert_signatures_sent_back,
    assert_signatures_stored,
    build_graph,
    function_responses,
    multiply,
    thread_config,
    tool_rounds_of,
    turns,
    unit_lookup,
    unit_toughness,
    write_signature_report,
)

pytestmark = pytest.mark.online

QUESTION = (
    "What is the toughness of the Zorvak Sentinel multiplied by 3? "
    "First look up the toughness, then use the multiply tool."
)
FOLLOW_UP = "And for the Quillon Brute, the same calculation?"


@pytest.fixture(scope="session")
def budget() -> RequestBudget:
    return RequestBudget(REQUEST_BUDGET, MIN_SECONDS_BETWEEN_CHAT_REQUESTS)


@pytest.fixture
def settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        if key_not_set(error):
            pytest.skip("GOOGLE_API_KEY is not set")
        raise


@pytest.fixture(scope="session")
def run_id() -> str:
    """Names the reports of one run: the time it started."""
    return time.strftime("%Y%m%d-%H%M%S")


@pytest.fixture
def recorder(budget: RequestBudget, run_id: str, request) -> Iterator[RecordingTransport]:
    """The traffic recorder of one test; its signature report is saved when the test ends."""
    transport = RecordingTransport(budget=budget)
    yield transport
    transport.close()
    report = ONLINE_CHECK_DIR / f"{run_id}_{request.node.name}.json"
    write_signature_report(transport.exchanges, report)


def _graph(settings: Settings, recorder: RecordingTransport, tools: list, max_tool_rounds: int):
    return build_graph(create_chat_model(settings, {"transport": recorder}), tools, max_tool_rounds)


def _assert_calls_paired(messages: list) -> None:
    call_ids = [call["id"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
    answer_ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert sorted(call_ids) == sorted(answer_ids)
    assert len(set(call_ids)) == len(call_ids)


def test_several_tool_calls_then_a_follow_up(settings, recorder):
    graph = _graph(settings, recorder, [unit_toughness, multiply], 3)

    messages = ask(graph, "t1", QUESTION)["messages"]
    assert tool_rounds_of(messages) >= 2
    assert "21" in messages[-1].text

    messages = ask(graph, "t1", FOLLOW_UP)["messages"]
    assert "27" in messages[-1].text
    _assert_calls_paired(messages)
    assert_signatures_sent_back(recorder.exchanges)  # the follow-up sends back the first answer
    assert_signatures_stored(turns(recorder.exchanges)[-1], messages[-1])  # last reply: stored only


def test_streaming_keeps_tool_calls_and_signatures(settings, recorder):
    graph = _graph(settings, recorder, [unit_toughness, multiply], 3)
    question = QUESTION + " Then explain the calculation in a paragraph of about 80 words."

    text_chunks = [
        chunk.text
        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(question)], "tool_rounds": 0},
            thread_config("t2"),
            stream_mode="messages",
        )
        if metadata["langgraph_node"] in ("agent", "final") and chunk.text
    ]
    assert len(text_chunks) > 1
    assert all(exchange.streamed for exchange in recorder.exchanges)

    messages = graph.get_state(thread_config("t2")).values["messages"]
    assert tool_rounds_of(messages) >= 1
    assert "21" in messages[-1].text
    _assert_calls_paired(messages)

    # The follow-up sends back the stored streamed replies, the streamed answer included.
    messages = ask(graph, "t2", FOLLOW_UP)["messages"]
    assert_signatures_sent_back(recorder.exchanges)
    assert_signatures_stored(turns(recorder.exchanges)[-1], messages[-1])  # last reply: stored only


def test_tool_error_is_given_back_to_the_model(settings, recorder):
    graph = _graph(settings, recorder, [unit_lookup], 2)

    state = ask(graph, "t3", "Look up the toughness of the Zorvak Sentinel in the unit database.")
    messages = state["messages"]
    errors = [m for m in messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert errors and "unit database offline" in errors[0].text
    assert not messages[-1].tool_calls and messages[-1].text
    # The first answered request after the failing call sends the error as the call's response.
    sent = next(
        responses
        for exchange in turns(recorder.exchanges)
        if (responses := function_responses(exchange.request))
    )
    assert all(response["name"] == "unit_lookup" for response in sent)
    assert "unit database offline" in json.dumps(sent)
    assert_signatures_sent_back(recorder.exchanges)
    assert_signatures_stored(turns(recorder.exchanges)[-1], messages[-1])  # last reply: stored only


def test_round_limit_gives_a_final_answer_without_tools(settings, recorder):
    graph = _graph(settings, recorder, [unit_toughness, multiply], 1)

    messages = ask(graph, "t4", QUESTION)["messages"]
    final = turns(recorder.exchanges)[-1]
    assert final.request["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
    assert final.request["contents"][-1] == {"role": "user", "parts": [{"text": FINAL_INSTRUCTION}]}
    assert all(m.text != FINAL_INSTRUCTION for m in messages)  # sent, not stored
    assert tool_rounds_of(messages) == 1  # no tool call after the limit
    reply = messages[-1]
    # An empty reply ends the graph (tested offline), but passing needs the model's own answer.
    reason = reply.response_metadata.get("finish_reason")
    assert reply.text, f"the final call returned no text (finish reason {reason})"
    # Whether the answer uses the tool result and names what is missing is read by a person:
    # `-rP` shows this output for a passed test; no file keeps it.
    print(f"final answer (finish reason {reason}):\n{reply.text}")
    assert_signatures_sent_back(recorder.exchanges)
    assert_signatures_stored(final, reply)  # last reply: stored only
