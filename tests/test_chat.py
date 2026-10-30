"""The command-line chat with the fake chat model: what is shown and what is answered (R27)."""

import argparse
import io
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from src.rag.chatbot.agent import ChatAgent
from src.rag.retriever.sql_retriever import SqlResult, SqlRetriever
from src.scripts import chat
from tests.fakes import FakeChatModel, FakeRules, tool_call

ANSWER = "Roll six D6 [Core Rules p. 34–35]."
BEFORE_TOOL = "Let me look this up."


def _agent(tiny_db: Path, *replies: AIMessage) -> ChatAgent:
    model = FakeChatModel(replies=list(replies))
    return ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "System prompt of the test.")


def _text_tool_answer(tiny_db: Path) -> ChatAgent:
    """A model that writes text, then calls a tool, then answers."""
    search = tool_call("search_rules", {"query": "grenade stratagem"}, "c1")
    return _agent(tiny_db, AIMessage(BEFORE_TOOL, tool_calls=[search]), AIMessage(ANSWER))


def _no_text(tiny_db: Path) -> ChatAgent:
    return _agent(tiny_db, AIMessage("", response_metadata={"finish_reason": "STOP"}))


NO_ANSWER = "No answer: the model returned no text (finish reason STOP)."


def _unknown_tool(tiny_db: Path) -> ChatAgent:
    """A model that calls a tool that does not exist, then answers."""
    call = tool_call("search_rule", {"query": "grenade"}, "c1")
    return _agent(tiny_db, AIMessage("", tool_calls=[call]), AIMessage(ANSWER))


def _console() -> tuple[Console, io.StringIO]:
    """A console writing to a string, not a terminal: a live area draws nothing on it."""
    output = io.StringIO()
    return Console(file=output, width=100, color_system=None), output


@pytest.mark.parametrize("failure", [ConnectionError, TimeoutError, KeyboardInterrupt])
def test_failed_turn_is_reported_and_next_turn_has_a_clean_thread(tiny_db, monkeypatch, failure):
    original = FakeChatModel._next_reply
    calls = []

    def reply(self, messages, options):
        calls.append(messages)
        if len(calls) == 1:
            raise failure("simulated failure")
        return original(self, messages, options)

    monkeypatch.setattr(FakeChatModel, "_next_reply", reply)
    agent = _agent(tiny_db, AIMessage("Recovered"))
    old_thread = agent._thread
    lines = iter(["First question", "/sources", "Second question", "/quit"])
    console, output = _console()
    chat.chat_loop(agent, console, lambda: next(lines))
    assert "New conversation" in output.getvalue()
    assert "No question asked yet" in output.getvalue()
    assert "Recovered" in output.getvalue()
    assert "First question" not in str(calls[1])
    assert old_thread not in agent._graph.checkpointer.storage


def test_one_shot_transport_failure_has_no_answer_and_returns_failure(tiny_db, monkeypatch):
    def fail(self, messages, options):
        raise ConnectionError("simulated failure")

    monkeypatch.setattr(FakeChatModel, "_next_reply", fail)
    console, errors = _console()
    output = io.StringIO()
    assert chat.answer_once(_agent(tiny_db), "Question", output, console) == 1
    assert output.getvalue() == ""
    assert "Model request failed (ConnectionError)" in errors.getvalue()


def test_programming_errors_are_not_hidden_by_cli_recovery(tiny_db):
    with pytest.raises(IndexError):  # empty fake replies, rather than an upstream failure
        chat.answer_once(_agent(tiny_db), "Question", io.StringIO(), _console()[0])


@pytest.fixture
def terminal(monkeypatch) -> tuple[Console, io.StringIO]:
    """A console that acts as a terminal whatever runs the tests, so that every state of a live
    area is drawn: Rich draws nothing on a terminal named dumb or unknown (`TERM`)."""
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(
        file=output, width=100, force_terminal=True, color_system=None, legacy_windows=False
    )
    return console, output


# --- interactive ---------------------------------------------------------------------------------


def test_only_the_final_answer_stays(tiny_db):
    console, output = _console()
    answer = chat.answer_live(_text_tool_answer(tiny_db), "How do grenades work?", console)

    assert answer.text == ANSWER
    assert output.getvalue().strip() == ANSWER


def test_on_a_terminal_the_live_area_streams_the_run_first(tiny_db, terminal):
    console, output = terminal
    chat.answer_live(_text_tool_answer(tiny_db), "How do grenades work?", console)
    shown = output.getvalue()
    assert shown.index(BEFORE_TOOL) < shown.index("searching the rules") < shown.rindex(ANSWER)


def test_interactive_an_unknown_tool_gets_a_generic_status_and_the_run_goes_on(tiny_db, terminal):
    console, output = terminal
    answer = chat.answer_live(_unknown_tool(tiny_db), "How do grenades work?", console)

    assert answer.text == ANSWER
    assert answer.tool_calls[0].error.startswith("Error: search_rule is not a valid tool")
    shown = output.getvalue()
    assert shown.index("calling search_rule…") < shown.rindex(ANSWER)


def test_interactive_without_text_shows_the_no_answer_line(tiny_db):
    console, output = _console()
    answer = chat.answer_live(_no_text(tiny_db), "question", console)
    assert not answer.answered
    assert output.getvalue().strip() == NO_ANSWER


def test_the_loop_answers_lists_sources_resets_and_quits(tiny_db):
    lines = iter(["/sources", "How do grenades work?", "/sources", "/reset", "/sources", "/quit"])
    console, output = _console()
    chat.chat_loop(_text_tool_answer(tiny_db), console, lambda: next(lines))

    shown = [line.rstrip() for line in output.getvalue().splitlines()]  # Markdown pads lines
    assert shown == [
        "No question asked yet.",
        ANSWER,
        "1. search_rules: grenade stratagem",
        "   pages: 34, 35, 44",
        "New conversation.",
        "No question asked yet.",
    ]


def test_the_loop_ends_at_the_end_of_input(tiny_db):
    def end() -> str:
        raise EOFError

    console, output = _console()
    chat.chat_loop(_text_tool_answer(tiny_db), console, end)
    assert output.getvalue() == ""


def test_sources_show_an_sql_error(tiny_db):
    query = tool_call("run_sql", {"sql": "SELECT colour FROM datasheets"}, "c1")
    agent = _agent(tiny_db, AIMessage("", tool_calls=[query]), AIMessage("No such data."))
    answer = agent.ask("question")
    assert chat.sources_text(answer).splitlines() == [
        "1. run_sql: SELECT colour FROM datasheets",
        "   error: OperationalError: no such column: colour",
    ]


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        (
            SqlResult(
                [
                    "datasheet_id",
                    "datasheet",
                    "model_line",
                    "model",
                    "attribute_value",
                    "source",
                    "edition",
                ],
                [("1", "Example Unit", "1", "Example Model", "4", "Book", "11")],
                False,
                None,
            ),
            '"rows": [["1", "Example Unit", "1", "Example Model", "4", "Book", "11"]]',
        ),
        (
            SqlResult([], [], False, "OperationalError: database unavailable"),
            '"error": "OperationalError: database unavailable"',
        ),
    ],
    ids=["success", "failure"],
)
def test_sources_show_application_profile_queries(lookup, expected):
    class LocalRetriever:
        def run(self, sql: str, parameters=()) -> SqlResult:
            return lookup

    agent = ChatAgent(
        FakeChatModel(replies=[AIMessage("First answer."), AIMessage("Evidence is incomplete.")]),
        FakeRules(),
        LocalRetriever(),
        "S",
    )
    agent.ask("What is the Toughness of a Space Marine?")
    answer = agent.ask("What about Example Unit?")

    sources = chat.sources_text(answer)
    assert "1. application profile_lookup: SELECT DISTINCT" in sources
    assert 'parameters: ["Example Unit"]' in sources
    assert expected in sources
    assert "No tool was called" not in sources


# --- one question --------------------------------------------------------------------------------


def test_question_writes_the_final_answer_once_and_the_status_to_standard_error(tiny_db):
    out = io.StringIO()
    err, err_output = _console()
    code = chat.answer_once(_text_tool_answer(tiny_db), "How do grenades work?", out, err)

    assert code == 0
    assert out.getvalue() == ANSWER + "\n"
    assert err_output.getvalue() == "searching the rules…\n"


def test_question_with_an_unknown_tool_gets_a_generic_status_and_the_answer(tiny_db):
    out = io.StringIO()
    err, err_output = _console()
    code = chat.answer_once(_unknown_tool(tiny_db), "How do grenades work?", out, err)

    assert code == 0
    assert out.getvalue() == ANSWER + "\n"
    assert err_output.getvalue() == "calling search_rule…\n"


def test_question_without_text_leaves_standard_output_empty_and_exits_1(tiny_db):
    out = io.StringIO()
    err, err_output = _console()
    code = chat.answer_once(_no_text(tiny_db), "question", out, err)

    assert code == 1
    assert out.getvalue() == ""
    assert err_output.getvalue() == NO_ANSWER + "\n"


# --- start-up ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "step"),
    [
        ("RULES_CHUNKS", "src.scripts.chunk"),
        ("CLEAN_DB", "src.scripts.prepare_db"),
        ("RULES_EMBEDDINGS", "src.scripts.build_index"),
    ],
)
def test_a_missing_file_stops_with_the_step_that_makes_it(
    tmp_path, monkeypatch, capsys, missing, step
):
    for name in ["RULES_CHUNKS", "CLEAN_DB", "RULES_EMBEDDINGS"]:
        path = tmp_path / name
        if name != missing:
            path.write_text("", encoding="utf-8")
        monkeypatch.setattr(chat, name, path)

    with pytest.raises(SystemExit) as stop:
        chat.build_agent(argparse.ArgumentParser(prog="chat"))
    assert stop.value.code == 2
    assert f"{tmp_path / missing} not found. Run {step}" in capsys.readouterr().err
