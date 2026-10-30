"""The online acceptance tool, offline: what it sends, refuses and saves (plan R46)."""

import json
import sys

import httpx
import pytest
from langchain_core.messages import AIMessage

from src.rag.chatbot.agent import ChatAgent, RunTrace
from src.rag.config import RAW_DIR, Settings
from src.rag.evaluation.testset import EvalQuestion
from src.rag.llm.budget import AttemptCounter, BudgetExhausted
from src.rag.llm.gemini import create_chat_model
from src.rag.retriever.sql_retriever import SqlRetriever
from src.rag.storage.evaluation_runs import save_testset
from tests import chat_acceptance
from tests.fakes import FakeChatModel, FakeRules, tool_call

ANSWER = "Roll six D6 [Core Rules p. 34–35]."


@pytest.fixture(autouse=True)
def input_artifacts(tmp_path_factory, monkeypatch):
    """Fingerprint tiny synthetic inputs, never the real rulebook or database in unit tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-placeholder")
    directory = tmp_path_factory.mktemp("acceptance-inputs")
    for name in ["RULES_CHUNKS", "RULES_EMBEDDINGS", "CLEAN_DB", "SQL_SCHEMA"]:
        path = directory / name
        path.write_text("synthetic input", encoding="utf-8")
        monkeypatch.setattr(chat_acceptance, name, path)
    testset = directory / "testset.jsonl"
    save_testset(
        testset,
        [
            EvalQuestion("q006", "If a unit wuz battle-shocked, wut's next?", "R", [], "multi"),
            EvalQuestion("q010", "How many Stratagems can Garrick use?", "R", [], "multi"),
        ],
    )
    lock_file = directory / "uv.lock"
    lock_file.write_text("locked", encoding="utf-8")
    source_dir = directory / "src"
    source_dir.mkdir()
    (source_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(chat_acceptance, "TESTSET", testset)
    monkeypatch.setattr(chat_acceptance, "LOCK_FILE", lock_file)
    monkeypatch.setattr(chat_acceptance, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(chat_acceptance, "_git_commit", lambda: "a" * 40)


def _agent(tiny_db) -> ChatAgent:
    """A model that calls `search_rules`, then answers."""
    search = tool_call("search_rules", {"query": "grenade stratagem"}, "c1")
    model = FakeChatModel(replies=[AIMessage("", tool_calls=[search]), AIMessage(ANSWER)])
    return ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "System prompt of the test.")


def _main(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["chat_acceptance", *argv])
    chat_acceptance.main()


def test_without_send_nothing_is_built_sent_or_written(tmp_path, monkeypatch, capsys):
    def refuse(*args):
        raise AssertionError("the agent must not be built without --send")

    monkeypatch.setattr(chat_acceptance, "build_agent", refuse)
    _main(monkeypatch, "grenades", "--budget", "4", "--output", str(tmp_path))

    assert "Nothing sent" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_wardens_is_an_independent_preview_case(tmp_path, monkeypatch, capsys):
    def refuse(*args):
        raise AssertionError("the agent must not be built without --send")

    monkeypatch.setattr(chat_acceptance, "build_agent", refuse)
    _main(monkeypatch, "wardens", "--budget", "3", "--output", str(tmp_path))

    output = capsys.readouterr().out
    assert "Wardens of Ultramar" in output
    assert "every model profile's Toughness" in output
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("question_id", "expected"),
    [("q006", "wuz battle-shocked, wut's"), ("q010", "Garrick")],
)
def test_frozen_questions_are_previewed_verbatim(
    question_id, expected, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        chat_acceptance,
        "build_agent",
        lambda *args, **kwargs: pytest.fail("preview must not build the agent"),
    )

    _main(monkeypatch, question_id, "--budget", "4", "--output", str(tmp_path))

    assert expected in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("contents", [[], ["q006", "q006"]])
def test_missing_or_duplicate_frozen_id_stops_before_the_agent(
    contents, tmp_path, monkeypatch, capsys
):
    save_testset(
        chat_acceptance.TESTSET,
        [
            EvalQuestion(item, f"Question {index}", "R", [], "multi")
            for index, item in enumerate(contents)
        ],
    )
    monkeypatch.setattr(
        chat_acceptance,
        "build_agent",
        lambda *args, **kwargs: pytest.fail("invalid test set must stop before the agent"),
    )

    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, "q006", "--budget", "4", "--output", str(tmp_path), "--send")

    assert stop.value.code == 2
    assert "q006" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_an_output_in_the_raw_folder_is_refused(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, "grenades", "--budget", "4", "--output", str(RAW_DIR / "acceptance"))
    assert stop.value.code == 2
    assert "holds the original inputs" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["interactive", "question"])
def test_a_run_saves_what_the_cli_showed_the_report_and_the_trace(
    tiny_db, tmp_path, monkeypatch, mode
):
    agent = _agent(tiny_db)
    given = []

    def build(parser, chat_args, embed_args, settings=None):
        given.append((chat_args["transport"], embed_args["transport"]))
        return agent

    monkeypatch.setattr(chat_acceptance, "build_agent", build)
    out = tmp_path / "out"
    _main(
        monkeypatch,
        "grenades",
        "--budget",
        "4",
        "--mode",
        mode,
        "--output",
        str(out),
        "--round",
        "R104",
        "--send",
    )

    (run_dir,) = out.iterdir()
    assert run_dir.name.endswith(f"_grenades_{mode}")
    chat_transport, embed_transport = given[0]
    assert isinstance(chat_transport, AttemptCounter)
    assert chat_transport._budget.limit == 4  # counted before each chat attempt is sent
    assert isinstance(embed_transport, AttemptCounter)
    assert embed_transport._budget.limit == 18

    for name in ("report.json", "trace.json"):  # the bytes written, LF on every system
        assert b"\r\n" not in (run_dir / name).read_bytes()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["question"] == "What is the rule for grenades?"
    assert (report["chat_budget"], report["chat_attempts"], report["embedding_attempts"]) == (
        4,
        [],
        [],
    )
    assert report["turn_budgets"] == [4]
    assert report["acceptance"] == {
        "question_id": "grenades",
        "round": "R104",
        "commit": "a" * 40,
        "model_settings": {"model": "fake-chat-model", "temperature": 0.0},
        "embedding_model": "gemini-embedding-001",
        "budgets": {"chat": 4, "turns": [4], "embedding": 18},
        "inputs": {
            "uv.lock": "14493f5f5470ed48c3f103d917ec52ae9005fa3913128031d0fac2a49ac3cc41",
            "testset": report["acceptance"]["inputs"]["testset"],
        },
    }
    assert report["complete"] and report["answer"]["text"] == ANSWER
    (call,) = report["answer"]["tool_calls"]
    assert (call["name"], call["pages"]) == ("search_rules", [34, 35, 44])
    assert call["output"].startswith("[Core Rules p. 34–35]")
    trace = RunTrace.from_dict(json.loads((run_dir / "trace.json").read_text(encoding="utf-8")))
    assert trace.messages[-1].text == ANSWER
    if mode == "interactive":
        assert report["exit_code"] is None
        shown = (run_dir / "console.txt").read_text(encoding="utf-8")
        assert ANSWER in shown and "1. search_rules: grenade stratagem" in shown
    else:
        assert report["exit_code"] == 0
        assert (run_dir / "stdout.txt").read_text(encoding="utf-8") == ANSWER + "\n"
        assert (run_dir / "stderr.txt").read_text(encoding="utf-8") == "searching the rules…\n"


def test_a_run_that_stops_midway_keeps_its_partial_trace_marked_incomplete(
    tiny_db, tmp_path, monkeypatch
):
    search = tool_call("search_rules", {"query": "grenade stratagem"}, "c1")
    model = FakeChatModel(replies=[AIMessage("", tool_calls=[search])])  # the second call fails
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "System prompt of the test.")
    monkeypatch.setattr(chat_acceptance, "build_agent", lambda *args, **kwargs: agent)
    out = tmp_path / "out"

    with pytest.raises(IndexError):
        _main(
            monkeypatch,
            "grenades",
            "--budget",
            "2",
            "--output",
            str(out),
            "--round",
            "R104",
            "--send",
        )
    (run_dir,) = out.iterdir()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["error"].startswith("IndexError") and not report["complete"]
    assert "answer" not in report
    trace = RunTrace.from_dict(json.loads((run_dir / "trace.json").read_text(encoding="utf-8")))
    question, call, result = trace.messages
    assert question.text == "What is the rule for grenades?"
    assert call.tool_calls[0]["args"] == {"query": "grenade stratagem"}
    assert result.text.startswith("[Core Rules p. 34–35]")  # the full result


def test_follow_up_uses_the_same_thread_and_records_both_answers(tiny_db, tmp_path, monkeypatch):
    model = FakeChatModel(replies=[AIMessage("Unit has Grenades"), AIMessage("It costs 1CP")])
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")
    monkeypatch.setattr(chat_acceptance, "build_agent", lambda *args, **kwargs: agent)
    out = tmp_path / "out"
    _main(
        monkeypatch,
        "unit_rules",
        "--budget",
        "4",
        "--embedding-budget",
        "2",
        "--follow-up",
        "And its cost?",
        "--turn-budget",
        "2",
        "--turn-budget",
        "2",
        "--mode",
        "question",
        "--output",
        str(out),
        "--round",
        "R104",
        "--send",
    )
    [run] = list(out.iterdir())
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert [turn["answer"]["text"] for turn in report["turns"]] == [
        "Unit has Grenades",
        "It costs 1CP",
    ]
    assert model.calls[1].messages[-2].text == "Unit has Grenades"
    assert report["embedding_budget"] == 2 and report["inputs"]
    assert [turn["status"] for turn in report["turns"]] == ["complete", "complete"]
    assert (run / "turn-01" / "stdout.txt").exists()
    assert (run / "turn-02" / "stdout.txt").exists()


def test_failed_follow_up_does_not_report_the_previous_answer_as_complete(
    tiny_db, tmp_path, monkeypatch
):
    model = FakeChatModel(replies=[AIMessage("Unit has Grenades")])
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")
    monkeypatch.setattr(chat_acceptance, "build_agent", lambda *args, **kwargs: agent)
    out = tmp_path / "out"
    with pytest.raises(IndexError):
        _main(
            monkeypatch,
            "unit_rules",
            "--budget",
            "4",
            "--follow-up",
            "And its cost?",
            "--turn-budget",
            "2",
            "--turn-budget",
            "2",
            "--output",
            str(out),
            "--round",
            "R104",
            "--send",
        )
    [run] = list(out.iterdir())
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert not report["complete"] and "answer" not in report
    assert report["turns"][0]["answer"]["text"] == "Unit has Grenades"
    assert report["turns"][1]["status"] == "failed"


def test_follow_up_requires_one_reserved_budget_per_turn(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        _main(
            monkeypatch,
            "unit_rules",
            "--budget",
            "4",
            "--follow-up",
            "And its cost?",
            "--output",
            str(tmp_path),
        )
    assert stop.value.code == 2
    assert "--turn-budget must be repeated once per turn" in capsys.readouterr().err


def test_transport_retries_count_and_budget_exhaustion_stops_sending(
    tiny_db, tmp_path, monkeypatch
):
    sent = []

    def server(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(429, json={"error": {"code": 429, "message": "retry"}})

    real_counter = AttemptCounter

    def counter(budget, capture_diagnostics=False):
        inner = httpx.MockTransport(server) if capture_diagnostics else httpx.MockTransport(server)
        return real_counter(budget, inner, capture_diagnostics)

    def build(parser, chat_args, embed_args, settings=None):
        model = create_chat_model(settings, chat_args)
        return ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")

    monkeypatch.setattr(chat_acceptance, "AttemptCounter", counter)
    monkeypatch.setattr(chat_acceptance, "build_agent", build)
    monkeypatch.setattr(
        chat_acceptance,
        "load_settings",
        lambda parser: Settings(google_api_key="test", _env_file=None),
    )
    out = tmp_path / "out"

    with pytest.raises(BudgetExhausted):
        _main(
            monkeypatch,
            "q006",
            "--budget",
            "2",
            "--embedding-budget",
            "1",
            "--output",
            str(out),
            "--round",
            "R104",
            "--send",
        )

    [run] = list(out.iterdir())
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert len(sent) == 2
    assert [attempt["status"] for attempt in report["chat_attempts"]] == [429, 429]
    assert not report["complete"]
