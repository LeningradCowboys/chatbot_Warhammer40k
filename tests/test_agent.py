"""The chat graph with the fake chat model and small retrievers: no request is sent (plan §6.8)."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.rag.chatbot import agent as agent_module
from src.rag.chatbot.agent import (
    AGENT_TOOL_CONFIG,
    MAX_SQL_CALLS_PER_QUESTION,
    ChatAgent,
    RulesSearchResult,
    RunTrace,
    answer_from_trace,
    build_tools,
    chunk_block,
)
from src.rag.chatbot.prompts import (
    APPLICABILITY_RULES,
    FINAL_INSTRUCTION,
    FINAL_PROMPT,
    TOOL_RULES,
    build_system_prompt,
)
from src.rag.config import MAX_TOOL_ROUNDS, TOP_K
from src.rag.retriever.rules_retriever import RetrievedChunk
from src.rag.retriever.sql_retriever import SqlResult, SqlRetriever
from tests.fakes import COVER, GRENADE, FakeChatModel, FakeRules, tool_call

SYSTEM = "System prompt of the test."
SIGNATURES = {"__gemini_function_call_thought_signatures__": {"c1": "c2lnbmF0dXJl"}}


class FailingRules:
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise ConnectionError("index unavailable")


def _search(call_id: str, query: str = "grenade") -> AIMessage:
    return AIMessage("", tool_calls=[tool_call("search_rules", {"query": query}, call_id)])


def _sql(call_id: str, sql: str) -> AIMessage:
    return AIMessage("", tool_calls=[tool_call("run_sql", {"sql": sql}, call_id)])


def _agent(model, tiny_db, rules=None) -> ChatAgent:
    return ChatAgent(model, rules or FakeRules(), SqlRetriever(tiny_db, 50), SYSTEM)


def _complete_profile_query(name: str = "Intercessor Squad") -> str:
    return (
        "SELECT d.Id, d.name, 'Space Marines' AS source, '11' AS edition, "
        "GROUP_CONCAT(m.name || ': T' || m.t, '; ') AS profiles "
        "FROM datasheets d JOIN datasheets_models m ON m.DatasheetId = d.Id "
        f"WHERE d.name = '{name}' GROUP BY d.Id, d.name"
    )


PROFILE_LOOKUP_COLUMNS = [
    "datasheet_id",
    "datasheet",
    "model_line",
    "model",
    "attribute_value",
    "source",
    "edition",
]
WARDENS_PROFILE_ROWS = [
    (
        "000004188",
        "Wardens of Ultramar",
        "1",
        "Ancient Gadriel, Veteran Sergeant Metaurus",
        "4",
        "Space Marines",
        "11",
    ),
    (
        "000004188",
        "Wardens of Ultramar",
        "2",
        "Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha",
        "3",
        "Space Marines",
        "11",
    ),
]


def _assert_rebuilt(trace: RunTrace, model: FakeChatModel) -> None:
    """Every model call of the trace gives back the input the model received, and its reply."""
    assert len(trace.model_calls) == len(model.calls)
    for call, received in zip(trace.model_calls, model.calls, strict=True):
        assert trace.model_input(call) == received.messages
        expected = "none" if call.node == "final" else AGENT_TOOL_CONFIG
        assert received.tool_choice == expected
        assert call.tool_choice == ("none" if call.node == "final" else "validated")
        assert isinstance(trace.messages[call.input_count], AIMessage)


# --- tools and state ---------------------------------------------------------------------------


def test_a_tool_call_is_run_and_answered_by_its_tool_message(tiny_db):
    rules = FakeRules()
    first = _search("c1", "grenade stratagem")
    first.additional_kwargs.update(SIGNATURES)
    model = FakeChatModel(replies=[first, AIMessage("Roll six D6 [Core Rules p. 34–35].")])
    answer = _agent(model, tiny_db, rules).ask("How do grenades work?")

    assert rules.queries == [("grenade stratagem", TOP_K)]
    stored = answer.trace.messages[1]
    assert (stored.tool_calls, stored.additional_kwargs) == (
        first.tool_calls,
        first.additional_kwargs,
    )
    response = answer.trace.messages[2]
    assert isinstance(response, ToolMessage) and response.tool_call_id == "c1"
    assert response.text == (
        chunk_block(GRENADE) + "\n\n" + chunk_block(COVER) + "\n\n" + APPLICABILITY_RULES
    )
    assert chunk_block(GRENADE).startswith("[Core Rules p. 34–35]\nSection: GRENADE")
    assert model.calls[1].messages[-1] == response  # the model reads the tool's text
    assert answer.text == "Roll six D6 [Core Rules p. 34–35]."


def test_observed_grenade_evidence_supplies_page_citation_instead_of_chunk_id():
    # The real acceptance copied [core_rules-0161] from the tool header into its answer.
    observed = replace(GRENADE, chunk_id="core_rules-0161", pages=[42])
    shown = chunk_block(observed)
    assert shown.startswith("[Core Rules p. 42]\n")
    assert "core_rules-0161" not in shown
    assert observed.chunk_id == "core_rules-0161"  # retained for the structured artifact


def test_a_chunk_shows_the_footer_printed_on_each_of_its_pages():
    # each footer stays with its page, outside the citation
    scoring = replace(COVER, pages=[59], page_footers=["CORE RULES | MISSION: ONLY WAR"])
    assert chunk_block(scoring).startswith(
        "[Core Rules p. 59] (page footer p. 59: CORE RULES | MISSION: ONLY WAR)\n"
    )
    both = replace(GRENADE, page_footers=["CORE RULES | FIGHT PHASE", "CORE RULES | FIGHT"])
    assert chunk_block(both).startswith(
        "[Core Rules p. 34–35] (page footer p. 34: CORE RULES | FIGHT PHASE; "
        "page footer p. 35: CORE RULES | FIGHT)\n"
    )
    one = replace(GRENADE, page_footers=["", "CORE RULES | FIGHT"])
    assert chunk_block(one).startswith(
        "[Core Rules p. 34–35] (page footer p. 35: CORE RULES | FIGHT)\n"
    )
    none_printed = replace(GRENADE, page_footers=["", ""])
    assert chunk_block(none_printed) == chunk_block(GRENADE)
    assert chunk_block(GRENADE) == "[Core Rules p. 34–35]\nSection: GRENADE\nRoll six D6."


def test_a_search_keeps_the_chunks_with_scores_pictures_and_pages(tiny_db):
    model = FakeChatModel(replies=[_search("c1"), AIMessage("Answer.")])
    (call,) = _agent(model, tiny_db).ask("question").tool_calls

    assert (call.name, call.arguments) == ("search_rules", {"query": "grenade"})
    assert call.result == RulesSearchResult(
        [RetrievedChunk(GRENADE, 0.032), RetrievedChunk(COVER, 0.016)], [], None
    )
    assert call.result.chunks[0].chunk.pictures == ["#/pictures/7"]
    assert call.pages == [34, 35, 44]
    assert call.error is None


def test_a_query_keeps_its_sql_result_and_its_error(tiny_db):
    query = "SELECT name, t FROM datasheets_models WHERE t = '4'"
    model = FakeChatModel(
        replies=[_sql("c1", query), _sql("c2", "SELECT nope FROM nowhere"), AIMessage("T4.")]
    )
    answer = _agent(model, tiny_db).ask("toughness?")
    calls = answer.tool_calls

    assert calls[0].result == SqlResult(["name", "t"], [("Intercessor", "4")], False, None)
    assert json.loads(calls[0].output) == {
        "columns": ["name", "t"],
        "rows": [["Intercessor", "4"]],
        "truncated": False,
        "error": None,
    }
    assert (calls[0].pages, calls[0].error) == ([], None)
    assert calls[1].error.startswith("OperationalError")  # given back, so the model can correct it
    assert calls[1].error in calls[1].output
    # Rows as the model's input held them: every call rebuilt, also from the trace read back.
    _assert_rebuilt(answer.trace, model)
    _assert_rebuilt(RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict()))), model)


def test_a_failing_tool_reaches_the_model_and_the_graph_goes_on(tiny_db):
    model = FakeChatModel(replies=[_search("c1"), AIMessage("The rules could not be searched.")])
    answer = _agent(model, tiny_db, FailingRules()).ask("question")

    received = model.calls[1].messages[-1]
    assert isinstance(received, ToolMessage)
    assert (received.status, received.tool_call_id) == ("error", "c1")
    assert "index unavailable" in received.text
    (call,) = answer.tool_calls
    assert call.result is None and call.error == received.text
    assert answer.text == "The rules could not be searched."


def test_a_follow_up_sees_the_thread_and_reset_starts_a_new_one(tiny_db):
    model = FakeChatModel(replies=[AIMessage("First."), AIMessage("Second."), AIMessage("Third.")])
    chat = _agent(model, tiny_db)
    chat.ask("one")
    chat.ask("two")
    assert [m.text for m in model.calls[1].messages[1:]] == ["one", "First.", "two"]

    chat.reset()
    chat.ask("three")
    assert [m.text for m in model.calls[2].messages[1:]] == ["three"]


def test_a_follow_up_can_answer_from_the_previous_sql_result_without_another_query(tiny_db):
    query = "SELECT name, t FROM datasheets_models WHERE name = 'Intercessor'"
    model = FakeChatModel(
        replies=[
            _sql("c1", query),
            AIMessage("Intercessor is T4."),
            AIMessage("It is the T4 profile from the earlier result."),
        ]
    )
    chat = _agent(model, tiny_db)
    chat.ask("What is the Intercessor profile?")
    follow_up = chat.ask("What about its Toughness?")

    assert follow_up.tool_calls == []
    assert any(
        isinstance(message, ToolMessage) and "Intercessor" in message.text
        for message in model.calls[-1].messages
    )
    assert follow_up.text == "It is the T4 profile from the earlier result."


def test_profile_follow_up_does_not_format_a_previous_model_name_as_toughness():
    class RecordingSqlRetriever:
        def __init__(self):
            self.calls = []
            self.results = iter(
                [
                    SqlResult(
                        ["Id", "name", "source", "edition", "payload"],
                        [
                            (
                                "001",
                                "Intercessor Squad",
                                "Space Marines",
                                "11",
                                "Intercessor",
                            )
                        ],
                        False,
                        None,
                    ),
                    SqlResult(
                        PROFILE_LOOKUP_COLUMNS,
                        [
                            (
                                "001",
                                "Intercessor Squad",
                                "1",
                                "Intercessor",
                                "4",
                                "Space Marines",
                                "11",
                            )
                        ],
                        False,
                        None,
                    ),
                ]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            self.calls.append((sql, parameters))
            return next(self.results)

    retriever = RecordingSqlRetriever()
    unsupported = AIMessage("Toughness: Intercessor.")
    model = FakeChatModel(
        replies=[
            _sql("c1", "SELECT the previous model-name payload"),
            AIMessage("The previous result contains the model name."),
            unsupported,
        ]
    )
    chat = ChatAgent(model, FakeRules(), retriever, SYSTEM)
    chat.ask("What is the Toughness profile for this infantry sample?")
    answer = chat.ask("What about Intercessor Squad?")

    assert answer.tool_calls == []
    assert answer.text == (
        "### Intercessor Squad [DB: datasheets]\n\n"
        "- Intercessor: Toughness 4. [DB: datasheets_models]\n\n"
        "- Recorded publication: Space Marines. [DB: source]\n"
        "- Recorded edition: 11. [DB: source]"
    )
    lookup_sql, parameters = retriever.calls[-1]
    assert parameters == ("Intercessor Squad",)
    assert "WHERE d.name = ? COLLATE NOCASE" in lookup_sql
    assert "m.t AS attribute_value" in lookup_sql
    assert "Intercessor Squad" not in lookup_sql
    assert model.replies == [unsupported]
    _assert_rebuilt(answer.trace, model)


def test_the_observed_wardens_aggregate_triggers_a_complete_local_lookup():
    class ObservedSqlRetriever:
        def __init__(self):
            self.calls = []
            self.results = iter(
                [
                    SqlResult(
                        ["Id", "name", "source", "edition", "profiles"],
                        [
                            (
                                "000004188",
                                "Wardens of Ultramar",
                                "Space Marines",
                                "11",
                                "Ancient Gadriel, Veteran Sergeant Metaurus: T4; "
                                "Gaius Silva, Aemelia Minervas, Dainal Kornelius, "
                                "Lucia Vestha: T3",
                            )
                        ],
                        False,
                        None,
                    ),
                    SqlResult(PROFILE_LOOKUP_COLUMNS, WARDENS_PROFILE_ROWS, False, None),
                ]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            self.calls.append((sql, parameters))
            return next(self.results)

    retriever = ObservedSqlRetriever()
    query = (
        "SELECT d.Id, d.name, s.name AS source, s.edition, "
        "GROUP_CONCAT(m.name || ': T' || m.t, '; ') AS profiles "
        "FROM datasheets d JOIN datasheets_models m ON m.DatasheetId = d.Id "
        "LEFT JOIN source s ON s.Id = d.sourceId WHERE d.name = 'Wardens of Ultramar' "
        "GROUP BY d.Id, d.name, s.name, s.edition"
    )
    unsupported = AIMessage("Space Marine models are T4; accompanying mortal models are T3.")
    model = FakeChatModel(
        replies=[
            _sql("c1", query),
            AIMessage("The category result includes the Wardens profiles."),
            unsupported,
        ]
    )
    chat = ChatAgent(model, FakeRules(), retriever, SYSTEM)
    chat.ask("What is the Toughness of a Space Marine?")
    answer = chat.ask("What about Wardens of Ultramar?")

    assert len(retriever.calls) == 2
    assert retriever.calls[-1][1] == ("Wardens of Ultramar",)
    assert answer.tool_calls == []
    assert answer.text == (
        "### Wardens of Ultramar [DB: datasheets]\n\n"
        "- Ancient Gadriel, Veteran Sergeant Metaurus: Toughness 4. "
        "[DB: datasheets_models]\n"
        "- Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha: Toughness 3. "
        "[DB: datasheets_models]\n\n"
        "- Recorded publication: Space Marines. [DB: source]\n"
        "- Recorded edition: 11. [DB: source]"
    )
    assert "human operative" not in answer.text
    assert "Space Marine models" not in answer.text
    assert answer.trace.messages[-1].additional_kwargs == {
        "answer_format": "structured_database_fields"
    }
    (query,) = answer.application_queries
    assert (query.origin, query.operation, query.parameters) == (
        "application",
        "profile_lookup",
        ("Wardens of Ultramar",),
    )
    assert query.result.rows == [
        (
            "000004188",
            "Wardens of Ultramar",
            "1",
            "Ancient Gadriel, Veteran Sergeant Metaurus",
            "4",
            "Space Marines",
            "11",
        ),
        (
            "000004188",
            "Wardens of Ultramar",
            "2",
            "Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha",
            "3",
            "Space Marines",
            "11",
        ),
    ]
    stored = RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict())))
    assert answer_from_trace(stored).application_queries == [query]
    assert model.replies == [unsupported]
    _assert_rebuilt(answer.trace, model)


def test_a_filtered_previous_result_cannot_hide_models_from_the_follow_up(tmp_path):
    database = tmp_path / "profiles.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(
            """
            CREATE TABLE datasheets (Id TEXT, name TEXT, sourceId TEXT);
            CREATE TABLE datasheets_models (
                DatasheetId TEXT, line TEXT, name TEXT, t TEXT
            );
            CREATE TABLE source (Id TEXT, name TEXT, edition TEXT);
            INSERT INTO source VALUES ('src', 'Space Marines', '11');
            INSERT INTO datasheets VALUES ('000004188', 'Wardens of Ultramar', 'src');
            INSERT INTO datasheets_models VALUES
                ('000004188', '1', 'Ancient Gadriel, Veteran Sergeant Metaurus', '4'),
                ('000004188', '2',
                 'Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha', '3');
            """
        )
    filtered_query = (
        "SELECT d.Id, d.name, s.name AS source, s.edition, "
        "GROUP_CONCAT(m.name || ': T' || m.t, '; ') AS profiles "
        "FROM datasheets d JOIN datasheets_models m ON m.DatasheetId = d.Id "
        "LEFT JOIN source s ON s.Id = d.sourceId "
        "WHERE d.name = 'Wardens of Ultramar' AND m.t = '4' "
        "GROUP BY d.Id, d.name, s.name, s.edition"
    )
    model = FakeChatModel(
        replies=[
            _sql("c1", filtered_query),
            AIMessage("Only one profile was returned."),
            AIMessage("This reply must not be used."),
        ]
    )
    chat = ChatAgent(model, FakeRules(), SqlRetriever(database, 50), SYSTEM)
    chat.ask("What is the Toughness of a Space Marine?")
    answer = chat.ask("What about Wardens of Ultramar?")

    assert "Ancient Gadriel, Veteran Sergeant Metaurus: Toughness 4" in answer.text
    assert (
        "Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha: Toughness 3" in answer.text
    )
    assert len(model.replies) == 1
    assert model.replies[0].text == "This reply must not be used."


def test_failed_complete_lookup_then_filtered_sql_uses_the_normal_final_answer():
    class ObservedSqlRetriever:
        def __init__(self):
            self.results = iter(
                [
                    SqlResult(
                        ["source", "edition", "t", "model_profiles", "datasheets"],
                        [("Space Marines", "11", "4", 3, 3)],
                        False,
                        None,
                    ),
                    SqlResult(
                        ["Id", "name", "source", "edition", "profiles"],
                        [
                            (
                                "000001606",
                                "Assault Intercessor Squad",
                                "Space Marines",
                                "11",
                                "Assault Intercessor Squad: T4",
                            )
                        ],
                        False,
                        None,
                    ),
                    SqlResult(
                        ["table", "Id", "name", "factionId"],
                        [("datasheets", "000004188", "Wardens of Ultramar", "SM")],
                        False,
                        None,
                    ),
                    SqlResult(
                        ["Id", "name", "model", "t", "source", "edition"],
                        [
                            (
                                "000004188",
                                "Wardens of Ultramar",
                                "Ancient Gadriel, Veteran Sergeant Metaurus",
                                "4",
                                "Space Marines",
                                "11",
                            ),
                            (
                                "000004188",
                                "Wardens of Ultramar",
                                "Gaius Silva, Aemelia Minervas, Dainal Kornelius, Lucia Vestha",
                                "3",
                                "Space Marines",
                                "11",
                            ),
                        ],
                        False,
                        None,
                    ),
                ]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            if parameters:
                return SqlResult([], [], False, "profile lookup unavailable")
            return next(self.results)

    filtered_sql = (
        "SELECT d.Id, d.name, m.name AS model, m.t, s.name AS source, s.edition "
        "FROM datasheets d JOIN datasheets_models m ON m.DatasheetId = d.Id "
        "LEFT JOIN source s ON s.Id = d.sourceId "
        "WHERE d.Id = '000004188' AND m.t = '4'"
    )
    bounded = AIMessage(
        "The filtered result establishes only the returned T4 profile; it does not establish "
        "every Wardens profile."
    )
    model = FakeChatModel(
        replies=[
            _sql("c1", "SELECT aggregate FROM profiles"),
            _sql("c2", "SELECT representatives FROM profiles"),
            AIMessage("The broad category has several Toughness values."),
            _sql("c3", "SELECT identity FROM datasheets"),
            _sql("c4", filtered_sql),
            bounded,
        ]
    )
    chat = ChatAgent(model, FakeRules(), ObservedSqlRetriever(), SYSTEM)
    chat.ask("What is the Toughness of a Space Marine?")
    events = list(chat.stream("What about Wardens of Ultramar?"))
    answer = chat.last_answer()

    assert answer.text == bounded.text
    assert "Gaius Silva" not in answer.text
    assert answer.trace.messages[-1].additional_kwargs.get("answer_format") is None
    assert "".join(event.text for event in events if event.kind == "text") == answer.text
    assert len(answer.tool_calls) == 2
    assert answer.tool_calls[-1].arguments == {"sql": filtered_sql}
    assert answer.application_queries[0].result.error == "profile lookup unavailable"
    assert len(model.calls) == 6
    assert model.replies == []
    _assert_rebuilt(answer.trace, model)


def test_incomplete_profile_fields_keep_normal_final_generation():
    class IncompleteSqlRetriever:
        def __init__(self):
            self.results = iter(
                [
                    SqlResult(
                        ["table", "Id", "name"],
                        [("datasheets", "1", "Example Unit")],
                        False,
                        None,
                    ),
                    SqlResult(
                        ["Id", "name", "model", "t", "edition"],
                        [("1", "Example Unit", "Example Model", "4", "11")],
                        False,
                        None,
                    ),
                ]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            if parameters:
                return SqlResult([], [], False, "profile lookup unavailable")
            return next(self.results)

    detail_sql = (
        "SELECT d.Id, d.name, m.name AS model, m.t, s.edition "
        "FROM datasheets d JOIN datasheets_models m ON m.DatasheetId = d.Id "
        "LEFT JOIN source s ON s.Id = d.sourceId WHERE d.Id = '1'"
    )
    model = FakeChatModel(
        replies=[
            AIMessage("A broad answer."),
            _sql("c1", "SELECT identity FROM datasheets"),
            _sql("c2", detail_sql),
            AIMessage("The available rows do not include the required publication field."),
        ]
    )
    chat = ChatAgent(model, FakeRules(), IncompleteSqlRetriever(), SYSTEM)
    chat.ask("What is the Toughness of a Space Marine?")
    answer = chat.ask("What about Example Unit?")

    assert answer.text == "The available rows do not include the required publication field."
    assert answer.trace.model_calls[-1].node == "final"
    assert model.calls[-1].tool_choice == "none"
    _assert_rebuilt(answer.trace, model)


def test_an_explicit_new_attribute_uses_normal_tool_routing(tiny_db):
    movement = (
        "SELECT d.name, 'Space Marines' AS source, '11' AS edition, '6' AS movement "
        "FROM datasheets d WHERE d.name = 'Intercessor Squad'"
    )
    model = FakeChatModel(
        replies=[
            _sql("c1", _complete_profile_query()),
            AIMessage("The result includes the Intercessor profile."),
            _sql("c2", movement),
            AIMessage("Its Movement is 6 inches."),
        ]
    )
    chat = _agent(model, tiny_db)
    chat.ask("What is the Toughness profile for this infantry sample?")
    answer = chat.ask("What about Intercessor Squad's Movement?")

    assert model.calls[2].tool_choice == AGENT_TOOL_CONFIG
    assert len(answer.tool_calls) == 1
    assert answer.tool_calls[0].arguments == {"sql": movement}


def test_an_ambiguous_inherited_attribute_uses_normal_routing(tiny_db):
    model = FakeChatModel(
        replies=[
            _sql("c1", _complete_profile_query()),
            AIMessage("The result includes the Intercessor profile."),
            AIMessage("The agent must resolve which attributes are requested."),
        ]
    )
    chat = _agent(model, tiny_db)
    chat.ask("What are the Movement and Toughness profiles for this infantry sample?")
    answer = chat.ask("What about Intercessor Squad?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"


def test_ambiguous_same_name_datasheets_keep_normal_agent_routing():
    class AmbiguousSqlRetriever:
        def __init__(self):
            self.results = iter(
                [
                    SqlResult(["name"], [("First result",)], False, None),
                    SqlResult(
                        PROFILE_LOOKUP_COLUMNS,
                        [
                            (
                                "001",
                                "Example Unit",
                                "1",
                                "First Model",
                                "4",
                                "Publication A",
                                "11",
                            ),
                            (
                                "002",
                                "Example Unit",
                                "1",
                                "Second Model",
                                "5",
                                "Publication B",
                                "11",
                            ),
                        ],
                        False,
                        None,
                    ),
                ]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            return next(self.results)

    model = FakeChatModel(
        replies=[
            _sql("c1", "SELECT a previous category result"),
            AIMessage("First answer."),
            AIMessage("The name is ambiguous, so the available evidence does not settle it."),
        ]
    )
    chat = ChatAgent(model, FakeRules(), AmbiguousSqlRetriever(), SYSTEM)
    chat.ask("What is the Toughness of this category?")
    answer = chat.ask("What about Example Unit?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"
    assert "ambiguous" in answer.text


@pytest.mark.parametrize(
    "lookup_result",
    [
        SqlResult([], [], False, "OperationalError: database unavailable"),
        SqlResult(
            PROFILE_LOOKUP_COLUMNS,
            [("001", "Example Unit", "1", "Example Model", "4", None, "11")],
            False,
            None,
        ),
        SqlResult(
            PROFILE_LOOKUP_COLUMNS,
            [("001", "Example Unit", "1", "Example Model", "4", "Book", "11")],
            True,
            None,
        ),
    ],
    ids=["query-failure", "missing-field", "truncated"],
)
def test_failed_or_incomplete_local_lookup_keeps_an_uncertain_agent_answer(lookup_result):
    class IncompleteSqlRetriever:
        def __init__(self):
            self.results = iter(
                [SqlResult(["name"], [("First result",)], False, None), lookup_result]
            )

        def run(self, sql: str, parameters=()) -> SqlResult:
            return next(self.results)

    model = FakeChatModel(
        replies=[
            _sql("c1", "SELECT a previous category result"),
            AIMessage("First answer."),
            AIMessage("The available database evidence does not establish the answer."),
        ]
    )
    chat = ChatAgent(model, FakeRules(), IncompleteSqlRetriever(), SYSTEM)
    chat.ask("What is the Toughness of this category?")
    answer = chat.ask("What about Example Unit?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"
    assert answer.text == "The available database evidence does not establish the answer."
    (query,) = answer.application_queries
    assert (query.origin, query.operation, query.parameters) == (
        "application",
        "profile_lookup",
        ("Example Unit",),
    )
    assert query.result == lookup_result
    stored = RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict())))
    assert answer_from_trace(stored).application_queries == [query]


def test_profile_follow_up_does_not_cross_a_conversation_reset(tiny_db):
    model = FakeChatModel(
        replies=[
            _sql("c1", _complete_profile_query()),
            AIMessage("The result includes the Intercessor profile."),
            AIMessage("A new conversation must retrieve its own evidence."),
        ]
    )
    chat = _agent(model, tiny_db)
    chat.ask("What is the Toughness profile for this infantry sample?")
    chat.reset()
    answer = chat.ask("What about Intercessor Squad?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"


def test_only_identical_sql_reuses_the_read_only_result():
    class CountingSqlRetriever:
        def __init__(self):
            self.calls: list[str] = []

        def run(self, sql: str, parameters=()) -> SqlResult:
            self.calls.append(sql)
            return SqlResult(["name", "t"], [("Intercessor", "4")], False, None)

    retriever = CountingSqlRetriever()
    run_sql = build_tools(FakeRules(), retriever)[1]
    query = "SELECT name, t FROM datasheets_models;"
    first = run_sql.invoke(tool_call("run_sql", {"sql": query}, "q1"))
    second = run_sql.invoke(tool_call("run_sql", {"sql": query}, "q2"))

    assert len(retriever.calls) == 1
    assert second.text == first.text
    assert second.artifact == first.artifact


def test_sql_cache_preserves_literal_whitespace_across_a_conversation_reset():
    class CountingSqlRetriever:
        def __init__(self):
            self.calls: list[str] = []

        def run(self, sql: str, parameters=()) -> SqlResult:
            self.calls.append(sql)
            return SqlResult(["value"], [(sql,)], False, None)

    first_query = "SELECT 'Ancient  Gadriel' AS value"
    second_query = "SELECT 'Ancient Gadriel' AS value"
    retriever = CountingSqlRetriever()
    model = FakeChatModel(
        replies=[
            _sql("c1", first_query),
            AIMessage("First answer."),
            _sql("c2", second_query),
            AIMessage("Second answer."),
        ]
    )
    chat = ChatAgent(model, FakeRules(), retriever, SYSTEM)

    chat.ask("Run the first literal query.")
    chat.reset()
    answer = chat.ask("Run the second literal query.")

    assert retriever.calls == [first_query, second_query]
    assert answer.tool_calls[0].result == SqlResult(["value"], [(second_query,)], False, None)


# --- the round limit and the final call --------------------------------------------------------


def _rounds(count: int, start: int = 0) -> list[AIMessage]:
    return [_search(f"c{start + i}") for i in range(count)]


def test_the_round_limit_leads_to_one_final_call_with_its_prompt_and_instruction(tiny_db):
    model = FakeChatModel(replies=[*_rounds(MAX_TOOL_ROUNDS), AIMessage("What the results show.")])
    answer = _agent(model, tiny_db).ask("question")  # the longest normal run: within the limit

    assert [c.tool_choice for c in model.calls] == [AGENT_TOOL_CONFIG] * MAX_TOOL_ROUNDS + ["none"]
    messages = answer.trace.messages
    assert model.calls[-1].messages[0].text == FINAL_PROMPT
    assert model.calls[-1].messages[-1] == HumanMessage(FINAL_INSTRUCTION)
    assert model.calls[-1].messages[1:-1] == messages[:-1]
    assert all(m.text != FINAL_INSTRUCTION for m in messages)  # sent, not stored
    assert len(answer.tool_calls) == MAX_TOOL_ROUNDS
    assert answer.text == "What the results show." and answer.answered
    _assert_rebuilt(answer.trace, model)


def test_a_follow_up_counts_its_rounds_from_zero(tiny_db):
    model = FakeChatModel(
        replies=[
            *_rounds(MAX_TOOL_ROUNDS),
            AIMessage("First."),
            *_rounds(MAX_TOOL_ROUNDS - 1, start=10),
            AIMessage("Second."),
        ]
    )
    chat = _agent(model, tiny_db)
    chat.ask("first question")
    answer = chat.ask("follow-up")  # under the limit: the agent answers, no final call

    assert [c.tool_choice for c in model.calls].count("none") == 1
    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert len(answer.tool_calls) == MAX_TOOL_ROUNDS - 1  # the last question's calls only
    assert answer.text == "Second."


def test_two_sql_calls_end_with_a_grounded_final_response(tiny_db):
    first = "SELECT name, t FROM datasheets_models"
    second = "SELECT name FROM datasheets"
    model = FakeChatModel(
        replies=[_sql("c1", first), _sql("c2", second), AIMessage("What the results support.")]
    )
    answer = _agent(model, tiny_db).ask("Compare the profiles and units.")

    assert MAX_SQL_CALLS_PER_QUESTION == 2
    assert len(answer.tool_calls) == 2
    assert model.calls[-1].tool_choice == "none"
    assert answer.trace.model_calls[-1].node == "final"
    assert answer.text == "What the results support."


def test_sql_limit_final_can_report_insufficient_evidence(tiny_db):
    bad = "SELECT missing FROM datasheets"
    model = FakeChatModel(
        replies=[
            _sql("c1", bad),
            _sql("c2", "SELECT absent FROM datasheets_models"),
            AIMessage("The available results do not establish the requested value."),
        ]
    )
    answer = _agent(model, tiny_db).ask("What is the missing value?")

    assert all(call.error for call in answer.tool_calls)
    assert answer.trace.model_calls[-1].node == "final"
    assert "do not establish" in answer.text


def test_one_step_less_than_the_recursion_limit_would_stop_the_longest_run(tiny_db, monkeypatch):
    monkeypatch.setattr(agent_module, "RECURSION_LIMIT", agent_module.RECURSION_LIMIT - 1)
    model = FakeChatModel(replies=[*_rounds(MAX_TOOL_ROUNDS), AIMessage("Answer.")])
    with pytest.raises(Exception, match="Recursion limit"):
        _agent(model, tiny_db).ask("question")


@pytest.mark.parametrize("node", ["final", "agent"])
def test_a_reply_without_text_ends_the_run_and_is_kept_as_returned(tiny_db, node):
    empty = AIMessage(
        "",
        additional_kwargs=dict(SIGNATURES),
        response_metadata={"finish_reason": "UNEXPECTED_TOOL_CALL"},
    )
    replies = [*_rounds(MAX_TOOL_ROUNDS), empty] if node == "final" else [empty]
    model = FakeChatModel(replies=replies)
    answer = _agent(model, tiny_db).ask("question")

    stored = answer.trace.messages[-1]
    assert (stored.content, stored.tool_calls) == ("", [])
    assert stored.additional_kwargs == empty.additional_kwargs
    assert stored.response_metadata["finish_reason"] == "UNEXPECTED_TOOL_CALL"
    assert len(model.calls) == len(replies)  # nothing called after it
    assert answer.text == "" and not answer.answered
    assert answer.finish_reason == "UNEXPECTED_TOOL_CALL"
    assert answer.trace.model_calls[-1].node == node


# --- streaming ---------------------------------------------------------------------------------


def _tool_chunk(name: str | None, args: str, call_id: str | None) -> dict:
    return {"name": name, "args": args, "id": call_id, "index": 0, "type": "tool_call_chunk"}


def test_events_by_kind_and_the_answer_is_the_last_message_only(tiny_db):
    model = FakeChatModel(
        replies=[
            [
                AIMessageChunk("Let me "),
                AIMessageChunk("look this up."),
                AIMessageChunk("", tool_call_chunks=[_tool_chunk("search_rules", '{"que', "c1")]),
                AIMessageChunk("", tool_call_chunks=[_tool_chunk(None, 'ry": "grenade"}', None)]),
            ],
            [AIMessageChunk("Roll "), AIMessageChunk(""), AIMessageChunk("six D6.")],
        ]
    )
    chat = _agent(model, tiny_db)
    events = [(event.kind, event.text) for event in chat.stream("How do grenades work?")]

    assert events == [
        ("text", "Let me "),
        ("text", "look this up."),
        ("tool_start", "search_rules"),
        ("tool_end", "search_rules"),
        ("text", "Roll "),
        ("text", "six D6."),
    ]
    answer = chat.last_answer()
    assert answer.text == "Roll six D6."  # not the text written before the tool call
    (call,) = answer.tool_calls
    assert call.arguments == {"query": "grenade"}  # the streamed pieces joined
    _assert_rebuilt(answer.trace, model)


# --- the trace -----------------------------------------------------------------------------------


def test_a_stored_trace_gives_back_the_same_answer(tiny_db):
    sql = "SELECT name FROM datasheets ORDER BY name"
    model = FakeChatModel(replies=[_search("c1"), _sql("c2", sql), AIMessage("Answer.")])
    answer = _agent(model, tiny_db).ask("question")

    stored = RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict())))
    again = answer_from_trace(stored)
    assert (again.text, again.tool_calls) == (answer.text, answer.tool_calls)
    assert again.tool_calls[1].result.rows == [("Heavy Intercessor Squad",), ("Intercessor Squad",)]


def test_the_trace_holds_prompts_settings_tools_and_every_message(tiny_db):
    model = FakeChatModel(replies=[_search("c1"), AIMessage("Answer.")])
    trace = _agent(model, tiny_db).ask("question").trace

    assert (trace.system_prompt, trace.final_prompt) == (SYSTEM, FINAL_PROMPT)
    assert trace.final_instruction == FINAL_INSTRUCTION
    assert trace.model_settings == {"model": "fake-chat-model", "temperature": 0.0}
    assert [t["function"]["name"] for t in trace.tools] == ["search_rules", "run_sql"]
    assert "query" in trace.tools[0]["function"]["parameters"]["properties"]
    sql_parameters = trace.tools[1]["function"]["parameters"]
    assert sql_parameters["required"] == ["sql"]
    assert sql_parameters["properties"]["sql"]["type"] == "string"
    assert [type(m) for m in trace.messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert trace.messages[2].artifact["chunks"][0]["chunk"]["chunk_id"] == "core_rules-0150"


@pytest.mark.parametrize("final", [False, True])
def test_a_trace_read_back_from_json_rebuilds_every_call(tiny_db, final):
    rounds = MAX_TOOL_ROUNDS if final else 1
    model = FakeChatModel(replies=[*_rounds(rounds), AIMessage("Answer.")])
    trace = _agent(model, tiny_db).ask("question").trace

    read_back = RunTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert read_back == trace
    assert [(c.node, c.tool_choice) for c in read_back.model_calls] == [
        ("agent", "validated")
    ] * rounds + [("final", "none") if final else ("agent", "validated")]
    _assert_rebuilt(read_back, model)


# --- the prompts ---------------------------------------------------------------------------------


def test_the_system_prompt_holds_the_schema_and_the_final_prompt_no_tool_rules():
    prompt = build_system_prompt("SCHEMA TEXT")
    assert TOOL_RULES in prompt and prompt.endswith("SCHEMA TEXT")
    assert "[Core Rules p. N]" in prompt and "[DB: table]" in prompt
    assert TOOL_RULES not in FINAL_PROMPT and "[Core Rules p. N]" in FINAL_PROMPT
    assert APPLICABILITY_RULES in prompt and APPLICABILITY_RULES in FINAL_PROMPT
    # the final call cannot use tools: its prompt never asks for one
    assert "search_rules" not in FINAL_PROMPT and "run_sql" not in FINAL_PROMPT


def test_answer_prompts_forbid_the_unsupported_tactical_consequence_observed_in_q006():
    prompt = build_system_prompt("SCHEMA TEXT")
    boundary = (
        "do not add tactical benefits, disadvantages, exposure, opportunities or causal claims "
        "unless the retrieved evidence explicitly states them"
    )

    assert boundary in prompt
    assert boundary in FINAL_PROMPT


def test_answer_prompts_forbid_the_unsupported_absence_claim_observed_in_q010():
    prompt = build_system_prompt("SCHEMA TEXT")
    boundary = (
        "unless the retrieved evidence explicitly states the absence, report the specific "
        "restriction it gives and say any broader limit is not established"
    )

    assert boundary in prompt
    assert boundary in FINAL_PROMPT
