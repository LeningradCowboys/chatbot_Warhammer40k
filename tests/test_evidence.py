"""Observed R77 failures: source combination and scope survive tool choice, replay and display."""

import io
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

from src.rag.chatbot.agent import (
    AGENT_TOOL_CONFIG,
    RELATED_RULE_SECTIONS,
    ChatAgent,
    RunTrace,
    answer_from_trace,
    build_tools,
    chunk_block,
    mission_scopes,
    source_notes,
)
from src.rag.chatbot.prompts import APPLICABILITY_RULES
from src.rag.chunker.markdown_chunker import Chunk, jsonl_to_chunks
from src.rag.config import CLEAN_DB, RULES_CHUNKS, TOP_K
from src.rag.retriever.rules_retriever import RetrievedChunk
from src.rag.retriever.sql_retriever import SqlRetriever
from src.scripts.chat import answer_live, answer_once, sources_text
from tests.chat_acceptance import answer_record
from tests.fakes import COVER, GRENADE, FakeChatModel, FakeRules, tool_call

CARD = replace(GRENADE, pages=[42], page_footers=["CORE RULES | STRATAGEMS"])
COMPLETE_CARD = replace(
    CARD,
    text="""Section: GRENADE
1CP

CORE - WARGEAR STRATAGEM

WHEN: Your Shooting phase.

TARGET: One Grenades unit that has not been selected to shoot this phase.

EFFECT: Select one visible enemy unit within 8\" and roll six D6.""",
)
SCORING = replace(
    COVER,
    pages=[59],
    section="CAPTURE AND CONTROL",
    page_footers=["CORE RULES | MISSION: ONLY WAR"],
)


@pytest.fixture
def cards_db(tiny_db: Path) -> Path:
    with closing(sqlite3.connect(tiny_db)) as db:
        for name in ("FactionId", "type", "cpCost", "turn", "phase", "detachment"):
            db.execute(f"ALTER TABLE stratagems ADD COLUMN \"{name}\" TEXT DEFAULT ''")
        db.execute("UPDATE stratagems SET cpCost = '1'")
        db.execute(
            "INSERT INTO stratagems (id, name, description, FactionId) "
            "VALUES ('s2', 'GRENADE', 'Faction variant, not the core rule', 'SM')"
        )
        db.commit()
    return tiny_db


@pytest.fixture
def complete_cards_db(cards_db: Path) -> Path:
    with closing(sqlite3.connect(cards_db)) as db:
        db.execute(
            "UPDATE stratagems SET type = 'Core - Wargear Stratagem', turn = 'Your turn', "
            "phase = 'Shooting phase' WHERE id = 's1'"
        )
        db.commit()
    return cards_db


def _agent(db: Path, chunks: list, *replies: AIMessage, max_rows: int = 50) -> ChatAgent:
    return ChatAgent(
        FakeChatModel(replies=list(replies)),
        FakeRules([RetrievedChunk(chunk, 0.03) for chunk in chunks]),
        SqlRetriever(db, max_rows),
        "Test system prompt",
    )


def _search() -> AIMessage:
    return AIMessage("", tool_calls=[tool_call("search_rules", {"query": "grenade"}, "s")])


def _sql(query: str) -> AIMessage:
    return AIMessage("", tool_calls=[tool_call("run_sql", {"sql": query}, "q")])


@pytest.mark.parametrize(
    "evidence",
    [
        "This unit can perform this action.",
        "This unit cannot perform this action.",
        "This unit can perform this action if it remained stationary.",
        "If it has Assault weapons, it is still eligible to shoot after it Advanced.",
    ],
)
def test_rules_search_preserves_permission_prohibition_and_conditions(tiny_db, evidence):
    chunk = replace(COVER, text=evidence)
    search = build_tools(FakeRules([RetrievedChunk(chunk, 0.03)]), SqlRetriever(tiny_db, 50))[0]
    message = search.invoke(tool_call("search_rules", {"query": "action eligibility"}, "s"))

    assert evidence in message.text
    assert APPLICABILITY_RULES in message.text
    assert message.artifact["chunks"][0]["chunk"]["text"] == evidence


def test_rules_search_marks_missing_evidence_without_inventing_a_conclusion(tiny_db):
    class EmptyRules:
        def search(self, query: str, top_k: int) -> list:
            return []

        def sections(self, names: list[str]) -> list:
            return []

    search = build_tools(EmptyRules(), SqlRetriever(tiny_db, 50))[0]
    message = search.invoke(tool_call("search_rules", {"query": "unknown action"}, "s"))

    assert message.text == APPLICABILITY_RULES
    assert message.artifact["chunks"] == []
    assert "undetermined" in message.text


def test_conflicting_rule_passages_remain_separate_for_source_grounding(tiny_db):
    allowed = replace(COVER, chunk_id="allowed", text="This unit can perform this action.")
    forbidden = replace(GRENADE, chunk_id="forbidden", text="This unit cannot perform this action.")
    search = build_tools(
        FakeRules([RetrievedChunk(allowed, 0.03), RetrievedChunk(forbidden, 0.02)]),
        SqlRetriever(tiny_db, 50),
    )[0]
    message = search.invoke(tool_call("search_rules", {"query": "action eligibility"}, "s"))

    assert message.text.index(allowed.text) < message.text.index(forbidden.text)
    assert [item["chunk"]["chunk_id"] for item in message.artifact["chunks"]] == [
        "allowed",
        "forbidden",
    ]
    assert "Keep conflicting sources separate" in message.text


def test_search_alone_supplies_both_sources_and_round_trips(cards_db):
    agent = _agent(
        cards_db, [replace(CARD, section=" grenade "), CARD], _search(), AIMessage("Answer")
    )
    answer = agent.ask("What is the rule for grenades?")
    (call,) = answer.tool_calls
    assert call.result.stratagem_names == ["GRENADE"]
    assert [row[0] for row in call.result.stratagems.rows] == ["s1"]
    assert "[Core Rules p. 42]" in call.output and "[DB: stratagems]" in call.output
    assert "publication/edition not recorded" in call.output
    assert "WHEN: Your Shooting phase" in call.output
    assert "Faction variant" not in call.output
    assert "database rows returned: 1" in sources_text(answer)
    rebuilt = answer_from_trace(RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict()))))
    assert rebuilt.tool_calls == answer.tool_calls
    assert source_notes(rebuilt) == source_notes(answer)
    assert answer_record(answer)["text"] == "Answer"
    assert answer_record(answer)["source_notes"] == source_notes(answer)


def test_complete_named_stratagem_evidence_routes_directly_to_final(complete_cards_db):
    model = FakeChatModel(replies=[_search(), AIMessage("The two sourced rules differ.")])
    rules = FakeRules([RetrievedChunk(COMPLETE_CARD, 0.03), RetrievedChunk(COVER, 0.01)])
    answer = ChatAgent(model, rules, SqlRetriever(complete_cards_db, 50), "System").ask(
        "What is the rule for grenades?"
    )

    assert len(answer.tool_calls) == 1
    assert [call.tool_choice for call in model.calls] == [AGENT_TOOL_CONFIG, "none"]
    assert answer.trace.model_calls[-1].node == "final"
    context = answer.trace.model_calls[-1].context
    assert context is not None
    assert "Complete named Stratagem evidence" in context
    assert "[Core Rules p. 42]" in context and "[DB: stratagems]" in context
    assert '"publication_and_edition": "not recorded"' in context
    assert "WHEN: Your Shooting phase" in context
    assert "Section: COVER" not in context
    final_input = model.calls[-1].messages
    assert [message.text for message in final_input[1:-1]] == ["What is the rule for grenades?"]
    rebuilt = RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict())))
    assert rebuilt.model_input(rebuilt.model_calls[-1]) == final_input


def test_incomplete_named_stratagem_card_keeps_normal_routing(complete_cards_db):
    model = FakeChatModel(replies=[_search(), AIMessage("The agent needs more evidence.")])
    answer = ChatAgent(
        model,
        FakeRules([RetrievedChunk(CARD, 0.03)]),
        SqlRetriever(complete_cards_db, 50),
        "System",
    ).ask("What is the rule for grenades?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"
    assert answer.trace.model_calls[-1].context is None


@pytest.mark.parametrize(("max_rows", "reason"), [(1, "truncated"), (50, "ambiguous")])
def test_incomplete_exact_database_result_keeps_normal_routing(complete_cards_db, max_rows, reason):
    with closing(sqlite3.connect(complete_cards_db)) as db:
        db.execute(
            "INSERT INTO stratagems "
            "(id, name, description, FactionId, type, cpCost, turn, phase, detachment) "
            "SELECT 's3', name, description, FactionId, type, cpCost, turn, phase, detachment "
            "FROM stratagems WHERE id = 's1'"
        )
        db.commit()
    model = FakeChatModel(replies=[_search(), AIMessage(f"Resolve {reason} evidence.")])
    answer = ChatAgent(
        model,
        FakeRules([RetrievedChunk(COMPLETE_CARD, 0.03)]),
        SqlRetriever(complete_cards_db, max_rows),
        "System",
    ).ask("What is the rule for grenades?")

    result = answer.tool_calls[0].result.stratagems
    assert result.truncated is (reason == "truncated")
    expected_rows = 1 if max_rows == 1 else 2
    assert len(result.rows) == expected_rows
    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"


@pytest.mark.parametrize("failure", ["missing-field", "conflicting-cost", "database-error"])
def test_incomplete_conflicting_or_failed_database_keeps_normal_routing(complete_cards_db, failure):
    with closing(sqlite3.connect(complete_cards_db)) as db:
        if failure == "missing-field":
            db.execute("UPDATE stratagems SET phase = '' WHERE id = 's1'")
        elif failure == "conflicting-cost":
            db.execute("UPDATE stratagems SET cpCost = '2' WHERE id = 's1'")
        else:
            db.execute("DROP TABLE stratagems")
        db.commit()
    model = FakeChatModel(replies=[_search(), AIMessage("Report the evidence gap.")])
    answer = ChatAgent(
        model,
        FakeRules([RetrievedChunk(COMPLETE_CARD, 0.03)]),
        SqlRetriever(complete_cards_db, 50),
        "System",
    ).ask("What is the rule for grenades?")

    result = answer.tool_calls[0].result.stratagems
    if failure == "missing-field":
        phase = result.rows[0][result.columns.index("phase")]
        assert phase == ""
    elif failure == "conflicting-cost":
        cost = result.rows[0][result.columns.index("cpCost")]
        assert cost == "2"
    else:
        assert result.error is not None
    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"


def test_named_stratagem_applicability_question_keeps_normal_routing(complete_cards_db):
    model = FakeChatModel(replies=[_search(), AIMessage("More facts are required.")])
    answer = ChatAgent(
        model,
        FakeRules([RetrievedChunk(COMPLETE_CARD, 0.03)]),
        SqlRetriever(complete_cards_db, 50),
        "System",
    ).ask("Can an Intercessor use the Grenade stratagem after advancing?")

    assert model.calls[-1].tool_choice == AGENT_TOOL_CONFIG
    assert answer.trace.model_calls[-1].node == "agent"
    assert answer.trace.model_calls[-1].context is None


def test_general_stratagem_search_adds_required_related_sections(cards_db):
    general = replace(
        CARD,
        section="STRATAGEMS",
        text="Section: STRATAGEMS\nPay the listed CP cost to use a Stratagem.",
    )
    battle_shock = Chunk(
        "core_rules-battle-shock",
        "Section: 2. BATTLE-SHOCK\nThe controlling player takes a Battle-shock test.",
        [11],
        "2. BATTLE-SHOCK",
        "text",
        [],
        ["CORE RULES | COMMAND PHASE"],
    )
    gaining_cp = Chunk(
        "core_rules-gaining-cp",
        "Section: GAINING COMMAND POINTS\n"
        "Outside of the CP players gain at the start of the Command phase, each player can only "
        "gain a total of 1CP per battle round, regardless of the source.",
        [11],
        "GAINING COMMAND POINTS",
        "text",
        [],
        ["CORE RULES | COMMAND PHASE"],
    )
    irrelevant = replace(COVER, chunk_id="irrelevant", section="BENEFIT OF COVER")
    fillers = [
        replace(COVER, chunk_id=f"filler-{index}", section=f"FILLER {index}")
        for index in range(TOP_K - 1)
    ]
    rules = FakeRules(
        [RetrievedChunk(general, 0.03), *[RetrievedChunk(item, 0.01) for item in fillers]],
        [
            RetrievedChunk(battle_shock, 0.0),
            RetrievedChunk(gaining_cp, 0.0),
            RetrievedChunk(irrelevant, 0.0),
        ],
    )
    model = FakeChatModel(replies=[_search(), AIMessage("Answer")])
    answer = ChatAgent(
        model,
        rules,
        SqlRetriever(cards_db, 50),
        "System",
    ).ask("How can I use a stratagem?")
    (call,) = answer.tool_calls
    sections = [item.chunk.section for item in call.result.chunks]
    chunk_ids = [item.chunk.chunk_id for item in call.result.chunks]

    assert TOP_K == 8
    assert rules.queries == [("grenade", TOP_K)]
    assert call.result.related_sections == RELATED_RULE_SECTIONS["STRATAGEMS"]
    assert sections[:TOP_K] == ["STRATAGEMS", *[f"FILLER {index}" for index in range(7)]]
    assert sections[TOP_K:] == ["2. BATTLE-SHOCK", "GAINING COMMAND POINTS"]
    assert len(chunk_ids) == TOP_K + len(RELATED_RULE_SECTIONS["STRATAGEMS"])
    assert len(chunk_ids) == len(set(chunk_ids))
    assert "BENEFIT OF COVER" not in sections
    final_context = model.calls[-1].messages[-1].text
    assert "Section: STRATAGEMS" in final_context
    assert "controlling player" in final_context
    assert "each player can only gain a total of 1CP per battle round" in final_context


def test_required_stratagem_sections_are_not_duplicated_when_already_retrieved(cards_db):
    general = replace(CARD, section="STRATAGEMS")
    gaining_cp = replace(
        COVER,
        chunk_id="core_rules-gaining-cp",
        section="GAINING COMMAND POINTS",
        text="Section: GAINING COMMAND POINTS\nEach player can gain only 1CP per battle round.",
    )
    battle_shock = replace(
        COVER,
        chunk_id="core_rules-battle-shock",
        section="2. BATTLE-SHOCK",
        text="Section: 2. BATTLE-SHOCK\nThe controlling player cannot use Stratagems.",
    )
    fillers = [
        replace(COVER, chunk_id=f"dedupe-filler-{index}", section=f"FILLER {index}")
        for index in range(TOP_K - 2)
    ]
    found = [general, gaining_cp, *fillers]
    rules = FakeRules(
        [RetrievedChunk(item, 0.03) for item in found],
        [RetrievedChunk(battle_shock, 0.0), RetrievedChunk(gaining_cp, 0.0)],
    )
    answer = ChatAgent(
        FakeChatModel(replies=[_search(), AIMessage("Answer")]),
        rules,
        SqlRetriever(cards_db, 50),
        "System",
    ).ask("How can I use a stratagem?")
    (call,) = answer.tool_calls
    chunk_ids = [item.chunk.chunk_id for item in call.result.chunks]

    assert len(chunk_ids) == TOP_K + 1
    assert chunk_ids.count("core_rules-gaining-cp") == 1
    assert chunk_ids.count("core_rules-battle-shock") == 1
    assert len(chunk_ids) <= TOP_K + len(RELATED_RULE_SECTIONS["STRATAGEMS"])


def test_multiple_exact_matches_are_retained_and_limits_are_visible(cards_db):
    with closing(sqlite3.connect(cards_db)) as db:
        db.execute(
            "INSERT INTO stratagems (id, name, description, FactionId) "
            "VALUES ('s3', 'GRENADE', 'Other core form', '')"
        )
        db.commit()
    answer = _agent(cards_db, [CARD], _search(), AIMessage("Answer")).ask("grenade")
    assert len(answer.tool_calls[0].result.stratagems.rows) == 2
    limited = _agent(cards_db, [CARD], _search(), AIMessage("Answer"), max_rows=1).ask("grenade")
    assert limited.tool_calls[0].result.stratagems.truncated
    assert any("incomplete" in note for note in source_notes(limited))


def test_no_match_is_distinct_from_unqueried_and_a_database_failure(cards_db, tiny_db):
    missing = replace(CARD, section="NOT A CARD' OR 1=1 --")
    answer = _agent(cards_db, [missing], _search(), AIMessage("Answer")).ask("grenade")
    assert answer.tool_calls[0].result.stratagems.rows == []
    assert any("no matching rows" in note for note in source_notes(answer))
    plain = _agent(cards_db, [COVER], _search(), AIMessage("Answer")).ask("cover")
    assert plain.tool_calls[0].result.stratagems is None
    assert not source_notes(plain)
    with closing(sqlite3.connect(tiny_db)) as db:
        db.execute("DROP TABLE stratagems")
        db.commit()
    failed = _agent(tiny_db, [CARD], _search(), AIMessage("PDF only")).ask("grenade")
    assert failed.tool_calls[0].pages == [42]
    assert failed.tool_calls[0].error
    assert "[Core Rules p. 42]" in failed.tool_calls[0].output
    assert any("query failed" in note for note in source_notes(failed))
    assert "pages: 42" in sources_text(failed)


def test_legacy_list_trace_keeps_its_original_text_and_no_invented_db_lookup(tiny_db):
    answer = _agent(tiny_db, [SCORING], _search(), AIMessage("Answer")).ask("scoring")
    message = next(m for m in answer.trace.messages if isinstance(m, ToolMessage))
    message.artifact = [asdict(RetrievedChunk(SCORING, 0.03))]
    message.content = "Historical tool text, not today's formatted context"
    rebuilt = answer_from_trace(RunTrace.from_dict(json.loads(json.dumps(answer.trace.to_dict()))))
    assert rebuilt.tool_calls[0].output == message.content
    assert rebuilt.tool_calls[0].result.stratagems is None
    assert rebuilt.tool_calls[0].pages == [59]


def test_scope_follows_page_labels_and_survives_follow_up_but_not_reset(tiny_db):
    mixed = replace(SCORING, pages=[59, 60], page_footers=["MISSION: ONLY WAR", "MISSION: SECOND"])
    assert mission_scopes(mixed) == ["ONLY WAR (Core Rules p. 59)", "SECOND (Core Rules p. 60)"]
    generic = replace(SCORING, page_footers=["CORE RULES | MISSIONS | OBJECTIVE MARKERS"])
    assert not mission_scopes(generic)
    assert "Mission scope: ONLY WAR" in chunk_block(SCORING)
    agent = _agent(
        tiny_db, [SCORING], _search(), AIMessage("First"), AIMessage("Second"), AIMessage("Third")
    )
    first = agent.ask("scoring")
    assert any(note.startswith("This turn:") and "ONLY WAR" in note for note in source_notes(first))
    follow = agent.ask("And the winner?")
    assert not follow.tool_calls
    assert all(note.startswith("Earlier in this conversation:") for note in source_notes(follow))
    agent.reset()
    assert source_notes(agent.ask("unrelated")) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT 'Space Marines' AS source, '11' AS edition", "edition 11"),
        ("SELECT 'Legends' AS source, '0' AS edition", "edition not recorded"),
        ("SELECT NULL AS source, NULL AS edition", "no publication value"),
        ("SELECT name FROM datasheets", "could not identify publication/edition"),
        ("SELECT name FROM datasheets WHERE 0", "no matching rows"),
        ("SELECT missing FROM datasheets", "query failed"),
    ],
)
def test_sql_publication_and_missing_evidence_states(tiny_db, query, expected):
    answer = _agent(tiny_db, [COVER], _sql(query), AIMessage("Answer")).ask("unit")
    assert expected in "\n".join(source_notes(answer))


def test_mixed_publications_stay_with_their_records(tiny_db):
    query = (
        "SELECT 'A' AS name, '1' AS Id, 'Book A' AS source, '10' AS edition "
        "UNION ALL SELECT 'B', '2', 'Book B', '11'"
    )
    answer = _agent(tiny_db, [COVER], _sql(query), AIMessage("Answer")).ask("unit")
    notes = source_notes(answer)
    assert any("for A (ID 1): Book A; edition 10" in note for note in notes)
    assert any("for B (ID 2): Book B; edition 11" in note for note in notes)


def test_observed_wardens_publication_alias_is_recognized(tiny_db):
    query = (
        "SELECT 'Wardens of Ultramar' AS unit_name, "
        "'Ancient Gadriel, Veteran Sergeant Metaurus' AS model_name, '4' AS toughness, "
        "'Space Marines' AS publication, '11' AS edition"
    )
    answer = _agent(tiny_db, [COVER], _sql(query), AIMessage("Answer")).ask("unit")
    notes = source_notes(answer)
    assert any("Space Marines; edition 11" in note for note in notes)
    assert all("could not identify" not in note for note in notes)


def test_two_publication_columns_are_not_guessed_between(tiny_db):
    query = "SELECT 'Book A' AS source, 'Book B' AS publication, '11' AS edition"
    answer = _agent(tiny_db, [COVER], _sql(query), AIMessage("Answer")).ask("unit")
    assert any("unique publication field" in note for note in source_notes(answer))


def test_category_statistics_are_rendered_from_sql_rows_without_recalculation(tiny_db):
    query = (
        "SELECT 'Book A' AS source, '10' AS edition, '4' AS t, "
        "94 AS model_profiles, 90 AS datasheets UNION ALL "
        "SELECT 'Book B', '11', '4', 10, 1"
    )
    answer = _agent(tiny_db, [COVER], _sql(query), AIMessage("The model says 104/91.")).ask(
        "toughness"
    )
    output = io.StringIO()
    shown = _agent(tiny_db, [COVER], _sql(query), AIMessage("The model says 104/91."))
    assert answer_once(shown, "toughness", output, Console(file=io.StringIO())) == 0
    text = output.getvalue()
    assert "The model says 104/91." in text
    assert "| Book A | 10 | 4 | 94 | 90 |" in text
    assert "| Book B | 11 | 4 | 10 | 1 |" in text
    assert "does not calculate cross-publication totals" in text
    statistics = answer_record(answer)["database_statistics"]
    assert statistics[0]["rows"] == [("Book A", "10", "4", 94, 90), ("Book B", "11", "4", 10, 1)]


def test_byte_limited_sql_is_also_disclosed(tiny_db):
    query = (
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<20) "
        "SELECT hex(zeroblob(200)) AS value FROM n"
    )
    agent = ChatAgent(
        FakeChatModel(replies=[_sql(query), AIMessage("Answer")]),
        FakeRules(),
        SqlRetriever(tiny_db, 50, max_bytes=4096),
        "System",
    )
    answer = agent.ask("query")
    assert answer.tool_calls[0].result.truncated
    assert any("incomplete" in note for note in source_notes(answer))


@pytest.mark.parametrize("interactive", [False, True])
def test_both_cli_modes_show_truncation_even_when_the_model_omits_it(tiny_db, interactive):
    agent = _agent(
        tiny_db,
        [COVER],
        _sql("SELECT name FROM datasheets"),
        AIMessage("Model original"),
        max_rows=1,
    )
    output = io.StringIO()
    console = Console(file=output, width=140, color_system=None)
    if interactive:
        answer_live(agent, "toughness", console)
    else:
        assert answer_once(agent, "toughness", output, Console(file=io.StringIO())) == 0
    assert "result is incomplete" in output.getvalue()
    assert "Source notes" in output.getvalue()
    assert agent.last_answer().text == "Model original"


@pytest.mark.slow
def test_real_grenade_metadata_supplies_both_source_versions_without_another_model_call():
    if not RULES_CHUNKS.is_file() or not CLEAN_DB.is_file():
        pytest.skip("prepared rule chunks and database required")
    chunks = jsonl_to_chunks(RULES_CHUNKS.read_text(encoding="utf-8"))
    grenade = next(chunk for chunk in chunks if chunk.section == "GRENADE")
    rules = FakeRules([RetrievedChunk(grenade, 0.03)])
    search = build_tools(rules, SqlRetriever(CLEAN_DB, 50))[0]
    message = search.invoke(tool_call("search_rules", {"query": "grenade"}, "s"))
    assert "[Core Rules p. 42]" in message.text
    assert "[DB: stratagems]" in message.text
    result = message.artifact["stratagems"]
    assert result["error"] is None and not result["truncated"]
    rows = [dict(zip(result["columns"], row, strict=True)) for row in result["rows"]]
    assert rows and all(row["name"] == "GRENADE" and row["FactionId"] == "" for row in rows)
    assert all("Advanced" in row["description"] for row in rows)
    assert "not been selected to shoot" in grenade.text


@pytest.mark.slow
def test_real_stratagem_context_includes_the_complete_cp_generation_rule():
    if not RULES_CHUNKS.is_file() or not CLEAN_DB.is_file():
        pytest.skip("prepared rule chunks and database required")
    chunks = jsonl_to_chunks(RULES_CHUNKS.read_text(encoding="utf-8"))
    general = next(chunk for chunk in chunks if chunk.section == "STRATAGEMS")
    related = [
        RetrievedChunk(chunk, 0.0)
        for chunk in chunks
        if chunk.section in RELATED_RULE_SECTIONS["STRATAGEMS"]
    ]
    rules = FakeRules([RetrievedChunk(general, 0.03)], related)
    search = build_tools(rules, SqlRetriever(CLEAN_DB, 50))[0]
    message = search.invoke(tool_call("search_rules", {"query": "STRATAGEMS"}, "s"))
    result = message.artifact
    chunk_ids = [item["chunk"]["chunk_id"] for item in result["chunks"]]

    assert result["related_sections"] == RELATED_RULE_SECTIONS["STRATAGEMS"]
    assert chunk_ids == [general.chunk_id, "core_rules-0036", "core_rules-0037"]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert "Section: STRATAGEMS" in message.text
    assert (
        "Outside of the CP players gain at the start of the Command phase, each player can only "
        "gain a total of 1CP per battle round, regardless of the source."
    ) in message.text
