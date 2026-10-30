"""The chat agent: two tools, a LangGraph graph, streamed events and the answer with its trace.

The graph (plan §6.8): `prepare` handles a narrow entity-only profile follow-up with a fresh,
complete local database query; otherwise `agent` calls the chat model with the tools.
After a tool call, a complete named-Stratagem result can also route to `final`; otherwise tools
continue up to `MAX_TOOL_ROUNDS`. The `final` node formats verified single-attribute model-profile
results directly from database fields; other answers still use the chat model. The CLI consumes
`stream()`; the evaluation calls `ask()`.
"""

import json
import operator
import re
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.rag.chatbot.prompts import APPLICABILITY_RULES, FINAL_INSTRUCTION, FINAL_PROMPT
from src.rag.chunker.markdown_chunker import Chunk
from src.rag.config import MAX_TOOL_ROUNDS, TOP_K
from src.rag.llm.budget import BudgetExhausted
from src.rag.retriever.rules_retriever import RetrievedChunk, RulesRetriever
from src.rag.retriever.sql_retriever import SqlResult, SqlRetriever

# Above the longest normal run (prepare, agent and tools per round, then final).
# Reaching this limit is a bug.
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 3
MODEL_NODES = ("agent", "final")  # nodes whose answer messages are streamed to the CLI
RELATED_RULE_SECTIONS = {
    "STRATAGEMS": ["2. BATTLE-SHOCK", "GAINING COMMAND POINTS"],
}
AGENT_TOOL_CHOICE: Literal["validated"] = "validated"
AGENT_TOOL_CONFIG = {"mode": AGENT_TOOL_CHOICE.upper()}
MAX_SQL_CALLS_PER_QUESTION = 2
ENTITY_FOLLOW_UP = re.compile(r"\s*what\s+about\s+(.+?)\s*[?.!]?\s*", re.IGNORECASE)
ATTRIBUTE_QUESTION = re.compile(
    r"\s*what\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:of|for)\s+.+?\s*[?.!]?\s*",
    re.IGNORECASE,
)
NAMED_RULE_QUESTION = re.compile(
    r"\s*(?:what(?:'s|\s+is|\s+are)\s+(?:the\s+)?rules?\s+(?:for|of)\s+(.+?)"
    r"|how\s+does\s+(.+?)\s+work)\s*[?.!]?\s*",
    re.IGNORECASE,
)
STRATAGEM_CARD_FIELDS = ("WHEN:", "TARGET:", "EFFECT:")
STRATAGEM_COLUMNS = {
    "id",
    "name",
    "factionid",
    "type",
    "cpcost",
    "turn",
    "phase",
    "detachment",
    "description",
}
STRATAGEM_COST = re.compile(r"\b(\d+)\s*CP\b", re.IGNORECASE)
PROFILE_ATTRIBUTES = {
    "movement": ("m", "Movement"),
    "toughness": ("t", "Toughness"),
    "save": ("sv", "Save"),
    "invulnerable save": ("invsv", "Invulnerable Save"),
    "wounds": ("w", "Wounds"),
    "leadership": ("ld", "Leadership"),
    "objective control": ("oc", "Objective Control"),
}
PROFILE_LOOKUP_SQL = """\
SELECT DISTINCT d.Id AS datasheet_id, d.name AS datasheet, m.line AS model_line,
       m.name AS model, m.{attribute} AS attribute_value,
       s.name AS source, s.edition
FROM datasheets d
JOIN datasheets_models m ON m.DatasheetId = d.Id
LEFT JOIN source s ON s.Id = d.sourceId
WHERE d.name = ? COLLATE NOCASE
ORDER BY d.Id, CAST(m.line AS INTEGER), m.line, m.name"""
STRUCTURED_DATABASE_ANSWER = "structured_database_fields"


@dataclass(frozen=True)
class RulesSearchResult:
    """Retrieved PDF passages, required related sections and a core Stratagem lookup."""

    chunks: list[RetrievedChunk]
    stratagem_names: list[str]
    stratagems: SqlResult | None
    related_sections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    """One tool call of the last question, with what the model received and the result behind it."""

    name: str
    arguments: dict
    output: str  # the text the model received
    result: RulesSearchResult | SqlResult | None  # None when the tool itself failed

    @property
    def pages(self) -> list[int]:
        """The pages of the returned chunks, in order, each once; [] for `run_sql`."""
        if not isinstance(self.result, RulesSearchResult):
            return []
        return list(
            dict.fromkeys(page for found in self.result.chunks for page in found.chunk.pages)
        )

    @property
    def error(self) -> str | None:
        """The failure the model received when the tool failed, or the SQL error, else None."""
        if self.result is None:
            return self.output
        if isinstance(self.result, SqlResult):
            return self.result.error
        if self.result.stratagems is not None:
            return self.result.stratagems.error
        return None


@dataclass(frozen=True)
class ModelCall:
    """One call of the chat model, located in the thread's messages."""

    node: Literal["agent", "final"]
    # input: the node's prompt + messages[:input_count], then for final the final instruction;
    # reply: messages[input_count]
    input_count: int
    tool_choice: Literal["auto", "validated", "none"]
    context: str | None = None  # deterministic evidence added to a final call only


@dataclass(frozen=True)
class ApplicationQuery:
    """One local database query initiated by the application rather than the chat model."""

    origin: Literal["application"]
    operation: Literal["profile_lookup"]
    question_index: int
    sql: str
    parameters: tuple[object, ...]
    result: SqlResult


@dataclass(frozen=True)
class RunTrace:
    """What every model call of the thread was given and returned, so each can be rebuilt."""

    system_prompt: str  # given to every agent call
    final_prompt: str  # given to the final call
    final_instruction: str  # sent after the messages in the final call, not stored in them
    model_settings: dict  # model name, temperature, thinking settings: shared by every call
    tools: list[dict]  # the declarations of the bound tools
    messages: list[BaseMessage]  # the thread's messages: questions, model replies, tool messages
    model_calls: list[ModelCall]  # one per call of agent or final, in order
    application_queries: list[ApplicationQuery] = field(default_factory=list)

    def model_input(self, call: ModelCall) -> list[BaseMessage]:
        """The messages a model call was given (the tools and settings are stored once)."""
        if call.node == "agent":
            return [SystemMessage(self.system_prompt), *self.messages[: call.input_count]]
        messages = _final_messages(self.messages[: call.input_count], call.context)
        return [
            SystemMessage(_final_prompt(self.final_prompt, call.context)),
            *messages,
            HumanMessage(self.final_instruction),
        ]

    def to_dict(self) -> dict:
        """The trace as JSON-compatible data; messages in LangChain's message format."""
        return {
            **{field.name: getattr(self, field.name) for field in fields(self)},
            "messages": messages_to_dict(self.messages),
            "model_calls": [asdict(call) for call in self.model_calls],
            "application_queries": [asdict(query) for query in self.application_queries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunTrace":
        """Read back a trace written by `to_dict`."""
        values = {
            **data,
            "messages": messages_from_dict(data["messages"]),
            "model_calls": [ModelCall(**call) for call in data["model_calls"]],
            "application_queries": [
                _application_query(query) for query in data.get("application_queries", [])
            ],
        }
        return cls(**values)


@dataclass(frozen=True)
class ChatAnswer:
    """The answer to the last question, with its tool calls and the thread's trace."""

    text: str  # the text of the last model message; "" when the model returned none
    tool_calls: list[ToolCall]  # the view the metrics use
    trace: RunTrace  # the full evidence, stored by the evaluation
    application_queries: list[ApplicationQuery] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        """Whether the model gave text; False: the run ended without an answer."""
        return bool(self.text.strip())

    @property
    def finish_reason(self) -> str | None:
        """The finish reason of the last model message, read from the trace."""
        return self.trace.messages[-1].response_metadata.get("finish_reason")


@dataclass(frozen=True)
class ChatEvent:
    """Something to show while a question runs."""

    kind: Literal["text", "tool_start", "tool_end"]
    text: str  # a piece of model text, or the tool's name


class ChatState(MessagesState):
    model_calls: Annotated[list[ModelCall], operator.add]
    application_queries: Annotated[list[ApplicationQuery], operator.add]
    tool_rounds: int  # tool rounds of the current question
    final_evidence: str | None
    profile_answer: str | None


def _normalized_value(value: object) -> str:
    """Normalize a question entity or database value for conservative exact matching."""
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def _previous_turn(messages: list[BaseMessage]) -> tuple[str, list[BaseMessage]] | None:
    """Return the immediately preceding question and its messages, if both turns exist."""
    questions = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(questions) < 2:
        return None
    previous, current = questions[-2:]
    return messages[previous].text, messages[previous + 1 : current]


def _profile_attribute_request(messages: list[BaseMessage]) -> tuple[str, str, str] | None:
    """Return entity, database column and label for a simple attribute follow-up."""
    current = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)), None
    )
    if current is None:
        return None
    entity_match = ENTITY_FOLLOW_UP.fullmatch(current.text)
    previous = _previous_turn(messages)
    if entity_match is None or previous is None:
        return None
    attribute_match = ATTRIBUTE_QUESTION.fullmatch(previous[0])
    if attribute_match is None:
        return None
    attribute = _normalized_value(attribute_match.group(1)).removesuffix(" profile")
    configured = PROFILE_ATTRIBUTES.get(attribute)
    if configured is None:
        return None
    entity = entity_match.group(1).strip()
    return (entity, *configured)


def _format_profile_answer(
    entity: str,
    attribute_label: str,
    profiles: list[tuple[object, object]],
    source: object,
    edition: object,
) -> str:
    """Format verified profile fields without adding model-generated descriptions."""
    recorded_edition = "not recorded" if str(edition) == "0" else str(edition)
    lines = [f"### {entity} [DB: datasheets]", ""]
    lines.extend(
        f"- {model}: {attribute_label} {value}. [DB: datasheets_models]"
        for model, value in profiles
    )
    lines.extend(
        [
            "",
            f"- Recorded publication: {source}. [DB: source]",
            f"- Recorded edition: {recorded_edition}. [DB: source]",
        ]
    )
    return "\n".join(lines)


def _profile_follow_up_answer(
    messages: list[BaseMessage], sql_retriever: SqlRetriever
) -> tuple[str | None, ApplicationQuery | None]:
    """Answer a simple profile follow-up from a fresh complete local lookup.

    The preceding turn supplies only the requested attribute. The entity value is bound as a SQL
    parameter, and the attribute column comes from ``PROFILE_ATTRIBUTES``. Ambiguous datasheets,
    incomplete fields, errors and truncated results leave the question to normal agent routing.
    """
    request = _profile_attribute_request(messages)
    if request is None:
        return None, None
    requested_entity, attribute_column, attribute_label = request
    sql = PROFILE_LOOKUP_SQL.format(attribute=attribute_column)
    parameters = (requested_entity,)
    result = sql_retriever.run(sql, parameters)
    query = ApplicationQuery(
        "application",
        "profile_lookup",
        max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage)),
        sql,
        parameters,
        result,
    )
    if result.error is not None or result.truncated or not result.rows:
        return None, query
    columns = {column.casefold(): index for index, column in enumerate(result.columns)}
    required = {
        "datasheet_id",
        "datasheet",
        "model_line",
        "model",
        "attribute_value",
        "source",
        "edition",
    }
    if len(columns) != len(result.columns) or not required <= columns.keys():
        return None, query
    required_indexes = [columns[name] for name in required]
    if any(any(row[index] in (None, "") for index in required_indexes) for row in result.rows):
        return None, query
    datasheet_ids = {row[columns["datasheet_id"]] for row in result.rows}
    datasheets = {_normalized_value(row[columns["datasheet"]]) for row in result.rows}
    metadata = {(row[columns["source"]], row[columns["edition"]]) for row in result.rows}
    if (
        len(datasheet_ids) != 1
        or datasheets != {_normalized_value(requested_entity)}
        or len(metadata) != 1
    ):
        return None, query
    profiles = list(
        dict.fromkeys(
            (row[columns["model"]], row[columns["attribute_value"]]) for row in result.rows
        )
    )
    source, edition = metadata.pop()
    entity = str(result.rows[0][columns["datasheet"]])
    return _format_profile_answer(entity, attribute_label, profiles, source, edition), query


def _named_rule_subject(question: str) -> str | None:
    """Return the sole subject of a direct named-rule question, if that is all it asks."""
    match = NAMED_RULE_QUESTION.fullmatch(question)
    if match is None:
        return None
    subject = next(group for group in match.groups() if group is not None).strip()
    subject = re.sub(r"^the\s+", "", subject, flags=re.IGNORECASE)
    return re.sub(r"\s+stratagems?$", "", subject, flags=re.IGNORECASE).strip()


def _same_rule_name(subject: str, name: str) -> bool:
    """Match an exact rule name while allowing only a final plural ``s`` in the question."""
    subject_name = _normalized_value(subject)
    record_name = _normalized_value(name)
    return bool(subject_name) and (
        subject_name == record_name
        or subject_name == f"{record_name}s"
        or record_name == f"{subject_name}s"
    )


def _complete_stratagem_text(text: str) -> bool:
    """Whether text contains the required fields of a Stratagem card."""
    upper = text.upper()
    return STRATAGEM_COST.search(text) is not None and all(
        field in upper for field in STRATAGEM_CARD_FIELDS
    )


def _sufficient_stratagem_evidence(messages: list[BaseMessage]) -> str | None:
    """Build final context for one complete, exact named-Stratagem result.

    The question must request only one named rule. The current turn must contain one matching
    rulebook card and one exact core database row with the complete card fields. Any ambiguity,
    missing field, error or truncation leaves the decision with the normal agent route.
    """
    start = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    question = messages[start].text
    subject = _named_rule_subject(question)
    if subject is None:
        return None
    calls = {
        call["id"]: call["name"]
        for message in messages[start + 1 :]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    candidates = []
    for message in messages[start + 1 :]:
        if (
            not isinstance(message, ToolMessage)
            or calls.get(message.tool_call_id) != "search_rules"
        ):
            continue
        result = _tool_result("search_rules", message)
        if not isinstance(result, RulesSearchResult) or len(result.stratagem_names) != 1:
            continue
        name = result.stratagem_names[0]
        if not _same_rule_name(subject, name):
            continue
        cards = [
            item.chunk
            for item in result.chunks
            if _normalized_value(item.chunk.section) == _normalized_value(name)
            and any(
                "STRATAGEMS" in {label.strip() for label in footer.split("|")}
                for footer in item.chunk.page_footers
            )
        ]
        if len(cards) != 1 or not _complete_stratagem_text(cards[0].text):
            continue
        database = result.stratagems
        if (
            database is None
            or database.error is not None
            or database.truncated
            or len(database.rows) != 1
        ):
            continue
        lowered = [column.casefold() for column in database.columns]
        if len(lowered) != len(set(lowered)) or not STRATAGEM_COLUMNS <= set(lowered):
            continue
        row = dict(zip(lowered, database.rows[0], strict=True))
        required = ("id", "name", "type", "cpcost", "turn", "phase", "description")
        if any(row[field] in (None, "") for field in required):
            continue
        if row["factionid"] not in (None, "") or not _same_rule_name(name, str(row["name"])):
            continue
        description = str(row["description"])
        if not _complete_stratagem_text(f"{row['cpcost']}CP\n{description}"):
            continue
        card_cost = STRATAGEM_COST.search(cards[0].text)
        database_cost = re.fullmatch(r"\s*(\d+)\s*(?:CP)?\s*", str(row["cpcost"]), re.IGNORECASE)
        if database_cost is None or database_cost.group(1) != card_cost.group(1):
            continue
        pages = cards[0].pages
        citation = str(pages[0]) if len(pages) == 1 else f"{pages[0]}–{pages[-1]}"
        candidates.append(
            {
                "question": question,
                "identity": name,
                "rulebook": {
                    "citation": f"[Core Rules p. {citation}]",
                    "section": cards[0].section,
                    "text": cards[0].text,
                },
                "database": {
                    "citation": "[DB: stratagems]",
                    "publication_and_edition": "not recorded",
                    "columns": database.columns,
                    "row": list(database.rows[0]),
                },
            }
        )
    if len(candidates) != 1:
        return None
    evidence = json.dumps(candidates[0], ensure_ascii=False)
    return (
        "## Complete named Stratagem evidence\n\n"
        "The application verified that the current question asks only for one named Stratagem, "
        "and that the evidence below contains one complete matching Core Rules card and one "
        "complete, exact, error-free and untruncated core database record.\n\n"
        f"{evidence}\n\n"
        "Answer from these two sources only. Attribute and compare them separately, preserving "
        "their cost, timing, target, effect and any stated restrictions. The Core Rules citation "
        "and database citation are supplied above; the database publication and edition are not "
        "recorded. Do not add classifications, relationships or other rules that are absent from "
        "this evidence."
    )


def _final_prompt(prompt: str, context: str | None) -> str:
    """Append deterministic evidence to the final prompt when its completeness was verified."""
    return prompt if context is None else f"{prompt}\n\n{context}"


def _final_messages(messages: list[BaseMessage], context: str | None) -> list[BaseMessage]:
    """Use only the current question when verified evidence replaces conversation prose."""
    if context is None:
        return messages
    return [next(message for message in reversed(messages) if isinstance(message, HumanMessage))]


def mission_scopes(chunk: Chunk) -> list[str]:
    """Explicit mission labels with their pages; ordinary section labels establish no scope."""
    scopes = []
    footers = chunk.page_footers or [""] * len(chunk.pages)
    for page, footer in zip(chunk.pages, footers, strict=True):
        for label in footer.split("|"):
            kind, _, name = label.strip().partition(":")
            if kind == "MISSION" and name.strip():
                scopes.append(f"{name.strip()} (Core Rules p. {page})")
    return list(dict.fromkeys(scopes))


def chunk_block(chunk: Chunk) -> str:
    """A chunk as the model reads it: its page citation, printed footers and section text.

    Internal IDs stay in the tool artifact; the text supplies the citation the answer should use.
    """
    pages = str(chunk.pages[0])
    if len(chunk.pages) > 1:
        pages += f"–{chunk.pages[-1]}"
    recorded = chunk.page_footers or [""] * len(chunk.pages)
    footers = "; ".join(
        f"page footer p. {page}: {footer}"
        for page, footer in zip(chunk.pages, recorded, strict=True)
        if footer
    )
    printed = f" ({footers})" if footers else ""
    scopes = mission_scopes(chunk)
    scope = f"Mission scope: {'; '.join(scopes)}\n" if scopes else ""
    return f"[Core Rules p. {pages}]{printed}\n{scope}{chunk.text}"


def build_tools(rules_retriever: RulesRetriever, sql_retriever: SqlRetriever) -> list[BaseTool]:
    """The two tools. Each gives the model a text and keeps its result as JSON-compatible data on
    the tool message, which the checkpointer stores as it is; `ChatAgent.last_answer` rebuilds
    the dataclasses from it."""

    sql_cache: dict[str, dict] = {}

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_rules(query: str) -> tuple[str, dict]:
        """Search the Core Rules: how the game works (phases, actions, keywords, terrain).

        Retrieved Stratagem sections also include exact-name core database records when found.

        Args:
            query: In English, with the rulebook's terms, whatever the question's language.
        """
        found = rules_retriever.search(query, TOP_K)
        required = list(
            dict.fromkeys(
                section
                for item in found[:1]
                for section in RELATED_RULE_SECTIONS.get(item.chunk.section.strip().upper(), [])
            )
        )
        related = rules_retriever.sections(required)
        known = {item.chunk.chunk_id for item in found}
        added = [item for item in related if item.chunk.chunk_id not in known]
        all_found = [*found, *added]
        names = []
        for item in all_found:
            labels = {
                label.strip() for footer in item.chunk.page_footers for label in footer.split("|")
            }
            name = item.chunk.section.strip().upper()
            if "STRATAGEMS" in labels and name and name not in names:
                names.append(name)
        stratagems = None
        blocks = [chunk_block(item.chunk) for item in found]
        blocks.extend(
            f"[Required related rule for {', '.join(required)}]\n{chunk_block(item.chunk)}"
            for item in added
        )
        if names:
            placeholders = ", ".join("?" for _ in names)
            stratagems = sql_retriever.run(
                "SELECT id, name, FactionId, type, cpCost, turn, phase, detachment, description "
                "FROM stratagems WHERE name COLLATE NOCASE IN (" + placeholders + ") "
                "AND FactionId = '' ORDER BY name, id",
                names,
            )
            blocks.append(
                "[DB: stratagems] Core records; publication/edition not recorded.\n"
                f"Exact-name candidates: {json.dumps(names, ensure_ascii=False)}\n"
                + json.dumps(asdict(stratagems), ensure_ascii=False)
            )
        result = RulesSearchResult(
            all_found, names, stratagems, [item.chunk.section for item in related]
        )
        blocks.append(APPLICABILITY_RULES)
        return "\n\n".join(blocks), asdict(result)

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def run_sql(sql: str) -> tuple[str, dict]:
        """Query units, weapons, abilities, stratagems, detachments and factions with read-only SQL.

        Args:
            sql: One SQLite SELECT or WITH query that follows the database schema.
        """
        cache_key = sql
        result = sql_cache.get(cache_key)
        if result is None:
            result = asdict(sql_retriever.run(sql))
            result["rows"] = [
                list(row) for row in result["rows"]
            ]  # as a checkpoint reads them back
            sql_cache[cache_key] = result
        return json.dumps(result, ensure_ascii=False), result

    return [search_rules, run_sql]


def _sql_result(data: dict) -> SqlResult:
    return SqlResult(
        data["columns"], [tuple(row) for row in data["rows"]], data["truncated"], data["error"]
    )


def _application_query(data: ApplicationQuery | dict) -> ApplicationQuery:
    """Rebuild an application query from its JSON-compatible trace form."""
    if isinstance(data, ApplicationQuery):
        result = data.result
        return ApplicationQuery(
            origin=data.origin,
            operation=data.operation,
            question_index=data.question_index,
            sql=data.sql,
            parameters=tuple(data.parameters),
            result=SqlResult(
                list(result.columns),
                [tuple(row) for row in result.rows],
                result.truncated,
                result.error,
            ),
        )
    return ApplicationQuery(
        origin=data["origin"],
        operation=data["operation"],
        question_index=data["question_index"],
        sql=data["sql"],
        parameters=tuple(data["parameters"]),
        result=_sql_result(data["result"]),
    )


def _tool_result(name: str, message: ToolMessage) -> RulesSearchResult | SqlResult | None:
    """The structured result kept on a tool message; None when the tool itself failed."""
    if message.status == "error":
        return None
    if name == "search_rules":
        data = message.artifact
        # Historical searches stored just the list, without an associated database query.
        if isinstance(data, list):
            data = {"chunks": data, "stratagem_names": [], "stratagems": None}
        return RulesSearchResult(
            [RetrievedChunk(Chunk(**item["chunk"]), item["score"]) for item in data["chunks"]],
            data["stratagem_names"],
            _sql_result(data["stratagems"]) if data["stratagems"] is not None else None,
            data.get("related_sections", []),
        )
    return _sql_result(message.artifact)


def _sql_notes(result: SqlResult, core_stratagems: bool = False) -> list[str]:
    if result.error:
        return [f"Database query failed: {result.error}"]
    notes = []
    if result.truncated:
        notes.append(
            "Database result is incomplete; returned rows do not establish an exhaustive list."
        )
    if not result.rows:
        notes.append(
            "Database query returned no matching rows; "
            "this does not establish that a rule is absent."
        )
        return notes
    if core_stratagems:
        return notes + ["DB stratagems: publication and edition are not recorded."]
    columns = {name.lower(): index for index, name in enumerate(result.columns)}
    publication_columns = [name for name in ("source", "publication") if name in columns]
    if len(publication_columns) > 1:
        return notes + [
            "Application could not identify a unique publication field in the returned columns."
        ]
    if len(publication_columns) != 1 or "edition" not in columns:
        return notes + [
            "Application could not identify publication/edition fields in the returned columns."
        ]
    publication_column = publication_columns[0]
    for row in result.rows:
        publication, edition = row[columns[publication_column]], row[columns["edition"]]
        record = f" for {row[columns['name']]}" if "name" in columns else ""
        if "id" in columns:
            record += f" (ID {row[columns['id']]})"
        if not publication:
            notes.append(f"Returned database record{record} has no publication value.")
        elif not edition or str(edition) == "0":
            notes.append(f"Database publication{record}: {publication}; edition not recorded.")
        else:
            notes.append(f"Database publication{record}: {publication}; edition {edition}.")
    return list(dict.fromkeys(notes))


def database_statistics(answer: ChatAnswer) -> list[SqlResult]:
    """Return current-turn SQL results that use the category-statistics output contract.

    The contract is deliberately narrow: rows must identify their publication, edition and
    Toughness and contain SQL-computed model-profile and datasheet counts. Values are returned
    unchanged; this function does not combine publication groups or calculate totals.

    Args:
        answer: The answer whose current-turn tool results are inspected.

    Returns:
        Non-error SQL results with every required statistics column.
    """
    required = {"source", "edition", "t", "model_profiles", "datasheets"}
    return [
        call.result
        for call in answer.tool_calls
        if isinstance(call.result, SqlResult)
        and call.result.error is None
        and bool(call.result.rows)
        and required <= {column.lower() for column in call.result.columns}
    ]


def source_notes(answer: ChatAnswer) -> list[str]:
    """Source scope and completeness from this thread, separate from the model's claims.

    Args:
        answer: The original model answer and its full conversation trace.

    Returns:
        Deduplicated notes labelled as current-turn or earlier-conversation evidence.
    """
    messages = answer.trace.messages
    start = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    names = {}
    notes = []
    for query in answer.trace.application_queries:
        period = "This turn" if query.question_index == start else "Earlier in this conversation"
        notes.extend(
            f"{period}: Application profile lookup: {note}" for note in _sql_notes(query.result)
        )
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            names.update({call["id"]: call["name"] for call in message.tool_calls})
        if not isinstance(message, ToolMessage):
            continue
        name = names.get(message.tool_call_id)
        if name not in {"search_rules", "run_sql"}:
            continue
        result = _tool_result(name, message)
        evidence = []
        if isinstance(result, RulesSearchResult):
            evidence.extend(
                f"Retrieved mission-specific evidence: {scope}."
                for item in result.chunks
                for scope in mission_scopes(item.chunk)
            )
            if result.stratagems is not None:
                evidence.extend(_sql_notes(result.stratagems, core_stratagems=True))
        elif isinstance(result, SqlResult):
            evidence.extend(_sql_notes(result))
        else:
            evidence.append(f"{name} failed: {message.text}")
        period = "This turn" if index > start else "Earlier in this conversation"
        notes.extend(f"{period}: {note}" for note in evidence)
    return list(dict.fromkeys(notes))


def answer_from_trace(trace: RunTrace) -> ChatAnswer:
    """The answer to the trace's last question: the text of its last message, and each tool call
    of that question paired with its tool message by call ID. A stored trace gives back the same
    answer as the run that wrote it, so the evaluation can score it again."""
    messages = trace.messages
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    question = messages[start:]
    responses = {m.tool_call_id: m for m in question if isinstance(m, ToolMessage)}
    tool_calls = [
        ToolCall(
            call["name"],
            call["args"],
            responses[call["id"]].text,
            _tool_result(call["name"], responses[call["id"]]),
        )
        for m in question
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    application_queries = [
        query for query in trace.application_queries if query.question_index == start
    ]
    return ChatAnswer(messages[-1].text, tool_calls, trace, application_queries)


class ChatAgent:
    """A conversation with the chat model over the rules and the database."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        rules_retriever: RulesRetriever,
        sql_retriever: SqlRetriever,
        system_prompt: str,
    ):
        """Build the graph.

        Args:
            chat_model: The chat model (`create_chat_model`); the tools are bound here.
            rules_retriever: Searches the rule chunks.
            sql_retriever: Runs queries on the clean database.
            system_prompt: `build_system_prompt(schema)`.
        """
        tools = build_tools(rules_retriever, sql_retriever)
        self._system_prompt = system_prompt
        # LangChain's identifying parameters of a model: its name and generation settings.
        self._model_settings = dict(chat_model._identifying_params)
        self._tool_declarations = [convert_to_openai_tool(t) for t in tools]
        self._graph = self._build_graph(chat_model, tools, sql_retriever)
        self._thread = str(uuid.uuid4())

    def _build_graph(
        self,
        chat_model: BaseChatModel,
        tools: list[BaseTool],
        sql_retriever: SqlRetriever,
    ):
        agent_model = chat_model.bind_tools(tools, tool_choice=AGENT_TOOL_CONFIG)
        final_model = chat_model.bind_tools(tools, tool_choice="none")

        def handle_tool_error(error: Exception) -> str:
            if isinstance(error, BudgetExhausted) or type(error).__module__.startswith(
                ("google.", "httpx", "langchain_google_genai")
            ):
                raise error
            return f"Error: {error!r}\n Please fix your mistakes."

        tool_node = ToolNode(tools, handle_tool_errors=handle_tool_error)

        def prepare(state: ChatState) -> dict:
            profile_answer, query = _profile_follow_up_answer(state["messages"], sql_retriever)
            return {
                "final_evidence": None,
                "profile_answer": profile_answer,
                "application_queries": [] if query is None else [query],
            }

        def after_prepare(state: ChatState) -> str:
            return "final" if state["profile_answer"] is not None else "agent"

        def agent(state: ChatState) -> dict:
            messages = state["messages"]
            reply = agent_model.invoke([SystemMessage(self._system_prompt), *messages])
            return {
                "messages": [reply],
                "model_calls": [ModelCall("agent", len(messages), AGENT_TOOL_CHOICE)],
            }

        def run_tools(state: ChatState) -> dict:
            result = tool_node.invoke(state)
            messages = [*state["messages"], *result["messages"]]
            return {
                "messages": result["messages"],
                "tool_rounds": state["tool_rounds"] + 1,
                "final_evidence": _sufficient_stratagem_evidence(messages),
            }

        def final(state: ChatState) -> dict:
            messages = state["messages"]
            context = state["final_evidence"]
            structured_answer = state["profile_answer"]
            if structured_answer is not None:
                return {
                    "messages": [
                        AIMessage(
                            structured_answer,
                            additional_kwargs={"answer_format": STRUCTURED_DATABASE_ANSWER},
                            response_metadata={"finish_reason": "STOP"},
                        )
                    ]
                }
            request = [
                SystemMessage(_final_prompt(FINAL_PROMPT, context)),
                *_final_messages(messages, context),
                HumanMessage(FINAL_INSTRUCTION),
            ]
            reply = final_model.invoke(request)  # stored as returned, an empty reply included
            return {
                "messages": [reply],
                "model_calls": [ModelCall("final", len(messages), "none", context)],
            }

        def after_tools(state: ChatState) -> str:
            if state["final_evidence"] is not None:
                return "final"
            messages = state["messages"]
            start = max(
                i for i, message in enumerate(messages) if isinstance(message, HumanMessage)
            )
            sql_calls = sum(
                call["name"] == "run_sql"
                for message in messages[start:]
                if isinstance(message, AIMessage)
                for call in message.tool_calls
            )
            if sql_calls >= MAX_SQL_CALLS_PER_QUESTION:
                return "final"
            return "agent" if state["tool_rounds"] < MAX_TOOL_ROUNDS else "final"

        graph = StateGraph(ChatState)
        graph.add_node("prepare", prepare)
        graph.add_node("agent", agent)
        graph.add_node("tools", run_tools)
        graph.add_node("final", final)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges("prepare", after_prepare, {"agent": "agent", "final": "final"})
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "final": "final"})
        graph.add_edge("final", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=[ModelCall, ApplicationQuery, SqlResult])
        return graph.compile(checkpointer=InMemorySaver(serde=serde))

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self._thread}, "recursion_limit": RECURSION_LIMIT}

    def stream(self, question: str) -> Iterator[ChatEvent]:
        """Ask a question on the current thread, yielding what to show while it runs.

        A model chunk's text becomes a `text` event, the first chunk of a tool call (the one
        that names the tool) a `tool_start`, a tool message a `tool_end`. Thinking parts and chunks
        without text give no event; the graph still stores them in the message.
        """
        structured_answer_streamed = False
        for chunk, metadata in self._graph.stream(
            {
                "messages": [HumanMessage(question)],
                "application_queries": [],
                "tool_rounds": 0,
                "final_evidence": None,
                "profile_answer": None,
            },
            self._config(),
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                yield ChatEvent("tool_end", chunk.name)
            elif isinstance(chunk, AIMessageChunk) and metadata["langgraph_node"] in MODEL_NODES:
                if (
                    chunk.additional_kwargs.get("answer_format") == STRUCTURED_DATABASE_ANSWER
                    and chunk.text
                ):
                    structured_answer_streamed = True
                if chunk.text:
                    yield ChatEvent("text", chunk.text)
                for call in chunk.tool_call_chunks:
                    if call.get("name"):
                        yield ChatEvent("tool_start", call["name"])
        last = self._graph.get_state(self._config()).values["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and last.additional_kwargs.get("answer_format") == STRUCTURED_DATABASE_ANSWER
            and not structured_answer_streamed
        ):
            yield ChatEvent("text", last.text)

    def trace(self) -> RunTrace:
        """The thread's trace as the graph's state holds it, also when a run stopped midway."""
        values = self._graph.get_state(self._config()).values
        return RunTrace(
            system_prompt=self._system_prompt,
            final_prompt=FINAL_PROMPT,
            final_instruction=FINAL_INSTRUCTION,
            model_settings=self._model_settings,
            tools=self._tool_declarations,
            messages=list(values.get("messages", [])),
            model_calls=list(values.get("model_calls", [])),
            application_queries=[
                _application_query(query) for query in values.get("application_queries", [])
            ],
        )

    def last_answer(self) -> ChatAnswer:
        """The answer to the last question of the thread, read from the graph's final state."""
        return answer_from_trace(self.trace())

    def ask(self, question: str) -> ChatAnswer:
        """Ask a question and return its answer: `stream()` to the end, then `last_answer()`."""
        for _ in self.stream(question):
            pass
        return self.last_answer()

    def reset(self) -> None:
        """Discard this conversation's checkpoints and start a new thread."""
        self._graph.checkpointer.delete_thread(self._thread)
        self._thread = str(uuid.uuid4())
