"""Command-line entry point: chat about the rules and the datasheets, or answer one question.

Usage:
    uv run python -m src.scripts.chat
    uv run python -m src.scripts.chat --question "How do grenades work?"

Interactive commands: /sources (the tool calls of the last answer), /reset (a new conversation),
/quit. The answer is streamed, then redrawn with separate source notes (plan §6.8, R79); with
`--question`, standard output receives the final answer and source notes once, without status lines.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, TextIO

from prompt_toolkit import PromptSession
from pydantic import ValidationError
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from src.rag.chatbot.agent import (
    ChatAgent,
    ChatAnswer,
    RulesSearchResult,
    database_statistics,
    source_notes,
)
from src.rag.chatbot.prompts import build_system_prompt
from src.rag.chunker.markdown_chunker import jsonl_to_chunks
from src.rag.config import (
    CLEAN_DB,
    RULES_CHUNKS,
    RULES_EMBEDDINGS,
    SQL_MAX_ROWS,
    SQL_SCHEMA,
    Settings,
    key_not_set,
)
from src.rag.llm.gemini import REQUEST_ERRORS, GeminiEmbedder, create_chat_model
from src.rag.retriever.rules_retriever import RulesRetriever
from src.rag.retriever.sql_retriever import SqlRetriever, load_entity_names
from src.rag.storage.rules_index import load_embeddings

STATUS = {"search_rules": "searching the rules…", "run_sql": "querying the database…"}
STATUS_STYLE = "dim italic"


def tool_status(name: str) -> str:
    """The status line while a tool runs; a name the model made up gets a generic one, and the
    graph answers the call with an error."""
    return STATUS.get(name, f"calling {name}…")


def load_settings(parser: argparse.ArgumentParser) -> Settings:
    """The runtime settings, or stop with one line if the API key is not set."""
    try:
        return Settings()
    except ValidationError as error:
        if not key_not_set(error):
            raise
        parser.error("GOOGLE_API_KEY is not set. See README.md, section API key.")


def build_agent(
    parser: argparse.ArgumentParser,
    chat_client_args: dict[str, Any] | None = None,
    embed_client_args: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> ChatAgent:
    """Load what the chat needs, or stop with one line naming the step or setting that fixes it.

    Args:
        parser: The script's parser, which reports a missing file or key and exits.
        chat_client_args: Extra `httpx` client arguments for the chat model, such as a transport
            that counts the requests (the online acceptance uses it).
        embed_client_args: The same for the embedder.
        settings: Runtime settings already loaded by a caller that must also record them. When
            omitted, this function loads them itself.
    """
    for path, step in [
        (RULES_CHUNKS, "src.scripts.chunk (README.md, Run, step 2)"),
        (CLEAN_DB, "src.scripts.prepare_db (README.md, Run, step 3)"),
        (RULES_EMBEDDINGS, "src.scripts.build_index (README.md, Run, step 4)"),
    ]:
        if not path.is_file():
            parser.error(f"{path} not found. Run {step}.")
    settings = settings or load_settings(parser)
    return ChatAgent(
        create_chat_model(settings, chat_client_args),
        load_rules_retriever(parser, settings, embed_client_args),
        SqlRetriever(CLEAN_DB, SQL_MAX_ROWS),
        build_system_prompt(SQL_SCHEMA.read_text(encoding="utf-8")),
    )


def load_rules_retriever(
    parser: argparse.ArgumentParser,
    settings: Settings,
    embed_client_args: dict[str, Any] | None = None,
) -> RulesRetriever:
    """The rules retriever over the saved chunks and index, or stop if the index was built from
    other chunks or another model.

    Args:
        parser: The script's parser, which reports a stale index and exits.
        settings: The runtime settings: API key and embedding model.
        embed_client_args: Extra `httpx` client arguments for the embedder, such as a transport
            that counts the requests.
    """
    chunks = jsonl_to_chunks(RULES_CHUNKS.read_text(encoding="utf-8"))
    embedder = GeminiEmbedder(settings, embed_client_args)
    try:
        vectors = load_embeddings(RULES_EMBEDDINGS, chunks, embedder.model)
    except ValueError as error:
        parser.error(f"{error} (README.md, Run, step 4).")
    return RulesRetriever(chunks, vectors, embedder, load_entity_names(CLEAN_DB))


def no_answer_line(answer: ChatAnswer) -> str:
    """The CLI's own line for a run that ended without text: a status, not an answer."""
    return f"No answer: the model returned no text (finish reason {answer.finish_reason})."


def displayed_answer(answer: ChatAnswer) -> str:
    """The model answer followed by direct SQL statistics and application source notes."""
    sections = [answer.text]
    for result in database_statistics(answer):
        indexes = {name.lower(): index for index, name in enumerate(result.columns)}
        columns = ["source", "edition", "t", "model_profiles", "datasheets"]
        headings = ["Publication", "Edition", "Toughness", "Model profiles", "Datasheets"]

        def cell(value: object) -> str:
            return ("" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")

        rows = [[cell(row[indexes[column]]) for column in columns] for row in result.rows]
        table = [
            "| " + " | ".join(headings) + " |",
            "| " + " | ".join("---" for _ in headings) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
        sections.append(
            "Database statistics (authoritative SQL rows; the model answer above is unchanged). "
            "Rows remain grouped by publication, edition and Toughness; the application does not "
            "calculate cross-publication totals.\n\n" + "\n".join(table)
        )
    notes = source_notes(answer)
    if notes:
        sections.append(
            "Source notes (retrieved evidence, not verification of the answer):\n\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return "\n\n".join(sections)


def answer_live(agent: ChatAgent, question: str, console: Console) -> ChatAnswer:
    """Show the run in a live area as it streams, then replace the area with the final answer.

    The streamed text may hold words written before a tool call; the area is cleared when the run
    ends, and what stays on screen is the model's answer with separate application source notes.
    """
    streamed, status = "", ""
    with Live(console=console, auto_refresh=False, transient=True) as live:
        for event in agent.stream(question):
            if event.kind == "text":
                streamed += event.text
            elif event.kind == "tool_start":
                status = tool_status(event.text)
            else:
                status = ""
            live.update(Group(Text(streamed), Text(status, style=STATUS_STYLE)), refresh=True)
    answer = agent.last_answer()
    if answer.answered:
        console.print(Markdown(displayed_answer(answer)))
    else:
        console.print(Text(no_answer_line(answer), style=STATUS_STYLE))
    return answer


def answer_once(agent: ChatAgent, question: str, out: TextIO, err: Console) -> int:
    """Answer `--question`: status lines to `err` while it runs, then the final answer to `out`.

    Returns:
        The exit code: 0 for an answer, 1 for no answer or a request failure, 130 for interruption.
        Without an answer, `out` stays empty.
    """
    try:
        for event in agent.stream(question):
            if event.kind == "tool_start":
                err.print(Text(tool_status(event.text), style=STATUS_STYLE))
    except (*REQUEST_ERRORS, KeyboardInterrupt) as error:
        agent.reset()
        message = (
            "Interrupted"
            if isinstance(error, KeyboardInterrupt)
            else f"Model request failed ({type(error).__name__})"
        )
        err.print(Text(f"{message}. Conversation cleared; try again.", style=STATUS_STYLE))
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    answer = agent.last_answer()
    if not answer.answered:
        err.print(Text(no_answer_line(answer), style=STATUS_STYLE))
        return 1
    out.write(displayed_answer(answer) + "\n")
    return 0


def sources_text(answer: ChatAnswer | None) -> str:
    """The model tool calls and application queries used for the last answer."""
    if answer is None:
        return "No question asked yet."
    if not answer.tool_calls and not answer.application_queries:
        return "No tool was called for the last answer."
    lines = []
    number = 1
    for query in answer.application_queries:
        lines.append(f"{number}. application {query.operation}: {query.sql}")
        lines.append(f"   parameters: {json.dumps(query.parameters, ensure_ascii=False)}")
        lines.append(f"   result: {json.dumps(asdict(query.result), ensure_ascii=False)}")
        number += 1
    for call in answer.tool_calls:
        lines.append(f"{number}. {call.name}: {' '.join(map(str, call.arguments.values()))}")
        if call.error:
            lines.append(f"   error: {call.error}")
        if call.pages:
            lines.append(f"   pages: {', '.join(map(str, call.pages))}")
        if isinstance(call.result, RulesSearchResult) and call.result.stratagems is not None:
            lines.append(f"   core stratagem lookup: {', '.join(call.result.stratagem_names)}")
            lines.append(f"   database rows returned: {len(call.result.stratagems.rows)}")
        if isinstance(call.result, RulesSearchResult) and call.result.related_sections:
            lines.append(f"   required related sections: {', '.join(call.result.related_sections)}")
        number += 1
    return "\n".join(lines)


def chat_loop(agent: ChatAgent, console: Console, read_line: Callable[[], str]) -> None:
    """Read questions and commands until /quit or the end of input."""
    last: ChatAnswer | None = None
    while True:
        try:
            line = read_line().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if line == "/quit":
            return
        if line == "/reset":
            agent.reset()
            last = None
            console.print(Text("New conversation.", style=STATUS_STYLE))
        elif line == "/sources":
            console.print(Text(sources_text(last)))
        elif line:
            try:
                last = answer_live(agent, line, console)
            except (*REQUEST_ERRORS, KeyboardInterrupt) as error:
                agent.reset()
                last = None
                message = (
                    "Interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else f"Model request failed ({type(error).__name__})"
                )
                console.print(Text(f"{message}. New conversation; try again.", style=STATUS_STYLE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat about the Warhammer 40,000 rules")
    parser.add_argument(
        "--question", help="answer this question and exit; standard output gets the answer only"
    )
    args = parser.parse_args()
    agent = build_agent(parser)

    if args.question is not None:
        sys.stdout.reconfigure(encoding="utf-8")  # a redirected answer is UTF-8 on every system
        sys.exit(answer_once(agent, args.question, sys.stdout, Console(stderr=True)))
    console = Console()
    console.print(
        Text("Ask about the rules or the datasheets. /sources /reset /quit", style=STATUS_STYLE)
    )
    session = PromptSession()
    chat_loop(agent, console, lambda: session.prompt("> "))


if __name__ == "__main__":
    main()
