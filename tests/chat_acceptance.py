"""Online acceptance of the chat: one reference question through the CLI's own code (plan R45, R46).

Run only on purpose, one question per run, with the most chat attempts it may send:

    uv run python -m tests.chat_acceptance toughness --budget 8           # shows what it would do
    uv run python -m tests.chat_acceptance toughness --budget 8 --round R1 --send  # sends

Without `--send` nothing is sent and nothing is written. With it, every chat attempt, retries
included, is counted before it is sent and refused past the budget; every attempt sent, chat or
embedding, is recorded before it goes, with its status or, without a response, its error. The run
is saved, failed or not, in its own folder under `data/checks/acceptance/`: `report.json`
(question, attempts, answer, tool calls with their full results), `trace.json` (also when the run
stopped midway, marked incomplete), and what the chat showed: `console.txt`, a plain file, so the
interactive commands and the final output but not the live area (the terminal tests of
`test_chat.py` cover it), or `stdout.txt` and `stderr.txt` with `--mode question`.
Neither the key nor the request headers are saved.
"""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from rich.console import Console

from src.rag.chatbot.agent import (
    ChatAgent,
    ChatAnswer,
    RunTrace,
    database_statistics,
    source_notes,
)
from src.rag.config import (
    CHAT_ACCEPTANCE_DIR,
    CLEAN_DB,
    LOCK_FILE,
    PROJECT_ROOT,
    RULES_CHUNKS,
    RULES_EMBEDDINGS,
    SOURCE_DIR,
    SQL_SCHEMA,
    TESTSET,
)
from src.rag.llm.budget import AttemptCounter, RequestBudget
from src.rag.llm.gemini import MIN_SECONDS_BETWEEN_CHAT_REQUESTS
from src.rag.storage.evaluation_runs import load_testset
from src.scripts.chat import answer_once, build_agent, chat_loop, load_settings
from src.scripts.outputs import refuse_unsafe_outputs

# Reference questions and focused comparison cases from the acceptance plans.
STATIC_QUESTIONS = {
    "grenades": "What is the rule for grenades?",
    "stratagem": "How can I use a stratagem?",
    "scoring": "What are the different ways to score points?",
    "toughness": "What is the toughness of a space marine?",
    "wardens": (
        "For Wardens of Ultramar, give every model profile's Toughness and the recorded "
        "publication and edition."
    ),
    "unit_rules": (
        "Can an Intercessor Squad use GRENADE, and what are its cost, timing, targets and effect? "
        "Compare the Core Rules with the database where they differ."
    ),
}
FROZEN_QUESTION_IDS = frozenset({"q006", "q010"})
QUESTION_CHOICES = (*STATIC_QUESTIONS, *sorted(FROZEN_QUESTION_IDS))


def _frozen_question(parser: argparse.ArgumentParser, question_id: str) -> str:
    """Return one uniquely identified frozen question, or stop before any request."""
    if not TESTSET.is_file():
        parser.error(f"{TESTSET} not found; the frozen acceptance question cannot be loaded.")
    try:
        questions = load_testset(TESTSET)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"{TESTSET}: {error}")
    matches = [question.question for question in questions if question.question_id == question_id]
    if len(matches) != 1:
        parser.error(f"{TESTSET}: expected one {question_id!r}, found {len(matches)}.")
    return matches[0]


def _question(parser: argparse.ArgumentParser, question_id: str) -> str:
    """Return a static acceptance question or its frozen test-set text."""
    if question_id in FROZEN_QUESTION_IDS:
        return _frozen_question(parser, question_id)
    return STATIC_QUESTIONS[question_id]


def _git_commit() -> str:
    """Return the full commit that the acceptance run executes."""
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_question(agent: ChatAgent, question: str, mode: str, run_dir: Path) -> int | None:
    """Ask once as the CLI does, writing what it shows into `run_dir`.

    Interactive: the question, then `/sources`, then `/quit`, on a console written to
    `console.txt`. Question: `answer_once` with `stdout.txt` and `stderr.txt`.

    Returns:
        The exit code of `--question`; None in interactive mode.
    """
    if mode == "interactive":
        lines = iter([question, "/sources", "/quit"])
        with (run_dir / "console.txt").open("w", encoding="utf-8") as file:
            chat_loop(agent, Console(file=file, width=100, color_system=None), lambda: next(lines))
        return None
    with (
        (run_dir / "stdout.txt").open("w", encoding="utf-8") as out,
        (run_dir / "stderr.txt").open("w", encoding="utf-8") as err,
    ):
        return answer_once(agent, question, out, Console(file=err, width=100, color_system=None))


def answer_record(answer: ChatAnswer) -> dict:
    """The answer as the report keeps it: its text and every tool call with its full result."""
    return {
        "text": answer.text,
        "source_notes": source_notes(answer),
        "database_statistics": [asdict(result) for result in database_statistics(answer)],
        "answered": answer.answered,
        "finish_reason": answer.finish_reason,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "pages": call.pages,
                "error": call.error,
                "output": call.output,
            }
            for call in answer.tool_calls
        ],
        "application_queries": [asdict(query) for query in answer.application_queries],
        "model_calls": [
            [call.node, call.input_count, call.tool_choice] for call in answer.trace.model_calls
        ],
    }


def write_results(
    run_dir: Path, report: dict, answer: ChatAnswer | None, trace: RunTrace | None
) -> None:
    """Save the report, with the answer when the run ended, and the trace the thread holds.

    A run that stopped midway has no answer: its trace, saved all the same, holds what the
    thread reached (the question, tool calls, tool results), and the report says it is incomplete.
    """
    report["complete"] = answer is not None
    if answer is not None:
        report["answer"] = answer_record(answer)
    if trace is not None:
        (run_dir / "trace.json").write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one reference question on the real API")
    parser.add_argument("question", choices=QUESTION_CHOICES)
    parser.add_argument("--mode", choices=["interactive", "question"], default="interactive")
    parser.add_argument(
        "--budget", type=int, required=True, help="the most chat attempts, retries included"
    )
    parser.add_argument("--send", action="store_true", help="send; without it, only show the plan")
    parser.add_argument(
        "--embedding-budget",
        type=int,
        default=18,
        help="maximum embedding attempts, retries included (default: 18)",
    )
    parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="ask another question on the same conversation (repeatable)",
    )
    parser.add_argument(
        "--turn-budget",
        action="append",
        type=int,
        default=[],
        help="chat attempts reserved for each turn; repeat once per turn when using follow-ups",
    )
    parser.add_argument("--output", type=Path, default=CHAT_ACCEPTANCE_DIR)
    parser.add_argument("--round", help="the explicitly authorized acceptance round")
    args = parser.parse_args()
    turn_count = 1 + len(args.follow_up)
    if args.follow_up and len(args.turn_budget) != turn_count:
        parser.error("--turn-budget must be repeated once per turn when --follow-up is used")
    if not args.follow_up and len(args.turn_budget) > 1:
        parser.error("a single-turn run accepts at most one --turn-budget")
    turn_budgets = args.turn_budget or [args.budget]
    if any(limit < 1 for limit in turn_budgets):
        parser.error("every --turn-budget must be at least 1")
    if sum(turn_budgets) > args.budget:
        parser.error("the sum of --turn-budget values cannot exceed --budget")
    question = _question(parser, args.question)
    started = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / f"{started}_{args.question}_{args.mode}"
    refuse_unsafe_outputs(parser, [run_dir], [RULES_CHUNKS, RULES_EMBEDDINGS, CLEAN_DB, SQL_SCHEMA])
    if not args.send:
        print(
            f"Would ask {question!r} ({args.mode}) with at most {args.budget} chat attempts, "
            f"turn budgets {turn_budgets}, and {args.embedding_budget} embedding attempts; "
            f"follow-ups {args.follow_up!r}; "
            f"saved in {run_dir}. Nothing sent: add --send to run it."
        )
        return
    if args.round is None:
        parser.error("--round is required with --send")

    chat_budget = RequestBudget(args.budget, MIN_SECONDS_BETWEEN_CHAT_REQUESTS)
    chat_transport = AttemptCounter(chat_budget, capture_diagnostics=True)
    embed_transport = AttemptCounter(RequestBudget(args.embedding_budget, 0))
    settings = load_settings(parser)
    agent = build_agent(
        parser,
        {"transport": chat_transport},
        {"transport": embed_transport},
        settings=settings,
    )
    run_dir.mkdir(parents=True)
    report = {"question_id": args.question, "question": question, "mode": args.mode}
    report |= {"started": started, "chat_budget": args.budget, "turn_budgets": turn_budgets}
    report |= {"embedding_budget": args.embedding_budget, "follow_ups": args.follow_up}
    report["acceptance"] = {
        "question_id": args.question,
        "round": args.round,
        "commit": _git_commit(),
        "model_settings": agent.trace().model_settings,
        "embedding_model": settings.embedding_model,
        "budgets": {
            "chat": args.budget,
            "turns": turn_budgets,
            "embedding": args.embedding_budget,
        },
        "inputs": {
            "uv.lock": hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest(),
            "testset": hashlib.sha256(TESTSET.read_bytes()).hexdigest(),
        },
    }
    report["inputs"] = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            RULES_CHUNKS,
            RULES_EMBEDDINGS,
            CLEAN_DB,
            SQL_SCHEMA,
            *sorted(SOURCE_DIR.rglob("*.py")),
        ]
    }
    answer = trace = None
    try:
        turns = [question, *args.follow_up]
        report["turns"] = [
            {"question": turn, "budget": budget, "status": "not_run", "answer": None}
            for turn, budget in zip(turns, turn_budgets, strict=True)
        ]
        for index, (turn, turn_budget) in enumerate(zip(turns, turn_budgets, strict=True), start=1):
            answer = None
            record = report["turns"][index - 1]
            record["status"] = "running"
            chat_budget.start_turn(turn_budget)
            turn_dir = run_dir if not args.follow_up else run_dir / f"turn-{index:02d}"
            turn_dir.mkdir(exist_ok=True)
            report["exit_code"] = run_question(agent, turn, args.mode, turn_dir)
            trace = agent.trace()
            answer = agent.last_answer() if trace.messages else None
            record["answer"] = answer_record(answer) if answer else None
            record["status"] = "complete" if answer and answer.answered else "no_answer"
            if answer is None or not answer.answered:
                for pending in report["turns"][index:]:
                    pending["reason"] = "an earlier turn did not produce an answer"
                break
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if "record" in locals():
            record["status"] = "failed"
            record["error"] = report["error"]
            for pending in report["turns"][index:]:
                pending["reason"] = "an earlier turn failed"
        trace = agent.trace()
        raise
    finally:
        report["chat_attempts"] = chat_transport.attempts
        report["embedding_attempts"] = embed_transport.attempts
        write_results(run_dir, report, answer, trace)
        print(
            f"chat attempts {report['chat_attempts']}, embedding attempts "
            f"{report['embedding_attempts']} -> {run_dir}"
        )


if __name__ == "__main__":
    main()
