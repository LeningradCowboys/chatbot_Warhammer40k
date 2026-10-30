"""Command-line entry point: answer the test set with the chat, then score the answers with RAGAS
(plan §6.9, steps 3 and 4).

Usage:
    uv run python -m src.scripts.evaluate answer --run trial --chat-budget 20 --embedding-budget 15
    uv run python -m src.scripts.evaluate score --run trial --chat-budget 70 --embedding-budget 6

A run is a folder of `data/eval/runs/`, named by `--run`: `answers/`, one file per question, and
`scores.jsonl`. A saved result is used only for the inputs it was made from; when the test set,
the chat's settings, an answer, the metrics or the models have changed, the run is refused before
any request, and a new `--run` starts again.

`answer` asks each question of the test set on a new thread through `ChatAgent.ask()`, searches
the rules retriever with the question as asked (its top `TOP_K` chunks, for the evaluation only,
never given to the model), and saves the answer as soon as it is given: its text, the chunks the
model received from `search_rules`, the retriever's chunks, the trace and the requests both took.
A question already answered is not asked again.

`score` computes RAGAS's metrics on the saved answers to the test set's questions, never asking
the chat: each score, or the metric's failure, is saved as soon as it is computed. A metric
already scored is not computed again; one that failed is computed again by the next run. An
answer without text is not scored.

Every request, retries included, is spent from the run's budgets before it is sent, the chat
model's and the embedding model's; a spent budget stops the run, and what it finished is kept.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from statistics import mean

from ragas.exceptions import RagasException
from ragas.metrics.base import SingleTurnMetric

from src.rag.chatbot.agent import ChatAgent, chunk_block, source_notes
from src.rag.config import (
    CLEAN_DB,
    EVAL_RUNS_DIR,
    LOCK_FILE,
    PROJECT_ROOT,
    RULES_CHUNKS,
    RULES_EMBEDDINGS,
    SOURCE_DIR,
    SQL_SCHEMA,
    TESTSET,
    TOP_K,
)
from src.rag.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM
from src.rag.evaluation.scoring import (
    EvalAnswer,
    MetricScore,
    agent_contexts,
    answers_problem,
    build_metrics,
    score,
    score_fingerprint,
    scores_problem,
)
from src.rag.evaluation.testset import EvalQuestion
from src.rag.llm.budget import AttemptCounter, BudgetExhausted, RequestBudget
from src.rag.llm.gemini import MIN_SECONDS_BETWEEN_CHAT_REQUESTS, GeminiEmbedder, create_chat_model
from src.rag.retriever.rules_retriever import RulesRetriever
from src.rag.storage.evaluation_runs import (
    answer_inputs,
    append_score,
    load_answers,
    load_scores,
    load_testset,
    save_answer,
    save_answer_inputs,
)
from src.scripts.chat import build_agent, load_rules_retriever, load_settings
from src.scripts.outputs import refuse_unsafe_outputs

RUN_NAME = re.compile(r"[A-Za-z0-9_-]+")  # a folder name of its own: no separator, no ".."

logger = logging.getLogger(__name__)


def answer_questions(
    agent: ChatAgent,
    retriever: RulesRetriever,
    questions: list[EvalQuestion],
    answers_dir: Path,
    chat: AttemptCounter,
    embedding: AttemptCounter,
) -> None:
    """Ask every question not answered yet, each on a new thread, and search the retriever with
    it apart from the chat, saving each answer at once.

    The saved answers are those of these questions, this chat and this k (`answers_problem`).

    Args:
        agent: The chat, its models sending through `chat` and `embedding`.
        retriever: The rules retriever, its embedder sending through `embedding`; its top
            `TOP_K` chunks for each question are saved, not given to the chat.
        questions: The test set.
        answers_dir: Where the answers are saved, one file per question.
        chat: The chat model's transport, whose attempts are saved with each answer.
        embedding: The embedder's transport, likewise.
    """
    answers_dir.mkdir(parents=True, exist_ok=True)
    answered = {
        answer.question.question_id for answer in load_answers(answers_dir) if answer.answer.strip()
    }
    for question in questions:
        if question.question_id in answered:
            continue
        chat_start, embedding_start = len(chat.attempts), len(embedding.attempts)
        # first the retriever's own search: a spent embedding budget then costs no chat request
        found = retriever.search(question.question, TOP_K)
        agent.reset()
        reply = agent.ask(question.question)
        if any(call.result is None for call in reply.tool_calls):
            raise RuntimeError(f"{question.question_id}: a tool call failed during answering")
        answer = EvalAnswer(
            question=question,
            answer=reply.text,
            agent_contexts=agent_contexts(reply),
            retriever_contexts=[chunk_block(item.chunk) for item in found],
            retriever_k=TOP_K,
            trace=reply.trace,
            chat_attempts=chat.attempts[chat_start:],
            embedding_attempts=embedding.attempts[embedding_start:],
            source_notes=source_notes(reply),
            context_policy="conversation-tool-outputs-v1",
        )
        save_answer(answers_dir, answer)
        logger.info(
            "%s answered (%d chunks received, %d retrieved apart)",
            question.question_id,
            len(answer.agent_contexts),
            len(answer.retriever_contexts),
        )


def score_answers(
    answers: list[EvalAnswer],
    metrics: list[SingleTurnMetric],
    models: list[str],
    scores_path: Path,
) -> None:
    """Compute every metric not scored yet on every answer with text, saving each at once.

    A score saved for other inputs (`score_fingerprint`) does not count as scored. A failure of
    the metric itself (a reply RAGAS cannot parse or that did not finish, no score) is saved as
    the score's error, and the next metric is computed; any other error, such as a spent budget
    or a request the API refused after its retries, stops the run.

    Args:
        answers: The answers to score.
        metrics: RAGAS's metrics (`build_metrics`).
        models: The names of the chat and embedding models the metrics use.
        scores_path: The run's scores, appended to.
    """
    saved = load_scores(scores_path)
    for answer in answers:
        question_id = answer.question.question_id
        if not answer.answer.strip():
            logger.info("%s not scored: the chat gave no answer", question_id)
            continue
        for metric in metrics:
            inputs = score_fingerprint(answer, metric, models)
            previous = saved.get((question_id, metric.name))
            if previous is not None and previous.value is not None and previous.inputs == inputs:
                continue
            try:
                value, error = score(metric, answer), None
            except (RagasException, ValueError) as failure:
                value, error = None, f"{type(failure).__name__}: {failure}"
            append_score(scores_path, MetricScore(question_id, metric.name, value, error, inputs))
            logger.info(
                "%s %s: %s", question_id, metric.name, error if value is None else f"{value:.3f}"
            )


def log_means(
    answers: list[EvalAnswer],
    metrics: list[SingleTurnMetric],
    scores_path: Path,
    expected: int | None = None,
) -> None:
    """Log each metric's mean and valid, failed and missing counts against the expected set."""
    scores = load_scores(scores_path)
    expected = len(answers) if expected is None else expected
    for metric in metrics:
        present = [
            scores[(answer.question.question_id, metric.name)]
            for answer in answers
            if (answer.question.question_id, metric.name) in scores
        ]
        values = [saved.value for saved in present if saved.value is not None]
        logger.info(
            "%s: mean %s; valid %d/%d, failed %d, missing %d",
            metric.name,
            f"{mean(values):.3f}" if values else "n/a",
            len(values),
            expected,
            len(present) - len(values),
            expected - len(present),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer the test set, then score the answers")
    parser.add_argument("step", choices=["answer", "score"])
    parser.add_argument("--run", required=True, help=f"the run's folder name, in {EVAL_RUNS_DIR}")
    parser.add_argument(
        "--chat-budget", type=int, required=True, help="the most chat requests, retries included"
    )
    parser.add_argument(
        "--embedding-budget",
        type=int,
        required=True,
        help="the most embedding requests, retries included",
    )
    args = parser.parse_args()
    if not RUN_NAME.fullmatch(args.run):
        parser.error(f"--run {args.run!r}: letters, digits, '-' and '_' only.")
    answers_dir, scores_path = (
        EVAL_RUNS_DIR / args.run / "answers",
        EVAL_RUNS_DIR / args.run / "scores.jsonl",
    )
    inputs_path = EVAL_RUNS_DIR / args.run / "inputs.json"
    inputs = [
        TESTSET,
        RULES_CHUNKS,
        RULES_EMBEDDINGS,
        CLEAN_DB,
        SQL_SCHEMA,
        LOCK_FILE,
        *sorted(SOURCE_DIR.rglob("*.py")),
    ]
    if not TESTSET.is_file():
        parser.error(f"{TESTSET} not found. Run src.scripts.generate_testset first.")
    if args.step == "answer":
        refuse_unsafe_outputs(parser, [answers_dir, inputs_path], inputs)
    else:
        if not answers_dir.is_dir():
            parser.error(f"{answers_dir} not found. Run the answer step of this run first.")
        refuse_unsafe_outputs(parser, [scores_path], [TESTSET, answers_dir])
    try:
        questions = load_testset(TESTSET)
    except ValueError as error:
        parser.error(f"{TESTSET}: {error}.")
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chat = AttemptCounter(RequestBudget(args.chat_budget, MIN_SECONDS_BETWEEN_CHAT_REQUESTS))
    embedding = AttemptCounter(RequestBudget(args.embedding_budget, 0))
    if args.step == "answer":
        agent = build_agent(parser, {"transport": chat}, {"transport": embedding})
        retriever = load_rules_retriever(parser, load_settings(parser), {"transport": embedding})
        problem = answers_problem(questions, load_answers(answers_dir), agent.trace(), TOP_K)
    else:
        settings = load_settings(parser)
        metrics = build_metrics(
            GeminiRagasLLM(create_chat_model(settings, {"transport": chat})),
            EmbedderRagasEmbeddings(GeminiEmbedder(settings, {"transport": embedding})),
        )
        models = [
            f"{settings.chat_model};temperature={settings.chat_temperature}",
            settings.embedding_model,
        ]
        saved = {answer.question.question_id: answer for answer in load_answers(answers_dir)}
        answers = [saved[q.question_id] for q in questions if q.question_id in saved]
        problem = scores_problem(questions, answers, metrics, models, load_scores(scores_path))
    if problem:
        parser.error(f"{problem}: start a new --run.")
    if args.step == "answer":
        try:
            save_answer_inputs(
                inputs_path,
                answers_dir,
                answer_inputs(
                    {path.relative_to(PROJECT_ROOT).as_posix(): path for path in inputs},
                    load_settings(parser).embedding_model,
                ),
            )
        except ValueError as error:
            parser.error(f"{error}: start a new --run.")

    try:
        try:
            if args.step == "answer":
                answer_questions(agent, retriever, questions, answers_dir, chat, embedding)
            else:
                score_answers(answers, metrics, models, scores_path)
                log_means(answers, metrics, scores_path, expected=len(questions))
        except BudgetExhausted as error:
            parser.exit(
                2,
                f"{parser.prog}: {error}; completed results were saved. "
                "Rerun with the same --run to continue.\n",
            )
    finally:
        logger.info(
            "%d chat requests, %d embedding requests sent",
            len(chat.attempts),
            len(embedding.attempts),
        )


if __name__ == "__main__":
    main()
