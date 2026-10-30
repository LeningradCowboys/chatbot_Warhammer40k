"""Command-line entry point: generate the RAGAS test set from the rulebook (plan §6.9, step 1).

Usage:
    uv run python -m src.scripts.generate_testset --chat-budget 40 --embedding-budget 10
    uv run python -m src.scripts.generate_testset --graph-only --chat-budget N --embedding-budget 10

RAGAS first builds its knowledge graph of the sections named in `TESTSET_SECTIONS`, one
transform step at a time; the graph is saved after each step, and also when a step stops midway
(a spent budget, an error, Ctrl+C), with the nodes it reached. Run again, the script continues
the saved graph. Once the graph is built, RAGAS writes the questions, saved together at the end:
a run stopped while writing them writes them all again the next time, from the saved graph. With
`--graph-only`, the run ends once the graph is built and saved, before the questions, so that
they can get a run with a whole budget of their own; a graph already built sends nothing.

Every request, retries included, is spent from the run's budgets before it is sent, the chat
model's and the embedding model's, and the run stops when one is spent. The test set is not
overwritten: it is checked by a person once generated (plan §6.9, step 2).
"""

import argparse
import logging
import os
from pathlib import Path

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.base import BaseGraphTransformation

from src.rag.config import (
    KNOWLEDGE_GRAPH,
    KNOWLEDGE_GRAPH_PROGRESS,
    RULES_MARKDOWN,
    SKIP_PAGES,
    TESTSET,
    TESTSET_SECTIONS,
    TESTSET_SIZE,
)
from src.rag.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM
from src.rag.evaluation.testset import (
    apply_step,
    generate_questions,
    graph_fingerprint,
    knowledge_graph,
    section_documents,
    select_sections,
    step_name,
    transform_steps,
)
from src.rag.llm.budget import AttemptCounter, RequestBudget
from src.rag.llm.gemini import MIN_SECONDS_BETWEEN_CHAT_REQUESTS, GeminiEmbedder, create_chat_model
from src.rag.storage.evaluation_runs import (
    GraphProgress,
    load_knowledge_graph,
    save_knowledge_graph,
    save_testset,
)
from src.scripts.chat import load_settings
from src.scripts.outputs import refuse_unsafe_outputs

logger = logging.getLogger(__name__)


def build_graph(
    kg: KnowledgeGraph,
    steps: list[BaseGraphTransformation],
    progress: GraphProgress,
    graph_path: Path,
    progress_path: Path,
) -> None:
    """Apply the steps not done yet, saving the graph after each one and when one stops.

    Args:
        kg: The graph, as saved or new; changed in place.
        steps: Every step of the graph, in order (`transform_steps`).
        progress: What `kg` is built from and how many steps it holds.
        graph_path: Where the graph is saved.
        progress_path: Where its progress is saved.
    """
    for index in range(progress.steps_done, len(steps)):
        logger.info("step %d of %d: %s", index + 1, len(steps), step_name(steps[index]))
        try:
            apply_step(kg, steps[index])
            progress = GraphProgress(progress.fingerprint, index + 1)
        finally:
            save_knowledge_graph(graph_path, progress_path, kg, progress)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RAGAS test set from the rulebook")
    parser.add_argument("--size", type=int, default=TESTSET_SIZE, help="questions to generate")
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="stop once the graph is built, before the questions",
    )
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
    if not RULES_MARKDOWN.is_file():
        parser.error(f"{RULES_MARKDOWN} not found. Run src.scripts.parse (README.md, Run).")
    refuse_unsafe_outputs(
        parser, [KNOWLEDGE_GRAPH, KNOWLEDGE_GRAPH_PROGRESS, TESTSET], [RULES_MARKDOWN]
    )
    if TESTSET.exists():
        parser.error(f"{TESTSET} exists and may have been checked: move it away to generate again.")
    sections = section_documents(RULES_MARKDOWN.read_text(encoding="utf-8"), SKIP_PAGES)
    try:
        documents = select_sections(sections, TESTSET_SECTIONS)
    except ValueError as error:
        parser.error(f"TESTSET_SECTIONS in config.py: {error}.")
    settings = load_settings(parser)
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chat = AttemptCounter(RequestBudget(args.chat_budget, MIN_SECONDS_BETWEEN_CHAT_REQUESTS))
    embedding = AttemptCounter(RequestBudget(args.embedding_budget, 0))
    llm = GeminiRagasLLM(create_chat_model(settings, {"transport": chat}))
    embeddings = EmbedderRagasEmbeddings(GeminiEmbedder(settings, {"transport": embedding}))
    steps = transform_steps(documents, llm, embeddings)
    fingerprint = graph_fingerprint(
        documents, steps, [settings.chat_model, settings.embedding_model]
    )
    if KNOWLEDGE_GRAPH_PROGRESS.is_file():
        kg, progress = load_knowledge_graph(KNOWLEDGE_GRAPH, KNOWLEDGE_GRAPH_PROGRESS)
        if progress.fingerprint != fingerprint:
            parser.error(
                f"{KNOWLEDGE_GRAPH} was built from other sections, steps or models: "
                "move it and its progress away to build it again."
            )
        logger.info(
            "continuing the saved graph: %d of %d steps done", progress.steps_done, len(steps)
        )
    else:
        kg, progress = knowledge_graph(documents), GraphProgress(fingerprint, 0)
        KNOWLEDGE_GRAPH.parent.mkdir(parents=True, exist_ok=True)

    try:
        build_graph(kg, steps, progress, KNOWLEDGE_GRAPH, KNOWLEDGE_GRAPH_PROGRESS)
        if args.graph_only:
            logger.info(
                "graph built, %d steps -> %s; no question written", len(steps), KNOWLEDGE_GRAPH
            )
            return
        questions = generate_questions(kg, llm, embeddings, args.size)
        save_testset(TESTSET, questions)
        logger.info("%d questions -> %s", len(questions), TESTSET)
    finally:
        logger.info(
            "%d chat requests, %d embedding requests sent",
            len(chat.attempts),
            len(embedding.attempts),
        )


if __name__ == "__main__":
    main()
