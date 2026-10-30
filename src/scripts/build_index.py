"""Command-line entry point: embed every rule chunk with Gemini and save the index.

Usage:
    uv run python -m src.scripts.build_index
"""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from src.rag.chunker.markdown_chunker import jsonl_to_chunks
from src.rag.config import EMBEDDING_DIM, RULES_CHUNKS, RULES_EMBEDDINGS, Settings, key_not_set
from src.rag.llm.budget import AttemptCounter, RequestBudget
from src.rag.llm.gemini import GeminiEmbedder
from src.rag.storage.rules_index import save_embeddings
from src.scripts.outputs import refuse_unsafe_outputs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the rule chunks and save the index")
    parser.add_argument("chunks", type=Path, nargs="?", default=RULES_CHUNKS)
    parser.add_argument("--output", type=Path, default=RULES_EMBEDDINGS)
    parser.add_argument(
        "--budget",
        type=int,
        default=18,
        help="maximum embedding HTTP attempts, retries included (default: 18)",
    )
    args = parser.parse_args()
    if not args.chunks.is_file():
        parser.error(f"{args.chunks} not found. Run src.scripts.chunk first (README.md, Run).")
    refuse_unsafe_outputs(parser, [args.output], [args.chunks])
    try:
        settings = Settings()
    except ValidationError as error:
        if not key_not_set(error):
            raise
        parser.error("GOOGLE_API_KEY is not set. See README.md, section API key.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = jsonl_to_chunks(args.chunks.read_text(encoding="utf-8"))
    attempts = AttemptCounter(RequestBudget(args.budget, 0))
    embedder = GeminiEmbedder(settings, {"transport": attempts})
    vectors = embedder.embed([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(args.output, vectors, chunks, embedder.model)
    logger.info(
        "%d chunks embedded with %s (%d dimensions) -> %s",
        len(chunks),
        embedder.model,
        EMBEDDING_DIM,
        args.output,
    )


if __name__ == "__main__":
    main()
