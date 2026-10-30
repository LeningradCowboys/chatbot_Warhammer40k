"""Command-line entry point: split the parsed rulebook into chunks and save them as JSON Lines.

Usage:
    uv run python -m src.scripts.chunk
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from src.rag.chunker.markdown_chunker import chunk_markdown, chunks_to_jsonl
from src.rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RULES_CHUNKS,
    RULES_JSON,
    RULES_MARKDOWN,
    SKIP_PAGES,
)
from src.rag.parser.docling_parser import (
    export_pages,
    page_footers,
    pages_to_markdown,
    picture_labels,
)
from src.scripts.outputs import refuse_unsafe_outputs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split page-tagged Markdown into chunks")
    parser.add_argument("markdown", type=Path, nargs="?", default=RULES_MARKDOWN)
    parser.add_argument(
        "--json",
        type=Path,
        help="the parsed JSON of the same parse, for the page footers (default: the rulebook's "
        "JSON with the rulebook's Markdown, none with other Markdown)",
    )
    parser.add_argument("--output", type=Path, default=RULES_CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    if not args.markdown.is_file():
        parser.error(f"{args.markdown} not found. Run src.scripts.parse first (README.md, Run).")
    json_path = args.json
    if json_path is None and args.markdown.resolve() == RULES_MARKDOWN.resolve():
        json_path = RULES_JSON
    if json_path is not None and not json_path.is_file():
        parser.error(f"{json_path} not found. Run src.scripts.parse first (README.md, Run).")
    inputs = [args.markdown] if json_path is None else [args.markdown, json_path]
    refuse_unsafe_outputs(parser, [args.output], inputs)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    markdown = args.markdown.read_text(encoding="utf-8")
    footers = None
    labels = None
    if json_path is not None:
        document = DoclingDocument.load_from_json(json_path)
        # the footers belong to this Markdown only if both come from one parse
        if pages_to_markdown(export_pages(document)) != markdown:
            parser.error(f"{json_path} is not the parse of {args.markdown}: their Markdown differs")
        footers = page_footers(document)
        labels = picture_labels(document)
    chunks = chunk_markdown(
        markdown,
        source=args.markdown.stem,  # "core_rules.md" -> IDs "core_rules-0000", ...
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        skip_pages=SKIP_PAGES,
        footers=footers,
        labels=labels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # "\n" on every system, so Windows and macOS write the same bytes
    args.output.write_text(chunks_to_jsonl(chunks), encoding="utf-8", newline="\n")

    types = Counter(chunk.content_type for chunk in chunks)
    longest = max(len(chunk.text) for chunk in chunks)
    logger.info(
        "%d chunks (%d text, %d table; %d with picture text, %d with a page footer), "
        "longest %d characters -> %s",
        len(chunks),
        types["text"],
        types["table"],
        sum(bool(chunk.pictures) for chunk in chunks),
        sum(any(chunk.page_footers) for chunk in chunks),
        longest,
        args.output,
    )


if __name__ == "__main__":
    main()
