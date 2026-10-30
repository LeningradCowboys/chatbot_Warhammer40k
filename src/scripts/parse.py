"""Command-line entry point: parse a PDF and save Markdown, JSON and tables.

Usage:
    uv run python -m src.scripts.parse "data/raw/Core Rules.pdf"
"""

import argparse
import json
import logging
from pathlib import Path

from src.rag.config import PROCESSED_DIR, TABLES_DIR
from src.rag.parser.docling_parser import export_pages, pages_to_markdown, parse_document
from src.scripts.outputs import refuse_unsafe_outputs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF into page-tagged Markdown")
    parser.add_argument("document", type=Path, help="Path to the PDF to parse")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    if not args.document.is_file():
        parser.error(f"{args.document} not found. See README.md, section Data.")
    name = args.document.stem.lower().replace(" ", "_")  # "Core Rules.pdf" -> "core_rules"
    markdown_path = args.output_dir / f"{name}.md"
    json_path = args.output_dir / f"{name}.json"
    # Before parsing: the folders and the files whose names are known.
    refuse_unsafe_outputs(
        parser, [args.output_dir, args.tables_dir, markdown_path, json_path], [args.document]
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per model file on a first run

    document = parse_document(args.document)
    pages = export_pages(document)
    table_paths = [
        args.tables_dir / f"{name}_p{table.prov[0].page_no:03d}_t{index}.csv"
        for index, table in enumerate(document.tables, start=1)
    ]
    # After parsing, before the first write: every output, the tables included.
    refuse_unsafe_outputs(parser, [markdown_path, json_path, *table_paths], [args.document])

    # Write canonical LF bytes for both artifacts on every platform.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(pages_to_markdown(pages), encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(document.export_to_dict(), indent=2), encoding="utf-8", newline="\n"
    )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for table, table_path in zip(document.tables, table_paths, strict=True):
        table.export_to_dataframe(doc=document).to_csv(
            table_path, index=False, encoding="utf-8", lineterminator="\n"
        )

    logger.info(
        "%d pages with text, %d tables -> %s", len(pages), len(document.tables), markdown_path
    )


if __name__ == "__main__":
    main()
