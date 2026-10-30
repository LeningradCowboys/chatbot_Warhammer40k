"""Command-line entry point: copy the datasheet database with its text repaired.

Usage:
    uv run python -m src.scripts.prepare_db
"""

import argparse
import logging
from pathlib import Path

from src.rag.config import CLEAN_DB, RAW_DB
from src.rag.storage.clean_db import clean_database
from src.scripts.outputs import refuse_unsafe_outputs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the database with its text repaired")
    parser.add_argument("database", type=Path, nargs="?", default=RAW_DB)
    parser.add_argument("--output", type=Path, default=CLEAN_DB)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"{args.database} not found. See README.md, section Data.")
    refuse_unsafe_outputs(parser, [args.output], [args.database])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = clean_database(args.database, args.output)
    logger.info(
        "%d text cells re-decoded, %d with card labels put on their own lines -> %s",
        report.re_decoded,
        report.card_labels,
        args.output,
    )


if __name__ == "__main__":
    main()
