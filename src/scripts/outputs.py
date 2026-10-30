"""The check every script makes before it writes: no output may be one of its inputs, or lie in
`data/raw/`, which holds the original inputs (AGENTS.md).

It runs before the work it guards (parsing, chunking, embedding, cleaning), so a refused command
spends nothing and changes no file.
"""

import argparse
from pathlib import Path

from src.rag.config import RAW_DIR


def output_problem(output: Path, inputs: list[Path]) -> str | None:
    """Why `output` must not be written, or None if it may.

    Paths are compared resolved, so `..` and symbolic links do not get past the check, and an
    existing output is compared with `Path.samefile`, which also sees a hard link to an input.

    Args:
        output: A file or folder the script will write.
        inputs: The files the script reads.
    """
    resolved = output.resolve()
    if resolved.is_relative_to(RAW_DIR.resolve()):
        return f"{output} is in {RAW_DIR}, which holds the original inputs"
    for source in inputs:
        same = output.exists() and source.exists() and output.samefile(source)
        if same or resolved == source.resolve():
            return f"{output} is the input {source}: it would be overwritten"
    return None


def refuse_unsafe_outputs(
    parser: argparse.ArgumentParser, outputs: list[Path], inputs: list[Path]
) -> None:
    """Stop the script with an error if any output must not be written.

    Args:
        parser: The script's parser, which reports the error and exits.
        outputs: Every file or folder the script will write.
        inputs: The files the script reads.
    """
    for output in outputs:
        problem = output_problem(output, inputs)
        if problem:
            parser.error(f"{problem}.")
