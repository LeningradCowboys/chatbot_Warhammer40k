"""Copy the datasheet database with its text repaired; the raw file is never modified."""

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# The labels of a rule card (stratagems, some abilities). The export glued most of them to the end
# of the sentence before ("phase.TARGET:", "phaseEFFECT:"); each one starts a line again.
CARD_LABEL = re.compile(r"(?<=\S) *(?=(?:WHEN|TARGET|EFFECT|RESTRICTIONS):)")


@dataclass(frozen=True)
class CleaningReport:
    """How many text cells each repair changed."""

    re_decoded: int
    card_labels: int


def repair_text(value: str) -> str:
    """Undo UTF-8 text that was decoded as Latin-1, e.g. 'Tâ\\x80\\x99au' -> 'T’au'.

    Text that is already correct cannot be re-encoded this way and is returned unchanged.
    """
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value


def separate_card_labels(text: str) -> str:
    """Start each label of a rule card (WHEN:, TARGET:, EFFECT:, RESTRICTIONS:) on its own line."""
    return CARD_LABEL.sub("\n", text)


def clean_database(source: Path, target: Path) -> CleaningReport:
    """Copy `source` to `target`, then repair every text cell of the copy.

    Args:
        source: The raw database, opened read-only.
        target: The cleaned copy; replaced if it exists, unless it is `source` itself.

    Returns:
        The number of cells changed by each repair.

    Raises:
        ValueError: `target` is the same file as `source`; nothing is deleted or written.
    """
    if target.exists() and source.samefile(target):
        raise ValueError(f"{target} is the source database itself: choose another output")
    target.unlink(missing_ok=True)
    with (
        closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as raw,
        closing(sqlite3.connect(target)) as clean,
    ):
        raw.backup(clean)
        re_decoded = card_labels = 0
        tables = [r[0] for r in clean.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            columns = [r[1] for r in clean.execute(f'PRAGMA table_info("{table}")')]
            updates = []
            for rowid, *values in clean.execute(f'SELECT rowid, * FROM "{table}"'):
                for column, value in zip(columns, values, strict=True):
                    if not isinstance(value, str):
                        continue
                    decoded = repair_text(value)
                    labelled = separate_card_labels(decoded)
                    re_decoded += decoded != value
                    card_labels += labelled != decoded
                    if labelled != value:
                        updates.append((column, labelled, rowid))
            for column, value, rowid in updates:
                clean.execute(
                    f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?', (value, rowid)
                )
        clean.commit()
    return CleaningReport(re_decoded=re_decoded, card_labels=card_labels)
