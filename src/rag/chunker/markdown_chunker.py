"""Split page-tagged Markdown into retrieval chunks.

Sections come from Markdown headings, tables stay whole, and long text is split with LangChain's
``RecursiveCharacterTextSplitter``. Every chunk records the pages it comes from, the running footer
printed on each of them, and the pictures whose text it holds.
This module is a library: it never reads or writes files.
The command-line entry point is ``src/scripts/chunk.py``.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.parser.docling_parser import PAGE_MARKER_PATTERN, PICTURE_END, PICTURE_START_PATTERN

HEADING_PATTERN = re.compile(r"^(#+)\s+(.+)$")  # any depth: the parser infers more than 6 levels
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # paragraph, line, sentence, word, character


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of the rulebook."""

    chunk_id: str  # "core_rules-0042": position in the document
    text: str  # what is embedded and shown to the LLM, starts with "Section: <section>"
    pages: list[int]  # [42], or [34, 35] when the chunk continues onto the next page
    section: str  # heading + the ancestors that tell it apart: "GRENADE", "WOODS > VISIBILITY"
    content_type: str  # "text" | "table"
    # pictures whose text is in the chunk (in its lines or its own heading), in document order:
    # ["#/pictures/12"], to look up in the parsed JSON of the same parse
    pictures: list[str]
    # the running footer printed on each page of ``pages``, in the same order ("" where the page
    # has none): shown to the LLM, never embedded; [] when not recorded (older chunk files, or
    # Markdown chunked without its parsed JSON)
    page_footers: list[str] = field(default_factory=list)
    picture_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.page_footers and len(self.page_footers) != len(self.pages):
            counts = f"{len(self.page_footers)} page footers for {len(self.pages)} pages"
            raise ValueError(f"{self.chunk_id}: {counts}")


@dataclass(frozen=True)
class Line:
    """One line of Markdown with the page it was printed on and the picture holding it."""

    text: str
    page_no: int
    picture: str | None  # "#/pictures/12" for text printed inside that picture


@dataclass
class Block:
    """Consecutive lines of one section that are all table rows, or all other text."""

    section: str
    content_type: str  # "text" | "table"
    heading_picture: str | None  # the picture holding the section's own heading, if any
    lines: list[Line] = field(default_factory=list)


def page_lines(markdown: str, skip_pages: frozenset[int]) -> list[Line]:
    """Return the lines of page-tagged Markdown with their page and picture, without markers.

    A page or picture marker only updates the current page or picture: a paragraph that continues
    onto the next page stays continuous, and a heading inside a picture stays the heading of the
    text after the picture. Lines of the pages in ``skip_pages`` are dropped, headings included.
    """
    lines = []
    page_no = 0  # the parser writes a page marker before any content
    picture = None
    for text in markdown.splitlines():
        if marker := PAGE_MARKER_PATTERN.fullmatch(text):
            page_no = int(marker.group(1))
        elif start := PICTURE_START_PATTERN.fullmatch(text):
            picture = start.group(1)
        elif text == PICTURE_END:
            picture = None
        elif page_no not in skip_pages:  # e.g. the p. 4 overview spread, which repeats other pages
            lines.append(Line(text=text, page_no=page_no, picture=picture))
    return lines


HeadingPath = tuple[tuple[int, str], ...]  # (level, name) of a heading and its ancestors


def heading_paths(lines: list[Line]) -> list[HeadingPath | None]:
    """Return, for each line, the path of headings that ends with it, or None if it is not one.

    A heading's level is its number of ``#``. It closes every open heading of the same or a
    deeper level, so the headings still open above it are its ancestors.
    """
    paths = []
    open_headings: HeadingPath = ()  # outermost first
    for line in lines:
        if heading := HEADING_PATTERN.fullmatch(line.text):
            level = len(heading.group(1))
            open_headings = tuple(item for item in open_headings if item[0] < level)
            open_headings += ((level, heading.group(2).strip()),)
            paths.append(open_headings)
        else:
            paths.append(None)
    return paths


def section_names(paths: list[HeadingPath]) -> dict[HeadingPath, str]:
    """Name each heading by the fewest of its ancestors that tell it apart from the others.

    A repeated name has a main heading when one of its headings is printed more prominently
    (a lower level) than all the others: ``MISSIONS``, the title of the missions part, against the
    ``MISSIONS`` paragraph of the core concepts. The main heading, like a unique one, is named by
    its name alone. Every other heading is named by the shortest end of its path that no other
    heading shares: ``VISIBILITY`` appears under every terrain feature at the same level, so each
    is named with its parent (``WOODS > VISIBILITY``), with two ancestors if the parent repeats
    too. Comparing the levels of two headings is reliable; the ancestors inferred from the levels
    are least reliable at the top, so names use as few as possible. Headings with the same whole
    path cannot be told apart, and share their name.
    """
    levels = defaultdict(list)
    for path in paths:
        level, name = path[-1]
        levels[name].append(level)

    def is_main(path: HeadingPath) -> bool:
        level, name = path[-1]
        return level == min(levels[name]) and levels[name].count(level) == 1

    # the names a heading can be told apart by: its own name if it is the main one, else its path
    keys = {path: (path[-1][1],) if is_main(path) else tuple(n for _, n in path) for path in paths}
    ends = Counter(key[-n:] for key in set(keys.values()) for n in range(1, len(key) + 1))
    names = {}
    for path, key in keys.items():
        length = next((n for n in range(1, len(key)) if ends[key[-n:]] == 1), len(key))
        names[path] = " > ".join(key[-length:])
    return names


def split_blocks(markdown: str, skip_pages: frozenset[int] = frozenset()) -> list[Block]:
    """Group the lines of page-tagged Markdown into blocks.

    A heading starts a new section, named by ``section_names``; its blocks keep the picture that
    holds the heading. Inside a section, a run of table rows (lines starting with ``|``) forms its
    own block.

    Args:
        markdown (str): page-tagged Markdown produced by the parser.
        skip_pages (frozenset[int]): pages left out of the index.

    Returns:
        list[Block]: blocks in document order. Text blocks may contain only blank lines.
    """
    lines = page_lines(markdown, skip_pages)
    paths = heading_paths(lines)
    names = section_names([path for path in paths if path])
    blocks = [Block(section="", content_type="text", heading_picture=None)]  # before any heading
    for line, path in zip(lines, paths, strict=True):
        if path:
            blocks.append(Block(names[path], "text", heading_picture=line.picture))
        else:
            content_type = "table" if line.text.startswith("|") else "text"
            if content_type != blocks[-1].content_type:
                blocks.append(Block(blocks[-1].section, content_type, blocks[-1].heading_picture))
            blocks[-1].lines.append(line)
    return blocks


def lines_between(lines: list[Line], start: int, end: int) -> list[Line]:
    """Return the lines that overlap characters ``start:end`` of the joined lines.

    Args:
        lines (list[Line]): the lines of one block, joined with "\\n" to form its text.
        start (int): index of the first character of the piece in that text.
        end (int): index just after the last character of the piece.

    Returns:
        list[Line]: the overlapping lines, in order.
    """
    overlapping = []
    line_start = 0
    for line in lines:
        line_end = line_start + len(line.text)
        if line_start < end and line_end > start:
            overlapping.append(line)
        line_start = line_end + 1  # skip the "\n" that joins two lines
    return overlapping


def chunk_markdown(
    markdown: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    skip_pages: frozenset[int] = frozenset(),
    footers: dict[int, str] | None = None,
    labels: dict[str, str] | None = None,
) -> list[Chunk]:
    """Split page-tagged Markdown into chunks with section, page, footer and picture metadata.

    Args:
        markdown (str): page-tagged Markdown produced by the parser.
        source (str): document name used in chunk IDs, e.g. "core_rules".
        chunk_size (int): maximum length of a text piece, in characters (tables are never split).
        chunk_overlap (int): characters shared by two consecutive pieces of the same block.
        skip_pages (frozenset[int]): pages left out of the index (``SKIP_PAGES`` in config).
        footers (dict[int, str] | None): the running footer of each page, from the same parse
            (``page_footers``); None leaves the chunks' footers unrecorded.
        labels (dict[str, str] | None): printed picture labels from the same parsed document.

    Returns:
        list[Chunk]: chunks in document order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",  # a cut after ". " keeps the full stop with its sentence
        add_start_index=True,  # each piece reports where it starts in the block text
    )
    chunks = []
    for block in split_blocks(markdown, skip_pages):
        text = "\n".join(line.text for line in block.lines)
        if block.content_type == "table":
            pieces = [(text, 0)]  # never split a table: a row without its header is meaningless
        else:
            documents = splitter.create_documents([text])
            pieces = [(doc.page_content, doc.metadata["start_index"]) for doc in documents]
        for piece, start in pieces:
            lines = lines_between(block.lines, start, start + len(piece))
            pictures = [block.heading_picture, *(line.picture for line in lines)]
            pictures = list(dict.fromkeys(ref for ref in pictures if ref))
            context = list(
                dict.fromkeys((labels or {})[ref] for ref in pictures if ref in (labels or {}))
            )
            pages = sorted({line.page_no for line in lines})
            text = f"Section: {block.section}\n{piece}" if block.section else piece
            if context:
                text = "\n".join(f"Figure label: {label}" for label in context) + "\n" + text
            chunks.append(
                Chunk(
                    chunk_id=f"{source}-{len(chunks):04d}",
                    text=text,
                    pages=pages,
                    section=block.section,
                    content_type=block.content_type,
                    pictures=pictures,
                    page_footers=[] if footers is None else [footers.get(p, "") for p in pages],
                    picture_labels=context,
                )
            )
    return chunks


def chunks_to_jsonl(chunks: list[Chunk]) -> str:
    """Serialise chunks as JSON Lines, one chunk per line (inverse of ``jsonl_to_chunks``)."""
    return "".join(json.dumps(asdict(chunk), ensure_ascii=False) + "\n" for chunk in chunks)


def jsonl_to_chunks(text: str) -> list[Chunk]:
    """Read chunks back from JSON Lines (inverse of ``chunks_to_jsonl``)."""
    return [Chunk(**json.loads(line)) for line in text.splitlines()]
