"""Parse the Core Rules PDF with Docling into page-tagged Markdown.

After the conversion, the document is corrected from its own layout, never from what its text
says: page furniture that Docling left in the body (page numbers, running footers, number badges)
is moved to the furniture layer, and table cells that span several columns are written once.
The text printed inside a picture is surrounded by picture markers that name the picture. The
running footer of each page is read back from its position, for the chunks to show.

This module is a library: it never parses command-line arguments and never prints.
The command-line entry point is ``src/scripts/parse.py``.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItem,
    DoclingDocument,
    FloatingItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

PAGE_MARKER = "<!-- page: {page_no} -->"
PAGE_MARKER_PATTERN = re.compile(r"^<!-- page: (\d+) -->$", re.MULTILINE)
PICTURE_START = "<!-- picture: {ref} -->"  # ref: the picture in the parsed JSON, "#/pictures/12"
PICTURE_START_PATTERN = re.compile(r"^<!-- picture: (#/pictures/\d+) -->$")
PICTURE_END = "<!-- /picture -->"
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f]")  # every control character except \t, \n
NUMBER_BADGE = re.compile(r"\d{1,2}")  # page number, phase tab or step badge printed on a graphic
FURNITURE_MIN_PAGES = 3  # Docling's furniture must recur on this many pages to mark a position
POSITION_TOLERANCE = 3.0  # points: a printed position varies by a point or two between pages
CARD_LABEL_GAP = 12.0  # the narrow side graphic ends immediately before its card's heading
DIE_SIDE_RANGE = (10.0, 30.0)  # printed dice in the rulebook's tables, in PDF points
DIE_GRID_TOLERANCE = 0.06
DIE_PATTERNS = (
    {(1, 1)},
    {(0, 0), (2, 2)},
    {(0, 2), (2, 0)},
    {(0, 0), (1, 1), (2, 2)},
    {(0, 2), (1, 1), (2, 0)},
    {(0, 0), (0, 2), (2, 0), (2, 2)},
    {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)},
    {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)},
)


@dataclass(frozen=True)
class ParsedPage:
    """Markdown content of one PDF page."""

    page_no: int
    markdown: str


def build_converter() -> DocumentConverter:
    """Create a Docling converter configured for a born-digital, multi-column rulebook.

    Returns:
        DocumentConverter: converter with OCR disabled, accurate table recognition, and heading
        levels inferred from font size.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # born-digital PDF: the embedded text layer is exact
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    # best of the five layout models compared on this rulebook (lab1_1003.md §7)
    pipeline_options.layout_options = LayoutObjectDetectionOptions.from_preset(
        "layout_heron_default"
    )
    # heading levels from the font: the numbers in this book's headings are step labels, not an
    # outline, and with more than 6 levels a terrain feature (14 pt) stays above its subsections
    # (9.5 pt) instead of sharing the last level with them (lab1_1003.md §7)
    pipeline_options.generate_parsed_pages = True  # the font sizes the inference needs
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True, use_numbering=False, max_level=100
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_document(document_path: Path) -> DoclingDocument:
    """Convert a PDF into a structured DoclingDocument, then correct it from its layout.

    Args:
        document_path (Path): path to the PDF to parse.

    Returns:
        DoclingDocument: the parsed document (layout labels, heading levels, reading order,
        tables, pages), with misplaced page furniture moved and spanning table cells written
        once (lab1_1003.md §12).

    Raises:
        RuntimeError: if the conversion did not fully succeed.
    """
    result = build_converter().convert(document_path)
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(
            f"Converting {document_path} ended with {result.status}: {result.errors}"
        )
    correct_furniture(result.document)
    write_spanning_cells_once(result.document)
    order_card_labels(result.document)
    with pdfium.PdfDocument(document_path) as pdf:
        for table in result.document.tables:
            page = pdf[table.prov[0].page_no - 1]
            try:
                paths = [
                    BoundingBox(
                        l=b[0],
                        t=page.get_height() - b[3],
                        r=b[2],
                        b=page.get_height() - b[1],
                        coord_origin=CoordOrigin.TOPLEFT,
                    )
                    for obj in page.get_objects(max_depth=20)
                    if obj.type == pdfium_raw.FPDF_PAGEOBJ_PATH
                    for b in [obj.get_bounds()]
                ]
                fill_dice_cells(table, dice_faces(paths))
            finally:
                page.close()
    return result.document


def dice_faces(paths: list[BoundingBox]) -> list[tuple[BoundingBox, int]]:
    """Read standard die faces from square outlines and the pip positions inside them.

    Args:
        paths: Vector-path bounds in top-left page coordinates.

    Returns:
        Die bounds and values, only for unambiguous standard pip patterns.
    """
    faces = []
    for box in paths:
        width, height = box.r - box.l, box.b - box.t
        if not (DIE_SIDE_RANGE[0] < width < DIE_SIDE_RANGE[1] and abs(width - height) < 0.5):
            continue
        pips = [
            p
            for p in paths
            if box.l < p.l < p.r < box.r
            and box.t < p.t < p.b < box.b
            and 0.15 * width < p.r - p.l < 0.25 * width
            and abs((p.r - p.l) - (p.b - p.t)) < 0.02 * width
        ]
        grid = []
        for pip in pips:
            x = ((pip.l + pip.r) / 2 - box.l) / width
            y = ((pip.t + pip.b) / 2 - box.t) / height
            col, row = round((x - 0.2) / 0.3), round((y - 0.2) / 0.3)
            if (
                abs(x - (0.2 + 0.3 * col)) > DIE_GRID_TOLERANCE
                or abs(y - (0.2 + 0.3 * row)) > DIE_GRID_TOLERANCE
            ):
                break
            grid.append((col, row))
        else:
            if len(grid) == len(set(grid)) and set(grid) in DIE_PATTERNS:
                faces.append((box, len(grid)))
    return faces


def fill_dice_cells(table: TableItem, faces: list[tuple[BoundingBox, int]]) -> None:
    """Insert graphical die values into their table cells by position, retaining existing text.

    Args:
        table: Table whose cell boxes enclose text, sometimes excluding adjacent dice.
        faces: Recognized dice in top-left page coordinates.
    """
    cells = [cell for cell in table.data.table_cells if cell.bbox is not None]
    additions: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for box, value in faces:
        x, y = (box.l + box.r) / 2, (box.t + box.b) / 2
        nearby = [
            (index, cell)
            for index, cell in enumerate(cells)
            if abs((cell.bbox.t + cell.bbox.b) / 2 - y) < (box.b - box.t) * 0.75
            and max(cell.bbox.l - box.r, box.l - cell.bbox.r, 0) < (box.r - box.l) * 1.5
        ]
        if not nearby:
            continue
        index, cell = min(nearby, key=lambda pair: abs((pair[1].bbox.l + pair[1].bbox.r) / 2 - x))
        additions[index].append((x, str(value)))
    for index, values in additions.items():
        cell = cells[index]
        values.append(((cell.bbox.l + cell.bbox.r) / 2, cell.text))
        cell.text = " ".join(text for _, text in sorted(values) if text)


def order_card_labels(document: DoclingDocument) -> None:
    """Place narrow side-label pictures after the card heading beside them.

    The pp. 41–42 labels were serialized ahead of several unrelated cards. Their top edges
    align with headings, and the label strip lies to the left of the card's text column.
    """
    headings = [
        ref.resolve(document)
        for ref in document.body.children
        if isinstance(ref.resolve(document), SectionHeaderItem)
    ]
    placements: dict[str, list[NodeItem]] = defaultdict(list)
    for picture in document.pictures:
        if picture.parent != document.body.get_ref() or not picture.prov:
            continue
        box = page_box(document, picture)
        children = [
            ref.resolve(document)
            for ref in picture.children
            if isinstance(ref.resolve(document), TextItem)
            and ref.resolve(document).content_layer == ContentLayer.BODY
        ]
        if len(children) != 1 or not 0 < box.r - box.l < 50:
            continue
        matches = [
            h
            for h in headings
            if h.prov
            and h.prov[0].page_no == picture.prov[0].page_no
            and 0 <= page_box(document, h).l - box.r <= CARD_LABEL_GAP
            and abs(page_box(document, h).t - box.t) <= 6
        ]
        if len(matches) == 1:
            placements[matches[0].self_ref].append(picture)
    for heading in headings:
        for node in placements[heading.self_ref]:
            document.body.children.remove(node.get_ref())
            position = document.body.children.index(heading.get_ref()) + 1
            document.body.children.insert(position, node.get_ref())


def picture_labels(document: DoclingDocument) -> dict[str, str]:
    """Return the topmost printed text of each picture as its context label.

    Physical position keeps a title such as the p. 37 example-datasheet label even when
    Docling places that title after the picture's internal subsections in reading order.
    """
    labels = {}
    texts: dict[str, list[TextItem]] = defaultdict(list)
    for item, _ in document.iterate_items(traverse_pictures=True):
        if isinstance(item, TextItem) and item.prov and item.content_layer == ContentLayer.BODY:
            if ref := picture_of(document, item):
                texts[ref].append(item)
    for ref, items in texts.items():
        if len(items) > 1:
            top = min(
                items, key=lambda item: (page_box(document, item).t, page_box(document, item).l)
            )
            labels[ref] = clean_markdown(top.text)
    return labels


def page_box(document: DoclingDocument, item: DocItem) -> BoundingBox:
    """Return the box of an item on its page, with the origin at the top left."""
    prov = item.prov[0]
    return prov.bbox.to_top_left_origin(document.pages[prov.page_no].size.height)


def same_position(box: BoundingBox, other: BoundingBox) -> bool:
    """Return True if two boxes are printed at the same place, within ``POSITION_TOLERANCE``."""
    corners = zip((box.l, box.t, box.r, box.b), (other.l, other.t, other.r, other.b), strict=True)
    return all(abs(a - b) <= POSITION_TOLERANCE for a, b in corners)


def furniture_zones(document: DoclingDocument) -> list[BoundingBox]:
    """Return the positions where Docling found page furniture on several pages.

    A position that holds furniture on at least ``FURNITURE_MIN_PAGES`` pages is part of the
    document's page template (a page number, a running footer).
    """
    furniture = [
        (item.prov[0].page_no, page_box(document, item))
        for item, _ in document.iterate_items(
            included_content_layers={ContentLayer.FURNITURE}, traverse_pictures=True
        )
        if isinstance(item, TextItem) and item.prov
    ]
    return [
        box
        for _, box in furniture
        if len({page for page, other in furniture if same_position(box, other)})
        >= FURNITURE_MIN_PAGES
    ]


def in_zone(box: BoundingBox, zone: BoundingBox) -> bool:
    """Return True if ``box`` lies within the rows of ``zone`` and overlaps it horizontally.

    Only the rows must match: a running footer is longer on some pages than on others.
    """
    return (
        zone.t - POSITION_TOLERANCE <= box.t
        and box.b <= zone.b + POSITION_TOLERANCE
        and box.l < zone.r
        and box.r > zone.l
    )


def page_footers(document: DoclingDocument) -> dict[int, str]:
    """Return the running footer printed on each page ("CORE RULES | MISSION: ONLY WAR").

    The rulebook prints its running footer on the lowest row of its page template, below the page
    number, at the left of some pages and the right of others. That row is the lowest of the
    ``furniture_zones``; a page's footer is the one furniture text in it that is not a number. A
    page with none, or with several, gets no footer, and a lowest row of page numbers only gives
    none: this is the rulebook's layout, checked on its pages, not a rule of every PDF.

    Args:
        document (DoclingDocument): the parsed document, its furniture corrected
            (``correct_furniture``), as the parse saves it.

    Returns:
        dict[int, str]: the footer of each page that has one, by page number.
    """
    zones = furniture_zones(document)
    if not zones:
        return {}
    lowest = max(zone.t for zone in zones)
    row = [zone for zone in zones if lowest - zone.t <= POSITION_TOLERANCE]
    found = defaultdict(list)
    for item, _ in document.iterate_items(
        included_content_layers={ContentLayer.FURNITURE}, traverse_pictures=True
    ):
        if not isinstance(item, TextItem) or not item.prov:
            continue
        text = item.text.strip()
        box = page_box(document, item)
        if not NUMBER_BADGE.fullmatch(text) and any(in_zone(box, zone) for zone in row):
            found[item.prov[0].page_no].append(text)
    return {page_no: texts[0] for page_no, texts in found.items() if len(texts) == 1}


def correct_furniture(document: DoclingDocument) -> None:
    """Move body text that is really page furniture to the furniture layer.

    Docling labels most page numbers and running footers as furniture, but not all: on the
    rulebook it left 12 of 24 page numbers and 3 of 51 footers in the body, which broke a sentence
    across pp. 34-35. A body text item is moved if it lies where furniture recurs on other pages
    (``furniture_zones``), or if it is only a one- or two-digit number, which in a layout like
    this one is a label printed on a graphic (page number, phase tab, step badge), not prose.
    Text inside pictures is checked too, since it is exported (``export_pages``).

    Args:
        document (DoclingDocument): the parsed document, changed in place.
    """
    zones = furniture_zones(document)
    for item, _ in document.iterate_items(
        included_content_layers={ContentLayer.BODY}, traverse_pictures=True
    ):
        if not isinstance(item, TextItem) or not item.prov:
            continue
        box = page_box(document, item)
        if NUMBER_BADGE.fullmatch(item.text.strip()) or any(in_zone(box, z) for z in zones):
            item.content_layer = ContentLayer.FURNITURE
            parent = item.parent.resolve(document)
            if isinstance(parent, FloatingItem):
                # a footer that Docling took for a picture's caption is not that caption: left in
                # the picture's captions, the export repeats it on other pages
                parent.captions = [ref for ref in parent.captions if ref.cref != item.self_ref]


def write_spanning_cells_once(document: DoclingDocument) -> None:
    """Give every table cell a single grid position, so that a spanning cell is written once.

    Markdown has no row or column spans: Docling's export repeats a spanning cell in every
    position it covers (``| ROLLING A D3 | ROLLING A D3 |``). Kept in its first row and column
    only, its text stays in the table and the other positions are left empty.

    Args:
        document (DoclingDocument): the parsed document, changed in place.
    """
    for table in document.tables:
        for cell in table.data.table_cells:
            cell.row_span, cell.end_row_offset_idx = 1, cell.start_row_offset_idx + 1
            cell.col_span, cell.end_col_offset_idx = 1, cell.start_col_offset_idx + 1


def picture_of(document: DoclingDocument, item: NodeItem) -> str | None:
    """Return the reference of the picture that holds ``item`` (or is ``item``), if any."""
    node: NodeItem | None = item
    while node is not None and not isinstance(node, PictureItem):
        node = node.parent.resolve(document) if node.parent else None
    return node.self_ref if node else None


class PictureMarkingSerializer(MarkdownDocSerializer):
    """Docling's Markdown serializer, with picture markers around the text of each picture.

    Docling serializes a picture's children (headings, paragraphs, lists, captions) as separate
    parts of the page, after the picture itself. Consecutive parts of the same picture are
    surrounded by its markers, each on its own line, so removing the marker lines gives Docling's
    own Markdown back. A picture without text gets no markers.
    """

    def serialize_doc(
        self, *, parts: list[SerializationResult], **kwargs: Any
    ) -> SerializationResult:
        """Join the parts of a page, the parts of each picture between its markers."""
        marked = []
        written = [part for part in parts if part.text]
        # every span of a part comes from the same picture, or from none (checked on the rulebook)
        for ref, group in groupby(written, key=lambda p: picture_of(self.doc, p.spans[0].item)):
            group = list(group)
            if ref is None:
                marked.extend(group)
            else:
                text = "\n\n".join(part.text for part in group)
                marked.append(
                    create_ser_result(
                        text=f"{PICTURE_START.format(ref=ref)}\n{text}\n{PICTURE_END}",
                        span_source=group,
                    )
                )
        return super().serialize_doc(parts=marked, **kwargs)


def export_pages(document: DoclingDocument) -> list[ParsedPage]:
    """Export each page of a parsed document to cleaned Markdown.

    The text inside pictures is exported too, between picture markers: Docling reads worked
    examples, diagrams and numbered step badges as pictures holding text items (headings,
    paragraphs, lists), and leaves them out of the Markdown by default. Pages without any text
    (full-page artwork) are skipped.

    Args:
        document (DoclingDocument): the parsed document.

    Returns:
        list[ParsedPage]: one entry per page that contains text, in page order.
    """
    pages = []
    for page_no in sorted(document.pages):
        params = MarkdownParams(pages={page_no}, image_placeholder="", traverse_pictures=True)
        markdown = clean_markdown(
            PictureMarkingSerializer(doc=document, params=params).serialize().text
        )
        if markdown:
            pages.append(ParsedPage(page_no=page_no, markdown=markdown))
    return pages


def clean_markdown(markdown: str) -> str:
    """Normalise the characters and whitespace of the Markdown export.

    The rulebook's bullet is a ``■`` glyph followed by a decorative glyph that the PDF font maps
    to a control character (``- ■ \\x07Text``). Both become a plain ``- `` list marker; a ``■``
    inside a line starts a list item that Docling merged into the previous one.

    Args:
        markdown (str): Markdown of one page.

    Returns:
        str: the cleaned content, without trailing spaces or runs of blank lines.
    """
    text = CONTROL_CHARACTERS.sub("", markdown)
    text = re.sub(r"^(?:- )?■ *", "- ", text, flags=re.MULTILINE)
    text = text.replace(" ■ ", "\n- ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pages_to_markdown(pages: list[ParsedPage]) -> str:
    """Join pages into one Markdown text, each page preceded by its page marker line.

    No blank line surrounds a marker, so a paragraph that continues onto the next page stays one
    paragraph once the marker line is removed.
    """
    blocks = [f"{PAGE_MARKER.format(page_no=page.page_no)}\n{page.markdown}\n" for page in pages]
    return "".join(blocks)


def markdown_to_pages(markdown: str) -> list[ParsedPage]:
    """Split page-tagged Markdown back into pages (inverse of ``pages_to_markdown``)."""
    parts = PAGE_MARKER_PATTERN.split(markdown)
    # re.split with one capturing group returns [text_before, no_1, body_1, no_2, body_2, ...]
    return [
        ParsedPage(page_no=int(page_no), markdown=body.strip())
        for page_no, body in zip(parts[1::2], parts[2::2], strict=True)
    ]
