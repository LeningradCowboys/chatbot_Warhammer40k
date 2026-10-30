import sys

import pytest
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from src.rag.config import RAW_PDF, RULES_JSON, SKIP_PAGES
from src.rag.parser.docling_parser import (
    PICTURE_END,
    PICTURE_START,
    PICTURE_START_PATTERN,
    ParsedPage,
    clean_markdown,
    correct_furniture,
    dice_faces,
    export_pages,
    fill_dice_cells,
    markdown_to_pages,
    order_card_labels,
    page_footers,
    pages_to_markdown,
    parse_document,
    picture_labels,
    picture_of,
    write_spanning_cells_once,
)
from src.scripts import parse as parse_script


def make_document(page_count: int) -> DoclingDocument:
    document = DoclingDocument(name="test")
    for page_no in range(1, page_count + 1):
        document.add_page(page_no=page_no, size=Size(width=600, height=800))
    return document


def at(page_no: int, left: float, top: float, right: float, bottom: float) -> ProvenanceItem:
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=page_no, bbox=box, charspan=(0, 0))


def table_cell(text: str, row: int, col: int, col_span: int = 1) -> TableCell:
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        col_span=col_span,
    )


def add_footers(document: DoclingDocument, lefts: list[float]) -> None:
    """Add a running footer that Docling labelled correctly, on pages 1, 2, 3, ..."""
    for page_no, left in enumerate(lefts, start=1):
        document.add_text(
            label=DocItemLabel.PAGE_FOOTER,
            text="CORE RULES | FIGHT PHASE",
            prov=at(page_no, left, 744, 255, 752),
            content_layer=ContentLayer.FURNITURE,
        )


def without_picture_markers(markdown: str) -> str:
    lines = markdown.splitlines()
    return "\n".join(
        line for line in lines if not PICTURE_START_PATTERN.fullmatch(line) and line != PICTURE_END
    )


def docling_export(document: DoclingDocument, page_no: int) -> str:
    """Docling's own Markdown of a page, cleaned: what the export gives without picture markers."""
    markdown = document.export_to_markdown(
        page_no=page_no, image_placeholder="", traverse_pictures=True
    )
    return clean_markdown(markdown)


def test_card_cost_pictures_follow_their_own_headings_by_position():
    document = make_document(1)
    for top, cost in [(98, "1CP"), (298, "2CP")]:
        picture = document.add_picture(prov=at(1, 70, top, 100, top + 90))
        document.add_text(
            label=DocItemLabel.TEXT,
            text=cost,
            parent=picture,
            prov=at(1, 80, top + 40, 95, top + 48),
        )
    first = document.add_heading(text="First card", prov=at(1, 103, 100, 180, 110))
    second = document.add_heading(text="Second card", prov=at(1, 103, 300, 180, 310))
    order_card_labels(document)
    children = [ref.resolve(document) for ref in document.body.children]
    assert children == [first, document.pictures[0], second, document.pictures[1]]
    assert document.pictures[0].children[0].resolve(document).text == "1CP"
    order_card_labels(document)
    assert children == [ref.resolve(document) for ref in document.body.children]


def test_dice_values_come_from_standard_pip_geometry_and_join_the_cell_text():
    # p.21: a ~20 pt square to the left of the '+' text; the two pips lie on a diagonal.
    face = BoundingBox(l=478.37, t=535.91, r=498.21, b=555.75)
    dots = [
        BoundingBox(l=492.26, t=537.89, r=496.22, b=541.86),
        BoundingBox(l=480.35, t=549.80, r=484.32, b=553.77),
    ]
    assert dice_faces([face, *dots]) == [(face, 2)]
    assert dice_faces([face, dots[0]]) == []  # one off-centre pip is not a die face
    document = make_document(1)
    cell = table_cell("+", 1, 1)
    cell.bbox = BoundingBox(l=499.98, t=539.67, r=508.09, b=552.33)
    table = document.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=[cell]))
    fill_dice_cells(table, dice_faces([face, *dots]))
    assert table.data.table_cells[0].text == "2 +"


def test_picture_label_uses_its_printed_position_not_serialization_order():
    document = make_document(1)
    picture = document.add_picture(prov=at(1, 100, 100, 500, 600))
    document.add_heading(text="UNIT COMPOSITION", parent=picture, prov=at(1, 200, 200, 300, 210))
    document.add_text(
        label=DocItemLabel.TEXT,
        text="EXAMPLE DATASHEET",
        parent=picture,
        prov=at(1, 120, 110, 300, 125),
    )
    assert picture_labels(document) == {picture.self_ref: "EXAMPLE DATASHEET"}


def test_page_markers_round_trip():
    pages = [ParsedPage(41, "## STRATAGEMS\n\ntext"), ParsedPage(42, "## GRENADE\n\nWHEN: ...")]
    assert markdown_to_pages(pages_to_markdown(pages)) == pages


def test_clean_markdown_collapses_blank_lines():
    assert clean_markdown("## A  \n\n\n\ntext\n") == "## A\n\ntext"


def test_clean_markdown_turns_bullet_glyphs_into_list_items():
    # real Docling output: "■" bullets, a control character from the bullet's font, two merged items
    page = (
        '- ■ \x07Targets must be within 12". ■ \x07If the distance rolled is insufficient\n'
        "■ Unit Fully Visible: every model is visible"
    )
    assert clean_markdown(page) == (
        '- Targets must be within 12".\n'
        "- If the distance rolled is insufficient\n"
        "- Unit Fully Visible: every model is visible"
    )


def test_correct_furniture_moves_body_text_where_furniture_recurs():
    # pp. 31-35: Docling labelled the running footer as furniture on three pages, as body text on
    # p. 34, where it is also longer; a page number and a step badge were labelled as body text too
    document = make_document(4)
    add_footers(document, [153, 153, 153])
    footer = document.add_text(
        DocItemLabel.TEXT, "CORE RULES | FIGHT PHASE | FIGHT", prov=at(4, 153, 744, 283, 752)
    )
    page_number = document.add_text(DocItemLabel.TEXT, "34", prov=at(4, 538, 693, 548, 701))
    rule = document.add_text(DocItemLabel.TEXT, "Roll six D6.", prov=at(4, 85, 600, 300, 742))
    beside = document.add_text(DocItemLabel.TEXT, "Other column.", prov=at(4, 300, 744, 520, 752))
    correct_furniture(document)
    assert footer.content_layer == ContentLayer.FURNITURE
    assert page_number.content_layer == ContentLayer.FURNITURE
    assert rule.content_layer == ContentLayer.BODY
    assert beside.content_layer == ContentLayer.BODY  # same height, but no horizontal overlap
    assert "CORE RULES" not in document.export_to_markdown(page_no=4)


def test_correct_furniture_matches_positions_within_the_tolerance():
    # measured positions vary by fractions of a point; 152.4 and 152.6 are the same position
    document = make_document(4)
    add_footers(document, [152.4, 152.6, 152.6])
    footer = document.add_text(DocItemLabel.TEXT, "CORE RULES", prov=at(4, 152.5, 744, 255, 752))
    correct_furniture(document)
    assert footer.content_layer == ContentLayer.FURNITURE


def test_correct_furniture_needs_the_position_on_several_pages():
    document = make_document(3)
    add_footers(document, [153, 153])
    text = document.add_text(DocItemLabel.TEXT, "CORE RULES", prov=at(3, 153, 744, 255, 752))
    correct_furniture(document)
    assert text.content_layer == ContentLayer.BODY


def test_correct_furniture_detaches_a_footer_read_as_a_caption():
    # pp. 11 and 35: Docling attached the running footer to a picture as its caption
    document = make_document(4)
    add_footers(document, [153, 153, 153])
    picture = document.add_picture(prov=at(4, 200, 400, 450, 740))
    footer = document.add_text(
        DocItemLabel.CAPTION, "CORE RULES | FIGHT", prov=at(4, 153, 744, 283, 752), parent=picture
    )
    picture.captions.append(footer.get_ref())
    correct_furniture(document)
    assert footer.content_layer == ContentLayer.FURNITURE
    assert picture.captions == []
    for page_no in (1, 2, 3, 4):
        assert "CORE RULES" not in document.export_to_markdown(page_no=page_no)


def add_page_numbers(document: DoclingDocument, pages: list[int]) -> None:
    """Add the page number of each page, above the running footer, as the rulebook prints it."""
    for page_no in pages:
        document.add_text(
            label=DocItemLabel.PAGE_FOOTER,
            text=str(page_no),
            prov=at(page_no, 538, 693, 548, 701),
            content_layer=ContentLayer.FURNITURE,
        )


def test_page_footers_reads_the_lowest_row_on_both_sides_of_the_page():
    # the footer is printed at the left of odd pages and the right of even ones (pp. 59-60), and
    # Docling labelled it a caption on p. 11 and a text on p. 31: only its position counts
    document = make_document(6)
    add_page_numbers(document, [1, 2, 3, 4, 5, 6])
    for page_no in range(1, 6):
        left, right = (153, 255) if page_no % 2 else (330, 456)  # right-aligned on even pages
        document.add_text(
            label=DocItemLabel.PAGE_FOOTER,
            text="CORE RULES | FIGHT PHASE",
            prov=at(page_no, left, 744, right, 752),
            content_layer=ContentLayer.FURNITURE,
        )
    document.add_text(
        label=DocItemLabel.CAPTION,
        text="CORE RULES | MISSION: ONLY WAR",
        prov=at(6, 330, 744, 456, 752),
        content_layer=ContentLayer.FURNITURE,
    )
    footers = page_footers(document)
    assert footers == {page_no: "CORE RULES | FIGHT PHASE" for page_no in range(1, 6)} | {
        6: "CORE RULES | MISSION: ONLY WAR"
    }


def test_page_footers_never_takes_a_row_of_page_numbers_for_footers():
    document = make_document(3)
    add_page_numbers(document, [1, 2, 3])
    assert page_footers(document) == {}


def test_page_footers_leaves_out_a_page_with_two_texts_in_the_row():
    # a footer read as two items cannot be told whole: the page gets none rather than a part
    document = make_document(4)
    add_footers(document, [153, 153, 153, 153])
    document.add_text(
        label=DocItemLabel.PAGE_FOOTER,
        text="| FIGHT",
        prov=at(4, 200, 744, 300, 752),
        content_layer=ContentLayer.FURNITURE,
    )
    assert sorted(page_footers(document)) == [1, 2, 3]


def test_export_pages_includes_the_text_inside_pictures():
    # p. 60: each mission step is a picture (the number badge) holding its heading and its text;
    # the badge's number alone is furniture
    document = make_document(1)
    step = document.add_picture(prov=at(1, 80, 100, 500, 300))
    document.add_heading("10 END THE BATTLE", prov=at(1, 120, 110, 300, 125), parent=step)
    document.add_text(
        DocItemLabel.TEXT, "The battle ends.", prov=at(1, 120, 130, 300, 140), parent=step
    )
    document.add_text(DocItemLabel.TEXT, "10", prov=at(1, 85, 110, 100, 125), parent=step)
    correct_furniture(document)
    [page] = export_pages(document)
    assert page.markdown == (
        f"{PICTURE_START.format(ref=step.self_ref)}\n"
        "## 10 END THE BATTLE\n\nThe battle ends.\n"
        f"{PICTURE_END}"
    )


def test_picture_markers_surround_only_the_text_of_each_picture():
    # p. 38: a numbered heading inside a picture, its text outside; an artwork without text
    document = make_document(1)
    document.add_text(DocItemLabel.TEXT, "Before.", prov=at(1, 80, 50, 500, 60))
    badge = document.add_picture(prov=at(1, 80, 100, 500, 120))
    document.add_heading("1 DATASHEET NAME", prov=at(1, 120, 105, 300, 115), parent=badge)
    document.add_text(DocItemLabel.TEXT, "The name of the unit.", prov=at(1, 80, 130, 500, 140))
    document.add_picture(prov=at(1, 80, 200, 500, 400))
    [page] = export_pages(document)
    assert page.markdown == (
        "Before.\n\n"
        f"{PICTURE_START.format(ref=badge.self_ref)}\n## 1 DATASHEET NAME\n{PICTURE_END}\n\n"
        "The name of the unit."
    )
    assert without_picture_markers(page.markdown) == docling_export(document, page_no=1)


def test_write_spanning_cells_once():
    # p. 9: a title cell spanning both columns, which Markdown would repeat in each column
    cells = [
        table_cell("ROLLING A D3", row=0, col=0, col_span=2),
        table_cell("DICE RESULT", row=1, col=0),
        table_cell("D3 RESULT", row=1, col=1),
    ]
    document = make_document(1)
    table = document.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells))
    write_spanning_cells_once(document)
    rows = [
        [text.strip() for text in line.strip("|").split("|")]
        for line in table.export_to_markdown(doc=document).splitlines()
    ]
    assert rows[0] == ["ROLLING A D3", ""]
    assert rows[2] == ["DICE RESULT", "D3 RESULT"]


def test_parse_script_ends_lines_with_lf(tmp_path, monkeypatch):
    # the Markdown and the tables are the same bytes on Windows and macOS (plan §15, R23)
    document = make_document(1)
    document.add_heading("GRENADE", prov=at(1, 80, 50, 500, 60))
    document.add_text(DocItemLabel.TEXT, "Roll six D6.", prov=at(1, 80, 70, 500, 80))
    cells = [table_cell("DICE RESULT", row=0, col=0), table_cell("4+", row=1, col=0)]
    document.add_table(
        data=TableData(num_rows=2, num_cols=1, table_cells=cells), prov=at(1, 80, 100, 500, 200)
    )
    pdf = tmp_path / "Core Rules.pdf"
    pdf.touch()
    output_dir, tables_dir = tmp_path / "processed", tmp_path / "tables"
    monkeypatch.setattr(parse_script, "parse_document", lambda path: document)  # no Docling models
    monkeypatch.setattr(
        sys,
        "argv",
        ["parse", str(pdf), "--output-dir", str(output_dir), "--tables-dir", str(tables_dir)],
    )
    parse_script.main()
    [table_file] = tables_dir.glob("*.csv")
    for path in (output_dir / "core_rules.md", table_file):
        assert b"\r" not in path.read_bytes(), path


@pytest.fixture(scope="module")
def document() -> DoclingDocument:
    if not RAW_PDF.exists():
        pytest.skip("Core Rules.pdf not available")
    return parse_document(RAW_PDF)


@pytest.fixture(scope="module")
def pages(document) -> dict[int, str]:
    return {page.page_no: page.markdown for page in export_pages(document)}


def flat(markdown: str) -> str:
    """Collapse line breaks and repeated spaces, so a phrase matches however Docling wraps it."""
    return " ".join(markdown.split())


@pytest.mark.slow
def test_grenade_rule_is_complete(pages):
    grenade = flat(pages[42])
    for fact in [
        "GRENADE",
        "Your Shooting phase",
        "not within Engagement Range",
        'within 8" of and visible',
        "Roll six D6",
        "4+",
        "mortal wound",
    ]:
        assert fact in grenade, fact


@pytest.mark.slow
def test_scoring_rule_is_complete(pages):
    scoring = flat(pages[59])
    for fact in [
        "Starting from the second battle round",
        "Command phase",
        "1 Victory point (VP) for each objective marker",
        "maximum of 3VP per turn",
    ]:
        assert fact in scoring, fact


@pytest.mark.slow
def test_card_costs_and_graphical_wound_thresholds_are_preserved(document, pages):
    from src.rag.chunker.markdown_chunker import chunk_markdown

    chunks = chunk_markdown(pages_to_markdown(export_pages(document)), "rules", 1500, 200)
    costs = {
        "COMMAND RE-ROLL": "1CP",
        "COUNTER-OFFENSIVE": "2CP",
        "EPIC CHALLENGE": "1CP",
        "INSANE BRAVERY": "1CP",
        "GRENADE": "1CP",
        "TANK SHOCK": "1CP",
        "RAPID INGRESS": "1CP",
        "FIRE OVERWATCH": "1CP",
        "GO TO GROUND": "1CP",
        "SMOKESCREEN": "1CP",
        "HEROIC INTERVENTION": "2CP",
    }
    for section, cost in costs.items():
        [chunk] = [c for c in chunks if c.section == section]
        assert [line for line in chunk.text.splitlines() if line in {"1CP", "2CP"}] == [cost]
    [table] = [t for t in document.tables if t.prov[0].page_no == 21]
    values = [
        c.text
        for c in table.data.table_cells
        if c.start_col_offset_idx == 1 and c.start_row_offset_idx > 0
    ]
    assert values == ["2 +", "3 +", "4 +", "5 +", "6 +"]


@pytest.mark.slow
def test_page_furniture_and_artwork_are_excluded(pages):
    assert 2 not in pages  # full-page artwork has no text
    # p. 4 reproduces other pages, footers included, and never reaches the index (SKIP_PAGES)
    indexed = {page_no: text for page_no, text in pages.items() if page_no not in SKIP_PAGES}
    for page_no, markdown in indexed.items():
        assert "CORE RULES |" not in markdown, page_no  # running footer
        assert str(page_no) not in markdown.splitlines(), page_no  # page number


@pytest.mark.slow
def test_text_inside_pictures_is_kept(pages):
    assert "The Termagant unit is selected to fight" in flat(pages[36])  # worked example
    assert "# 11 DETERMINE VICTOR" in pages[60]  # mission step heading, inside its badge picture


@pytest.mark.slow
def test_picture_markers_only_add_lines(document, pages):
    # every page without its marker lines is Docling's own export: the markers change no text
    for page_no, markdown in pages.items():
        assert without_picture_markers(markdown) == docling_export(document, page_no), page_no


@pytest.mark.slow
def test_every_exported_part_comes_from_one_picture_or_none(document):
    # PictureMarkingSerializer takes a part's picture from its first item: every item of a part
    # must belong to the same picture, or all to none, for the markers to name the right picture
    for page_no in document.pages:
        params = MarkdownParams(pages={page_no}, image_placeholder="", traverse_pictures=True)
        for part in MarkdownDocSerializer(doc=document, params=params).get_parts():
            if part.text:
                pictures = {picture_of(document, span.item) for span in part.spans}
                assert len(pictures) == 1, (page_no, part.text[:60], pictures)


@pytest.mark.slow
def test_page_footers_of_the_saved_parse():
    # read from the saved JSON, as the chunk script does; checked against the PDF (plan R71)
    if not RULES_JSON.exists():
        pytest.skip("core_rules.json not available: run src.scripts.parse first")
    footers = page_footers(DoclingDocument.load_from_json(RULES_JSON))
    assert len(footers) == 51
    assert footers[53] == "CORE RULES | AIRCRAFT"
    assert footers[59] == footers[60] == "CORE RULES | MISSION: ONLY WAR"
    assert footers[11] == "CORE RULES | COMMAND PHASE"  # labelled a caption
    assert footers[31] == "CORE RULES | FIGHT PHASE"  # labelled a text
    assert footers[35] == "CORE RULES | FIGHT PHASE | FIGHT"  # labelled a caption
    assert 40 not in footers and 55 not in footers  # no footer printed on these pages
    assert not any(footer.isdigit() for footer in footers.values())  # never a page number
