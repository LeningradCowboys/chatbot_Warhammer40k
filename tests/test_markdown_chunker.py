import json
import re
import sys
from dataclasses import replace
from itertools import groupby, pairwise

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TextItem,
)

from src.rag.chunker.markdown_chunker import (
    Chunk,
    HeadingPath,
    chunk_markdown,
    chunks_to_jsonl,
    jsonl_to_chunks,
    page_lines,
    section_names,
)
from src.rag.config import CHUNK_OVERLAP, CHUNK_SIZE, RULES_JSON, RULES_MARKDOWN, SKIP_PAGES
from src.rag.parser.docling_parser import (
    PICTURE_END,
    PICTURE_START,
    PICTURE_START_PATTERN,
    ParsedPage,
    export_pages,
    page_footers,
    pages_to_markdown,
)
from src.scripts import chunk as chunk_script


def make_markdown(*pages: tuple[int, str]) -> str:
    return pages_to_markdown([ParsedPage(page_no, markdown) for page_no, markdown in pages])


def in_picture(ref: str, markdown: str) -> str:
    return f"{PICTURE_START.format(ref=ref)}\n{markdown}\n{PICTURE_END}"


def test_headings_become_sections():
    markdown = make_markdown((41, "## STRATAGEMS\n\nSpend CP."), (42, "## GRENADE\n\nRoll six D6."))
    chunks = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert [(c.chunk_id, c.section, c.pages, c.text) for c in chunks] == [
        ("doc-0000", "STRATAGEMS", [41], "Section: STRATAGEMS\nSpend CP."),
        ("doc-0001", "GRENADE", [42], "Section: GRENADE\nRoll six D6."),
    ]


def test_section_continuing_onto_next_page_stays_one_chunk():
    markdown = make_markdown(
        (34, "## MAKE ATTACKS\n\nno models in the target unit of that attack"),
        (35, "remain within Engagement Range."),
    )
    [chunk] = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert chunk.pages == [34, 35]
    assert "that attack\nremain within" in chunk.text
    assert "<!--" not in chunk.text


def test_table_is_one_chunk_even_when_longer_than_chunk_size():
    rows = [f'| row {i} | 44" x 60" |' for i in range(20)]
    table = "\n".join(["| SIZE | BATTLEFIELD |", "|---|---|", *rows])
    markdown = make_markdown((59, f"## BATTLE SIZE\n\nIntro text.\n\n{table}\n\nAfter text."))
    chunks = chunk_markdown(markdown, source="doc", chunk_size=100, chunk_overlap=0)
    assert [c.content_type for c in chunks] == ["text", "table", "text"]
    assert chunks[1].text == f"Section: BATTLE SIZE\n{table}"
    assert all(c.section == "BATTLE SIZE" and c.pages == [59] for c in chunks)


def test_long_section_is_split_and_each_piece_keeps_its_pages():
    page_34 = "\n\n".join(f"Paragraph {i} on page 34 ends here." for i in range(10))
    page_35 = "\n\n".join(f"Paragraph {i} on page 35 ends here." for i in range(10))
    markdown = make_markdown((34, f"## MAKE ATTACKS\n\n{page_34}"), (35, page_35))
    chunks = chunk_markdown(markdown, source="doc", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 2
    for chunk in chunks:
        body = chunk.text.removeprefix("Section: MAKE ATTACKS\n")
        assert len(body) <= 200
        assert chunk.pages == sorted({int(p) for p in re.findall(r"on page (\d+)", body)})
    assert any(chunk.pages == [34, 35] for chunk in chunks)


def test_long_text_is_cut_after_a_full_stop():
    markdown = make_markdown((10, "## A\n\n" + "This sentence is complete. " * 10))
    chunks = chunk_markdown(markdown, source="doc", chunk_size=60, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_text_before_first_heading_and_heading_without_text():
    cover = "WARHAMMER 40,000\n\n## CORE RULES\n## MOVEMENT PHASE\n\nMove units."
    markdown = make_markdown((1, cover))
    chunks = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert [(c.section, c.text) for c in chunks] == [
        ("", "WARHAMMER 40,000"),
        ("MOVEMENT PHASE", "Section: MOVEMENT PHASE\nMove units."),
    ]


def test_section_names_use_the_fewest_ancestors_that_tell_them_apart():
    # pp. 44-48: every terrain feature has its own VISIBILITY, at the same level; here WOODS also
    # appears in two biomes, so one parent is not enough for its VISIBILITY; SELECT TARGETS appears
    # twice under the same phase, as on p. 19, and cannot be told apart
    markdown = make_markdown(
        (1, "## FOREST\n\nTrees.\n### WOODS\n#### VISIBILITY\n\nA."),
        (2, "## JUNGLE\n### WOODS\n#### VISIBILITY\n\nB.\n### RUINS\n#### VISIBILITY\n\nC."),
        (3, "## SHOOTING\n### SELECT TARGETS\n\nD.\n### SELECT TARGETS\n\nE."),
        (4, "## CHARGE\n### SELECT TARGETS\n\nF."),
    )
    chunks = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert [c.section for c in chunks] == [
        "FOREST",  # unique: its own name
        "FOREST > WOODS > VISIBILITY",  # WOODS > VISIBILITY is shared with the jungle
        "JUNGLE > WOODS > VISIBILITY",
        "RUINS > VISIBILITY",  # one parent is enough
        "SHOOTING > SELECT TARGETS",  # same whole path twice: the same name
        "SHOOTING > SELECT TARGETS",
        "CHARGE > SELECT TARGETS",
    ]
    assert chunks[3].text == "Section: RUINS > VISIBILITY\nC."


def test_the_most_prominent_of_a_repeated_name_keeps_the_name():
    # p. 5 mentions MISSIONS among the core concepts; p. 57 is the MISSIONS part, printed larger.
    # p. 10 lists the phases under THE BATTLE ROUND; p. 11 is the COMMAND PHASE itself. The
    # large-print heading of p. 3 sits above everything, and must not appear in any name.
    markdown = make_markdown(
        (3, "## Introduction\n\nWelcome."),
        (5, "### CORE CONCEPTS\n##### MISSIONS\n\nA mission sets the battle."),
        (10, "### THE BATTLE ROUND\n##### COMMAND PHASE\n\nOverview."),
        (11, "#### COMMAND PHASE\n\nBoth players gain 1CP."),
        (57, "### MISSIONS\n\nSelect a mission."),
    )
    chunks = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert [c.section for c in chunks] == [
        "Introduction",
        "CORE CONCEPTS > MISSIONS",
        "THE BATTLE ROUND > COMMAND PHASE",
        "COMMAND PHASE",
        "MISSIONS",
    ]


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        pytest.param(
            [((2, "A"), (3, "X")), ((2, "B"), (3, "X")), ((2, "C"), (5, "X"))],
            ["A > X", "B > X", "C > X"],
            id="two most prominent at the same level: no main heading",
        ),
        pytest.param(
            [((4, "X"),), ((2, "A"), (4, "X"))],
            ["X", "A > X"],
            id="a whole path that is the end of another",
        ),
        pytest.param(
            [((2, "X"),), ((2, "P"), (4, "X"))],
            ["X", "P > X"],
            id="a main heading without ancestors",
        ),
        pytest.param(
            [((2, "A"), (4, "X")), ((2, "A"), (4, "X"))],
            ["X", "X"],
            id="only the same whole path twice: the name alone, shared",
        ),
    ],
)
def test_section_names_edge_cases(paths: list[HeadingPath], expected: list[str]):
    names = section_names(paths)
    assert [names[path] for path in paths] == expected


def test_chunks_record_the_pictures_their_text_comes_from():
    # p. 37: a diagram and an example datasheet (one picture, split by its own headings); p. 38: a
    # numbered heading inside a badge picture, its text outside; p. 59: a step heading inside a
    # picture, whose only text is under a subheading printed outside
    datasheet = "### WARGEAR OPTIONS\n\n- Bolt pistol.\n\n### UNIT COMPOSITION\n\n- 1 Sergeant"
    markdown = make_markdown(
        (
            37,
            "## DATASHEETS\n\nEach unit has a datasheet.\n\n"
            + in_picture("#/pictures/76", "Datasheet layout.")
            + "\n\n"
            + in_picture("#/pictures/77", datasheet),
        ),
        (38, in_picture("#/pictures/78", "## 1 DATASHEET NAME") + "\n\nThe name of the unit."),
        (
            59,
            in_picture("#/pictures/135", "## 2 MISSION OBJECTIVES")
            + "\n\n### CAPTURE AND CONTROL\n\nScore 1VP.",
        ),
    )
    chunks = chunk_markdown(markdown, source="doc", chunk_size=500, chunk_overlap=0)
    assert [(c.section, c.pictures) for c in chunks] == [
        ("DATASHEETS", ["#/pictures/76"]),
        ("WARGEAR OPTIONS", ["#/pictures/77"]),
        ("UNIT COMPOSITION", ["#/pictures/77"]),
        ("1 DATASHEET NAME", ["#/pictures/78"]),  # its own heading is in the picture
        ("CAPTURE AND CONTROL", []),  # a parent heading in a picture does not count
    ]
    assert chunks[0].text == "Section: DATASHEETS\nEach unit has a datasheet.\n\nDatasheet layout."


@pytest.mark.parametrize(
    ("body", "chunk_size", "expected"),
    [
        pytest.param(
            "## RULES\n\nRule one.\n\n"
            + in_picture("#/pictures/9", "Example one.")
            + "\n\nRule two.\n\n"
            + in_picture("#/pictures/2", "Example two."),
            500,
            [["#/pictures/9", "#/pictures/2"]],
            id="two pictures in one chunk: both, in document order",
        ),
        pytest.param(
            in_picture("#/pictures/3", "## CARD\n\nCard text.\n\n- Option."),
            500,
            [["#/pictures/3"]],
            id="heading and text in the same picture: listed once",
        ),
        pytest.param(
            "## RULES\n\n"
            + in_picture("#/pictures/1", "First example sentence here.")
            + "\n\nPlain rule sentence here.\n\n"
            + in_picture("#/pictures/2", "Second example sentence here."),
            30,
            [["#/pictures/1"], [], ["#/pictures/2"]],
            id="a section cut into pieces: each piece lists only its own",
        ),
    ],
)
def test_pictures_list_every_picture_of_the_chunk_once(
    body: str, chunk_size: int, expected: list[list[str]]
):
    chunks = chunk_markdown(make_markdown((8, body)), "doc", chunk_size, chunk_overlap=0)
    assert [chunk.pictures for chunk in chunks] == expected


def test_skipped_page_produces_no_chunk():
    markdown = make_markdown(
        (3, "## INTRODUCTION\n\nWelcome."),
        (4, "## FAKE OVERVIEW HEADING\n\nDuplicated text."),
        (5, "## MISSIONS\n\nSelect a mission."),
    )
    chunks = chunk_markdown(markdown, "doc", 500, 0, skip_pages=frozenset({4}))
    assert [(c.section, c.pages) for c in chunks] == [("INTRODUCTION", [3]), ("MISSIONS", [5])]
    assert not any("Duplicated" in c.text for c in chunks)


def test_chunks_jsonl_round_trip():
    stratagem = "## GRENADE\n\n" + in_picture("#/pictures/1", "CORE – WARGEAR STRATAGEM")
    markdown = make_markdown((42, stratagem), (43, "Roll six D6."))
    footers = {42: "CORE RULES | STRATAGEMS"}
    chunks = chunk_markdown(markdown, "doc", chunk_size=500, chunk_overlap=0, footers=footers)
    assert chunks[0].page_footers == ["CORE RULES | STRATAGEMS", ""]
    assert jsonl_to_chunks(chunks_to_jsonl(chunks)) == chunks


def test_chunks_record_the_footer_of_each_of_their_pages():
    # a chunk over two pages keeps each page's footer: one page's footer never stands for both
    markdown = make_markdown(
        (34, "## FIGHT\n\nEach model can pile in"), (35, 'up to 3".'), (36, "## NEXT\n\nText.")
    )
    footers = {34: "CORE RULES | FIGHT PHASE", 35: "CORE RULES | FIGHT PHASE | FIGHT"}
    chunks = chunk_markdown(markdown, "doc", 500, 0, footers=footers)
    assert [(chunk.pages, chunk.page_footers) for chunk in chunks] == [
        ([34, 35], ["CORE RULES | FIGHT PHASE", "CORE RULES | FIGHT PHASE | FIGHT"]),
        ([36], [""]),  # recorded: this page has none
    ]


def test_chunks_without_footers_leave_them_unrecorded():
    chunks = chunk_markdown(make_markdown((1, "## A\n\nOne.")), "doc", 500, 0)
    assert [chunk.page_footers for chunk in chunks] == [[]]


def test_page_footers_must_match_the_pages():
    with pytest.raises(ValueError, match="1 page footers for 2 pages"):
        Chunk("doc-0000", "Section: A\nOne.", [1, 2], "A", "text", [], ["CORE RULES | A"])


def test_an_older_chunk_line_reads_without_footers():
    # chunk files and traces saved before footers were recorded read as they were saved
    line = json.dumps(
        {
            "chunk_id": "doc-0000",
            "text": "Section: A\nOne.",
            "pages": [59],
            "section": "A",
            "content_type": "text",
            "pictures": [],
        }
    )
    [chunk] = jsonl_to_chunks(line + "\n")
    assert chunk.page_footers == []


def at(page_no: int, top: float, bottom: float, left: float, right: float) -> ProvenanceItem:
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=page_no, bbox=box, charspan=(0, 0))


def footed_parse(page_count: int) -> DoclingDocument:
    """A parse whose pages each hold a section and a running footer at the same place."""
    document = DoclingDocument(name="rules")
    for page_no in range(1, page_count + 1):
        document.add_page(page_no=page_no, size=Size(width=600, height=800))
        document.add_heading(f"PART {page_no}", prov=at(page_no, 50, 60, 80, 500))
        document.add_text(DocItemLabel.TEXT, f"Rule {page_no}.", prov=at(page_no, 70, 80, 80, 500))
        document.add_text(
            DocItemLabel.PAGE_FOOTER,
            f"CORE RULES | PART {page_no}",
            prov=at(page_no, 744, 752, 153, 255),
            content_layer=ContentLayer.FURNITURE,
        )
    return document


def test_chunk_script_takes_the_footers_of_the_same_parse(tmp_path, monkeypatch):
    document = footed_parse(3)
    markdown, parsed = tmp_path / "rules.md", tmp_path / "rules.json"
    markdown.write_text(pages_to_markdown(export_pages(document)), encoding="utf-8", newline="\n")
    document.save_as_json(parsed)
    output = tmp_path / "chunks.jsonl"
    argv = ["chunk", str(markdown), "--json", str(parsed), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    chunk_script.main()
    chunks = jsonl_to_chunks(output.read_text(encoding="utf-8"))
    assert [chunk.page_footers for chunk in chunks] == [
        [f"CORE RULES | PART {n}"] for n in (1, 2, 3)
    ]


def test_chunk_script_refuses_the_json_of_another_parse(tmp_path, monkeypatch):
    # a JSON whose Markdown differs would give its footers to the wrong pages
    document = footed_parse(3)
    markdown, parsed = tmp_path / "rules.md", tmp_path / "rules.json"
    edited = pages_to_markdown(export_pages(document)).replace("Rule 2.", "Rule two.")
    markdown.write_text(edited, encoding="utf-8", newline="\n")
    document.save_as_json(parsed)
    output = tmp_path / "chunks.jsonl"
    argv = ["chunk", str(markdown), "--json", str(parsed), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        chunk_script.main()
    assert not output.exists()


def test_chunk_script_ends_lines_with_lf(tmp_path, monkeypatch):
    # the chunk file is the same bytes on Windows and macOS (plan §15, R23)
    markdown = tmp_path / "rules.md"
    markdown.write_bytes(make_markdown((1, "## A\n\nOne."), (2, "## B\n\nTwo.")).encode())
    output = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(sys, "argv", ["chunk", str(markdown), "--output", str(output)])
    chunk_script.main()
    assert output.read_bytes().count(b"\n") == 2
    assert b"\r" not in output.read_bytes()
    # Markdown other than the rulebook's, without --json: no footers, never the rulebook's
    assert [chunk.page_footers for chunk in jsonl_to_chunks(output.read_text())] == [[], []]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    if not RULES_MARKDOWN.exists():
        pytest.skip("core_rules.md not available: run src.scripts.parse first")
    markdown = RULES_MARKDOWN.read_text(encoding="utf-8")
    return chunk_markdown(markdown, "core_rules", CHUNK_SIZE, CHUNK_OVERLAP, SKIP_PAGES)


@pytest.fixture(scope="module")
def parsed() -> DoclingDocument:
    if not RULES_JSON.exists():
        pytest.skip("core_rules.json not available: run src.scripts.parse first")
    return DoclingDocument.load_from_json(RULES_JSON)


def picture_text(document: DoclingDocument, ref: str) -> str:
    [picture] = [picture for picture in document.pictures if picture.self_ref == ref]
    items = document.iterate_items(
        root=picture, traverse_pictures=True, included_content_layers={ContentLayer.BODY}
    )
    return "\n".join(item.text for item, _ in items if isinstance(item, TextItem))


@pytest.mark.slow
def test_every_chunk_has_pages_and_no_marker(chunks):
    assert all(chunk.pages for chunk in chunks)
    assert not any("<!-- page:" in chunk.text for chunk in chunks)


@pytest.mark.slow
def test_make_attacks_sentence_is_in_one_chunk(chunks):
    # the whole sentence, from its condition to its example: pp. 34-35 break in the middle of it
    sentence = (
        "Note that all of the attacks you have declared targets for are always resolved against"
        " the target units, even if, when you come to resolve an attack, no models in the target"
        " unit of that attack remain within Engagement Range of the attacking model's unit"
        " (because of models being destroyed as the result of other attacks made by the"
        " attacking model's unit, for example)."
    )
    matching = [chunk for chunk in chunks if sentence in " ".join(chunk.text.split())]
    assert matching, "the p. 34-35 sentence was cut"
    assert all(chunk.pages == [34, 35] for chunk in matching)


@pytest.mark.slow
def test_text_is_joined_at_every_page_break_inside_a_sentence(chunks):
    # wherever a page ends mid-sentence and the next starts in lower case, the words on both sides
    # of the break are adjacent in one chunk; a continuation that starts with a capital ("Space
    # Marines ...") cannot be told from a new sentence and is not checked
    lines = page_lines(RULES_MARKDOWN.read_text(encoding="utf-8"), SKIP_PAGES)
    pages = [
        " ".join(line.text for line in page).split()
        for _, page in groupby(lines, key=lambda line: line.page_no)
    ]
    texts = [" ".join(chunk.text.split()) for chunk in chunks]
    breaks = [
        (before[-3:], after[:3])
        for before, after in pairwise(pages)
        if before[-1][-1] not in ".!?:" and after[0][0].islower()
    ]
    assert breaks, "no sentence crosses a page break: the p. 34-35 one is missing"
    for end, start in breaks:
        assert any(" ".join(end + start) in text for text in texts), (end, start)


@pytest.mark.slow
def test_terrain_subsections_name_their_terrain(chunks):
    # pp. 45-48: VISIBILITY is a subsection of every terrain feature
    sections = {chunk.section for chunk in chunks}
    for terrain in ("WOODS", "RUINS", "CRATERS AND RUBBLE"):
        assert f"{terrain} > VISIBILITY" in sections, terrain
    assert "VISIBILITY" not in sections


@pytest.mark.slow
def test_part_titles_keep_their_name(chunks):
    # the parts repeat a name of the core concepts (pp. 5-7); the large-print Introduction of p. 3,
    # inferred above every part, must not appear in any name
    sections = {chunk.section for chunk in chunks}
    for part in ("DATASHEETS", "TERRAIN FEATURES", "MISSIONS"):
        assert part in sections, part
    assert not any(section.startswith("Introduction >") for section in sections)


@pytest.mark.slow
def test_stratagem_name_and_rule_in_same_chunk(chunks):
    matching = [chunk for chunk in chunks if "Roll six D6" in chunk.text]
    assert matching, "the Grenade rule text is missing"
    assert all("GRENADE" in chunk.text for chunk in matching)


@pytest.mark.slow
def test_no_chunk_comes_from_a_skipped_page(chunks):
    assert not any(set(chunk.pages) & SKIP_PAGES for chunk in chunks)


@pytest.mark.slow
def test_picture_markers_change_no_chunk(chunks):
    # chunked without its picture markers, the rulebook gives the same chunks, without pictures
    markdown = RULES_MARKDOWN.read_text(encoding="utf-8")
    unmarked = "\n".join(
        line
        for line in markdown.splitlines()
        if not PICTURE_START_PATTERN.fullmatch(line) and line != PICTURE_END
    )
    plain = chunk_markdown(unmarked, "core_rules", CHUNK_SIZE, CHUNK_OVERLAP, SKIP_PAGES)
    assert [replace(chunk, pictures=[]) for chunk in chunks] == plain


@pytest.mark.slow
def test_every_picture_reference_leads_to_its_picture(chunks, parsed):
    # saved and read back, each reference names a picture of the same parse, on the chunk's pages
    pages = {picture.self_ref: picture.prov[0].page_no for picture in parsed.pictures}
    for chunk in jsonl_to_chunks(chunks_to_jsonl(chunks)):
        for ref in chunk.pictures:
            assert pages.get(ref) in chunk.pages, (chunk.chunk_id, ref)


@pytest.mark.slow
def test_example_datasheet_leads_back_to_the_whole_card(chunks, parsed):
    # p. 37: the example datasheet is cut by its own headings, and its title comes last; both
    # parts lead to the one picture that holds the whole card, title included
    sections = ("WARGEAR OPTIONS", "UNIT COMPOSITION")
    parts = [c for c in chunks if 37 in c.pages and c.section in sections]
    assert len(parts) == 2
    [[ref], [other]] = [part.pictures for part in parts]
    assert ref == other
    assert "EXAMPLE DATASHEET (WARGEAR AND COMPOSITION)" in picture_text(parsed, ref)


@pytest.mark.slow
def test_heading_inside_a_picture_keeps_its_picture(chunks, parsed):
    # p. 38: each numbered heading is printed inside a badge picture, its text outside it; the
    # reference must be that heading's own badge, not another picture of the page
    [chunk] = [c for c in chunks if c.section == "1 DATASHEET NAME"]
    [ref] = chunk.pictures
    assert picture_text(parsed, ref) == "1 DATASHEET NAME"
    assert "Here you will find the name of the unit." in chunk.text


@pytest.mark.slow
def test_chunk_with_several_pictures_lists_each_of_them(chunks, parsed):
    # p. 8: one chunk holds the text of the four visibility examples, each a picture of its own;
    # their first lines tell them apart, so the whole list is checked, in order, without relying
    # on Docling's picture numbers (a missing, extra or wrong reference fails)
    [chunk] = [c for c in chunks if c.pages == [8] and c.section == "UNIT FULLY VISIBLE"]
    first_lines = [picture_text(parsed, ref).splitlines()[0] for ref in chunk.pictures]
    assert first_lines == [
        "Model Visible",
        "Unit Visible",
        "Model Fully Visible",
        "Unit Fully Visible",
    ]


@pytest.mark.slow
def test_chunks_show_the_mission_their_pages_belong_to(parsed):
    markdown = RULES_MARKDOWN.read_text(encoding="utf-8")
    footers = page_footers(parsed)
    chunks = chunk_markdown(markdown, "core_rules", CHUNK_SIZE, CHUNK_OVERLAP, SKIP_PAGES, footers)
    by_section = {chunk.section: chunk for chunk in chunks}
    assert by_section["CAPTURE AND CONTROL"].page_footers == ["CORE RULES | MISSION: ONLY WAR"]
    assert by_section["11 DETERMINE VICTOR"].page_footers == ["CORE RULES | MISSION: ONLY WAR"]
    assert by_section["DETERMINE VICTOR"].page_footers == ["CORE RULES | MISSIONS"]  # p. 57
    assert by_section["SELECT ARMY FACTION"].page_footers == [""]  # p. 55 prints none


def test_picture_context_is_embedded_and_survives_round_trip():
    markdown = (
        "<!-- page: 37 -->\n<!-- picture: #/pictures/77 -->\n"
        "## UNIT COMPOSITION\n1 Sergeant\n<!-- /picture -->"
    )
    [chunk] = chunk_markdown(
        markdown, "rules", 1500, 200, labels={"#/pictures/77": "EXAMPLE DATASHEET"}
    )
    assert chunk.text.startswith("Figure label: EXAMPLE DATASHEET\nSection: UNIT COMPOSITION")
    assert chunk.picture_labels == ["EXAMPLE DATASHEET"]
    assert jsonl_to_chunks(chunks_to_jsonl([chunk])) == [chunk]
