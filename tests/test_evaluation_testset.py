"""The RAGAS test set, offline: the rulebook's sections, RAGAS's graph built step by step and
continued after a stop, and the questions, with the fake model answering RAGAS's prompts; the
script itself, with and without `--graph-only`."""

import pytest

from src.rag.config import RULES_MARKDOWN, SKIP_PAGES, TESTSET_SECTIONS, Settings

pytest.importorskip("ragas", reason="the evaluation group is not installed: uv sync --group eval")

from ragas.utils import num_tokens_from_string  # noqa: E402

from src.rag.evaluation.ragas_adapters import (  # noqa: E402
    EmbedderRagasEmbeddings,
    GeminiRagasLLM,
)
from src.rag.evaluation.testset import (  # noqa: E402
    FILTER_KEPT,
    generate_questions,
    graph_fingerprint,
    knowledge_graph,
    section_documents,
    select_sections,
    step_name,
    transform_steps,
)
from src.rag.llm.budget import BudgetExhausted  # noqa: E402
from src.rag.storage.evaluation_runs import (  # noqa: E402
    GraphProgress,
    load_knowledge_graph,
    load_testset,
)
from src.scripts import generate_testset  # noqa: E402
from src.scripts.generate_testset import build_graph  # noqa: E402
from tests.fakes import FakeEmbedder, RagasChatModel  # noqa: E402

# Long enough for RAGAS's default transforms of sections of 101 to 500 tokens.
RULE = "When a unit makes an attack, roll one D6 for each attack and compare it with the target. "
SECTIONS = ["DICE", "RE-ROLLS", "SEQUENCING", "COMMAND"]
MARKDOWN = "\n".join(
    line
    for page, section in enumerate(SECTIONS, 1)
    for line in [f"<!-- page: {page} -->", f"## {section}", "", RULE * 8, ""]
)
STEPS = [
    "SummaryExtractor(summary)",
    "CheckedNodeFilter",
    "EmbeddingExtractor(summary_embedding)",
    "ThemesExtractor(themes)",
    "NERExtractor(entities)",
    "CosineSimilarityBuilder",
    "OverlapScoreBuilder",
]
# The model calls of the whole graph: four documents, four LLM steps.
GRAPH_CALLS = 16


def _adapters(model: RagasChatModel) -> tuple[GeminiRagasLLM, EmbedderRagasEmbeddings]:
    return GeminiRagasLLM(model), EmbedderRagasEmbeddings(FakeEmbedder())


def test_sections_are_documents_with_their_pages():
    markdown = "\n".join(
        [
            "<!-- page: 3 -->",
            "## GRENADE",
            "Roll six D6.",
            "<!-- page: 4 -->",
            "Skipped page.",
            "<!-- page: 5 -->",
            "| ROLL | RESULT |",
            "## EMPTY",
            "",
            "## COVER",
            "+1 to saves.",
        ]
    )

    documents = section_documents(markdown, frozenset({4}))

    assert [(doc.page_content, doc.metadata) for doc in documents] == [
        (
            "Section: GRENADE\nRoll six D6.\n| ROLL | RESULT |",
            {"section": "GRENADE", "pages": [3, 5]},
        ),
        ("Section: COVER\n+1 to saves.", {"section": "COVER", "pages": [5]}),
    ]


def test_the_steps_are_ragas_default_transforms_one_at_a_time():
    documents = section_documents(MARKDOWN, frozenset())
    steps = transform_steps(documents, *_adapters(RagasChatModel()))
    assert [step_name(step) for step in steps] == STEPS


def test_the_questions_are_written_from_the_built_graph(tmp_path):
    model = RagasChatModel()
    llm, embeddings = _adapters(model)
    documents = section_documents(MARKDOWN, frozenset())
    kg = knowledge_graph(documents)
    progress = GraphProgress(graph_fingerprint(documents, [], []), 0)
    steps = transform_steps(documents, llm, embeddings)
    build_graph(kg, steps, progress, tmp_path / "graph.json", tmp_path / "progress.json")

    questions = generate_questions(kg, llm, embeddings, 4)

    assert len(model.titles) > GRAPH_CALLS  # personas, scenarios and questions after the graph
    assert questions[0].question_id == "q001"
    assert [q.question_id for q in questions] == [f"q{n:03d}" for n in range(1, len(questions) + 1)]
    assert all(q.question and q.reference and q.reference_contexts for q in questions)
    assert {q.synthesizer for q in questions} >= {
        "single_hop_specific_query_synthesizer",
        "multi_hop_abstract_query_synthesizer",
    }
    assert RULE.strip() in questions[0].reference_contexts[0]


def test_a_graph_stopped_midway_is_continued_without_asking_again(tmp_path):
    graph_path, progress_path = tmp_path / "graph.json", tmp_path / "progress.json"
    documents = section_documents(MARKDOWN, frozenset())
    progress = GraphProgress("fingerprint", 0)
    stopped = RagasChatModel(limit=2)  # two summaries, then a spent budget

    with pytest.raises(BudgetExhausted):
        steps = transform_steps(documents, *_adapters(stopped))
        build_graph(knowledge_graph(documents), steps, progress, graph_path, progress_path)

    kg, saved = load_knowledge_graph(graph_path, progress_path)
    assert saved == progress  # the first step is not done
    assert sum(node.get_property("summary") is not None for node in kg.nodes) == 2

    resumed = RagasChatModel()
    build_graph(
        kg, transform_steps(documents, *_adapters(resumed)), saved, graph_path, progress_path
    )

    assert resumed.titles.count("StringIO") == 2  # only the two summaries missing
    assert len(stopped.titles) + len(resumed.titles) == GRAPH_CALLS
    kg, saved = load_knowledge_graph(graph_path, progress_path)
    assert saved.steps_done == len(STEPS)
    assert all(node.get_property("entities") for node in kg.nodes)
    assert kg.relationships


def test_the_fingerprint_follows_the_documents_and_the_models():
    documents = section_documents(MARKDOWN, frozenset())
    steps = transform_steps(documents, *_adapters(RagasChatModel()))
    fingerprint = graph_fingerprint(documents, steps, ["chat", "embedding"])

    assert graph_fingerprint(documents, steps, ["chat", "embedding"]) == fingerprint
    assert graph_fingerprint(documents[1:], steps, ["chat", "embedding"]) != fingerprint
    assert graph_fingerprint(documents, steps, ["other chat", "embedding"]) != fingerprint


def test_a_graph_built_over_many_small_budgets_moves_on_each_time(tmp_path):
    graph_path, progress_path = tmp_path / "graph.json", tmp_path / "progress.json"
    documents = section_documents(MARKDOWN, frozenset())
    kg, progress = knowledge_graph(documents), GraphProgress("fingerprint", 0)
    runs = calls = 0
    while progress.steps_done < len(STEPS) and runs < GRAPH_CALLS:
        model = RagasChatModel(limit=2)  # fewer requests than nodes, each step included
        runs += 1
        try:
            build_graph(
                kg,
                transform_steps(documents, *_adapters(model)),
                progress,
                graph_path,
                progress_path,
            )
        except BudgetExhausted:
            pass
        calls += len(model.titles)
        kg, progress = load_knowledge_graph(graph_path, progress_path)
        if progress.steps_done == 1:  # in the filter: the nodes checked so far are marked
            assert sum(bool(node.get_property(FILTER_KEPT)) for node in kg.nodes) in {0, 2, 4}

    assert progress.steps_done == len(STEPS)
    assert (runs, calls) == (GRAPH_CALLS // 2, GRAPH_CALLS)  # two new calls in every run
    assert len(kg.nodes) == len(SECTIONS)


def test_the_sections_of_the_test_set_are_chosen_by_name_in_document_order():
    documents = section_documents(MARKDOWN, frozenset())

    chosen = select_sections(documents, ("SEQUENCING", "DICE"))

    assert [doc.metadata["section"] for doc in chosen] == ["DICE", "SEQUENCING"]
    with pytest.raises(ValueError, match="TERRAIN"):
        select_sections(documents, ("DICE", "TERRAIN"))


def test_a_group_of_sections_is_one_document_in_document_order():
    documents = {doc.metadata["section"]: doc for doc in section_documents(MARKDOWN, frozenset())}

    chosen = select_sections(list(documents.values()), ("COMMAND", ("SEQUENCING", "DICE")))

    assert [(doc.metadata["section"], doc.metadata["pages"]) for doc in chosen] == [
        ("DICE + SEQUENCING", [1, 3]),
        ("COMMAND", [4]),
    ]
    dice, sequencing = documents["DICE"].page_content, documents["SEQUENCING"].page_content
    assert chosen[0].page_content == f"{dice}\n\n{sequencing}"  # each under its own Section: line
    with pytest.raises(ValueError, match="TERRAIN"):
        select_sections(list(documents.values()), (("DICE", "TERRAIN"),))


@pytest.mark.slow
def test_the_configured_sections_are_in_the_parsed_rulebook_and_long_enough_to_summarise():
    if not RULES_MARKDOWN.is_file():
        pytest.skip(f"{RULES_MARKDOWN} has not been parsed")
    documents = section_documents(RULES_MARKDOWN.read_text(encoding="utf-8"), SKIP_PAGES)

    chosen = select_sections(documents, TESTSET_SECTIONS)

    pages = [[11], [11], [13], [14], [14], [19], [20], [29], [32], [33, 35, 36], [41], [58], [59]]
    assert [doc.metadata["pages"] for doc in chosen] == pages
    assert all(num_tokens_from_string(doc.page_content) > 100 for doc in chosen)
    scoring = chosen[-1].page_content  # the scoring rule with the mission it belongs to
    assert scoring.startswith("Section: ONLY WAR") and "Section: CAPTURE AND CONTROL" in scoring


class Script:
    """`generate_testset.main()` on the test's sections, its files in a temporary folder, the fake
    model and embedder in place of Gemini's; each call of `generate_questions` is recorded."""

    def __init__(self, tmp_path, monkeypatch):
        self.monkeypatch = monkeypatch
        self.graph, self.progress = tmp_path / "graph.json", tmp_path / "progress.json"
        self.testset = tmp_path / "testset.jsonl"
        self.generated = []
        markdown = tmp_path / "core_rules.md"
        markdown.write_text(MARKDOWN, encoding="utf-8")
        for name, value in {
            "RULES_MARKDOWN": markdown,
            "SKIP_PAGES": frozenset(),
            "TESTSET_SECTIONS": tuple(SECTIONS),
            "KNOWLEDGE_GRAPH": self.graph,
            "KNOWLEDGE_GRAPH_PROGRESS": self.progress,
            "TESTSET": self.testset,
        }.items():
            monkeypatch.setattr(generate_testset, name, value)
        settings = Settings(google_api_key="test", _env_file=None)
        monkeypatch.setattr(generate_testset, "load_settings", lambda parser: settings)
        monkeypatch.setattr(generate_testset, "GeminiEmbedder", lambda *args: FakeEmbedder())
        real = generate_testset.generate_questions

        def generate(*args, **kwargs):
            self.generated.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(generate_testset, "generate_questions", generate)

    def run(self, model: RagasChatModel, *flags: str) -> None:
        self.monkeypatch.setattr(generate_testset, "create_chat_model", lambda *args: model)
        budgets = ["--chat-budget", "100", "--embedding-budget", "100"]
        self.monkeypatch.setattr("sys.argv", ["generate_testset", *flags, *budgets])
        generate_testset.main()

    @property
    def steps_done(self) -> int:
        return load_knowledge_graph(self.graph, self.progress)[1].steps_done


@pytest.fixture
def script(tmp_path, monkeypatch) -> Script:
    return Script(tmp_path, monkeypatch)


def test_graph_only_builds_a_new_graph_and_stops_before_the_questions(script):
    model = RagasChatModel()

    script.run(model, "--graph-only")

    assert script.steps_done == len(STEPS)
    assert len(model.titles) == GRAPH_CALLS
    assert script.generated == []
    assert not script.testset.exists()


def test_graph_only_continues_a_saved_graph_and_stops_before_the_questions(script):
    stopped = RagasChatModel(limit=2)
    with pytest.raises(BudgetExhausted):
        script.run(stopped, "--graph-only")
    assert script.steps_done == 0

    resumed = RagasChatModel()
    script.run(resumed, "--graph-only")

    assert script.steps_done == len(STEPS)
    assert len(stopped.titles) + len(resumed.titles) == GRAPH_CALLS
    assert script.generated == []
    assert not script.testset.exists()


def test_graph_only_on_a_built_graph_sends_nothing(script):
    script.run(RagasChatModel(), "--graph-only")
    model = RagasChatModel()

    script.run(model, "--graph-only")

    assert model.titles == []
    assert script.generated == []
    assert not script.testset.exists()


def test_without_graph_only_the_questions_follow_the_graph(script):
    model = RagasChatModel()

    script.run(model)

    assert len(script.generated) == 1
    assert len(model.titles) > GRAPH_CALLS
    assert [q.question_id for q in load_testset(script.testset)][:1] == ["q001"]


def test_the_questions_of_a_graph_built_with_graph_only_ask_nothing_of_the_graph(script):
    script.run(RagasChatModel(), "--graph-only")
    model = RagasChatModel()

    script.run(model)

    graph_schemas = {"StringIO", "QuestionPotentialOutput", "ThemesAndConcepts", "NEROutput"}
    assert model.titles and not graph_schemas & set(model.titles)
    assert len(script.generated) == 1
    assert load_testset(script.testset)
