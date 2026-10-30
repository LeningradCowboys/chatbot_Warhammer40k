"""The RAGAS test set: the rulebook's sections, RAGAS's knowledge graph of them, its questions.

RAGAS builds the graph with its default transforms (summaries, themes, entities, embeddings, then
the relationships between nodes), then writes the questions from the graph (plan §6.9). The graph
takes the most model calls, so it is built one step at a time and can be saved between steps;
each step that calls the model skips the nodes it already reached (an extractor those holding its
property, the node filter those it kept), so a step stopped midway and run again only asks for the
nodes it had not reached.
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from itertools import groupby

import ragas
from langchain_core.documents import Document
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import Extractor, Parallel, apply_transforms, default_transforms
from ragas.testset.transforms.base import BaseGraphTransformation
from ragas.testset.transforms.filters import CustomNodeFilter

from src.rag.chunker.markdown_chunker import split_blocks

# One call at a time: the chat model's quota allows one request every few seconds
# (`MIN_SECONDS_BETWEEN_CHAT_REQUESTS`), so concurrent calls would only queue in the request
# budget, and the last of them would pass RAGAS's timeout while still waiting to be sent.
RUN_CONFIG = RunConfig(max_workers=1)
QUESTION_ID = re.compile(r"q\d{3,}")  # "q001": also the name of the question's answer file
FILTER_KEPT = "kept_by_filter"  # the property `CheckedNodeFilter` gives each node it keeps


@dataclass(frozen=True)
class EvalQuestion:
    """One question of the test set, with what RAGAS wrote it from."""

    question_id: str  # "q001": its place in the generated test set
    question: str
    reference: str  # RAGAS's reference answer
    reference_contexts: list[str]  # the passages RAGAS wrote the question and reference from
    synthesizer: str  # RAGAS's kind of question: single-hop, multi-hop, ...


def section_documents(markdown: str, skip_pages: frozenset[int]) -> list[Document]:
    """The rulebook's sections as LangChain documents, in document order.

    A section is the text under one heading (`split_blocks`), its tables included; a section with
    no text is left out. Each document starts with its section's name, as the chunks do.

    Args:
        markdown: The page-tagged Markdown of the parser.
        skip_pages: The pages left out of the index (`SKIP_PAGES`).

    Returns:
        One document per section, with the section's name and pages as metadata.
    """
    documents = []
    for section, blocks in groupby(split_blocks(markdown, skip_pages), key=lambda b: b.section):
        lines = [line for block in blocks for line in block.lines]
        text = "\n".join(line.text for line in lines).strip()
        if text:
            content = f"Section: {section}\n{text}" if section else text
            pages = sorted({line.page_no for line in lines})
            documents.append(Document(content, metadata={"section": section, "pages": pages}))
    return documents


def select_sections(
    documents: list[Document], sections: tuple[str | tuple[str, ...], ...]
) -> list[Document]:
    """The documents of the named sections, in document order.

    A group of names (a tuple) gives one document: its sections joined in document order, each
    under its own `Section:` line, so that a rule is generated with the context it belongs to (a
    mission's scoring with the mission, a step with the sequence that orders it).

    Raises:
        ValueError: A name is not the name of one section (the rulebook was parsed again).
    """
    groups = [(entry,) if isinstance(entry, str) else entry for entry in sections]
    names = [name for group in groups for name in group]
    found = [doc for doc in documents if doc.metadata["section"] in names]
    missing = set(names) - {doc.metadata["section"] for doc in found}
    if missing or len(found) != len(names):
        raise ValueError(f"not one section each: {sorted(missing) or names}")
    position = {doc.metadata["section"]: index for index, doc in enumerate(found)}
    selected = []
    for group in sorted(groups, key=lambda group: min(position[name] for name in group)):
        parts = [doc for doc in found if doc.metadata["section"] in group]
        selected.append(
            Document(
                "\n\n".join(doc.page_content for doc in parts),
                metadata={
                    "section": " + ".join(doc.metadata["section"] for doc in parts),
                    "pages": sorted({page for doc in parts for page in doc.metadata["pages"]}),
                },
            )
        )
    return selected


def knowledge_graph(documents: list[Document]) -> KnowledgeGraph:
    """A graph with one document node per document, as RAGAS's generator makes it."""
    return KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
            )
            for doc in documents
        ]
    )


def transform_steps(
    documents: list[Document], llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings
) -> list[BaseGraphTransformation]:
    """RAGAS's default transforms for these documents, one step per transform.

    RAGAS applies the transforms of a `Parallel` one after the other, so flattening them keeps
    its order. Each step that calls the model skips the nodes it already reached.
    """
    steps = []
    for transform in default_transforms(documents, llm, embeddings):
        steps.extend(transform.transformations if isinstance(transform, Parallel) else [transform])
    return [_resumable(step) for step in steps]


@dataclass
class CheckedNodeFilter(CustomNodeFilter):
    """RAGAS's node filter, which also marks each node it keeps with `FILTER_KEPT`. The nodes it
    removes are gone from the graph, so a filter run again checks only the nodes left unmarked."""

    async def custom_filter(self, node: Node, kg: KnowledgeGraph) -> bool:
        removed = await super().custom_filter(node, kg)
        if not removed:
            node.add_property(FILTER_KEPT, True)
        return removed


def _resumable(step: BaseGraphTransformation) -> BaseGraphTransformation:
    """The step, skipping the nodes it already reached: an extractor those holding its property,
    the node filter (as a `CheckedNodeFilter` with the same settings) those it kept."""
    if isinstance(step, CustomNodeFilter):
        settings = {
            f.name: getattr(step, f.name) for f in fields(step) if f.init and f.name != "name"
        }
        step = CheckedNodeFilter(**settings)
        step.filter_nodes = _missing(step.filter_nodes, FILTER_KEPT)
    elif isinstance(step, Extractor):
        step.filter_nodes = _missing(step.filter_nodes, step.property_name)
    return step


def _missing(filter_nodes: Callable[[Node], bool], property_name: str) -> Callable[[Node], bool]:
    """`filter_nodes`, without the nodes that already hold `property_name`."""
    return lambda node: filter_nodes(node) and node.get_property(property_name) is None


def step_name(step: BaseGraphTransformation) -> str:
    """The step's class and, for an extractor, the property it extracts."""
    name = type(step).__name__
    return f"{name}({step.property_name})" if isinstance(step, Extractor) else name


def graph_fingerprint(
    documents: list[Document], steps: list[BaseGraphTransformation], models: list[str]
) -> str:
    """What a saved graph was built from: the documents, the steps, the models and RAGAS's
    version. A graph is continued only by a run with the same fingerprint."""
    content = {
        "documents": [[doc.page_content, doc.metadata] for doc in documents],
        "steps": [step_name(step) for step in steps],
        "models": models,
        "ragas": ragas.__version__,
    }
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def apply_step(kg: KnowledgeGraph, step: BaseGraphTransformation) -> None:
    """Apply one step to the graph, in place, one model call at a time.

    RAGAS runs every node of the step and then raises the first error: the nodes that succeeded
    keep their properties in `kg`, which the caller can save.
    """
    apply_transforms(kg, step, RUN_CONFIG)


def generate_questions(
    kg: KnowledgeGraph, llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings, size: int
) -> list[EvalQuestion]:
    """RAGAS's test set from a built graph: personas, scenarios, then the questions.

    Args:
        kg: The graph, every step applied.
        llm: The model RAGAS writes with.
        embeddings: The embeddings RAGAS compares with.
        size: The number of questions.

    Returns:
        The questions, numbered in the order RAGAS returned them.
    """
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
    testset = generator.generate(size, run_config=RUN_CONFIG)
    return [
        EvalQuestion(
            question_id=f"q{number:03d}",
            question=sample.eval_sample.user_input,
            reference=sample.eval_sample.reference,
            reference_contexts=list(sample.eval_sample.reference_contexts),
            synthesizer=sample.synthesizer_name,
        )
        for number, sample in enumerate(testset.samples, 1)
    ]
