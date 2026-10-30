"""Save and load the files of the RAGAS evaluation (plan §6.9).

Each file is written as soon as its work is done, so a run stopped midway keeps what it finished
and the next run continues from it:

- the knowledge graph, in RAGAS's own JSON, and its progress: what it is built from and the steps
  applied;
- the test set: one question per line (JSON Lines), which a person checks and may shorten;
- the answers: one JSON file per question, named by its ID (`QUESTION_ID`, checked when the test set
  is read; the file's path is checked again when it is written);
- the scores: one line per metric computed on an answer, appended; a later line for the same
  question and metric replaces an earlier one.

A file that replaces an earlier version is written beside it and then moved over it, so an
interrupted write leaves the earlier version whole.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ragas.testset.graph import KnowledgeGraph

from src.rag.chatbot.agent import RunTrace
from src.rag.evaluation.scoring import EvalAnswer, MetricScore
from src.rag.evaluation.testset import QUESTION_ID, EvalQuestion


@dataclass(frozen=True)
class GraphProgress:
    """How far a saved knowledge graph is built."""

    fingerprint: str  # `graph_fingerprint` of what it is built from
    steps_done: int  # the transform steps applied, in order


def _replace_text(path: Path, text: str) -> None:
    """Write `text` beside `path`, then move it over `path`."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)


def answer_inputs(files: dict[str, Path], embedding_model: str) -> dict[str, str]:
    """Fingerprint the data, source and dependency files used to produce answers.

    Args:
        files: Stable project-relative names mapped to the actual files.
        embedding_model: Runtime query embedding model, which is not a chat-model setting.

    Returns:
        File hashes and the embedding model; never API credentials.
    """
    inputs = {"embedding_model": embedding_model}
    for name, path in sorted(files.items()):
        content = path.read_bytes()
        if path.suffix in {".py", ".md", ".json", ".jsonl", ".lock"}:
            content = content.replace(b"\r\n", b"\n")
        inputs[f"file:{name}"] = hashlib.sha256(content).hexdigest()
    return inputs


def save_answer_inputs(path: Path, answers_dir: Path, inputs: dict[str, str]) -> None:
    """Record a new answer run's inputs, or refuse continuation under different inputs.

    Args:
        path: The run's input manifest.
        answers_dir: Existing answer files; legacy runs without a manifest cannot be extended.
        inputs: Current file hashes and embedding model.

    Raises:
        ValueError: The manifest differs or answers already exist without one. Historical
            answers can still be scored without this function.
    """
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != inputs:
            raise ValueError("the run's data, retrieval implementation or embedding model changed")
    elif any(answers_dir.glob("*.json")):
        raise ValueError(
            "the saved answers have no input manifest; their retrieval inputs are unknown"
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace_text(path, json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")


def save_knowledge_graph(
    graph_path: Path, progress_path: Path, kg: KnowledgeGraph, progress: GraphProgress
) -> None:
    """Write the graph, then its progress.

    A run stopped between the two leaves a graph ahead of its progress: the step it records as
    not done is run again, and its extractors skip the nodes it already reached.
    """
    partial = graph_path.with_name(graph_path.name + ".partial")
    kg.save(partial)
    partial.replace(graph_path)
    _replace_text(progress_path, json.dumps(asdict(progress), indent=2) + "\n")


def load_knowledge_graph(
    graph_path: Path, progress_path: Path
) -> tuple[KnowledgeGraph, GraphProgress]:
    """Read back a graph and its progress written by `save_knowledge_graph`."""
    progress = GraphProgress(**json.loads(progress_path.read_text(encoding="utf-8")))
    return KnowledgeGraph.load(graph_path), progress


def save_testset(path: Path, questions: list[EvalQuestion]) -> None:
    """Write the questions, one JSON object per line."""
    lines = [json.dumps(asdict(question), ensure_ascii=False) + "\n" for question in questions]
    _replace_text(path, "".join(lines))


def load_testset(path: Path) -> list[EvalQuestion]:
    """Read the questions back, in order; blank lines are skipped.

    Raises:
        ValueError: A question's ID is not of the form `q001`, or two questions share one.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion(**json.loads(line)) for line in lines if line.strip()]
    seen = set()
    for question in questions:
        if not QUESTION_ID.fullmatch(question.question_id) or question.question_id in seen:
            raise ValueError(f"question ID {question.question_id!r} is not a new ID like 'q001'")
        seen.add(question.question_id)
    return questions


def save_answer(directory: Path, answer: EvalAnswer) -> None:
    """Write one answer to `<directory>/<question_id>.json`.

    Raises:
        ValueError: That path is not a file directly in `directory`.
    """
    path = directory / f"{answer.question.question_id}.json"
    if path.resolve().parent != directory.resolve():
        raise ValueError(f"{path} is not a file of {directory}")
    data = {**asdict(answer), "trace": answer.trace.to_dict()}
    text = json.dumps(data, ensure_ascii=False, indent=1, default=str) + "\n"
    _replace_text(path, text)


def load_answers(directory: Path) -> list[EvalAnswer]:
    """Read back every answer in `directory`, by question ID; none if it does not exist."""
    answers = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["question"] = EvalQuestion(**data["question"])
        data["trace"] = RunTrace.from_dict(data["trace"])
        answers.append(EvalAnswer(**data))
    return answers


def append_score(path: Path, score: MetricScore) -> None:
    """Add one score as a line at the end of the file."""
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")


def load_scores(path: Path) -> dict[tuple[str, str], MetricScore]:
    """The latest score of each (question ID, metric); none if the file does not exist."""
    if not path.is_file():
        return {}
    scores = (
        MetricScore(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
    )
    return {(score.question_id, score.metric): score for score in scores}
