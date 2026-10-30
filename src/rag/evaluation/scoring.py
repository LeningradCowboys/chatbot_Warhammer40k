"""RAGAS's scores of the chat's answers: what a saved answer holds, and its metrics (plan §6.9).

An answer is saved with two sets of contexts, so the scores can be computed, and computed
again, from the saved answers without asking the chat again: the tool outputs the model received
(rules and SQL across its conversation, for faithfulness), and the retriever's
own top-k chunks for the question as asked, the retrieval alone (for context precision and
recall). A saved result is used only for the inputs it was made from: an answer for the same
question, chat settings and k, a score for the same answer, contexts, metric and models (their
fingerprints); otherwise the run is refused.
"""

import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields

import ragas
from langchain_core.messages import ToolMessage
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics.base import SingleTurnMetric

from src.rag.chatbot.agent import ChatAnswer, RunTrace
from src.rag.evaluation.testset import EvalQuestion


@dataclass(frozen=True)
class EvalAnswer:
    """The chat's answer to one question of the test set, with what it was given."""

    question: EvalQuestion
    answer: str  # "" when the model returned no text
    agent_contexts: list[str]  # actual successful tool outputs, including SQL, across the thread
    # the retriever's top `retriever_k` chunks for the question as asked, best first; searched
    # for the evaluation only, never given to the model
    retriever_contexts: list[str]
    retriever_k: int
    trace: RunTrace
    chat_attempts: list[dict]  # the requests the answer took, retries included (`AttemptCounter`)
    embedding_attempts: list[dict]
    source_notes: list[str] = field(default_factory=list)  # application notes, not model prose
    context_policy: str = "legacy-rule-chunks"  # historical files omit these new fields


@dataclass(frozen=True)
class MetricScore:
    """One metric computed on one answer: its value, or why there is none."""

    question_id: str
    metric: str
    value: float | None
    error: str | None
    inputs: str  # `score_fingerprint` of the answer, metric and models it was computed from


def agent_contexts(answer: ChatAnswer) -> list[str]:
    """Successful tool outputs exactly as received, including earlier turns and SQL evidence.

    Stored text is never rebuilt using today's context formatter. Application source notes are
    not model evidence and are excluded; independently retrieved rule contexts stay separate.
    """
    return list(
        dict.fromkeys(
            message.text
            for message in answer.trace.messages
            if isinstance(message, ToolMessage) and message.status != "error"
        )
    )


def _sha256(content: object) -> str:
    text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def chat_fingerprint(question: EvalQuestion, trace: RunTrace) -> str:
    """What an answer was made from: the question with its reference, and the chat's settings
    (prompts, model settings, tool declarations), as its trace holds them."""
    chat = [trace.system_prompt, trace.final_prompt, trace.final_instruction]
    return _sha256([asdict(question), chat, trace.model_settings, trace.tools])


def metric_contexts(metric: SingleTurnMetric, answer: EvalAnswer) -> tuple[str, list[str]]:
    """The contexts a metric is given, and where they come from: the retriever's own results for
    context precision and recall, what the model received for faithfulness, none for the metrics
    that judge the answer against the question or the reference."""
    if isinstance(metric, LLMContextPrecisionWithReference | LLMContextRecall):
        return f"retriever top {answer.retriever_k}", answer.retriever_contexts
    if isinstance(metric, Faithfulness):
        return "agent", answer.agent_contexts
    return "none", []


def score_fingerprint(answer: EvalAnswer, metric: SingleTurnMetric, models: list[str]) -> str:
    """What a score is computed from: the question, reference and answer, the contexts the metric
    is given with their source, the metric with its settings (its fields of plain values, such as
    relevancy's strictness), the models and RAGAS's version."""
    settings = {
        f.name: value
        for f in fields(metric)
        if isinstance(value := getattr(metric, f.name), int | float | str | bool)
    }
    sample = [asdict(answer.question), answer.answer, metric_contexts(metric, answer)]
    return _sha256([sample, type(metric).__name__, settings, models, ragas.__version__])


def answers_problem(
    questions: list[EvalQuestion], answers: list[EvalAnswer], trace: RunTrace, k: int
) -> str | None:
    """Why the saved answers cannot be continued with these questions, this chat and this k, or
    None.

    Args:
        questions: The test set.
        answers: The answers saved in the run.
        trace: A trace of the current chat (`ChatAgent.trace()`), for its settings.
        k: The number of chunks the retriever returns for the evaluation (`TOP_K`).
    """
    current = {question.question_id: question for question in questions}
    for answer in answers:
        question = current.get(answer.question.question_id)
        if question is None:
            continue
        if chat_fingerprint(answer.question, answer.trace) != chat_fingerprint(question, trace):
            return (
                f"the saved answer to {question.question_id} answers another version of the "
                "question, or was given with other chat settings"
            )
        if answer.retriever_k != k:
            return f"the saved answer to {question.question_id} kept the top {answer.retriever_k}"
    return None


def scores_problem(
    questions: list[EvalQuestion],
    answers: list[EvalAnswer],
    metrics: list[SingleTurnMetric],
    models: list[str],
    scores: dict[tuple[str, str], MetricScore],
) -> str | None:
    """Why the saved answers and scores cannot be scored further with this test set, these
    metrics and models, or None."""
    current = {question.question_id: question for question in questions}
    for answer in answers:
        question_id = answer.question.question_id
        if question_id in current and answer.question != current[question_id]:
            return f"the saved answer to {question_id} answers another version of the question"
        for metric in metrics:
            saved = scores.get((question_id, metric.name))
            if saved is not None and saved.inputs != score_fingerprint(answer, metric, models):
                return (
                    f"the saved {metric.name} of {question_id} was computed on another answer, "
                    "or with other metric settings or models"
                )
    return None


def build_metrics(llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings) -> list[SingleTurnMetric]:
    """RAGAS's metrics of an answer: faithfulness to the contexts the model received, relevancy
    to the question, the precision and recall of the retriever's contexts against the reference,
    and the answer's factual correctness against the reference."""
    return [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        FactualCorrectness(llm=llm),
    ]


def score(metric: SingleTurnMetric, answer: EvalAnswer) -> float:
    """Compute one metric on one saved answer, given the contexts of `metric_contexts`.

    Raises:
        ValueError: RAGAS returned no number (NaN), for example when no statement could be
            drawn from the answer.
    """
    sample = SingleTurnSample(
        user_input=answer.question.question,
        response=answer.answer,
        retrieved_contexts=metric_contexts(metric, answer)[1],
        reference=answer.question.reference,
    )
    value = float(asyncio.run(metric.single_turn_ascore(sample)))
    if math.isnan(value):
        raise ValueError(f"{metric.name}: RAGAS returned no score")
    return value
