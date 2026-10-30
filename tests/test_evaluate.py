"""The evaluation, offline: the chat's answers saved with their two sets of contexts, RAGAS's
scores computed from the saved answers, each metric given its own contexts, runs stopped midway
continued without asking again, and saved results refused when their inputs have changed."""

import json
from dataclasses import dataclass, replace

import httpx
import pytest

pytest.importorskip("ragas", reason="the evaluation group is not installed: uv sync --group eval")

from langchain_core.messages import AIMessage  # noqa: E402
from ragas.exceptions import RagasOutputParserException  # noqa: E402
from ragas.metrics import ResponseRelevancy  # noqa: E402

from src.rag.chatbot.agent import ChatAgent, chunk_block  # noqa: E402
from src.rag.chatbot.prompts import APPLICABILITY_RULES  # noqa: E402
from src.rag.chunker.markdown_chunker import Chunk  # noqa: E402
from src.rag.config import EMBEDDING_DIM, TOP_K, Settings  # noqa: E402
from src.rag.evaluation.ragas_adapters import (  # noqa: E402
    EmbedderRagasEmbeddings,
    GeminiRagasLLM,
)
from src.rag.evaluation.scoring import (  # noqa: E402
    EvalAnswer,
    MetricScore,
    agent_contexts,
    answers_problem,
    build_metrics,
    score_fingerprint,
    scores_problem,
)
from src.rag.evaluation.testset import EvalQuestion  # noqa: E402
from src.rag.llm.budget import AttemptCounter, BudgetExhausted, RequestBudget  # noqa: E402
from src.rag.llm.gemini import GeminiEmbedder, create_chat_model  # noqa: E402
from src.rag.retriever.rules_retriever import RetrievedChunk, RulesRetriever  # noqa: E402
from src.rag.retriever.sql_retriever import SqlRetriever  # noqa: E402
from src.rag.storage.evaluation_runs import (  # noqa: E402
    answer_inputs,
    append_score,
    load_answers,
    load_scores,
    load_testset,
    save_answer,
    save_answer_inputs,
    save_testset,
)
from src.scripts.evaluate import answer_questions, log_means, score_answers  # noqa: E402
from tests.fakes import (  # noqa: E402
    COVER,
    GRENADE,
    FakeChatModel,
    FakeEmbedder,
    FakeRules,
    RagasChatModel,
    ragas_reply,
    tool_call,
)

QUESTIONS = [
    EvalQuestion("q001", "How do grenades work?", "Roll six D6.", ["Section: GRENADE"], "single"),
    EvalQuestion("q002", "What does cover give?", "+1 to saves.", ["Section: COVER"], "single"),
]
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
    "factual_correctness",
]
MODELS = ["chat-model", "embedding-model"]
# RAGAS's calls for one answer with two contexts: faithfulness 2, relevancy 3 (its strictness),
# context precision 1 per context, context recall 1, factual correctness 4 (the claims of the
# answer and of the reference, each checked against the other).
CALLS_PER_ANSWER = 12
# What the retriever returns for the question as asked, apart from the chat: other chunks than
# those the chat's searches get (GRENADE and COVER), so that a metric given the wrong set shows.
RANGE = Chunk("core_rules-0020", "Section: RANGE\nMeasure in inches.", [7], "RANGE", "text", [])
DEPLOY = Chunk("core_rules-0230", "Section: DEPLOY\nSet up units.", [57], "DEPLOY", "text", [])


def _replies(answer: str) -> list[AIMessage]:
    """The chat model's turns for one question: a search of the rules, then the answer."""
    search = tool_call("search_rules", {"query": "grenade"}, "c1")
    return [AIMessage("", tool_calls=[search]), AIMessage(answer)]


def _agent(replies: list[AIMessage], tiny_db, system_prompt: str = "S") -> ChatAgent:
    model = FakeChatModel(replies=replies)
    return ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), system_prompt)


def _direct() -> FakeRules:
    """The retriever the answer step searches with the question as asked."""
    return FakeRules([RetrievedChunk(RANGE, 0.9), RetrievedChunk(DEPLOY, 0.5)])


def _answer(question: EvalQuestion, text: str, tiny_db) -> EvalAnswer:
    """A saved answer as the answer step makes it."""
    answers_dir = tiny_db.parent / f"answers-{question.question_id}-{len(text)}"
    agent = _agent(_replies(text), tiny_db)
    answer_questions(agent, _direct(), [question], answers_dir, *_counters())
    return load_answers(answers_dir)[0]


def _counters() -> tuple[AttemptCounter, AttemptCounter]:
    return AttemptCounter(), AttemptCounter()


def _adapters(model) -> tuple[GeminiRagasLLM, EmbedderRagasEmbeddings]:
    return GeminiRagasLLM(model), EmbedderRagasEmbeddings(FakeEmbedder())


def _score(answers, metrics, scores_path) -> None:
    score_answers(answers, metrics, MODELS, scores_path)


# --- answers ---------------------------------------------------------------------------------


def test_contexts_keep_sql_and_prior_turn_outputs_without_reformatting(tiny_db):
    query = tool_call("run_sql", {"sql": "SELECT name FROM datasheets"}, "q")
    agent = _agent(
        [
            *_replies("First"),
            AIMessage("", tool_calls=[query]),
            AIMessage("Second"),
            AIMessage("Third"),
        ],
        tiny_db,
    )
    first = agent.ask("grenades")
    original = first.tool_calls[0].output
    second = agent.ask("unit names")
    sql_output = second.tool_calls[0].output
    third = agent.ask("follow-up")
    assert agent_contexts(third) == [original, sql_output]
    assert not third.tool_calls


def test_legacy_answer_files_load_without_replacing_their_saved_contexts(tmp_path, tiny_db):
    answer = _answer(QUESTIONS[0], "Answer", tiny_db)
    save_answer(tmp_path, answer)
    path = tmp_path / "q001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("context_policy")
    data.pop("source_notes")
    data["agent_contexts"] = ["Historical exact context"]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_answers(tmp_path)[0]
    assert loaded.context_policy == "legacy-rule-chunks"
    assert loaded.agent_contexts == ["Historical exact context"]
    assert loaded.source_notes == []


def test_each_answer_is_saved_with_the_chunks_received_and_the_retriever_s_own(tmp_path, tiny_db):
    model = FakeChatModel(replies=_replies("Roll six D6.") + _replies("+1 to saves."))
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")
    direct = _direct()

    answer_questions(agent, direct, QUESTIONS, tmp_path, *_counters())

    answers = load_answers(tmp_path)
    assert [(a.question, a.answer) for a in answers] == [
        (QUESTIONS[0], "Roll six D6."),
        (QUESTIONS[1], "+1 to saves."),
    ]
    assert answers[0].agent_contexts == [
        chunk_block(GRENADE) + "\n\n" + chunk_block(COVER) + "\n\n" + APPLICABILITY_RULES
    ]
    assert answers[0].context_policy == "conversation-tool-outputs-v1"
    assert answers[0].retriever_contexts == [chunk_block(RANGE), chunk_block(DEPLOY)]  # in order
    assert {a.retriever_k for a in answers} == {TOP_K}
    assert direct.queries == [(q.question, TOP_K) for q in QUESTIONS]  # the question as asked
    assert "Measure in inches." not in str([call.messages for call in model.calls])  # not given
    assert len(answers[1].trace.messages) == 4  # a new thread: only the second question's messages


def test_a_stopped_answer_step_keeps_its_answers_and_asks_only_the_others(tmp_path, tiny_db):
    with pytest.raises(IndexError):  # the fake has no reply for the second question
        answer_questions(
            _agent(_replies("Roll six D6."), tiny_db), _direct(), QUESTIONS, tmp_path, *_counters()
        )
    assert [a.question.question_id for a in load_answers(tmp_path)] == ["q001"]

    model = FakeChatModel(replies=_replies("+1 to saves."))
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")
    answer_questions(agent, _direct(), QUESTIONS, tmp_path, *_counters())

    assert [call.messages[1].content for call in model.calls] == [QUESTIONS[1].question] * 2
    assert [a.answer for a in load_answers(tmp_path)] == ["Roll six D6.", "+1 to saves."]


def test_the_requests_of_each_answer_are_saved_with_it(tmp_path, tiny_db):
    chat, embedding = _counters()
    chat.attempts.extend([{"status": 200, "error": None}] * 3)  # an earlier answer's

    answer_questions(
        _agent(_replies("Roll six D6."), tiny_db),
        _direct(),
        QUESTIONS[:1],
        tmp_path,
        chat,
        embedding,
    )
    chat.attempts.append({"status": 200, "error": None})  # after the answer

    assert load_answers(tmp_path)[0].chat_attempts == []  # the fake model sent nothing


def test_saved_answers_are_refused_for_another_question_or_other_chat_settings(tmp_path, tiny_db):
    answers = [_answer(QUESTIONS[0], "Roll six D6.", tiny_db)]
    chat = _agent([], tiny_db).trace()

    assert answers_problem(QUESTIONS, answers, chat, TOP_K) is None
    changed = [replace(QUESTIONS[0], question="How far can a grenade be thrown?"), QUESTIONS[1]]
    assert "q001" in answers_problem(changed, answers, chat, TOP_K)
    other_prompt = _agent([], tiny_db, system_prompt="Another prompt.").trace()
    assert "q001" in answers_problem(QUESTIONS, answers, other_prompt, TOP_K)
    assert f"top {TOP_K}" in answers_problem(QUESTIONS, answers, chat, TOP_K + 3)  # another k
    assert answers_problem(QUESTIONS[1:], answers, other_prompt, TOP_K) is None  # not in the set


def _embedding_server(request: httpx.Request) -> httpx.Response:
    count = len(json.loads(request.read())["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": [1.0] * EMBEDDING_DIM}] * count})


def test_the_retriever_s_own_search_is_spent_from_the_embedding_budget(tmp_path, tiny_db):
    budget = CountingBudget(limit=1)
    embedding = AttemptCounter(budget, httpx.MockTransport(_embedding_server))
    settings = Settings(google_api_key="test", _env_file=None)
    chunks = [RANGE, DEPLOY]
    vectors = FakeEmbedder().embed([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")
    retriever = RulesRetriever(
        chunks, vectors, GeminiEmbedder(settings, {"transport": embedding}), []
    )
    model = FakeChatModel(replies=_replies("Roll six D6."))
    agent = ChatAgent(model, FakeRules(), SqlRetriever(tiny_db, 50), "S")

    answer_questions(agent, retriever, QUESTIONS[:1], tmp_path, AttemptCounter(), embedding)

    (answer,) = load_answers(tmp_path)
    assert [a["status"] for a in answer.embedding_attempts] == [200]  # one request, no retry
    assert budget.spent == 1
    assert len(answer.retriever_contexts) == len(chunks)
    with pytest.raises(BudgetExhausted):  # the budget spent, the next question asks no chat
        answer_questions(agent, retriever, QUESTIONS, tmp_path, AttemptCounter(), embedding)
    assert len(model.calls) == 2  # the first question's two turns only


# --- scores ----------------------------------------------------------------------------------


def test_every_metric_is_scored_from_the_saved_answer(tmp_path, tiny_db):
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)
    model = RagasChatModel()
    metrics = build_metrics(*_adapters(model))

    _score([answer], metrics, tmp_path / "scores.jsonl")

    scores = load_scores(tmp_path / "scores.jsonl")
    assert sorted(metric for _, metric in scores) == sorted(METRICS)
    assert all(s.error is None and s.value is not None for s in scores.values())
    # the fake model finds every statement in the contexts, and each context useful
    assert scores[("q001", "faithfulness")].value == 1.0
    assert scores[("q001", "context_recall")].value == 1.0
    assert scores[("q001", "factual_correctness")].value == 1.0  # the answer is the reference
    assert scores[("q001", "faithfulness")].inputs == score_fingerprint(answer, metrics[0], MODELS)
    assert len(model.titles) == CALLS_PER_ANSWER


def test_a_stopped_score_step_keeps_its_scores_and_computes_only_the_others(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    stopped = RagasChatModel(limit=5)  # faithfulness (2 calls) and relevancy (3), then spent

    with pytest.raises(BudgetExhausted):
        _score([answer], build_metrics(*_adapters(stopped)), scores_path)
    assert sorted(m for _, m in load_scores(scores_path)) == ["answer_relevancy", "faithfulness"]

    resumed = RagasChatModel()
    _score([answer], build_metrics(*_adapters(resumed)), scores_path)

    assert len(resumed.titles) == CALLS_PER_ANSWER - 5
    assert sorted(metric for _, metric in load_scores(scores_path)) == sorted(METRICS)


@dataclass
class FailingMetric:
    """A metric that fails as RAGAS fails on a reply it cannot parse, or with another error."""

    name: str = "failing"
    error: type[Exception] = RagasOutputParserException

    async def single_turn_ascore(self, sample):
        raise self.error()


def test_a_failed_metric_is_saved_and_computed_again_by_the_next_run(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    faithfulness = build_metrics(*_adapters(RagasChatModel()))[0]

    _score([answer], [FailingMetric(), faithfulness], scores_path)

    failed = load_scores(scores_path)[("q001", "failing")]
    assert (failed.value, failed.error.split(":")[0]) == (None, "RagasOutputParserException")
    assert load_scores(scores_path)[("q001", "faithfulness")].value == 1.0  # the next one ran

    model = RagasChatModel()
    _score([answer], [FailingMetric(), build_metrics(*_adapters(model))[0]], scores_path)
    lines = scores_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["metric"] for line in lines] == ["failing", "faithfulness", "failing"]
    assert model.titles == []  # faithfulness was not computed again


def test_another_error_stops_the_score_step(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    model = RagasChatModel()
    refused = FailingMetric(error=RuntimeError)  # as a request the API refused after its retries

    with pytest.raises(RuntimeError):
        _score([answer], [refused, build_metrics(*_adapters(model))[0]], scores_path)

    assert load_scores(scores_path) == {}
    assert model.titles == []  # the next metric was not computed


def _sentences(text: str) -> list[str]:
    return [f"{sentence.strip()}." for sentence in text.split(".") if sentence.strip()]


def test_factual_correctness_checks_the_answer_and_the_reference_against_each_other(
    tmp_path, tiny_db
):
    question = replace(QUESTIONS[0], reference="Roll six D6. Re-roll ones.")
    answer = _answer(question, "Roll six D6. Add 1 to each roll. Re-roll ones.", tiny_db)
    # a claim per sentence, true when its sentence is in the text it is checked against
    model = RagasChatModel(
        replies={
            "ClaimDecompositionOutput": lambda data: {"claims": _sentences(data["response"])},
            "NLIStatementOutput": lambda data: {
                "statements": [
                    {
                        "statement": claim,
                        "reason": "Checked.",
                        "verdict": int(claim in data["context"]),
                    }
                    for claim in data["statements"]
                ]
            },
        }
    )
    factual_correctness = build_metrics(*_adapters(model))[-1]

    _score([answer], [factual_correctness], tmp_path / "scores.jsonl")

    # the answer's 3 claims against the reference: 2 true, 1 false (precision 2/3); the
    # reference's 2 claims against the answer: both true (recall 1); F1 0.8
    assert load_scores(tmp_path / "scores.jsonl")[("q001", "factual_correctness")].value == 0.8
    calls = list(zip(model.titles, model.inputs, strict=True))
    decomposed = [data["response"] for title, data in calls if title == "ClaimDecompositionOutput"]
    assert sorted(decomposed) == sorted([answer.answer, question.reference])
    checked = {
        (data["context"], tuple(data["statements"]))
        for title, data in calls
        if title == "NLIStatementOutput"
    }
    assert checked == {
        (question.reference, tuple(_sentences(answer.answer))),
        (answer.answer, tuple(_sentences(question.reference))),
    }
    contexts = answer.agent_contexts + answer.retriever_contexts
    assert not any(context in str(model.inputs) for context in contexts)


def test_an_answer_without_text_is_not_scored(tmp_path, tiny_db):
    answer = _answer(QUESTIONS[0], "", tiny_db)
    model = RagasChatModel()

    _score([answer], build_metrics(*_adapters(model)), tmp_path / "scores.jsonl")

    assert model.titles == []
    assert load_scores(tmp_path / "scores.jsonl") == {}


def test_saved_scores_are_refused_for_another_answer_question_or_metric(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    metrics = build_metrics(*_adapters(RagasChatModel()))
    _score([answer], metrics, scores_path)
    scores = load_scores(scores_path)

    assert scores_problem(QUESTIONS, [answer], metrics, MODELS, scores) is None
    another = _answer(QUESTIONS[0], "Roll six D6, then add 1.", tiny_db)
    assert "q001" in scores_problem(QUESTIONS, [another], metrics, MODELS, scores)
    assert "q001" in scores_problem(QUESTIONS, [answer], metrics, ["other", "models"], scores)
    stricter = ResponseRelevancy(llm=metrics[1].llm, embeddings=metrics[1].embeddings, strictness=2)
    assert "answer_relevancy" in scores_problem(QUESTIONS, [answer], [stricter], MODELS, scores)
    changed = [replace(QUESTIONS[0], reference="Roll one D6."), QUESTIONS[1]]
    assert "q001" in scores_problem(changed, [answer], metrics, MODELS, scores)


def test_a_score_saved_for_other_inputs_is_computed_again(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    append_score(scores_path, MetricScore("q001", "faithfulness", 0.0, None, "old inputs"))
    model = RagasChatModel()

    _score([answer], build_metrics(*_adapters(model))[:1], scores_path)

    assert load_scores(scores_path)[("q001", "faithfulness")].value == 1.0
    assert len(model.titles) == 2


def _inputs_of(metric_name: str, answer: EvalAnswer, scores_path) -> str:
    """Everything the fake model received while scoring one metric, as one text."""
    model = RagasChatModel()
    metric = next(m for m in build_metrics(*_adapters(model)) if m.name == metric_name)
    score_answers([answer], [metric], MODELS, scores_path)
    return json.dumps(model.inputs, ensure_ascii=False)


@pytest.mark.parametrize(
    ("metric_name", "given"),
    [
        ("faithfulness", "agent"),
        ("llm_context_precision_with_reference", "retriever"),
        ("context_recall", "retriever"),
        ("answer_relevancy", None),
        ("factual_correctness", None),
    ],
)
def test_each_metric_is_given_its_own_contexts(tmp_path, tiny_db, metric_name, given):
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)

    inputs = _inputs_of(metric_name, answer, tmp_path / "scores.jsonl")

    # COVER's text is only in the chunks the chat received; RANGE's and DEPLOY's only in the
    # retriever's (GRENADE's is also the answer and the reference)
    texts = {"agent": ["+1 to saves."], "retriever": ["Measure in inches.", "Set up units."]}
    expected = texts.get(given, [])
    assert all(text in inputs for text in expected)
    assert not any(
        text in inputs for kind, kind_texts in texts.items() if kind != given for text in kind_texts
    )


def test_a_score_s_fingerprint_follows_only_the_contexts_its_metric_is_given(tiny_db):
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)
    metrics = {m.name: m for m in build_metrics(*_adapters(RagasChatModel()))}
    other_agent = replace(answer, agent_contexts=["Something else."])
    other_retriever = replace(answer, retriever_contexts=["Something else."])
    other_k = replace(answer, retriever_k=answer.retriever_k + 1)

    def changed(other: EvalAnswer) -> set[str]:
        return {
            name
            for name, metric in metrics.items()
            if score_fingerprint(other, metric, MODELS) != score_fingerprint(answer, metric, MODELS)
        }

    assert changed(other_agent) == {"faithfulness"}
    assert changed(other_retriever) == {"llm_context_precision_with_reference", "context_recall"}
    assert changed(other_k) == {"llm_context_precision_with_reference", "context_recall"}


class CountingBudget(RequestBudget):
    """A request budget that also counts the requests asked of it, those it refuses included."""

    def __init__(self, limit: int):
        super().__init__(limit, min_interval=0)
        self.asked = 0

    def spend(self) -> None:
        self.asked += 1
        super().spend()


def _gemini(request: httpx.Request) -> httpx.Response:
    """Answers RAGAS's prompts and embedding requests like the Gemini API (`ragas_reply`)."""
    body = json.loads(request.read())
    if request.url.path.endswith(":batchEmbedContents"):
        vector = {"values": [1.0] + [0.0] * (EMBEDDING_DIM - 1)}
        return httpx.Response(200, json={"embeddings": [vector] * len(body["requests"])})
    _, text = ragas_reply(body["contents"][-1]["parts"][0]["text"])
    content = {"role": "model", "parts": [{"text": text}]}
    return httpx.Response(200, json={"candidates": [{"content": content, "finishReason": "STOP"}]})


def test_scoring_spends_every_request_and_retry_from_the_budgets(tmp_path, tiny_db):
    answer, scores_path = _answer(QUESTIONS[0], "Roll six D6.", tiny_db), tmp_path / "scores.jsonl"
    refused = [httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}})]
    chat_budget, embedding_budget = CountingBudget(CALLS_PER_ANSWER), CountingBudget(10)
    chat = AttemptCounter(
        chat_budget, httpx.MockTransport(lambda r: refused.pop() if refused else _gemini(r))
    )
    embedding = AttemptCounter(embedding_budget, httpx.MockTransport(_gemini))
    settings = Settings(google_api_key="test", _env_file=None)
    metrics = build_metrics(
        GeminiRagasLLM(create_chat_model(settings, {"transport": chat})),
        EmbedderRagasEmbeddings(GeminiEmbedder(settings, {"transport": embedding})),
    )

    with pytest.raises(BudgetExhausted):  # the retry of the 429 took the last request
        _score([answer], metrics, scores_path)

    assert [a["status"] for a in chat.attempts] == [429] + [200] * (CALLS_PER_ANSWER - 1)
    assert (chat_budget.spent, chat_budget.asked) == (CALLS_PER_ANSWER, CALLS_PER_ANSWER + 1)
    # relevancy embeds the question, then the three questions written back, in one request
    assert embedding_budget.spent == len(embedding.attempts) == 2
    assert sorted(m for _, m in load_scores(scores_path)) == sorted(METRICS[:4])


# --- files -----------------------------------------------------------------------------------


def test_the_test_set_reads_back_without_the_lines_a_person_removed(tmp_path):
    path = tmp_path / "testset.jsonl"
    save_testset(path, QUESTIONS)
    first, second = path.read_text(encoding="utf-8").splitlines()
    path.write_text(f"{second}\n\n", encoding="utf-8")

    assert load_testset(path) == [QUESTIONS[1]]
    assert json.loads(first)["question_id"] == "q001"


@pytest.mark.parametrize("question_id", ["../outside", "q1", "q001.json", "Q001"])
def test_a_test_set_with_a_question_id_not_like_q001_is_refused(tmp_path, question_id):
    path = tmp_path / "testset.jsonl"
    save_testset(path, [replace(QUESTIONS[0], question_id=question_id)])
    with pytest.raises(ValueError, match="like 'q001'"):
        load_testset(path)


def test_a_test_set_with_a_question_id_twice_is_refused(tmp_path):
    path = tmp_path / "testset.jsonl"
    save_testset(path, [QUESTIONS[0], replace(QUESTIONS[1], question_id="q001")])
    with pytest.raises(ValueError, match="'q001'"):
        load_testset(path)


def test_an_answer_is_never_written_outside_its_folder(tmp_path, tiny_db):
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)
    run_dir = tmp_path / "run"
    answers_dir = run_dir / "answers"
    answers_dir.mkdir(parents=True)
    outside = replace(answer, question=replace(answer.question, question_id="../outside"))

    with pytest.raises(ValueError, match="is not a file of"):
        save_answer(answers_dir, outside)

    assert [p.name for p in run_dir.iterdir()] == ["answers"]
    assert list(answers_dir.iterdir()) == []


def test_an_answer_reads_back_whole(tmp_path, tiny_db):
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)
    save_answer(tmp_path, answer)

    (loaded,) = load_answers(tmp_path)

    assert loaded == answer
    assert (tmp_path / "q001.json").is_file()
    assert not list(tmp_path.glob("*.partial"))


def test_the_latest_score_of_a_metric_counts(tmp_path):
    path = tmp_path / "scores.jsonl"
    assert load_scores(path) == {}
    append_score(path, MetricScore("q001", "faithfulness", None, "ValueError: no score", "a"))
    append_score(path, MetricScore("q001", "faithfulness", 0.5, None, "a"))

    assert load_scores(path) == {
        ("q001", "faithfulness"): MetricScore("q001", "faithfulness", 0.5, None, "a")
    }


@pytest.mark.parametrize(
    "changed",
    [
        "chunks.jsonl",
        "vectors.npz",
        "db.sqlite",
        "agent.py",
        "retriever.py",
        "uv.lock",
        "embedding_model",
    ],
)
def test_answer_run_rejects_changed_retrieval_inputs_before_continuing(tmp_path, changed):
    files = {}
    for name in ["chunks.jsonl", "vectors.npz", "db.sqlite", "agent.py", "retriever.py", "uv.lock"]:
        path = tmp_path / name
        path.write_text("original", encoding="utf-8")
        files[name] = path
    manifest, answers = tmp_path / "run" / "inputs.json", tmp_path / "run" / "answers"
    original = answer_inputs(files, "embedding-model")
    save_answer_inputs(manifest, answers, original)
    save_answer_inputs(manifest, answers, original)
    if changed != "embedding_model":
        files[changed].write_text("footer or data or algorithm changed", encoding="utf-8")
    current = answer_inputs(
        files, "other-model" if changed == "embedding_model" else "embedding-model"
    )
    with pytest.raises(ValueError, match="changed"):
        save_answer_inputs(manifest, answers, current)
    assert json.loads(manifest.read_text(encoding="utf-8")) == original


def test_legacy_answers_can_be_read_but_cannot_be_extended_without_input_manifest(
    tmp_path, tiny_db
):
    answer = _answer(QUESTIONS[0], "Six dice", tiny_db)
    answers = tmp_path / "answers"
    answers.mkdir()
    save_answer(answers, answer)
    assert load_answers(answers) == [answer]
    with pytest.raises(ValueError, match="no input manifest"):
        save_answer_inputs(tmp_path / "inputs.json", answers, {})
    assert not (tmp_path / "inputs.json").exists()


def test_input_hashes_ignore_checkout_line_endings_but_not_footer_changes(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_bytes(b'{"page_footers": []}\r\n')
    first = answer_inputs({"chunks": chunks}, "model")
    chunks.write_bytes(b'{"page_footers": []}\n')
    assert answer_inputs({"chunks": chunks}, "model") == first
    chunks.write_bytes(b'{"page_footers": ["MISSION: ONLY WAR"]}\n')
    assert answer_inputs({"chunks": chunks}, "model") != first


def test_metric_summary_reports_missing_and_failed_samples(tmp_path, tiny_db, caplog):
    answers = [_answer(QUESTIONS[0], "Six dice", tiny_db), _answer(QUESTIONS[1], "Cover", tiny_db)]
    metric = build_metrics(*_adapters(RagasChatModel()))[0]
    path = tmp_path / "scores.jsonl"
    append_score(path, MetricScore("q001", metric.name, 0.8, None, "a"))
    append_score(path, MetricScore("q002", metric.name, None, "judge failed", "b"))
    with caplog.at_level("INFO"):
        log_means(answers, [metric], path, expected=3)
    assert "mean 0.800; valid 1/3, failed 1, missing 1" in caplog.text


def test_http_402_and_tool_budget_errors_stop_at_once_without_saving_partial_results(
    tmp_path, tiny_db
):
    from google.genai import errors

    chunks = [RANGE, DEPLOY]
    vectors = FakeEmbedder().embed([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")
    settings = Settings(google_api_key="test", _env_file=None)

    # 1. Embedding budget spent during agent's search_rules call stops immediately without saving
    embedding_budget = CountingBudget(limit=1)  # enough for direct search, not for search_rules
    embedding = AttemptCounter(embedding_budget, httpx.MockTransport(_embedding_server))
    retriever = RulesRetriever(
        chunks, vectors, GeminiEmbedder(settings, {"transport": embedding}), []
    )
    model = FakeChatModel(replies=_replies("Roll six D6."))
    agent = ChatAgent(model, retriever, SqlRetriever(tiny_db, 50), "S")

    with pytest.raises(BudgetExhausted):
        answer_questions(agent, retriever, QUESTIONS[:1], tmp_path, AttemptCounter(), embedding)

    assert load_answers(tmp_path) == []
    assert len(model.calls) == 1  # stopped inside ToolNode before making a second chat call

    # 2. HTTP 402 during scoring stops on the first attempt without retry and keeps earlier scores
    answer = _answer(QUESTIONS[0], "Roll six D6.", tiny_db)
    scores_path = tmp_path / "scores.jsonl"
    calls = 0

    def billing_fail_after_faithfulness(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls > 2:  # faithfulness takes 2 calls, then 402 Payment Required on relevancy
            return httpx.Response(402, json={"error": {"code": 402, "message": "billing required"}})
        return _gemini(request)

    chat_budget = CountingBudget(limit=20)
    chat = AttemptCounter(chat_budget, httpx.MockTransport(billing_fail_after_faithfulness))
    embed = AttemptCounter(CountingBudget(10), httpx.MockTransport(_gemini))
    metrics = build_metrics(
        GeminiRagasLLM(create_chat_model(settings, {"transport": chat})),
        EmbedderRagasEmbeddings(GeminiEmbedder(settings, {"transport": embed})),
    )

    with pytest.raises((errors.ClientError, Exception)):
        _score([answer], metrics, scores_path)

    assert [a["status"] for a in chat.attempts] == [200, 200, 402]  # 402 not retried
    saved = load_scores(scores_path)
    assert list(saved) == [("q001", "faithfulness")]
    assert saved[("q001", "faithfulness")].value == 1.0


def test_an_empty_saved_answer_is_retried_when_answer_step_resumes(tmp_path, tiny_db):
    answer_questions(
        _agent(_replies(""), tiny_db), _direct(), QUESTIONS[:1], tmp_path, *_counters()
    )
    assert load_answers(tmp_path)[0].answer == ""

    answer_questions(
        _agent(_replies("Roll six D6."), tiny_db), _direct(), QUESTIONS[:1], tmp_path, *_counters()
    )
    assert load_answers(tmp_path)[0].answer == "Roll six D6."
