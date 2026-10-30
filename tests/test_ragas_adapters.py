"""The RAGAS adapters, offline: the fakes stand in for Gemini, RAGAS's own metrics call them, and a
mock transport answers in place of the API where the SDK's retries and the budget are tested."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from google.genai import errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

pytest.importorskip("ragas", reason="the evaluation group is not installed: uv sync --group eval")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.prompt_values import StringPromptValue  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.exceptions import LLMDidNotFinishException  # noqa: E402
from ragas.metrics import Faithfulness, ResponseRelevancy  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from src.rag.config import EMBEDDING_DIM, Settings  # noqa: E402
from src.rag.evaluation.ragas_adapters import (  # noqa: E402
    SIMILARITY_TASK,
    EmbedderRagasEmbeddings,
    GeminiRagasLLM,
)
from src.rag.llm.budget import AttemptCounter, BudgetExhausted, RequestBudget  # noqa: E402
from src.rag.llm.gemini import GeminiEmbedder, create_chat_model  # noqa: E402
from tests.fakes import FakeChatModel, FakeEmbedder  # noqa: E402

QUESTION = "Can a unit throw a grenade?"
STATEMENTS = ["A unit can throw a grenade.", "A grenade is thrown in the Fight phase."]
SETTINGS = Settings(google_api_key="test", _env_file=None)
# The calls RAGAS makes through the adapters: a completion, and texts embedded from its event loop
# or from a worker thread (`embed_texts(is_async=False)`); each has its own retry wrapper in RAGAS.
CALLS = ["completion", "embeddings, async", "embeddings, sync"]


def _reply(content: object, finish_reason: str = "STOP") -> AIMessage:
    text = content if isinstance(content, str) else json.dumps(content)
    return AIMessage(text, response_metadata={"finish_reason": finish_reason})


def test_one_model_call_per_completion_keeps_factory_sampling_and_ragas_stop():
    model = FakeChatModel(replies=[_reply("first"), _reply("second")])
    prompt = StringPromptValue(text="Say something.")

    result = asyncio.run(GeminiRagasLLM(model).generate(prompt, n=2, temperature=0.3, stop=["END"]))

    assert [generation.text for generation in result.generations[0]] == ["first", "second"]
    assert [call.messages[0].content for call in model.calls] == ["Say something."] * 2
    assert [call.options for call in model.calls] == [{"stop": ["END"]}] * 2


def test_a_reply_cut_by_the_token_limit_is_not_finished():
    model = FakeChatModel(replies=[_reply('{"statements": [', "MAX_TOKENS")])

    with pytest.raises(LLMDidNotFinishException):
        asyncio.run(GeminiRagasLLM(model).generate(StringPromptValue(text="Say something.")))


def test_texts_are_embedded_for_similarity_in_one_call():
    embedder = FakeEmbedder()
    embeddings = EmbedderRagasEmbeddings(embedder)

    vectors = asyncio.run(embeddings.embed_texts(["a", "b"]))
    query = embeddings.embed_query("a")

    assert embedder.calls == [(["a", "b"], SIMILARITY_TASK), (["a"], SIMILARITY_TASK)]
    assert query == vectors[0]
    assert all(isinstance(value, float) for value in query)


def test_faithfulness_is_computed_by_ragas_through_the_adapter():
    model = FakeChatModel(
        replies=[
            _reply({"statements": STATEMENTS}),
            _reply(
                {
                    "statements": [
                        {"statement": STATEMENTS[0], "reason": "In the context.", "verdict": 1},
                        {"statement": STATEMENTS[1], "reason": "Not in the context.", "verdict": 0},
                    ]
                }
            ),
        ]
    )
    sample = SingleTurnSample(
        user_input=QUESTION,
        response=" ".join(STATEMENTS),
        retrieved_contexts=["[Core Rules p. 34] A unit can throw one grenade per turn."],
    )

    score = asyncio.run(Faithfulness(llm=GeminiRagasLLM(model)).single_turn_ascore(sample))

    assert score == 0.5  # one statement of two is supported by the context
    assert len(model.calls) == 2


def test_response_relevancy_uses_both_adapters():
    model = FakeChatModel(replies=[_reply({"question": QUESTION, "noncommittal": 0})] * 3)
    embedder = FakeEmbedder()
    metric = ResponseRelevancy(
        llm=GeminiRagasLLM(model), embeddings=EmbedderRagasEmbeddings(embedder)
    )
    sample = SingleTurnSample(user_input=QUESTION, response=STATEMENTS[0])

    score = asyncio.run(metric.single_turn_ascore(sample))

    assert score == pytest.approx(1.0)  # the questions generated back are the question asked
    assert len(model.calls) == 3  # RAGAS's strictness: three completions
    assert {task for _, task in embedder.calls} == {SIMILARITY_TASK}


# --- one attempt per call: the SDK retries the requests, under the budget ---------------------


class CountingBudget(RequestBudget):
    """A request budget that also counts the requests asked of it, those it refuses included."""

    def __init__(self, limit: int):
        super().__init__(limit, min_interval=0)
        self.asked = 0

    def spend(self) -> None:
        self.asked += 1
        super().spend()


def _server(statuses: list[int]) -> Callable[[httpx.Request], httpx.Response]:
    """Answers chat and embedding requests like the Gemini API, from a list of statuses to play."""

    def answer(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0) if statuses else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": "refused"}})
        if request.url.path.endswith(":batchEmbedContents"):
            count = len(json.loads(request.read())["requests"])
            vector = {"values": [1.0] + [0.0] * (EMBEDDING_DIM - 1)}
            return httpx.Response(200, json={"embeddings": [vector] * count})
        content = {"role": "model", "parts": [{"text": "Done."}]}
        return httpx.Response(
            200, json={"candidates": [{"content": content, "finishReason": "STOP"}]}
        )

    return answer


def _call(call: str, counter: AttemptCounter, run_config: RunConfig | None = None) -> object:
    """Make one of RAGAS's `CALLS` through an adapter over the real Gemini objects, whose requests
    go through `counter`; `run_config`, when given, is set first, as a metric sets its run's."""
    client_args = {"transport": counter}
    if call == "completion":
        adapter = GeminiRagasLLM(create_chat_model(SETTINGS, client_args))
    else:
        adapter = EmbedderRagasEmbeddings(GeminiEmbedder(SETTINGS, client_args))
    if run_config is not None:
        adapter.set_run_config(run_config)
    if call == "completion":
        return asyncio.run(adapter.generate(StringPromptValue(text="Say something.")))
    return asyncio.run(adapter.embed_texts(["a"], is_async=call == "embeddings, async"))


@pytest.mark.parametrize("run_config", [None, RunConfig()], ids=["default", "set by a metric"])
@pytest.mark.parametrize("call", CALLS)
def test_an_error_the_sdk_does_not_retry_is_sent_once(call, run_config):
    budget = CountingBudget(limit=10)
    counter = AttemptCounter(budget, httpx.MockTransport(_server([400])))

    with pytest.raises((ChatGoogleGenerativeAIError, errors.ClientError)):
        _call(call, counter, run_config)

    assert [attempt["status"] for attempt in counter.attempts] == [400]
    assert budget.asked == 1


@pytest.mark.parametrize("run_config", [None, RunConfig()], ids=["default", "set by a metric"])
@pytest.mark.parametrize("call", CALLS)
def test_a_spent_budget_is_raised_at_once(call, run_config):
    budget = CountingBudget(limit=0)
    counter = AttemptCounter(budget, httpx.MockTransport(_server([])))

    with pytest.raises(BudgetExhausted):
        _call(call, counter, run_config)

    assert counter.attempts == []  # nothing sent
    assert budget.asked == 1  # and refused once, not asked again by RAGAS


@pytest.mark.parametrize("call", CALLS)
def test_the_sdk_retries_a_rate_limit_within_the_budget(call):
    budget = CountingBudget(limit=10)
    counter = AttemptCounter(budget, httpx.MockTransport(_server([429])))

    _call(call, counter)

    assert [attempt["status"] for attempt in counter.attempts] == [429, 200]
    assert (budget.spent, budget.asked) == (2, 2)


@pytest.mark.parametrize(
    "adapter",
    [
        lambda: GeminiRagasLLM(FakeChatModel(replies=[])),
        lambda: EmbedderRagasEmbeddings(FakeEmbedder()),
    ],
    ids=["llm", "embeddings"],
)
def test_ragas_configuration_is_kept_with_one_attempt(adapter):
    created = adapter()
    run_config = RunConfig(timeout=7, max_retries=5, max_wait=3, max_workers=2, seed=9)

    created_config = created.run_config
    created.set_run_config(run_config)

    assert created_config == RunConfig(max_retries=1)
    assert created.run_config == RunConfig(
        timeout=7, max_retries=1, max_wait=3, max_workers=2, seed=9
    )
    assert run_config.max_retries == 5  # the caller's configuration is not changed
