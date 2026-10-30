"""The request budget and the attempt counter: what a run may send, and what it records."""

import gzip
import json
import threading
import time

import httpx
import pytest

from src.rag.llm.budget import AttemptCounter, BudgetExhausted, RequestBudget


def test_budget_spaces_requests():
    now = [0.0]
    waits = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    budget = RequestBudget(limit=5, min_interval=12, clock=lambda: now[0], sleep=sleep)
    budget.spend()
    now[0] += 3
    budget.spend()
    now[0] += 20
    budget.spend()
    assert waits == [9]


def test_threads_sharing_a_budget_take_turns():
    now = [0.0]

    def sleep(seconds: float) -> None:
        time.sleep(0.01)  # while one thread waits for its slot, the others try to spend
        now[0] += seconds

    budget = RequestBudget(limit=3, min_interval=12, clock=lambda: now[0], sleep=sleep)
    refused = []

    def spend() -> None:
        try:
            budget.spend()
        except BudgetExhausted:
            refused.append(True)

    threads = [threading.Thread(target=spend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert (budget.spent, len(refused)) == (3, 5)
    assert now[0] == 24  # two waits of 12 s, one after the other


def test_an_attempt_is_recorded_before_it_is_sent():
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("no response", request=request)
        return httpx.Response(429 if request.url.path == "/busy" else 200)

    budget = RequestBudget(limit=3, min_interval=0)
    counter = AttemptCounter(budget, httpx.MockTransport(answer))
    with httpx.Client(transport=counter) as client:
        client.get("https://example.test/busy")
        with pytest.raises(httpx.ReadTimeout):
            client.get("https://example.test/slow")
        client.get("https://example.test/ok")
        with pytest.raises(BudgetExhausted):
            client.get("https://example.test/refused")  # past the budget: not sent

    assert counter.attempts == [
        {"status": 429, "error": None},
        {"status": None, "error": "ReadTimeout"},
        {"status": 200, "error": None},
    ]


def test_one_budget_shared_by_two_transports_counts_both():
    budget = RequestBudget(limit=2, min_interval=0)
    inner = httpx.MockTransport(lambda request: httpx.Response(200))
    chat, judge = AttemptCounter(budget, inner), AttemptCounter(budget, inner)
    with httpx.Client(transport=chat) as chat_client, httpx.Client(transport=judge) as judge_client:
        chat_client.get("https://example.test/chat")
        judge_client.get("https://example.test/judge")
        assert budget.exhausted
        with pytest.raises(BudgetExhausted, match="request budget of 2 reached"):
            chat_client.get("https://example.test/chat")
    assert (len(chat.attempts), len(judge.attempts)) == (1, 1)


def test_turn_limits_reserve_requests_without_resetting_the_run_or_spacing():
    now = [0.0]
    waits = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    budget = RequestBudget(5, 10, clock=lambda: now[0], sleep=sleep)
    budget.start_turn(2)
    budget.spend()
    budget.spend()
    with pytest.raises(BudgetExhausted, match="turn request budget of 2 reached"):
        budget.spend()
    budget.start_turn(3)
    budget.spend()
    assert budget.spent == 3 and waits == [10, 10]
    budget.spend()
    budget.spend()
    assert budget.exhausted


def test_diagnostics_keep_function_fields_without_prompts_keys_or_headers():
    reply = {
        "candidates": [
            {
                "finishReason": "MALFORMED_FUNCTION_CALL",
                "finishMessage": "The generated function call was malformed.",
                "content": {
                    "parts": [
                        {"text": "hidden response"},
                        {"functionCall": {"name": "run_sql", "args": {"sql": "SELECT 1"}}},
                    ]
                },
            }
        ]
    }
    body = f"data: {json.dumps(reply)}\r\n\r\n".encode()
    inner = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-encoding": "gzip"}, content=gzip.compress(body)
        )
    )
    counter = AttemptCounter(RequestBudget(1, 0), inner, capture_diagnostics=True)
    payload = {
        "tools": [
            {"functionDeclarations": [{"name": "run_sql", "parameters": {"type": "OBJECT"}}]}
        ],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0},
        "contents": [{"role": "user", "parts": [{"text": "private question"}]}],
    }
    with httpx.Client(transport=counter) as client:
        response = client.post(
            "https://example.test/models/test:streamGenerateContent?key=private-key",
            headers={"authorization": "private-header"},
            json=payload,
        )
        assert response.text.startswith("data: ")

    diagnostic = counter.attempts[0]["diagnostic"]
    saved = json.dumps(diagnostic)
    assert diagnostic["request"]["path"] == "/models/test:streamGenerateContent"
    assert diagnostic["request"]["tools"] == payload["tools"]
    assert diagnostic["request"]["contents"] == [
        {"role": "user", "parts": [{"kind": "text", "characters": 16}]}
    ]
    assert diagnostic["response"]["candidates"][0]["finish_reason"] == ("MALFORMED_FUNCTION_CALL")
    assert diagnostic["response"]["candidates"][0]["parts"][1] == {
        "kind": "functionCall",
        "name": "run_sql",
        "args": {"sql": "SELECT 1"},
    }
    assert "private question" not in saved
    assert "private-key" not in saved
    assert "private-header" not in saved
    assert "hidden response" not in saved
