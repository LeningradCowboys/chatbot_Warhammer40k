"""Offline tests of the compatibility check's tools (tests/gemini_check.py): no request is sent."""

import base64
import gzip
import hashlib
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.rag.config import Settings
from src.rag.llm.budget import BudgetExhausted, RequestBudget
from src.rag.llm.gemini import create_chat_model
from tests import gemini_check
from tests.fakes import FakeChatModel, tool_call
from tests.gemini_check import (
    FINAL_INSTRUCTION,
    FINAL_PROMPT,
    PLACEHOLDER_SIGNATURE,
    SIGNATURE_MAP_KEY,
    SYSTEM_PROMPT,
    Exchange,
    RecordingTransport,
    ask,
    assert_signatures_sent_back,
    assert_signatures_stored,
    build_graph,
    multiply,
    signature_report,
    turns,
    unit_lookup,
    unit_toughness,
    write_signature_report,
)

BASE = "https://generativelanguage.googleapis.com/v1beta"
GENERATE = f"{BASE}/models/gemini-3.8-flash:generateContent"
STREAM = f"{BASE}/models/gemini-3.8-flash:streamGenerateContent?alt=sse"


def _reply(parts: list[dict]) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


def _sse(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events).encode()


def _gzip_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(body))


# --- recording -------------------------------------------------------------------------------


def test_records_generate_and_stream_requests_through_their_real_paths():
    reply = _reply([{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "S1"}])
    events = [_reply([{"text": "Hel"}]), _reply([{"text": "lo", "thoughtSignature": "S2"}])]

    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":generateContent"):
            return _gzip_response(json.dumps(reply).encode())
        if request.url.path.endswith(":streamGenerateContent"):
            return _gzip_response(_sse(events))
        return httpx.Response(200, json={"models": []})

    recorder = RecordingTransport(inner=httpx.MockTransport(server))
    with httpx.Client(transport=recorder) as client:
        assert client.post(GENERATE, json={"contents": [1]}).json() == reply
        assert client.post(STREAM, json={"contents": [2]}).text.startswith("data: ")
        client.get(f"{BASE}/models")

    assert [(e.request, e.streamed, e.status, e.encoding) for e in recorder.exchanges] == [
        ({"contents": [1]}, False, 200, "gzip"),
        ({"contents": [2]}, True, 200, "gzip"),
    ]
    assert recorder.exchanges[0].reply_parts() == reply["candidates"][0]["content"]["parts"]
    assert recorder.exchanges[1].reply_parts() == [
        {"text": "Hel"},
        {"text": "lo", "thoughtSignature": "S2"},
    ]


def test_an_unknown_encoding_is_an_error():
    exchange = Exchange(
        request={}, streamed=False, status=200, encoding="br", reply=bytearray(b"x")
    )
    with pytest.raises(ValueError, match="br"):
        exchange.reply_payloads()


def test_budget_stops_before_sending_once_spent():
    sent = []

    def server(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_reply([]))

    budget = RequestBudget(limit=2, min_interval=0)
    recorder = RecordingTransport(budget=budget, inner=httpx.MockTransport(server))
    with httpx.Client(transport=recorder) as client:
        client.post(GENERATE, json={"contents": []})
        client.post(GENERATE, json={"contents": []})  # a retry is a request like any other
        with pytest.raises(BudgetExhausted, match="incomplete"):
            client.post(GENERATE, json={"contents": []})
    assert len(sent) == 2
    assert budget.spent == 2


# --- signatures ------------------------------------------------------------------------------


def _exchange(request_contents: list[dict], reply: dict | list[dict], streamed=False) -> Exchange:
    body = _sse(reply) if streamed else json.dumps(reply).encode()
    return Exchange({"contents": request_contents}, streamed, 200, "", bytearray(body))


def _signature(n: int) -> bytes:
    """Synthetic signature bytes, whose standard base64 holds both `+` and `/`."""
    return bytes([0xFB, 0xFF, 0xBF, n]) * 6


def std(signature: bytes) -> str:
    """The form Gemini returns."""
    return base64.b64encode(signature).decode()


def url(signature: bytes) -> str:
    """The form the SDK sends back."""
    return base64.urlsafe_b64encode(signature).decode()


S0, S1 = _signature(0), _signature(1)
CALL = {"functionCall": {"name": "unit_toughness", "args": {"name": "Zorvak Sentinel"}}}
RESULT = {"role": "user", "parts": [{"functionResponse": {"name": "unit_toughness"}}]}
QUESTION = {"role": "user", "parts": [{"text": "question"}]}


def _model(parts: list[dict]) -> dict:
    return {"role": "model", "parts": parts}


def test_the_synthetic_signatures_differ_between_the_two_alphabets():
    assert std(S1) != url(S1) and "+" in std(S1) and "/" in std(S1)


def test_signatures_sent_back_whatever_part_and_stream_cut():
    first_reply = _reply([{"text": "Let me check.", "thoughtSignature": std(S0)}, CALL])
    # Streamed answer: the signature comes on a last piece with no text.
    stream = [
        _reply([{"text": "It is "}]),
        _reply([{"text": "7."}]),
        _reply([{"text": "", "thoughtSignature": std(S1)}]),
    ]
    first = [QUESTION]
    sent_first = _model([{"text": "Let me check.", "thoughtSignature": url(S0)}, CALL])
    second = [QUESTION, sent_first, RESULT]
    for answer_parts in (
        [{"text": "It is 7.", "thoughtSignature": url(S1)}],  # merged by the adapter
        [{"text": "It is "}, {"text": "7."}, {"thoughtSignature": url(S1)}],  # kept as received
    ):
        third = second + [_model(answer_parts), QUESTION]
        assert_signatures_sent_back(
            [
                _exchange(first, first_reply),
                _exchange(second, stream, streamed=True),
                _exchange(third, _reply([{"text": "done"}])),
            ]
        )


def _check_second_request(reply_parts: list[dict], sent_parts: list[dict]) -> None:
    assert_signatures_sent_back(
        [
            _exchange([QUESTION], _reply(reply_parts)),
            _exchange([QUESTION, _model(sent_parts), RESULT], _reply([{"text": "done"}])),
        ]
    )


def test_the_standard_and_url_safe_forms_of_one_signature_match():
    _check_second_request(
        [{**CALL, "thoughtSignature": std(S1)}], [{**CALL, "thoughtSignature": url(S1)}]
    )


def test_a_signature_of_other_bytes_fails():
    with pytest.raises(AssertionError, match="request 1"):
        _check_second_request(
            [{**CALL, "thoughtSignature": std(S1)}], [{**CALL, "thoughtSignature": url(S0)}]
        )


def test_a_signature_that_is_not_base64_fails():
    with pytest.raises(AssertionError, match="not base64"):
        _check_second_request(
            [{**CALL, "thoughtSignature": std(S1)}], [{**CALL, "thoughtSignature": "not base64!"}]
        )


def test_a_refused_attempt_is_spent_but_is_no_turn():
    replies = [
        httpx.Response(429, json={"error": {"code": 429}}),
        httpx.Response(200, json=_reply([{**CALL, "thoughtSignature": std(S1)}])),
        httpx.Response(200, json=_reply([{"text": "done"}])),
    ]
    budget = RequestBudget(limit=5, min_interval=0)
    recorder = RecordingTransport(budget, httpx.MockTransport(lambda request: replies.pop(0)))
    sent_back = _model([{**CALL, "thoughtSignature": url(S1)}])
    with httpx.Client(transport=recorder) as client:
        client.post(GENERATE, json={"contents": [QUESTION]})  # refused
        client.post(GENERATE, json={"contents": [QUESTION]})  # the retry
        client.post(GENERATE, json={"contents": [QUESTION, sent_back, RESULT]})

    assert budget.spent == 3
    assert len(recorder.exchanges) == 3
    assert len(turns(recorder.exchanges)) == 2
    assert_signatures_sent_back(recorder.exchanges)


def test_adjacent_texts_sent_back_unchanged_pass():
    _check_second_request(
        [{"text": "Same.", "thoughtSignature": std(S1)}, {"text": "Same."}],
        [{"text": "Same.", "thoughtSignature": url(S1)}, {"text": "Same."}],
    )


@pytest.mark.parametrize(
    ("reply", "sent"),
    [
        (
            [{"text": "First.", "thoughtSignature": std(S1)}, {"text": "Second."}],
            [{"text": "First."}, {"text": "Second.", "thoughtSignature": url(S1)}],
        ),
        (
            [{"text": "Same.", "thoughtSignature": std(S1)}, {"text": "Same."}],
            [{"text": "Same."}, {"text": "Same.", "thoughtSignature": url(S1)}],
        ),
    ],
    ids=["different-texts", "identical-texts"],
)
def test_a_signature_moved_between_adjacent_texts_of_a_reply_fails(reply, sent):
    with pytest.raises(AssertionError, match="request 1"):
        _check_second_request(reply, sent)


def test_a_signature_moved_to_another_text_fails():
    reply = [{"text": "First.", "thoughtSignature": std(S1)}, CALL, {"text": "Second."}]
    sent = [{"text": "First."}, CALL, {"text": "Second.", "thoughtSignature": url(S1)}]
    with pytest.raises(AssertionError, match="request 1"):
        _check_second_request(reply, sent)


def test_a_signature_moved_from_text_to_a_call_fails():
    reply = [{"text": "Let me check.", "thoughtSignature": std(S1)}, CALL]
    sent = [{"text": "Let me check."}, {**CALL, "thoughtSignature": url(S1)}]
    with pytest.raises(AssertionError, match="request 1"):
        _check_second_request(reply, sent)


def test_a_lost_signature_fails():
    with pytest.raises(AssertionError, match="request 1"):
        _check_second_request([{**CALL, "thoughtSignature": std(S1)}], [CALL])


def test_the_placeholder_in_place_of_a_signature_fails():
    with pytest.raises(AssertionError, match="placeholder"):
        _check_second_request(
            [{**CALL, "thoughtSignature": std(S1)}],
            [{**CALL, "thoughtSignature": PLACEHOLDER_SIGNATURE}],
        )


def test_replies_without_any_signature_fail():
    with pytest.raises(AssertionError, match="no reply carried"):
        assert_signatures_sent_back([_exchange([QUESTION], _reply([{"text": "hi"}]))])


def _stored_check(reply_parts: list[dict], message: AIMessage) -> None:
    assert_signatures_stored(_exchange([QUESTION], _reply(reply_parts)), message)


def test_stored_signatures_are_read_from_the_adapter_fields():
    reply = [
        {"text": "Let me check.", "thoughtSignature": std(S0)},
        {**CALL, "thoughtSignature": std(S1)},
    ]
    for call_signature in (std(S1), S1):  # a base64 text, or bytes
        message = AIMessage(
            content=[{"type": "text", "text": "Let me check.", "extras": {"signature": url(S0)}}],
            additional_kwargs={SIGNATURE_MAP_KEY: {"call-1": call_signature}},
        )
        _stored_check(reply, message)


@pytest.mark.parametrize(
    ("message", "problem"),
    [
        (AIMessage(content=""), "not stored"),
        (AIMessage(content=std(S1)), "not stored"),  # in the text, not in a signature field
        (
            AIMessage(content="", additional_kwargs={SIGNATURE_MAP_KEY: {"c": url(S0)}}),
            "not stored",
        ),
        (
            AIMessage(
                content="", additional_kwargs={SIGNATURE_MAP_KEY: {"c": PLACEHOLDER_SIGNATURE}}
            ),
            "placeholder",
        ),
        (
            AIMessage(content="", additional_kwargs={SIGNATURE_MAP_KEY: {"c": "not base64!"}}),
            "not base64",
        ),
        (
            AIMessage(content="", additional_kwargs={SIGNATURE_MAP_KEY: {"c": "签名"}}),
            "not base64",
        ),
        # A field named `signature` where the adapter does not keep signatures.
        (AIMessage(content="", response_metadata={"signature": std(S1)}), "not stored"),
        (
            AIMessage(
                content="",
                tool_calls=[tool_call("unit_toughness", {"signature": std(S1)}, "call-1")],
            ),
            "not stored",
        ),
        (AIMessage(content="", additional_kwargs={"signature": std(S1)}), "not stored"),
    ],
    ids=[
        "missing",
        "only-as-text",
        "other-bytes",
        "placeholder",
        "not-base64",
        "not-ascii",
        "in-response-metadata",
        "in-tool-arguments",
        "elsewhere-in-additional-kwargs",
    ],
)
def test_a_signature_not_stored_in_a_field_fails(message, problem):
    with pytest.raises(AssertionError, match=problem):
        _stored_check([{**CALL, "thoughtSignature": std(S1)}], message)


def test_the_report_describes_each_turn_by_sizes_and_hashes_only():
    exchanges = [
        _exchange([QUESTION], _reply([{**CALL, "thoughtSignature": std(S1)}])),
        _exchange(
            [QUESTION, _model([{**CALL, "thoughtSignature": url(S1)}]), RESULT],
            _reply([{"text": "done"}]),
        ),
    ]
    report = signature_report(exchanges)

    assert report["attempts"] == [{"status": 200, "streamed": False}] * 2
    [first_part] = report["turns"][0]["reply_parts"]
    assert first_part == {
        "position": 0,
        "kind": "call",
        "call": "unit_toughness",
        "signatures": [{"bytes": len(S1), "sha256": hashlib.sha256(S1).hexdigest()}],
    }
    [sent] = report["turns"][1]["request_sends_back"]
    assert sent["turn"] == 0 and sent["matches"] is True
    text = json.dumps(report)
    assert std(S1) not in text and url(S1) not in text


def test_the_report_marks_a_turn_sent_back_with_other_bytes():
    exchanges = [
        _exchange([QUESTION], _reply([{**CALL, "thoughtSignature": std(S1)}])),
        _exchange(
            [QUESTION, _model([{**CALL, "thoughtSignature": url(S0)}]), RESULT],
            _reply([{"text": "done"}]),
        ),
    ]
    [sent] = signature_report(exchanges)["turns"][1]["request_sends_back"]
    assert sent["matches"] is False


SECRET = "a reply text that must not reach the report"


@pytest.mark.parametrize(
    ("bad_reply", "error"),
    [
        (json.dumps(_reply([{**CALL, "thoughtSignature": "签名"}])).encode(), "AssertionError"),
        (
            json.dumps(_reply([{**CALL, "thoughtSignature": PLACEHOLDER_SIGNATURE}])).encode(),
            "AssertionError",
        ),
        (
            f'{{"candidates": [{{"content": {{"parts": [{{"text": "{SECRET}'.encode(),
            "JSONDecodeError",
        ),
    ],
    ids=["not-ascii-signature", "placeholder", "cut-json"],
)
def test_the_report_keeps_what_it_can_read_and_names_only_the_kind_of_error(
    tmp_path, bad_reply, error
):
    good = _exchange([QUESTION], _reply([{**CALL, "thoughtSignature": std(S1)}]))
    bad = Exchange(
        {"contents": [QUESTION, _model([{**CALL, "thoughtSignature": url(S1)}]), RESULT]},
        False,
        200,
        "",
        bytearray(bad_reply),
    )
    path = tmp_path / "checks" / "run.json"
    write_signature_report([good, bad], path)

    text = path.read_text(encoding="utf-8")
    report = json.loads(text)
    first, second = report["turns"]
    assert first["reply_parts"][0]["signatures"][0]["sha256"] == hashlib.sha256(S1).hexdigest()
    assert second["reply_parts"] is None and second["reply_error"] == error
    assert second["request_sends_back"][0]["matches"] is True  # the request is still read
    assert SECRET not in text and "签名" not in text and PLACEHOLDER_SIGNATURE not in text


def test_an_error_the_report_cannot_place_is_a_report_level_error(tmp_path, monkeypatch):
    def fail(exchanges):
        raise KeyError("data that must not be saved")

    monkeypatch.setattr(gemini_check, "signature_report", fail)
    path = tmp_path / "run.json"
    write_signature_report([_exchange([QUESTION], _reply([{"text": "hi"}]))], path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report == {"attempts": [{"status": 200, "streamed": False}], "report_error": "KeyError"}


def test_a_report_that_cannot_be_written_is_an_error(tmp_path):
    folder = tmp_path / "run.json"
    folder.mkdir()  # a folder where the file should go
    with pytest.raises(OSError):
        write_signature_report([_exchange([QUESTION], _reply([{"text": "hi"}]))], folder)


# --- the graph, with the fake chat model -----------------------------------------------------


def test_validated_tool_choice_reaches_the_real_adapter_request():
    recorder = RecordingTransport(
        inner=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_reply([{"text": "done"}]))
        )
    )
    model = create_chat_model(
        Settings(google_api_key="test", _env_file=None),
        {"transport": recorder},
    )

    reply = model.bind_tools([unit_toughness], tool_choice={"mode": "VALIDATED"}).invoke(
        "toughness?"
    )

    assert reply.text == "done"
    [request] = [exchange.request for exchange in turns(recorder.exchanges)]
    assert request["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"
    assert "temperature" not in request.get("generationConfig", {})


def test_a_tool_error_reaches_the_next_model_call():
    model = FakeChatModel(
        replies=[
            AIMessage("", tool_calls=[tool_call("unit_lookup", {"name": "Zorvak Sentinel"}, "c1")]),
            AIMessage("The database is offline."),
        ]
    )
    state = ask(build_graph(model, [unit_lookup], 2), "t", "toughness?")

    received = model.calls[1].messages[-1]
    assert isinstance(received, ToolMessage)
    assert (received.status, received.tool_call_id) == ("error", "c1")
    assert "unit database offline" in received.text
    assert state["messages"][-1].text == "The database is offline."


def test_the_round_limit_goes_to_a_final_call_with_tools_disabled():
    model = FakeChatModel(
        replies=[
            AIMessage(
                "", tool_calls=[tool_call("unit_toughness", {"name": "Quillon Brute"}, "c1")]
            ),
            AIMessage("Final answer."),
        ]
    )
    state = ask(build_graph(model, [unit_toughness], 1), "t", "toughness?")

    assert [call.tool_choice for call in model.calls] == [None, "none"]
    assert model.calls[0].messages[0].text == SYSTEM_PROMPT
    # The final call is rebuilt from the state: final prompt, the messages before its reply, the
    # instruction; the instruction itself is not stored.
    messages = state["messages"]
    assert model.calls[1].messages == [
        SystemMessage(FINAL_PROMPT),
        *messages[:-1],
        HumanMessage(FINAL_INSTRUCTION),
    ]
    assert all(m.text != FINAL_INSTRUCTION for m in messages)
    assert state["tool_rounds"] == 1
    assert messages[-1].text == "Final answer."


def test_an_empty_final_reply_ends_the_graph_and_is_kept_as_returned():
    """The reply seen on the real API at the round limit (R40, R41), through the real adapter."""
    signature = base64.b64encode(bytes(range(40))).decode()
    replies = [
        _reply(
            [
                {
                    "functionCall": {"name": "unit_toughness", "args": {"name": "Zorvak Sentinel"}},
                    "thoughtSignature": signature,
                }
            ]
        ),
        {"candidates": [{"finishReason": "UNEXPECTED_TOOL_CALL", "index": 0}]},
    ]
    recorder = RecordingTransport(
        inner=httpx.MockTransport(lambda request: httpx.Response(200, json=replies.pop(0)))
    )
    model = create_chat_model(
        Settings(google_api_key="test", _env_file=None), {"transport": recorder}
    )

    messages = ask(build_graph(model, [unit_toughness, multiply], 1), "t", "toughness?")["messages"]

    assert len(turns(recorder.exchanges)) == 2  # the graph ended after the final call
    final = turns(recorder.exchanges)[-1].request
    assert final["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
    assert final["systemInstruction"]["parts"][0]["text"] == FINAL_PROMPT
    assert final["contents"][-1] == {"role": "user", "parts": [{"text": FINAL_INSTRUCTION}]}
    assert all(m.text != FINAL_INSTRUCTION for m in messages)
    reply = messages[-1]
    assert (reply.text, reply.tool_calls) == ("", [])
    assert reply.response_metadata["finish_reason"] == "UNEXPECTED_TOOL_CALL"
    assert_signatures_sent_back(recorder.exchanges)


def test_several_calls_in_one_turn_are_one_round():
    calls = [
        tool_call("unit_toughness", {"name": "Quillon Brute"}, "c1"),
        tool_call("multiply", {"a": 2, "b": 3}, "c2"),
    ]
    model = FakeChatModel(replies=[AIMessage("", tool_calls=calls), AIMessage("Answer.")])
    state = ask(build_graph(model, [unit_toughness, multiply], 2), "t", "question")

    assert state["tool_rounds"] == 1
    assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 2
    assert [call.tool_choice for call in model.calls] == [None, None]  # no final call


def test_a_follow_up_on_the_same_thread_counts_rounds_from_zero():
    def calls_tool(call_id: str) -> AIMessage:
        return AIMessage(
            "", tool_calls=[tool_call("unit_toughness", {"name": "Mirethan Scout"}, call_id)]
        )

    model = FakeChatModel(
        replies=[
            calls_tool("c1"),
            calls_tool("c2"),
            AIMessage("First."),
            calls_tool("c3"),
            AIMessage("Second."),
        ]
    )
    graph = build_graph(model, [unit_toughness], 2)
    ask(graph, "t", "first question")  # two rounds, then the final call
    state = ask(graph, "t", "follow-up")  # one round: still under the limit, so no final call

    assert state["tool_rounds"] == 1
    assert [call.tool_choice for call in model.calls] == [None, None, "none", None, None]
    assert state["messages"][-1].text == "Second."
