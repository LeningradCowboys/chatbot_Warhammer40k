"""Counting and capping the requests a run sends to Gemini, at the HTTP level.

A run that calls the API (the RAGAS evaluation, the online checks) passes an `AttemptCounter` as the
`httpx` transport of its Gemini objects: every attempt, retries included, is spent from a
`RequestBudget` before it is sent and recorded, so a run never sends more than it was allowed.
"""

import gzip
import json
import threading
import time
import zlib
from collections.abc import Callable
from typing import Any

import httpx


class BudgetExhausted(Exception):
    """The run's request budget is spent: the run is incomplete."""


class RequestBudget:
    """Counts requests before they are sent, retries included, and refuses them past the limit.

    It also spaces them by at least `min_interval` seconds, so that a per-minute rate limit is
    not hit and no retry is spent on it. One budget can be shared by several transports, such as
    those of the chat model that answers and of the one RAGAS calls, which use the same model's
    quota, and by several threads (RAGAS calls its LLM from several at once): a lock makes them
    take their turns, so the spacing and the count hold.
    """

    def __init__(
        self,
        limit: int,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.limit = limit
        self.min_interval = min_interval
        self.spent = 0
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._turn_limit: int | None = None
        self._turn_spent = 0
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        """Whether no request can be sent any more."""
        return self.spent >= self.limit

    def spend(self) -> None:
        """Wait for the next slot and count one request, or raise if the budget is spent."""
        with self._lock:
            if self._turn_limit is not None and self._turn_spent >= self._turn_limit:
                raise BudgetExhausted(
                    f"turn request budget of {self._turn_limit} reached: the turn is incomplete"
                )
            if self.exhausted:
                raise BudgetExhausted(
                    f"request budget of {self.limit} reached: the run is incomplete"
                )
            if self._last is not None:
                wait = self._last + self.min_interval - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            self.spent += 1
            self._turn_spent += 1

    def start_turn(self, limit: int) -> None:
        """Start a turn with its own limit without resetting the run count or request spacing.

        Args:
            limit: Maximum requests for this turn, including retries.
        """
        if limit < 1:
            raise ValueError("turn request budget must be at least 1")
        with self._lock:
            self._turn_limit = limit
            self._turn_spent = 0


def _part_summary(part: dict[str, Any]) -> dict[str, Any]:
    """Describe a Gemini content part without retaining its text or thought signature."""
    if call := part.get("functionCall"):
        return {"kind": "functionCall", "name": call.get("name"), "args": call.get("args")}
    if response := part.get("functionResponse"):
        value = response.get("response")
        keys = sorted(value) if isinstance(value, dict) else []
        return {"kind": "functionResponse", "name": response.get("name"), "response_keys": keys}
    if "text" in part:
        return {
            "kind": "thought" if part.get("thought") else "text",
            "characters": len(part.get("text", "")),
        }
    return {"kind": next((key for key in part if key != "thoughtSignature"), "unknown")}


def _request_diagnostic(request: httpx.Request) -> dict[str, Any]:
    """Return the request fields needed to diagnose function calling, without prompt text."""
    diagnostic: dict[str, Any] = {"path": request.url.path}
    try:
        body = json.loads(request.read())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        diagnostic["parse_error"] = type(error).__name__
        return diagnostic
    diagnostic["tools"] = body.get("tools")
    diagnostic["tool_config"] = body.get("toolConfig")
    diagnostic["generation_config"] = body.get("generationConfig")
    diagnostic["contents"] = [
        {
            "role": content.get("role"),
            "parts": [_part_summary(part) for part in content.get("parts", [])],
        }
        for content in body.get("contents", [])
    ]
    return diagnostic


def _decoded_body(body: bytes, encoding: str) -> bytes:
    """Decode the two content encodings used by the Gemini HTTP responses."""
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        return zlib.decompress(body)
    if encoding not in ("", "identity"):
        raise ValueError(f"unsupported content encoding: {encoding}")
    return body


def _response_diagnostic(body: bytes, encoding: str) -> dict[str, Any]:
    """Return finish/error/function-call fields from a Gemini response body."""
    try:
        text = _decoded_body(body, encoding).decode("utf-8")
        sse = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        decoded = [json.loads(value) for value in sse] if sse else json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError, zlib.error) as error:
        return {"parse_error": type(error).__name__, "content_encoding": encoding}
    events = decoded if isinstance(decoded, list) else [decoded]
    candidates = []
    errors = []
    prompt_feedback = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if error := event.get("error"):
            errors.append(
                {key: error.get(key) for key in ("code", "status", "message") if key in error}
            )
        if feedback := event.get("promptFeedback"):
            prompt_feedback.append(
                {
                    key: feedback.get(key)
                    for key in ("blockReason", "blockReasonMessage")
                    if key in feedback
                }
            )
        for candidate in event.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])
            candidates.append(
                {
                    "finish_reason": candidate.get("finishReason"),
                    "finish_message": candidate.get("finishMessage"),
                    "parts": [_part_summary(part) for part in parts],
                }
            )
    return {"candidates": candidates, "errors": errors, "prompt_feedback": prompt_feedback}


class _DiagnosticStream(httpx.SyncByteStream):
    """Pass a response through unchanged and save a sanitized summary after it is consumed."""

    def __init__(
        self,
        stream: httpx.SyncByteStream,
        encoding: str,
        completed: Callable[[dict[str, Any]], None],
    ):
        self._stream = stream
        self._encoding = encoding
        self._completed = completed

    def __iter__(self):
        body = bytearray()
        try:
            for piece in self._stream:
                body.extend(piece)
                yield piece
        finally:
            self._completed(_response_diagnostic(bytes(body), self._encoding))

    def close(self) -> None:
        self._stream.close()


class AttemptCounter(httpx.BaseTransport):
    """An httpx transport that records every attempt before it is sent.

    Each attempt keeps its HTTP status, or, when no response came (a timeout), `status` None and
    the error's class. An attempt the budget refuses is not sent and not recorded.

    Args:
        budget: Spent before each attempt; `None` counts nothing.
        inner: The transport that sends the requests; the network by default.
        capture_diagnostics: Whether to retain sanitized function-call request and response fields.
    """

    def __init__(
        self,
        budget: RequestBudget | None = None,
        inner: httpx.BaseTransport | None = None,
        capture_diagnostics: bool = False,
    ):
        self._budget = budget
        self._inner = inner or httpx.HTTPTransport()
        self._capture_diagnostics = capture_diagnostics
        self.attempts: list[dict] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._budget is not None:
            self._budget.spend()
        attempt = {"status": None, "error": None}
        if self._capture_diagnostics:
            attempt["diagnostic"] = {"request": _request_diagnostic(request), "response": None}
        self.attempts.append(attempt)
        try:
            response = self._inner.handle_request(request)
        except Exception as error:
            attempt["error"] = type(error).__name__
            raise
        attempt["status"] = response.status_code
        if self._capture_diagnostics:
            diagnostic = attempt["diagnostic"]
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_DiagnosticStream(
                    response.stream,
                    response.headers.get("content-encoding", ""),
                    lambda value: diagnostic.__setitem__("response", value),
                ),
                extensions=response.extensions,
                request=request,
            )
        return response

    def close(self) -> None:
        self._inner.close()
