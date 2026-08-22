"""Tests for the LLM client: pricing, error classification, JSON parsing,
retry logic, and the shared rate limiter."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config import Settings
from utils.llm_client import (
    LLMClient,
    RateLimiter,
    classify_error,
    _extract_status_code,
)


def _bare_client(provider: str = "openai", **settings_kwargs) -> LLMClient:
    """Build a client without initializing a real provider SDK."""
    settings = Settings(OPENAI_API_KEY="x", **settings_kwargs)
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.provider = provider
    client._rate_limiter = None
    return client


class _FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestEstimateCost:
    def test_model_aware_pricing(self):
        c = _bare_client(OPENAI_MODEL="gpt-4o")
        expensive = c.estimate_cost(50, 8)["estimated_cost_usd"]
        c.settings.openai_model = "gpt-4o-mini"
        cheap = c.estimate_cost(50, 8)["estimated_cost_usd"]
        assert cheap < expensive  # mini must be cheaper than gpt-4o

    def test_priced_by_model_flag(self):
        c = _bare_client(OPENAI_MODEL="gpt-4o")
        assert c.estimate_cost(10, 5)["priced_by_model"] is True
        c.settings.openai_model = "totally-made-up-model"
        result = c.estimate_cost(10, 5)
        assert result["priced_by_model"] is False
        assert result["estimated_cost_usd"] > 0  # falls back to provider price


class TestClassifyError:
    def test_rate_limit(self):
        diag = classify_error(_FakeHTTPError(429), "openai", "gpt-4o")
        assert diag["retryable"] is True
        assert "Rate Limit" in diag["error_type"]

    def test_auth_error(self):
        diag = classify_error(_FakeHTTPError(401), "openai", "gpt-4o")
        assert diag["retryable"] is False
        assert "Authentication" in diag["error_type"]

    def test_model_not_found(self):
        diag = classify_error(_FakeHTTPError(404), "groq", "dead-model")
        assert "Model Not Found" in diag["error_type"]

    def test_server_error_retryable(self):
        diag = classify_error(_FakeHTTPError(503), "openai", "gpt-4o")
        assert diag["retryable"] is True

    def test_invalid_json(self):
        diag = classify_error(json.JSONDecodeError("x", "doc", 0), "openai", "gpt-4o")
        assert "JSON" in diag["error_type"]


class TestExtractStatusCode:
    def test_from_attribute(self):
        assert _extract_status_code(_FakeHTTPError(429)) == 429

    def test_from_message(self):
        assert _extract_status_code(Exception("got a 503 from server")) == 503

    def test_none_when_absent(self):
        assert _extract_status_code(Exception("something odd")) is None


class TestGenerateJson:
    def test_strips_code_fences(self):
        c = _bare_client()
        c.generate = lambda *a, **k: '```json\n{"a": 1}\n```'
        assert c.generate_json("s", "u") == {"a": 1}

    def test_plain_json(self):
        c = _bare_client()
        c.generate = lambda *a, **k: '{"b": 2}'
        assert c.generate_json("s", "u") == {"b": 2}


class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda *_: None)  # no real waiting
        c = _bare_client(API_CALL_DELAY=0.0)
        calls = {"n": 0}

        def flaky(system_prompt, user_prompt, temperature, max_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeHTTPError(429)
            return "ok"

        c._generate_openai = flaky
        assert c.generate("s", "u") == "ok"
        assert calls["n"] == 2

    def test_non_retryable_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        c = _bare_client(API_CALL_DELAY=0.0)
        calls = {"n": 0}

        def bad(system_prompt, user_prompt, temperature, max_tokens):
            calls["n"] += 1
            raise _FakeHTTPError(401)  # auth error, not retryable

        c._generate_openai = bad
        with pytest.raises(_FakeHTTPError):
            c.generate("s", "u")
        assert calls["n"] == 1


class TestRateLimiter:
    def test_zero_rate_is_noop(self):
        rl = RateLimiter(0)
        start = time.monotonic()
        for _ in range(5):
            rl.acquire()
        assert time.monotonic() - start < 0.05

    def test_spacing(self):
        rl = RateLimiter(20.0)  # 0.05s apart
        start = time.monotonic()
        for _ in range(4):
            rl.acquire()
        # 4 acquisitions => at least 3 intervals of 0.05s
        assert time.monotonic() - start >= 0.15 - 0.02
