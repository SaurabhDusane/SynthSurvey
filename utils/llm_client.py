"""Abstraction over OpenAI, Anthropic, Gemini, Groq, Mistral, and Cohere LLM APIs."""

import json
import logging
import threading
import time
from typing import Any, Optional

from config import Settings

log = logging.getLogger(__name__)

# HTTP status codes that are retryable
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


class RateLimiter:
    """Thread-safe token-bucket-style limiter that paces the *start* of calls.

    Each caller reserves the next available time slot and sleeps outside the
    lock, so concurrent workers are globally spaced by ``min_interval`` while
    their (slow) network calls still overlap.
    """

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec and rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_time)
            self._next_time = scheduled + self.min_interval
            wait = scheduled - now
        if wait > 0:
            time.sleep(wait)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Try to pull an HTTP status code from various SDK exception types."""
    for attr in ("status_code", "code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Some SDKs nest it inside .response
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    # Check string representation for common patterns
    msg = str(exc)
    for code in (429, 500, 502, 503, 529):
        if str(code) in msg:
            return code
    return None


def classify_error(exc: Exception, provider: str, model: str) -> dict:
    """Classify an error into a human-readable diagnosis with fix suggestions.

    Returns:
        dict with keys: error_type, message, suggestion, retryable
    """
    msg = str(exc).strip()
    status = _extract_status_code(exc)

    if status == 429 or "rate" in msg.lower() or "quota" in msg.lower():
        return {
            "error_type": "Rate Limit / Quota Exceeded",
            "message": msg,
            "suggestion": (
                f"Your {provider} API quota is exhausted. Options:\n"
                f"  1. Wait a few minutes and retry\n"
                f"  2. Upgrade your {provider} plan for higher limits\n"
                f"  3. Switch to a different provider in the sidebar\n"
                f"  4. Increase the 'API Call Delay' slider to space out requests"
            ),
            "retryable": True,
        }
    if status == 401 or "auth" in msg.lower() or ("invalid" in msg.lower() and "key" in msg.lower()):
        return {
            "error_type": "Authentication Error",
            "message": msg,
            "suggestion": (
                f"Your {provider} API key is invalid or expired.\n"
                f"  1. Double-check the key in the sidebar\n"
                f"  2. Generate a new key from {provider}'s dashboard"
            ),
            "retryable": False,
        }
    if status == 404 or "not found" in msg.lower():
        return {
            "error_type": "Model Not Found",
            "message": msg,
            "suggestion": (
                f"The model '{model}' is not available.\n"
                f"  1. Select a different model in Model Settings\n"
                f"  2. Check {provider}'s docs for valid model names"
            ),
            "retryable": False,
        }
    if status in (500, 502, 503, 529):
        return {
            "error_type": "Server Error",
            "message": msg,
            "suggestion": (
                f"The {provider} API returned a server error (HTTP {status}).\n"
                f"  1. This is usually temporary — retry in a minute\n"
                f"  2. Check {provider}'s status page for outages"
            ),
            "retryable": True,
        }
    if isinstance(exc, json.JSONDecodeError):
        return {
            "error_type": "Invalid JSON Response",
            "message": msg,
            "suggestion": (
                "The model returned text that isn't valid JSON.\n"
                "  1. Try lowering the temperature for more predictable output\n"
                "  2. Retry — this is often intermittent"
            ),
            "retryable": True,
        }
    return {
        "error_type": "Unknown Error",
        "message": msg,
        "suggestion": (
            "An unexpected error occurred.\n"
            "  1. Check the full error message above\n"
            "  2. Try a different provider or model\n"
            "  3. If persistent, report this as a bug"
        ),
        "retryable": False,
    }


# Approximate pricing per 1M tokens, keyed by exact model name: (input, output).
# Falls back to _PROVIDER_PRICING when a model isn't listed here.
_MODEL_PRICING = {
    # OpenAI
    "gpt-4o":                       (2.50, 10.00),
    "gpt-4o-mini":                  (0.15, 0.60),
    "gpt-4-turbo":                  (10.00, 30.00),
    "gpt-3.5-turbo":                (0.50, 1.50),
    # Anthropic
    "claude-3-5-sonnet-20241022":   (3.00, 15.00),
    "claude-3-opus-20240229":       (15.00, 75.00),
    "claude-3-haiku-20240307":      (0.25, 1.25),
    # Google Gemini
    "gemini-2.0-flash":             (0.10, 0.40),
    "gemini-2.0-flash-lite":        (0.075, 0.30),
    "gemini-1.5-pro-latest":        (1.25, 5.00),
    "gemini-1.5-flash-latest":      (0.075, 0.30),
    # Groq
    "llama-3.3-70b-versatile":      (0.59, 0.79),
    "llama-3.1-8b-instant":         (0.05, 0.08),
    "gemma2-9b-it":                 (0.20, 0.20),
    # Mistral
    "mistral-large-latest":         (2.00, 6.00),
    "mistral-medium-latest":        (2.75, 8.10),
    "mistral-small-latest":         (0.20, 0.60),
    "open-mixtral-8x22b":           (2.00, 6.00),
    # Cohere
    "command-r-plus":               (2.50, 10.00),
    "command-r":                    (0.15, 0.60),
    "command-light":                (0.30, 0.60),
}

# Per-provider fallback pricing when the exact model isn't in _MODEL_PRICING.
_PROVIDER_PRICING = {
    "openai":    (2.50, 10.00),
    "anthropic": (3.00, 15.00),
    "gemini":    (1.25, 5.00),
    "groq":      (0.59, 0.79),
    "mistral":   (2.00, 6.00),
    "cohere":    (2.50, 10.00),
}


# Provider-to-key/model mapping, derived from the central provider registry.
from providers import PROVIDERS as _PROVIDERS

_PROVIDER_META = {
    pid: {"key_attr": spec.key_attr, "model_attr": spec.model_attr}
    for pid, spec in _PROVIDERS.items()
}


class LLMClient:
    """Unified LLM client supporting OpenAI, Anthropic, Gemini, Groq, Mistral, and Cohere."""

    def __init__(self, settings: Settings, rate_limiter: Optional[RateLimiter] = None):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self._rate_limiter = rate_limiter
        self._client = None
        # Keep legacy attrs for backward compat
        self._openai_client = None
        self._anthropic_client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        meta = _PROVIDER_META.get(self.provider)
        if meta is None:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        api_key = getattr(self.settings, meta["key_attr"], None)
        if not api_key:
            raise ValueError(
                f"{meta['key_attr'].upper()} is required when using {self.provider} provider"
            )

        if self.provider == "openai":
            import openai
            self._openai_client = openai.OpenAI(api_key=api_key)
            self._client = self._openai_client
        elif self.provider == "anthropic":
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            self._client = self._anthropic_client
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)
        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=api_key)
        elif self.provider == "mistral":
            from mistralai import Mistral
            self._client = Mistral(api_key=api_key)
        elif self.provider == "cohere":
            import cohere
            self._client = cohere.ClientV2(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            system_prompt: System/context prompt.
            user_prompt: User/task prompt.
            temperature: Sampling temperature (higher = more creative).
            max_tokens: Maximum tokens in the response.

        Returns:
            The raw text response from the LLM.
        """
        dispatch = {
            "openai": self._generate_openai,
            "anthropic": self._generate_anthropic,
            "gemini": self._generate_gemini,
            "groq": self._generate_groq,
            "mistral": self._generate_mistral,
            "cohere": self._generate_cohere,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError(f"Unsupported provider: {self.provider}")

        max_retries = getattr(self.settings, "max_retries", 2)
        last_exc = None
        for attempt in range(max_retries + 1):
            # Rate limiting: a shared limiter paces concurrent workers; without
            # one, fall back to a simple per-call delay.
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            elif self.settings.api_call_delay > 0:
                time.sleep(self.settings.api_call_delay)
            try:
                return fn(system_prompt, user_prompt, temperature, max_tokens)
            except Exception as exc:
                last_exc = exc
                status = _extract_status_code(exc)
                if status in _RETRYABLE_STATUS and attempt < max_retries:
                    wait = min(2 ** attempt * 2, 60)  # 2s, 4s, 8s, ... capped at 60s
                    log.warning(
                        "Retryable error (HTTP %s) on attempt %d/%d — waiting %ds",
                        status, attempt + 1, max_retries + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                raise  # non-retryable or out of retries
        raise last_exc  # should not reach here, but just in case

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using OpenAI API."""
        response = self._openai_client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Anthropic API."""
        response = self._anthropic_client.messages.create(
            model=self.settings.anthropic_model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text

    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using the Google Gemini API (google-genai SDK)."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text

    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Groq API (Llama, Mixtral, etc.)."""
        response = self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _generate_mistral(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Mistral API."""
        response = self._client.chat.complete(
            model=self.settings.mistral_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _generate_cohere(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Cohere API."""
        response = self._client.chat(
            model=self.settings.cohere_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.message.content[0].text

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the LLM.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
        """
        raw = self.generate(system_prompt, user_prompt, temperature, max_tokens)
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)

    def test_connection(self) -> dict:
        """Send a minimal request to verify the API key and model work.

        Returns:
            dict with keys: success, message, provider, model
        """
        meta = _PROVIDER_META.get(self.provider, {})
        model = getattr(self.settings, meta.get("model_attr", ""), "unknown")
        try:
            result = self.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Respond with exactly: OK",
                temperature=0.0,
                max_tokens=10,
            )
            return {
                "success": True,
                "message": f"Connected successfully. Response: {result.strip()[:50]}",
                "provider": self.provider,
                "model": model,
            }
        except Exception as exc:
            diag = classify_error(exc, self.provider, model)
            return {
                "success": False,
                "message": diag["message"][:200],
                "error_type": diag["error_type"],
                "suggestion": diag["suggestion"],
                "provider": self.provider,
                "model": model,
            }

    def estimate_cost(
        self, num_responses: int, num_questions: int
    ) -> dict[str, float]:
        """Estimate the cost of generating responses.

        Returns a dict with estimated input/output tokens and cost in USD.
        """
        # Rough estimates per response
        persona_input_tokens = 500
        persona_output_tokens = 300
        form_fill_input_tokens = 300 + (num_questions * 80)
        form_fill_output_tokens = num_questions * 50

        total_input = num_responses * (persona_input_tokens + form_fill_input_tokens)
        total_output = num_responses * (persona_output_tokens + form_fill_output_tokens)

        # Resolve the exact selected model, then price by model (falling back
        # to a per-provider default) so estimates track the chosen model.
        model = getattr(
            self.settings,
            _PROVIDER_META.get(self.provider, {}).get("model_attr", "openai_model"),
            "unknown",
        )
        input_cost_per_m, output_cost_per_m = _MODEL_PRICING.get(
            model, _PROVIDER_PRICING.get(self.provider, (2.50, 10.00))
        )
        priced_by_model = model in _MODEL_PRICING

        estimated_cost = (
            (total_input / 1_000_000) * input_cost_per_m
            + (total_output / 1_000_000) * output_cost_per_m
        )

        return {
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(estimated_cost, 4),
            "provider": self.provider,
            "model": model,
            "priced_by_model": priced_by_model,
        }
