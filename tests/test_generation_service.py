"""Tests for the GenerationService: concurrency, checkpointing, and resume."""

import itertools
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config import Settings
from models.form_schema import FormQuestion, FormSchema, QuestionType
from services.generation import GenerationService, peek_checkpoint, run_key


def _schema():
    return FormSchema(
        form_url="https://docs.google.com/forms/d/e/x/viewform",
        form_title="Test Form",
        questions=[
            FormQuestion(
                question_id="q_1", entry_id="entry.1", question_text="Fav color?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                options=["Red", "Blue"], is_required=True,
            ),
        ],
    )


class _FakeLLM:
    """Routes persona vs. form-fill by the persona prompt's unique marker."""

    def __init__(self):
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def generate_json(self, system_prompt, user_prompt, temperature=0.9, max_tokens=4096):
        with self._lock:
            n = next(self._counter)
        if "respondent persona" in system_prompt.lower():
            return {
                "name": f"Person {n}", "age": 20 + n % 5, "gender": "nb",
                "major": f"Major {n}", "year": "Junior", "university": "ASU",
                "interests": ["a", "b"], "personality_traits": ["curious"],
                "engagement_level": "high", "attitude_toward_topic": "positive",
                "background_context": "bg",
            }
        return {"q_1": "Red"}


@pytest.fixture
def settings():
    return Settings(OPENAI_API_KEY="x", API_CALL_DELAY=0.0, CONCURRENCY=4)


def test_run_generates_requested_count(tmp_path, settings):
    svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                            checkpoint_dir=tmp_path)
    events = list(svc.run(10))
    assert len(events) == 10
    assert svc.completed_count() == 10
    assert all(r.generation_success for r in svc.dataset.responses)
    # events carry monotonically increasing completion counts
    assert [e.completed for e in events] == list(range(1, 11))


def test_unique_personas_under_concurrency(tmp_path, settings):
    svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                            checkpoint_dir=tmp_path)
    list(svc.run(20))
    ids = [r.persona_id for r in svc.dataset.responses]
    assert len(set(ids)) == 20  # no lost updates in the shared persona set


def test_stop_halts_early(tmp_path, settings):
    svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                            checkpoint_dir=tmp_path)
    seen = {"n": 0}

    def should_stop():
        return seen["n"] >= 5

    for _ in svc.run(50, should_stop=should_stop):
        seen["n"] += 1

    assert svc.stopped_reason == "user"
    assert svc.completed_count() < 50


def test_checkpoint_written_and_resumed(tmp_path, settings):
    schema = _schema()
    # First run: stop after 6.
    svc = GenerationService(schema, settings, llm_client=_FakeLLM(),
                            checkpoint_dir=tmp_path)
    seen = {"n": 0}
    for _ in svc.run(10, should_stop=lambda: seen["n"] >= 6):
        seen["n"] += 1
    saved = peek_checkpoint(schema, settings, None, checkpoint_dir=tmp_path)
    assert saved == svc.completed_count() >= 6

    # Resume: should load prior work and only top up to 10.
    svc2 = GenerationService(schema, settings, llm_client=_FakeLLM(),
                             checkpoint_dir=tmp_path, resume=True)
    loaded = svc2.completed_count()
    assert loaded == saved
    remainder = list(svc2.run(10))
    assert len(remainder) == 10 - loaded
    assert svc2.completed_count() == 10
    # Completing the run clears the checkpoint.
    assert peek_checkpoint(schema, settings, None, checkpoint_dir=tmp_path) == 0


def test_fresh_run_clears_stale_checkpoint(tmp_path, settings):
    schema = _schema()
    svc = GenerationService(schema, settings, llm_client=_FakeLLM(),
                            checkpoint_dir=tmp_path)
    list(svc.run(3, should_stop=lambda: True))  # write nothing / stop immediately
    # A brand-new non-resume service on the same key starts empty.
    svc2 = GenerationService(schema, settings, llm_client=_FakeLLM(),
                             checkpoint_dir=tmp_path, resume=False)
    assert svc2.completed_count() == 0


def test_run_key_stable_and_config_sensitive():
    a = run_key("url", "openai", "gpt-4o", None)
    b = run_key("url", "openai", "gpt-4o", None)
    c = run_key("url", "openai", "gpt-4o-mini", None)
    assert a == b
    assert a != c
