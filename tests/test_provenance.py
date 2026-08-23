"""Tests for Phase 4: provenance manifest, seeding, and the persona-context trace."""

import itertools
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from config import Settings
from models.form_schema import FormQuestion, FormSchema, QuestionType
from models.response import GeneratedResponse, SurveyDataset
from services.generation import GenerationService
from services.provenance import (
    APP_VERSION,
    DISCLAIMER,
    build_manifest,
    manifest_to_bytes,
)
from services.quota import build_assignment_plan


def _schema():
    return FormSchema(
        form_url="https://docs.google.com/forms/d/e/x/viewform", form_title="T",
        questions=[FormQuestion(question_id="q_1", entry_id="e", question_text="Fav?",
                                question_type=QuestionType.MULTIPLE_CHOICE,
                                options=["Red", "Blue"], is_required=True)],
    )


def _dataset(n=5):
    ds = SurveyDataset(form_title="T", form_url="u")
    for i in range(n):
        ds.add_response(GeneratedResponse(persona_id=f"p{i}", persona_summary="s",
                                          answers={"q_1": "Red"}, generation_success=True))
    ds.generation_completed = "2026-01-01T00:00:00Z"
    return ds


class _FakeLLM:
    def __init__(self):
        self._c = itertools.count()
        self._l = threading.Lock()

    def generate_json(self, system_prompt, user_prompt, temperature=0.9, max_tokens=4096):
        with self._l:
            n = next(self._c)
        if "respondent persona" in system_prompt.lower():
            return {"name": f"P{n}", "age": 22, "gender": "nb", "major": "CS", "year": "Junior",
                    "university": "ASU", "interests": ["a", "b"], "personality_traits": ["curious"],
                    "engagement_level": "high", "attitude_toward_topic": "positive", "background_context": "bg"}
        return {"q_1": "Red"}


class TestProvenance:
    def test_manifest_structure(self):
        s = Settings(OPENAI_API_KEY="secret", LLM_PROVIDER="openai", OPENAI_MODEL="gpt-4o", TEMPERATURE=0.5)
        m = build_manifest(_dataset(3), _schema(), s, study_config={"num_responses": 3}, seed=7)
        assert m["tool"] == "SynthSurvey"
        assert m["version"] == APP_VERSION
        assert m["disclaimer"] == DISCLAIMER
        assert m["generation"]["seed"] == 7
        assert m["generation"]["provider"] == "openai"
        assert m["counts"]["successful"] == 3

    def test_config_hash_stable_and_sensitive(self):
        s = Settings(OPENAI_API_KEY="x")
        h1 = build_manifest(_dataset(1), _schema(), s, {"a": 1, "b": 2})["config_hash"]
        h2 = build_manifest(_dataset(1), _schema(), s, {"b": 2, "a": 1})["config_hash"]  # order-independent
        h3 = build_manifest(_dataset(1), _schema(), s, {"a": 9})["config_hash"]
        assert h1 == h2
        assert h1 != h3

    def test_no_api_key_in_manifest(self):
        s = Settings(OPENAI_API_KEY="super-secret-key")
        blob = manifest_to_bytes(build_manifest(_dataset(1), _schema(), s, {"num_responses": 1})).decode()
        assert "super-secret-key" not in blob


class TestSeedReproducibility:
    def test_quota_plan_reproducible(self):
        spec = {"gender": {"Female": 50, "Male": 50}}
        assert build_assignment_plan(spec, 40, seed=123) == build_assignment_plan(spec, 40, seed=123)

    def test_different_seed_differs(self):
        spec = {"gender": {"Female": 50, "Male": 50}}
        a = build_assignment_plan(spec, 40, seed=1)
        b = build_assignment_plan(spec, 40, seed=2)
        assert a != b  # ordering differs, though marginals match


class TestPersonaContextTrace:
    def test_response_carries_context(self, tmp_path):
        s = Settings(OPENAI_API_KEY="x", API_CALL_DELAY=0.0)
        svc = GenerationService(_schema(), s, llm_client=_FakeLLM(), checkpoint_dir=tmp_path)
        list(svc.run(3))
        for r in svc.dataset.responses:
            ctx = r.persona_context
            assert ctx.get("engagement_level") == "high"
            assert ctx.get("attitude_toward_topic")
            assert "background_context" in ctx
