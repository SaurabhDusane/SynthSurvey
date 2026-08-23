"""Tests for Phase 2 research features: latent traits, A/B stimulus, waves."""

import itertools
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config import Settings
from models.form_schema import FormQuestion, FormSchema, QuestionType
from services.generation import GenerationService
from services.stimulus import build_stimulus_plan, clean_variants
from services.traits import DEFAULT_TRAIT_NAMES, sample_traits, traits_directive


def _schema():
    return FormSchema(
        form_url="https://docs.google.com/forms/d/e/x/viewform", form_title="T",
        questions=[FormQuestion(
            question_id="q_1", entry_id="e", question_text="Fav?",
            question_type=QuestionType.MULTIPLE_CHOICE, options=["Red", "Blue"], is_required=True)],
    )


class _FakeLLM:
    def __init__(self):
        self._c = itertools.count()
        self._l = threading.Lock()

    def generate_json(self, system_prompt, user_prompt, temperature=0.9, max_tokens=4096):
        with self._l:
            n = next(self._c)
        if "respondent persona" in system_prompt.lower():
            return {"name": f"P{n}", "age": 22, "gender": "nb", "major": "CS", "year": "Junior",
                    "university": "ASU", "interests": ["a", "b"], "personality_traits": ["x"],
                    "engagement_level": "high", "attitude_toward_topic": "pos", "background_context": "bg"}
        return {"q_1": "Red"}


@pytest.fixture
def settings():
    return Settings(OPENAI_API_KEY="x", API_CALL_DELAY=0.0, CONCURRENCY=4)


class TestTraits:
    def test_sample_deterministic(self):
        assert sample_traits(3, seed=7) == sample_traits(3, seed=7)

    def test_different_index_differs(self):
        assert sample_traits(1) != sample_traits(2)

    def test_values_in_range_and_named(self):
        t = sample_traits(0)
        assert set(t) == set(DEFAULT_TRAIT_NAMES)
        assert all(0.0 <= v <= 1.0 for v in t.values())

    def test_directive_mentions_traits(self):
        d = traits_directive({"openness": 0.9})
        assert "openness" in d and "0.9" in d

    def test_directive_empty(self):
        assert traits_directive({}) == ""


class TestStimulus:
    def test_clean_drops_empty(self):
        v = clean_variants([{"name": "A", "text": "hi"}, {"name": "B", "text": "  "}])
        assert len(v) == 1 and v[0]["name"] == "A"

    def test_default_names(self):
        v = clean_variants([{"text": "x"}, {"text": "y"}])
        assert v[0]["name"] == "Variant A" and v[1]["name"] == "Variant B"

    def test_single_variant_applies_to_all(self):
        plan = build_stimulus_plan([{"name": "A", "text": "x"}], 5)
        assert len(plan) == 5 and all(p["name"] == "A" for p in plan)

    def test_even_split(self):
        plan = build_stimulus_plan([{"name": "A", "text": "x"}, {"name": "B", "text": "y"}], 10)
        counts = Counter(p["name"] for p in plan)
        assert counts["A"] == 5 and counts["B"] == 5

    def test_empty_when_no_variants(self):
        assert build_stimulus_plan([], 10) == []


class TestServiceIntegration:
    def test_traits_attached_to_personas_and_responses(self, tmp_path, settings):
        svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                                checkpoint_dir=tmp_path, enable_traits=True)
        list(svc.run(5))
        personas = svc.persona_gen.generated_personas
        assert all(p.latent_traits for p in personas)
        assert all(r.latent_traits for r in svc.dataset.responses)

    def test_stimulus_recorded_and_split(self, tmp_path, settings):
        plan = build_stimulus_plan(
            [{"name": "A", "text": "concept a"}, {"name": "B", "text": "concept b"}], 8)
        svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                                checkpoint_dir=tmp_path, stimulus_plan=plan)
        list(svc.run(8))
        variants = Counter(r.stimulus_variant for r in svc.dataset.responses)
        assert variants["A"] == 4 and variants["B"] == 4

    def test_waves_reuse_personas(self, tmp_path, settings):
        svc = GenerationService(_schema(), settings, llm_client=_FakeLLM(),
                                checkpoint_dir=tmp_path, waves=3)
        events = list(svc.run(4))
        assert len(events) == 12  # 4 personas x 3 waves
        waves = Counter(r.wave for r in svc.dataset.responses)
        assert waves == {1: 4, 2: 4, 3: 4}
        # Each persona_id appears once per wave -> 3 times total.
        id_counts = Counter(r.persona_id for r in svc.dataset.responses)
        assert all(c == 3 for c in id_counts.values())
        assert len(id_counts) == 4
