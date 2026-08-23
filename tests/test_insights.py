"""Tests for Phase 3: stats/significance, survey linting, and study configs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from analysis.stats import (
    chi2_sf,
    chi_square_test,
    crosstab_test,
    effect_size_label,
)
from analysis.survey_lint import lint_survey
from models.form_schema import FormQuestion, FormSchema, QuestionType, ScaleConfig
from services.study import (
    STUDY_KIND,
    build_study_config,
    parse_study_config,
    study_to_bytes,
)


class TestStats:
    @pytest.mark.parametrize("chi2,dof,expected", [
        (3.841, 1, 0.05), (5.991, 2, 0.05), (11.345, 3, 0.01),
    ])
    def test_chi2_sf_matches_critical_values(self, chi2, dof, expected):
        assert round(chi2_sf(chi2, dof), 3) == expected

    def test_chi2_sf_zero(self):
        assert chi2_sf(0, 1) == 1.0

    def test_strong_association_low_p(self):
        r = chi_square_test([[40, 10], [10, 40]])
        assert r["p_value"] < 0.001
        assert r["cramers_v"] > 0.5

    def test_independence_high_p(self):
        r = chi_square_test([[25, 25], [25, 25]])
        assert r["p_value"] == pytest.approx(1.0)
        assert r["chi2"] == 0.0

    def test_degenerate_table(self):
        r = chi_square_test([[5, 3, 2]])  # single row
        assert r["p_value"] is None

    def test_crosstab_test(self):
        df = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "answer": ["Yes"] * 40 + ["No"] * 10 + ["Yes"] * 10 + ["No"] * 40,
        })
        r = crosstab_test(df, "group", "answer")
        assert r["n"] == 100
        assert r["p_value"] < 0.001
        assert not r["table"].empty

    def test_effect_size_labels(self):
        assert effect_size_label(0.05) == "negligible"
        assert effect_size_label(0.6) == "large"
        assert effect_size_label(None) == "n/a"


class TestSurveyLint:
    def test_leading_question_flagged(self):
        schema = FormSchema(form_url="u", form_title="T", questions=[
            FormQuestion(question_id="q1", entry_id="e", question_text="Don't you agree our service is great?",
                         question_type=QuestionType.LINEAR_SCALE, scale_config=ScaleConfig()),
        ])
        issues = [f["issue"] for f in lint_survey(schema)["findings"]]
        assert any("Leading" in i for i in issues)

    def test_double_barreled_flagged(self):
        schema = FormSchema(form_url="u", form_title="T", questions=[
            FormQuestion(question_id="q1", entry_id="e", question_text="Rate the food and the service",
                         question_type=QuestionType.LINEAR_SCALE,
                         scale_config=ScaleConfig(min_label="bad", max_label="good")),
        ])
        issues = [f["issue"] for f in lint_survey(schema)["findings"]]
        assert any("double-barreled" in i.lower() for i in issues)

    def test_missing_opt_out_flagged(self):
        schema = FormSchema(form_url="u", form_title="T", questions=[
            FormQuestion(question_id="q1", entry_id="e", question_text="Pick one",
                         question_type=QuestionType.MULTIPLE_CHOICE, options=["Red", "Blue"]),
        ])
        issues = [f["issue"] for f in lint_survey(schema)["findings"]]
        assert any("opt-out" in i.lower() for i in issues)

    def test_clean_survey(self):
        schema = FormSchema(form_url="u", form_title="T", questions=[
            FormQuestion(question_id="q1", entry_id="e", question_text="What is your favorite color?",
                         question_type=QuestionType.MULTIPLE_CHOICE, options=["Red", "Blue", "Prefer not to say"]),
        ])
        result = lint_survey(schema)
        assert result["warn_count"] == 0


class TestStudyConfig:
    def _state(self):
        return {
            "form_url": "https://docs.google.com/forms/d/e/x/viewform",
            "num_responses": 30, "persona_constraints": "grad students",
            "llm_provider": "groq", "temperature": 0.8, "api_call_delay": 0.5,
            "concurrency": 4, "max_retries": 2, "quota_targets": {"gender": {"Female": 50, "Male": 50}},
            "enable_traits": True, "waves": 2, "stimuli": [{"name": "A", "text": "x"}],
            "groq_model": "llama-3.3-70b-versatile", "openai_model": "gpt-4o",
            # Secrets that must NOT be exported:
            "api_key_groq": "gsk_secret", "api_key_openai": "sk-secret",
        }

    def test_round_trip(self):
        cfg = build_study_config(self._state())
        blob = study_to_bytes(cfg)
        parsed = parse_study_config(blob)
        assert parsed["kind"] == STUDY_KIND
        assert parsed["num_responses"] == 30
        assert parsed["enable_traits"] is True
        assert parsed["models"]["groq_model"] == "llama-3.3-70b-versatile"

    def test_no_api_keys_exported(self):
        cfg = build_study_config(self._state())
        blob = study_to_bytes(cfg).decode()
        assert "gsk_secret" not in blob and "sk-secret" not in blob
        assert not any("api_key" in k for k in cfg)

    def test_rejects_non_study_file(self):
        with pytest.raises(ValueError):
            parse_study_config('{"kind": "something_else"}')
