"""Tests for fidelity scoring and quality checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from analysis.fidelity import compute_fidelity, fidelity_verdict
from analysis.quality import analyze_quality
from models.form_schema import FormQuestion, FormSchema, QuestionType, ScaleConfig


def _schema(questions):
    return FormSchema(form_url="u", form_title="T", questions=questions)


def _mc(text):
    return FormQuestion(
        question_id=f"q_{text}", entry_id="e", question_text=text,
        question_type=QuestionType.MULTIPLE_CHOICE, options=["Red", "Blue"],
    )


def _scale(text, lo=1, hi=5):
    return FormQuestion(
        question_id=f"q_{text}", entry_id="e", question_text=text,
        question_type=QuestionType.LINEAR_SCALE,
        scale_config=ScaleConfig(min_value=lo, max_value=hi),
    )


class TestFidelity:
    def test_identical_distributions_score_high(self):
        schema = _schema([_mc("Color")])
        ref = pd.DataFrame({"Color": ["Red", "Blue", "Red", "Blue"]})
        syn = pd.DataFrame({"Color": ["Red", "Blue", "Red", "Blue"]})
        result = compute_fidelity(syn, ref, schema)
        assert result["overall_score"] == 100.0
        assert result["scored"] == 1

    def test_disjoint_distributions_score_low(self):
        schema = _schema([_mc("Color")])
        ref = pd.DataFrame({"Color": ["Red", "Red", "Red", "Red"]})
        syn = pd.DataFrame({"Color": ["Blue", "Blue", "Blue", "Blue"]})
        result = compute_fidelity(syn, ref, schema)
        assert result["overall_score"] == 0.0

    def test_text_questions_skipped(self):
        q = FormQuestion(
            question_id="q1", entry_id="e", question_text="Comments",
            question_type=QuestionType.PARAGRAPH,
        )
        schema = _schema([q])
        ref = pd.DataFrame({"Comments": ["a", "b"]})
        syn = pd.DataFrame({"Comments": ["c", "d"]})
        result = compute_fidelity(syn, ref, schema)
        assert result["overall_score"] is None
        assert result["skipped"] == 1

    def test_verdict_labels(self):
        assert fidelity_verdict(90) == "Excellent match"
        assert fidelity_verdict(None) == "Not scored"
        assert fidelity_verdict(40) == "Poor match"


class TestQuality:
    def test_straightlining_flagged(self):
        schema = _schema([_scale("q1"), _scale("q2"), _scale("q3")])
        # Every respondent answers all 3s -> 100% straightlining.
        df = pd.DataFrame({"q1": [3, 3, 3], "q2": [3, 3, 3], "q3": [3, 3, 3]})
        result = analyze_quality(df, schema)
        sl = next(f for f in result["findings"] if f["check"] == "Straightlining")
        assert sl["severity"] == "warn"
        assert result["warn_count"] >= 1

    def test_no_straightlining_when_varied(self):
        schema = _schema([_scale("q1"), _scale("q2"), _scale("q3")])
        df = pd.DataFrame({"q1": [1, 4, 2], "q2": [3, 1, 5], "q3": [2, 5, 1]})
        result = analyze_quality(df, schema)
        sl = next(f for f in result["findings"] if f["check"] == "Straightlining")
        assert sl["severity"] == "pass"

    def test_high_skew_flagged(self):
        schema = _schema([_scale("q1"), _scale("q2"), _scale("q3")])
        df = pd.DataFrame({"q1": [5, 5, 5], "q2": [5, 4, 5], "q3": [4, 5, 5]})
        result = analyze_quality(df, schema)
        skew = next(f for f in result["findings"] if f["check"] == "Response skew")
        assert skew["severity"] == "warn"

    def test_duplicate_rows_flagged(self):
        schema = _schema([_mc("Color")])
        df = pd.DataFrame({"Color": ["Red", "Red", "Blue"]})
        result = analyze_quality(df, schema)
        dup = next(f for f in result["findings"] if f["check"] == "Duplicate responses")
        assert int(dup["metric"]) >= 1
        assert dup["severity"] == "warn"

    def test_clean_data_no_warnings(self):
        schema = _schema([_scale("q1"), _scale("q2"), _scale("q3")])
        df = pd.DataFrame({
            "q1": [1, 2, 3, 4, 5],
            "q2": [5, 4, 3, 2, 1],
            "q3": [2, 4, 1, 5, 3],
        })
        result = analyze_quality(df, schema)
        assert result["warn_count"] == 0
