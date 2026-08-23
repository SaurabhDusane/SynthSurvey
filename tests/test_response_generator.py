"""Tests for the response validator and response models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from models.form_schema import (
    FormQuestion,
    FormSchema,
    QuestionType,
    ScaleConfig,
)
from models.response import GeneratedResponse, SurveyDataset
from utils.validators import ResponseValidator, fix_response


def make_schema(questions: list[FormQuestion]) -> FormSchema:
    """Helper to create a FormSchema with given questions."""
    return FormSchema(
        form_url="https://example.com/form",
        form_title="Test Form",
        questions=questions,
    )


class TestResponseValidator:
    """Tests for ResponseValidator."""

    def test_valid_multiple_choice(self):
        """Valid multiple choice answer should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Favorite color?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    options=["Red", "Blue", "Green"],
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "Red"})
        assert is_valid
        assert len(errors) == 0

    def test_invalid_multiple_choice(self):
        """Invalid multiple choice answer should fail."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Favorite color?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    options=["Red", "Blue", "Green"],
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "Purple"})
        assert not is_valid
        assert len(errors) == 1

    def test_missing_required(self):
        """Missing required answer should fail."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Name?",
                    question_type=QuestionType.SHORT_TEXT,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({})
        assert not is_valid

    def test_optional_missing_ok(self):
        """Missing optional answer should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Comments?",
                    question_type=QuestionType.PARAGRAPH,
                    is_required=False,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({})
        assert is_valid

    def test_valid_scale(self):
        """Valid scale answer should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Rate 1-5",
                    question_type=QuestionType.LINEAR_SCALE,
                    scale_config=ScaleConfig(min_value=1, max_value=5),
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": 3})
        assert is_valid

    def test_invalid_scale_out_of_range(self):
        """Scale answer out of range should fail."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Rate 1-5",
                    question_type=QuestionType.LINEAR_SCALE,
                    scale_config=ScaleConfig(min_value=1, max_value=5),
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": 7})
        assert not is_valid

    def test_valid_checkbox(self):
        """Valid checkbox answer should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Select all",
                    question_type=QuestionType.CHECKBOX,
                    options=["A", "B", "C"],
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": ["A", "C"]})
        assert is_valid

    def test_invalid_checkbox_option(self):
        """Checkbox with invalid option should fail."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Select all",
                    question_type=QuestionType.CHECKBOX,
                    options=["A", "B", "C"],
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": ["A", "D"]})
        assert not is_valid

    def test_valid_short_text(self):
        """Valid short text should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Name?",
                    question_type=QuestionType.SHORT_TEXT,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "John Doe"})
        assert is_valid

    def test_valid_date(self):
        """Valid date format should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Date?",
                    question_type=QuestionType.DATE,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "2024-03-15"})
        assert is_valid

    def test_invalid_date_format(self):
        """Invalid date format should fail."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Date?",
                    question_type=QuestionType.DATE,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "March 15, 2024"})
        assert not is_valid

    def test_valid_time(self):
        """Valid time format should pass."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Time?",
                    question_type=QuestionType.TIME,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "14:30"})
        assert is_valid

    def test_case_insensitive_choice(self):
        """Multiple choice validation should be case-insensitive."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Color?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    options=["Red", "Blue"],
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "red"})
        assert is_valid

    def test_other_option_allowed(self):
        """'Other' option should allow free text."""
        schema = make_schema(
            [
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="Color?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    options=["Red", "Blue"],
                    has_other_option=True,
                    is_required=True,
                )
            ]
        )
        validator = ResponseValidator(schema)
        is_valid, errors = validator.validate({"q_1": "Purple (custom)"})
        assert is_valid


class TestFixResponse:
    """Tests for the fix_response utility."""

    def test_fix_case_mismatch(self):
        """Should fix case mismatches in multiple choice."""
        q = FormQuestion(
            question_id="q_1",
            entry_id="entry.1",
            question_text="Color?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=["Red", "Blue", "Green"],
        )
        fixed = fix_response(q, "red")
        assert fixed == "Red"

    def test_fix_scale_clamping(self):
        """Should clamp scale values to valid range."""
        q = FormQuestion(
            question_id="q_1",
            entry_id="entry.1",
            question_text="Rate",
            question_type=QuestionType.LINEAR_SCALE,
            scale_config=ScaleConfig(min_value=1, max_value=5),
        )
        fixed = fix_response(q, 7)
        assert fixed == 5

    def test_fix_checkbox_string_to_list(self):
        """Should convert string checkbox answer to list."""
        q = FormQuestion(
            question_id="q_1",
            entry_id="entry.1",
            question_text="Select",
            question_type=QuestionType.CHECKBOX,
            options=["A", "B", "C"],
        )
        fixed = fix_response(q, "A")
        assert fixed == ["A"]


class TestSurveyDataset:
    """Tests for the SurveyDataset model."""

    def test_add_response(self):
        """Adding a response should increment total_generated."""
        dataset = SurveyDataset(
            form_title="Test",
            form_url="https://example.com",
        )
        resp = GeneratedResponse(
            persona_id="abc",
            persona_summary="Test persona",
            answers={"q_1": "answer"},
        )
        dataset.add_response(resp)
        assert dataset.total_generated == 1
        assert len(dataset.responses) == 1

    def test_response_is_synthetic(self):
        """Generated responses should always be marked synthetic."""
        resp = GeneratedResponse(
            persona_id="abc",
            persona_summary="Test",
            answers={},
        )
        assert resp.is_synthetic is True

    def test_response_timestamp(self):
        """Generated responses should have a timestamp."""
        resp = GeneratedResponse(
            persona_id="abc",
            persona_summary="Test",
            answers={},
        )
        assert resp.generation_timestamp is not None
        assert len(resp.generation_timestamp) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
