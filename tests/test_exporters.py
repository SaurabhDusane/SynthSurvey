"""Tests for CSV and JSON exporters."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from models.form_schema import FormSchema, FormQuestion, QuestionType
from models.response import GeneratedResponse, SurveyDataset
from exporters.csv_exporter import CSVExporter
from exporters.json_exporter import JSONExporter


def _make_schema() -> FormSchema:
    return FormSchema(
        form_url="https://example.com/form",
        form_title="Test Form",
        questions=[
            FormQuestion(
                question_id="q_1",
                entry_id="entry.1",
                question_text="Name",
                question_type=QuestionType.SHORT_TEXT,
                is_required=True,
            ),
            FormQuestion(
                question_id="q_2",
                entry_id="entry.2",
                question_text="Favorite color",
                question_type=QuestionType.MULTIPLE_CHOICE,
                options=["Red", "Blue", "Green"],
            ),
            FormQuestion(
                question_id="q_3",
                entry_id="entry.3",
                question_text="Hobbies",
                question_type=QuestionType.CHECKBOX,
                options=["Reading", "Gaming", "Hiking"],
            ),
        ],
    )


def _make_dataset() -> SurveyDataset:
    ds = SurveyDataset(form_title="Test Form", form_url="https://example.com/form")
    ds.add_response(
        GeneratedResponse(
            persona_id="p1",
            persona_summary="Alice, 20, CS student",
            answers={"q_1": "Alice", "q_2": "Red", "q_3": ["Reading", "Gaming"]},
        )
    )
    ds.add_response(
        GeneratedResponse(
            persona_id="p2",
            persona_summary="Bob, 22, Biology student",
            answers={"q_1": "Bob", "q_2": "Blue", "q_3": ["Hiking"]},
        )
    )
    return ds


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_to_dataframe_basic(self):
        schema = _make_schema()
        dataset = _make_dataset()
        df = CSVExporter.to_dataframe(dataset, schema)

        assert len(df) == 2
        assert "Name" in df.columns
        assert "Favorite color" in df.columns
        assert "is_synthetic" in df.columns
        assert "persona_id" in df.columns
        assert "persona_summary" in df.columns
        assert "generation_timestamp" in df.columns
        assert df["is_synthetic"].all()

    def test_to_dataframe_checkbox_semicolon(self):
        """Checkbox answers should be joined with semicolons."""
        schema = _make_schema()
        dataset = _make_dataset()
        df = CSVExporter.to_dataframe(dataset, schema)

        hobbies_col = df["Hobbies"]
        assert hobbies_col.iloc[0] == "Reading; Gaming"
        assert hobbies_col.iloc[1] == "Hiking"

    def test_to_dataframe_with_existing_data(self):
        """Existing data should be prepended with is_synthetic=False."""
        schema = _make_schema()
        dataset = _make_dataset()
        existing = pd.DataFrame(
            {"Name": ["Charlie"], "Favorite color": ["Green"], "Hobbies": ["Reading"]}
        )
        df = CSVExporter.to_dataframe(dataset, schema, existing_data=existing)

        assert len(df) == 3
        assert df["is_synthetic"].iloc[0] == False
        assert df["is_synthetic"].iloc[1] == True

    def test_to_csv_string(self):
        schema = _make_schema()
        dataset = _make_dataset()
        df = CSVExporter.to_dataframe(dataset, schema)
        csv_str = CSVExporter.to_csv_string(df)

        assert isinstance(csv_str, str)
        assert "Name" in csv_str
        assert "Alice" in csv_str

    def test_to_csv_bytes(self):
        schema = _make_schema()
        dataset = _make_dataset()
        df = CSVExporter.to_dataframe(dataset, schema)
        csv_bytes = CSVExporter.to_csv_bytes(df)

        assert isinstance(csv_bytes, bytes)
        assert b"Name" in csv_bytes


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_to_dict_structure(self):
        schema = _make_schema()
        dataset = _make_dataset()
        result = JSONExporter.to_dict(dataset, schema)

        assert "metadata" in result
        assert "questions" in result
        assert "responses" in result
        assert result["metadata"]["form_title"] == "Test Form"
        assert result["metadata"]["total_generated"] == 2
        assert len(result["questions"]) == 3
        assert len(result["responses"]) == 2

    def test_to_dict_readable_answers(self):
        """Answers should use question text as keys, not question IDs."""
        schema = _make_schema()
        dataset = _make_dataset()
        result = JSONExporter.to_dict(dataset, schema)

        first_resp = result["responses"][0]
        assert "Name" in first_resp["answers"]
        assert first_resp["answers"]["Name"] == "Alice"
        assert first_resp["is_synthetic"] is True

    def test_to_json_string(self):
        schema = _make_schema()
        dataset = _make_dataset()
        json_str = JSONExporter.to_json_string(dataset, schema)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["form_title"] == "Test Form"

    def test_to_json_bytes(self):
        schema = _make_schema()
        dataset = _make_dataset()
        json_bytes = JSONExporter.to_json_bytes(dataset, schema)

        assert isinstance(json_bytes, bytes)
        parsed = json.loads(json_bytes.decode("utf-8"))
        assert len(parsed["responses"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
