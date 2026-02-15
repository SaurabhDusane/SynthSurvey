"""Tests for the Google Form parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from models.form_schema import (
    FormSchema,
    FormQuestion,
    QuestionType,
    ScaleConfig,
    GridConfig,
)
from parsers.google_form_parser import GoogleFormParser, _safe_get


class TestGoogleFormParser:
    """Tests for GoogleFormParser."""

    def test_normalize_url_viewform(self):
        """URL already ending with /viewform should remain unchanged."""
        url = "https://docs.google.com/forms/d/e/abc123/viewform"
        parser = GoogleFormParser(url)
        assert parser.url == url

    def test_normalize_url_no_suffix(self):
        """URL without /viewform should get it appended."""
        url = "https://docs.google.com/forms/d/e/abc123"
        parser = GoogleFormParser(url)
        assert parser.url.endswith("/viewform")

    def test_normalize_url_with_query_params(self):
        """URL with query params should have them stripped."""
        url = "https://docs.google.com/forms/d/e/abc123/viewform?usp=sf_link"
        parser = GoogleFormParser(url)
        assert "?" not in parser.url
        assert parser.url.endswith("/viewform")

    def test_normalize_url_formresponse(self):
        """URL with /formResponse should be converted to /viewform."""
        url = "https://docs.google.com/forms/d/e/abc123/formResponse"
        parser = GoogleFormParser(url)
        assert parser.url.endswith("/viewform")
        assert "/formResponse" not in parser.url

    def test_normalize_url_trailing_slash(self):
        """URL with trailing slash should be handled."""
        url = "https://docs.google.com/forms/d/e/abc123/"
        parser = GoogleFormParser(url)
        assert parser.url.endswith("/viewform")


class TestFormSchema:
    """Tests for the FormSchema model."""

    def test_form_schema_creation(self):
        """FormSchema should be creatable with basic fields."""
        schema = FormSchema(
            form_url="https://example.com/form",
            form_title="Test Form",
            form_description="A test form",
            questions=[
                FormQuestion(
                    question_id="q_1",
                    entry_id="entry.1",
                    question_text="What is your name?",
                    question_type=QuestionType.SHORT_TEXT,
                    is_required=True,
                ),
                FormQuestion(
                    question_id="q_2",
                    entry_id="entry.2",
                    question_text="Pick a color",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    options=["Red", "Blue", "Green"],
                ),
            ],
        )
        assert schema.total_questions == 2
        assert schema.questions[0].is_required is True
        assert len(schema.questions[1].options) == 3

    def test_question_type_enum(self):
        """All expected question types should be in the enum."""
        expected_types = [
            "short_text",
            "paragraph",
            "multiple_choice",
            "checkbox",
            "dropdown",
            "linear_scale",
            "multiple_choice_grid",
            "checkbox_grid",
            "date",
            "time",
            "unsupported",
        ]
        for t in expected_types:
            assert QuestionType(t) is not None

    def test_scale_config(self):
        """ScaleConfig should store min/max values and labels."""
        sc = ScaleConfig(min_value=1, max_value=10, min_label="Low", max_label="High")
        assert sc.min_value == 1
        assert sc.max_value == 10
        assert sc.min_label == "Low"

    def test_grid_config(self):
        """GridConfig should store rows and columns."""
        gc = GridConfig(
            rows=["Row A", "Row B"],
            columns=["Col 1", "Col 2", "Col 3"],
        )
        assert len(gc.rows) == 2
        assert len(gc.columns) == 3


class TestSafeGet:
    """Tests for the _safe_get helper."""

    def test_simple_index(self):
        assert _safe_get([10, 20, 30], 1) == 20

    def test_nested_index(self):
        data = [["a", "b"], ["c", "d"]]
        assert _safe_get(data, 1, 0) == "c"

    def test_out_of_bounds_returns_default(self):
        assert _safe_get([1, 2], 5, default="nope") == "nope"

    def test_non_list_returns_default(self):
        assert _safe_get("hello", 0, default=None) is None

    def test_deeply_nested(self):
        data = [None, [None, [["deep"]]]]
        assert _safe_get(data, 1, 1, 0, 0) == "deep"

    def test_none_element_mid_path(self):
        data = [None, None]
        assert _safe_get(data, 1, 0, default="x") == "x"


class TestFBPublicLoadDataExtraction:
    """Tests for extracting and parsing FB_PUBLIC_LOAD_DATA_."""

    def _make_html(self, js_data: list) -> str:
        payload = json.dumps(js_data)
        return f'<html><script>var FB_PUBLIC_LOAD_DATA_ = {payload};</script></html>'

    def test_extract_basic(self):
        data = [None, ["desc", [], None, None, None, None, None, None, "My Form"]]
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = self._make_html(data)
        extracted = parser._extract_fb_public_load_data(parser.raw_html)
        assert extracted == data

    def test_extract_missing_raises(self):
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = "<html><body>No data here</body></html>"
        with pytest.raises(ValueError, match="Could not find FB_PUBLIC_LOAD_DATA_"):
            parser._extract_fb_public_load_data(parser.raw_html)

    def test_parse_from_fb_data_minimal(self):
        """A minimal FB_PUBLIC_LOAD_DATA_ with one short-text question."""
        # Structure: data[1][8] = title, data[1][1] = questions list
        # Each question item: [id, title, desc, ?, [[entry_id, options, ?, type, constraints]]]
        question_item = [
            None,                       # [0] id
            "What is your name?",       # [1] title
            None,                       # [2] description
            None,                       # [3]
            [[12345, None, None, 0, [None, None, 1]]]  # [4] sub-questions: entry=12345, type=0 (short_text), required
        ]
        fb_data = [
            None,
            [
                "Form description",     # [0] description
                [question_item],        # [1] questions
                None, None, None, None, None, None,
                "Test Form Title",      # [8] title
            ]
        ]
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = self._make_html(fb_data)
        schema = parser.parse()

        assert schema.form_title == "Test Form Title"
        assert schema.form_description == "Form description"
        assert len(schema.questions) == 1
        q = schema.questions[0]
        assert q.question_text == "What is your name?"
        assert q.question_type == QuestionType.SHORT_TEXT
        assert q.is_required is True
        assert q.entry_id == "entry.12345"

    def test_parse_multiple_choice(self):
        """Parse a multiple-choice question with options."""
        question_item = [
            None,
            "Favorite color?",
            None,
            None,
            [[67890, [["Red"], ["Blue"], ["Green"]], None, 2, None]]  # type=2 (MC)
        ]
        fb_data = [
            None,
            [
                None,
                [question_item],
                None, None, None, None, None, None,
                "Color Survey",
            ]
        ]
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = self._make_html(fb_data)
        schema = parser.parse()

        assert len(schema.questions) == 1
        q = schema.questions[0]
        assert q.question_type == QuestionType.MULTIPLE_CHOICE
        assert q.options == ["Red", "Blue", "Green"]

    def test_parse_section_break(self):
        """Items without sub-questions should be treated as section breaks."""
        section_item = [None, "Section 2", "Description of section 2", None, None]
        question_item = [
            None, "Q1", None, None,
            [[111, None, None, 0, None]]
        ]
        fb_data = [
            None,
            [
                None,
                [question_item, section_item],
                None, None, None, None, None, None,
                "Sectioned Form",
            ]
        ]
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = self._make_html(fb_data)
        schema = parser.parse()

        assert len(schema.questions) == 1
        assert len(schema.sections) >= 1
        # The section break should have been detected
        section_titles = [s.title for s in schema.sections]
        assert "Section 2" in section_titles


class TestHTMLFallbackParser:
    """Tests for the HTML DOM fallback parser."""

    def test_fallback_with_data_params(self):
        """HTML with data-params divs should be parsed."""
        html = '''
        <html>
        <head>
            <meta property="og:title" content="Fallback Form">
            <meta property="og:description" content="A fallback test">
        </head>
        <body>
            <div data-params="[null,&quot;Question One&quot;,null,null,[[100,null,null,0,null]]]"></div>
        </body>
        </html>
        '''
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = html
        schema = parser._parse_from_html()

        assert schema.form_title == "Fallback Form"
        assert schema.form_description == "A fallback test"

    def test_fallback_with_input_elements(self):
        """HTML with entry.* input elements should be parsed."""
        html = '''
        <html>
        <head>
            <meta property="og:title" content="Input Form">
        </head>
        <body>
            <input name="entry.12345" type="text">
            <textarea name="entry.67890"></textarea>
        </body>
        </html>
        '''
        parser = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        parser.raw_html = html
        schema = parser._parse_from_html()

        assert schema.form_title == "Input Form"
        assert len(schema.questions) == 2
        entry_ids = {q.entry_id for q in schema.questions}
        assert "entry.12345" in entry_ids
        assert "entry.67890" in entry_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
