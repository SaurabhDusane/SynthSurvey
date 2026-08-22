"""Pydantic models for Google Form structure."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Supported Google Form question types."""

    SHORT_TEXT = "short_text"
    PARAGRAPH = "paragraph"
    MULTIPLE_CHOICE = "multiple_choice"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    LINEAR_SCALE = "linear_scale"
    MULTIPLE_CHOICE_GRID = "multiple_choice_grid"
    CHECKBOX_GRID = "checkbox_grid"
    DATE = "date"
    TIME = "time"
    UNSUPPORTED = "unsupported"


# Mapping from Google Forms internal type IDs to our QuestionType enum
GOOGLE_FORM_TYPE_MAP = {
    0: QuestionType.SHORT_TEXT,
    1: QuestionType.PARAGRAPH,
    2: QuestionType.MULTIPLE_CHOICE,
    3: QuestionType.DROPDOWN,
    4: QuestionType.CHECKBOX,
    5: QuestionType.LINEAR_SCALE,
    7: QuestionType.MULTIPLE_CHOICE_GRID,
    8: QuestionType.CHECKBOX_GRID,
    9: QuestionType.DATE,
    10: QuestionType.TIME,
}


class GridConfig(BaseModel):
    """Configuration for grid-type questions."""

    rows: list[str] = Field(default_factory=list, description="Row labels")
    columns: list[str] = Field(default_factory=list, description="Column labels")


class ScaleConfig(BaseModel):
    """Configuration for linear scale questions."""

    min_value: int = Field(default=1, description="Minimum scale value")
    max_value: int = Field(default=5, description="Maximum scale value")
    min_label: Optional[str] = Field(default=None, description="Label for min value")
    max_label: Optional[str] = Field(default=None, description="Label for max value")


class ValidationRule(BaseModel):
    """Validation rules for a question."""

    rule_type: Optional[str] = Field(
        default=None, description="Type of validation (e.g., 'number', 'text_length')"
    )
    condition: Optional[str] = Field(
        default=None, description="Validation condition"
    )
    value: Optional[Any] = Field(default=None, description="Validation value")
    error_message: Optional[str] = Field(
        default=None, description="Custom error message"
    )


class FormQuestion(BaseModel):
    """A single question in a Google Form."""

    question_id: str = Field(description="Unique identifier for the question")
    entry_id: str = Field(description="Google Form entry ID for submission")
    question_text: str = Field(description="The question text")
    question_type: QuestionType = Field(description="Type of the question")
    options: list[str] = Field(
        default_factory=list, description="Available options for choice questions"
    )
    is_required: bool = Field(default=False, description="Whether the question is required")
    validation: Optional[ValidationRule] = Field(
        default=None, description="Validation rules"
    )
    scale_config: Optional[ScaleConfig] = Field(
        default=None, description="Scale configuration for linear scale questions"
    )
    grid_config: Optional[GridConfig] = Field(
        default=None, description="Grid configuration for grid questions"
    )
    has_other_option: bool = Field(
        default=False, description="Whether 'Other' is an option"
    )
    section_index: int = Field(
        default=0, description="Which section/page this question belongs to"
    )


class FormSection(BaseModel):
    """A section/page in a Google Form."""

    section_index: int = Field(description="Section index")
    title: Optional[str] = Field(default=None, description="Section title")
    description: Optional[str] = Field(default=None, description="Section description")


class FormSchema(BaseModel):
    """Complete parsed structure of a Google Form."""

    form_url: str = Field(description="Original form URL")
    form_title: str = Field(description="Title of the form")
    form_description: Optional[str] = Field(
        default=None, description="Description of the form"
    )
    questions: list[FormQuestion] = Field(
        default_factory=list, description="All questions in the form"
    )
    sections: list[FormSection] = Field(
        default_factory=list, description="Sections/pages in the form"
    )
    total_questions: int = Field(default=0, description="Total number of questions")
    parse_method: str = Field(
        default="structured",
        description="How the form was parsed: 'structured' or 'html_fallback'",
    )

    def model_post_init(self, __context: Any) -> None:
        """Set total_questions after initialization."""
        if self.total_questions == 0:
            self.total_questions = len(self.questions)
