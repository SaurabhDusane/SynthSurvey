"""Pydantic models for generated responses and the survey dataset."""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class GeneratedResponse(BaseModel):
    """A single generated survey response tied to a persona."""

    persona_id: str = Field(description="ID of the persona who generated this response")
    persona_summary: str = Field(description="Short text summary of the persona")
    answers: dict[str, Any] = Field(
        description="Mapping of question_id to answer value"
    )
    is_synthetic: bool = Field(default=True, description="Always True for generated data")
    generation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of when this response was generated",
    )
    generation_success: bool = Field(
        default=True, description="Whether the response passed validation"
    )
    retry_count: int = Field(
        default=0, description="Number of retries needed for this response"
    )


class SurveyDataset(BaseModel):
    """Complete dataset of generated survey responses."""

    form_title: str = Field(description="Title of the form")
    form_url: str = Field(description="Original form URL")
    responses: list[GeneratedResponse] = Field(
        default_factory=list, description="All generated responses"
    )
    total_generated: int = Field(default=0, description="Total responses generated")
    total_failed: int = Field(default=0, description="Total failed generations")
    generation_started: Optional[str] = Field(default=None)
    generation_completed: Optional[str] = Field(default=None)

    def add_response(self, response: GeneratedResponse) -> None:
        """Add a response to the dataset."""
        self.responses.append(response)
        self.total_generated = len(self.responses)
