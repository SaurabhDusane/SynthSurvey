"""Pydantic model for persona cards."""

import uuid

from pydantic import BaseModel, Field


class Persona(BaseModel):
    """A unique fictional persona for survey response generation."""

    persona_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for the persona",
    )
    name: str = Field(description="Fictional name")
    age: int = Field(description="Age of the persona")
    gender: str = Field(description="Gender identity")
    major: str = Field(description="Major or field of study")
    year: str = Field(
        description="Academic year (Freshman, Sophomore, Junior, Senior, Grad Student)"
    )
    university: str = Field(
        default="Arizona State University", description="University name"
    )
    interests: list[str] = Field(
        description="List of 2-4 interests", min_length=2, max_length=4
    )
    personality_traits: list[str] = Field(
        description="Personality traits (e.g., introverted, detail-oriented)"
    )
    engagement_level: str = Field(
        description="How seriously they take surveys: low / medium / high"
    )
    attitude_toward_topic: str = Field(
        description="One-line summary of their stance on the survey's subject"
    )
    background_context: str = Field(
        description="1-2 sentence backstory that informs their perspective"
    )

    def summary(self) -> str:
        """Return a short text summary of the persona."""
        return (
            f"{self.name}, {self.age}, {self.gender}, {self.year} {self.major} "
            f"at {self.university}. Interests: {', '.join(self.interests)}. "
            f"Engagement: {self.engagement_level}. {self.attitude_toward_topic}"
        )

    def attribute_tuple(self) -> tuple:
        """Return a tuple of key attributes for uniqueness checking."""
        return (
            self.name.lower(),
            self.age,
            self.gender.lower(),
            self.major.lower(),
            self.year.lower(),
        )
