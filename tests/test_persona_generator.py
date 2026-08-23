"""Tests for the persona generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from models.persona import Persona


class TestPersona:
    """Tests for the Persona model."""

    def test_persona_creation(self):
        """Persona should be creatable with all required fields."""
        persona = Persona(
            name="Jane Doe",
            age=21,
            gender="Female",
            major="Computer Science",
            year="Junior",
            university="Arizona State University",
            interests=["coding", "hiking"],
            personality_traits=["introverted", "detail-oriented"],
            engagement_level="high",
            attitude_toward_topic="Very interested in technology surveys",
            background_context="Jane is a CS student who loves building web apps.",
        )
        assert persona.name == "Jane Doe"
        assert persona.age == 21
        assert persona.engagement_level == "high"
        assert len(persona.interests) == 2

    def test_persona_summary(self):
        """Persona summary should contain key information."""
        persona = Persona(
            name="John Smith",
            age=22,
            gender="Male",
            major="Biology",
            year="Senior",
            interests=["research", "gaming"],
            personality_traits=["outgoing"],
            engagement_level="medium",
            attitude_toward_topic="Neutral about campus dining",
            background_context="John is a pre-med student.",
        )
        summary = persona.summary()
        assert "John Smith" in summary
        assert "Senior" in summary
        assert "Biology" in summary
        assert "medium" in summary

    def test_persona_attribute_tuple(self):
        """Attribute tuple should be lowercase for comparison."""
        persona = Persona(
            name="Alice Johnson",
            age=20,
            gender="Female",
            major="English",
            year="Sophomore",
            interests=["reading", "writing"],
            personality_traits=["creative"],
            engagement_level="high",
            attitude_toward_topic="Loves literature surveys",
            background_context="Alice writes for the school paper.",
        )
        t = persona.attribute_tuple()
        assert t[0] == "alice johnson"
        assert t[1] == 20
        assert t[2] == "female"
        assert t[3] == "english"
        assert t[4] == "sophomore"

    def test_persona_uniqueness_check(self):
        """Two personas with different attributes should have different tuples."""
        p1 = Persona(
            name="A B",
            age=20,
            gender="Male",
            major="CS",
            year="Freshman",
            interests=["a", "b"],
            personality_traits=["x"],
            engagement_level="low",
            attitude_toward_topic="...",
            background_context="...",
        )
        p2 = Persona(
            name="C D",
            age=21,
            gender="Female",
            major="Math",
            year="Junior",
            interests=["c", "d"],
            personality_traits=["y"],
            engagement_level="high",
            attitude_toward_topic="...",
            background_context="...",
        )
        assert p1.attribute_tuple() != p2.attribute_tuple()

    def test_persona_id_auto_generated(self):
        """Persona ID should be auto-generated."""
        persona = Persona(
            name="Test",
            age=18,
            gender="Other",
            major="Art",
            year="Freshman",
            interests=["drawing", "music"],
            personality_traits=["quiet"],
            engagement_level="low",
            attitude_toward_topic="Indifferent",
            background_context="Just started college.",
        )
        assert persona.persona_id is not None
        assert len(persona.persona_id) == 8

    def test_persona_interests_min_length(self):
        """Persona should require at least 2 interests."""
        with pytest.raises(Exception):
            Persona(
                name="Test",
                age=18,
                gender="Other",
                major="Art",
                year="Freshman",
                interests=["only_one"],
                personality_traits=["quiet"],
                engagement_level="low",
                attitude_toward_topic="...",
                background_context="...",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
