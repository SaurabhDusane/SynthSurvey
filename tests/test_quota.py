"""Tests for deterministic quota planning."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.persona import Persona
from services.quota import (
    apply_assignment,
    assignment_directive,
    build_assignment_plan,
    largest_remainder,
)


class TestLargestRemainder:
    def test_sums_to_n(self):
        counts = largest_remainder({"a": 0.2, "b": 0.5, "c": 0.3}, 10)
        assert sum(counts.values()) == 10

    def test_exact_split(self):
        counts = largest_remainder({"a": 0.5, "b": 0.5}, 10)
        assert counts == {"a": 5, "b": 5}

    def test_handles_zero(self):
        assert largest_remainder({"a": 1.0}, 0) == {"a": 0}

    def test_accepts_raw_percentages(self):
        counts = largest_remainder({"a": 20, "b": 30}, 10)
        assert sum(counts.values()) == 10
        assert counts["b"] > counts["a"]


class TestBuildPlan:
    def test_empty_spec_returns_empty(self):
        assert build_assignment_plan(None, 10) == []
        assert build_assignment_plan({}, 10) == []

    def test_marginals_hit_exactly(self):
        spec = {"engagement_level": {"low": 0.2, "medium": 0.5, "high": 0.3}}
        plan = build_assignment_plan(spec, 100)
        assert len(plan) == 100
        counts = Counter(a["engagement_level"] for a in plan)
        assert counts["low"] == 20
        assert counts["medium"] == 50
        assert counts["high"] == 30

    def test_multiple_attributes_each_marginal_exact(self):
        spec = {
            "gender": {"Female": 0.5, "Male": 0.5},
            "year": {"Freshman": 0.25, "Sophomore": 0.25, "Junior": 0.25, "Senior": 0.25},
        }
        plan = build_assignment_plan(spec, 20)
        assert len(plan) == 20
        assert Counter(a["gender"] for a in plan)["Female"] == 10
        assert Counter(a["year"] for a in plan)["Junior"] == 5

    def test_age_range_produces_age_in_band(self):
        spec = {"age_range": {"18-20": 1.0}}
        plan = build_assignment_plan(spec, 15)
        assert all(18 <= a["age"] <= 20 for a in plan)
        assert all(a["age_range"] == "18-20" for a in plan)

    def test_deterministic_with_seed(self):
        spec = {"gender": {"Female": 0.5, "Male": 0.5}}
        assert build_assignment_plan(spec, 30, seed=7) == build_assignment_plan(spec, 30, seed=7)

    def test_unknown_attr_ignored(self):
        plan = build_assignment_plan({"height": {"tall": 1.0}}, 10)
        assert plan == []


class TestDirectiveAndApply:
    def test_directive_mentions_fields(self):
        d = assignment_directive({"gender": "Female", "year": "Senior", "age": 22})
        assert "Female" in d and "Senior" in d and "22" in d

    def test_apply_overrides_persona(self):
        p = Persona(
            name="X", age=30, gender="Male", major="CS", year="Freshman",
            university="ASU", interests=["a", "b"], personality_traits=["x"],
            engagement_level="low", attitude_toward_topic="pos", background_context="bg",
        )
        apply_assignment(p, {"gender": "Female", "year": "Senior", "engagement_level": "high", "age": 22})
        assert (p.gender, p.year, p.engagement_level, p.age) == ("Female", "Senior", "high", 22)

    def test_apply_noop_on_empty(self):
        p = Persona(
            name="X", age=30, gender="Male", major="CS", year="Freshman",
            university="ASU", interests=["a", "b"], personality_traits=["x"],
            engagement_level="low", attitude_toward_topic="pos", background_context="bg",
        )
        apply_assignment(p, {})
        assert p.gender == "Male"
