"""Deterministic quota / distribution planning for persona generation.

Rather than *asking* the LLM to approximate a target demographic mix (which it
does unreliably), we pre-assign each response's key attributes so the requested
marginals are hit exactly, feed them as hard constraints, and reconcile the
generated persona to the assignment. This is what makes quota'd output
trustworthy.

A ``QuotaSpec`` maps an attribute to the desired proportion of each bucket::

    {"engagement_level": {"low": 0.2, "medium": 0.5, "high": 0.3},
     "gender": {"Female": 0.5, "Male": 0.5}}

Proportions per attribute are normalized, so raw percentages (20/50/30) work too.
"""

from __future__ import annotations

import random
from typing import Optional

# Attributes that can be quota-controlled and their canonical bucket sets.
YEAR_BUCKETS = ["Freshman", "Sophomore", "Junior", "Senior", "Grad Student"]
ENGAGEMENT_BUCKETS = ["low", "medium", "high"]
GENDER_BUCKETS = ["Female", "Male", "Non-binary"]
AGE_BANDS: dict[str, tuple[int, int]] = {
    "18-20": (18, 20),
    "21-23": (21, 23),
    "24-27": (24, 27),
    "28-35": (28, 35),
}

# attribute -> ordered bucket labels (used to build UI and validate specs)
SUPPORTED_ATTRS: dict[str, list[str]] = {
    "gender": GENDER_BUCKETS,
    "year": YEAR_BUCKETS,
    "engagement_level": ENGAGEMENT_BUCKETS,
    "age_range": list(AGE_BANDS.keys()),
}

QuotaSpec = dict[str, dict[str, float]]


def largest_remainder(proportions: dict[str, float], n: int) -> dict[str, int]:
    """Allocate ``n`` integer counts across buckets to match proportions exactly.

    Uses the largest-remainder (Hamilton) method so the counts sum to ``n``.
    """
    total = sum(max(0.0, v) for v in proportions.values())
    if total <= 0 or n <= 0:
        return {k: 0 for k in proportions}

    exact = {k: (max(0.0, v) / total) * n for k, v in proportions.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(floors.values())

    # Hand out the leftover to the largest fractional parts.
    order = sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def build_assignment_plan(
    spec: Optional[QuotaSpec],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Return ``n`` per-response attribute assignments matching ``spec``.

    Each attribute is allocated independently (matching its marginal exactly) and
    shuffled, so the joint distribution is randomized while every marginal is
    honored. Deterministic for a given ``seed``.

    Returns an empty list if ``spec`` is empty — callers treat that as
    "no quotas, generate freely".
    """
    if not spec or n <= 0:
        return []

    rng = random.Random(seed)
    per_attr_seq: dict[str, list[str]] = {}
    for attr, buckets in spec.items():
        if attr not in SUPPORTED_ATTRS or not buckets:
            continue
        counts = largest_remainder(buckets, n)
        if sum(counts.values()) == 0:
            continue
        seq: list[str] = []
        for bucket, c in counts.items():
            seq.extend([bucket] * c)
        rng.shuffle(seq)
        per_attr_seq[attr] = seq

    if not per_attr_seq:
        return []

    plan: list[dict] = []
    for i in range(n):
        assignment: dict = {}
        for attr, seq in per_attr_seq.items():
            bucket = seq[i]
            if attr == "age_range":
                lo, hi = AGE_BANDS.get(bucket, (18, 25))
                assignment["age"] = rng.randint(lo, hi)
                assignment["age_range"] = bucket
            else:
                assignment[attr] = bucket
        plan.append(assignment)
    return plan


def assignment_directive(assignment: dict) -> str:
    """Render a hard-constraint instruction for the persona prompt."""
    parts = []
    if "gender" in assignment:
        parts.append(f'gender = "{assignment["gender"]}"')
    if "year" in assignment:
        parts.append(f'year = "{assignment["year"]}"')
    if "engagement_level" in assignment:
        parts.append(f'engagement_level = "{assignment["engagement_level"]}"')
    if "age" in assignment:
        parts.append(f'age = {assignment["age"]}')
    if not parts:
        return ""
    return (
        "HARD REQUIREMENTS — this persona MUST have exactly these attributes "
        "(do not deviate): " + "; ".join(parts) + "."
    )


def apply_assignment(persona, assignment: dict) -> None:
    """Force a persona's fields to match its quota assignment exactly.

    Guarantees quota adherence regardless of whether the LLM complied.
    """
    if not assignment:
        return
    for field in ("gender", "year", "engagement_level", "age"):
        if field in assignment:
            setattr(persona, field, assignment[field])
