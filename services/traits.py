"""Latent-trait personas.

Real respondents' answers correlate because the same underlying dispositions
drive many questions at once. We give each persona a small set of stable hidden
traits (0-1 scores), inject them into the form-filling prompt, and instruct the
model to answer consistently with them — so related questions correlate the way
they do for real people, instead of being independently random-but-plausible.

Traits are sampled deterministically (seeded per persona index) so a run is
reproducible.
"""

from __future__ import annotations

import random
from typing import Optional

# Default general-purpose latent dimensions that plausibly drive survey answers.
DEFAULT_TRAIT_DIMENSIONS: list[tuple[str, str]] = [
    ("openness", "openness to new ideas, change, and experiences"),
    ("conscientiousness", "diligence, care, and attention to detail"),
    ("skepticism", "how critical/skeptical vs. trusting and accepting they are"),
    ("enthusiasm", "how positive/enthusiastic vs. indifferent about the topic"),
    ("price_sensitivity", "how much cost and value drive their judgments"),
    ("tech_savviness", "comfort and familiarity with technology"),
]

DEFAULT_TRAIT_NAMES = [name for name, _ in DEFAULT_TRAIT_DIMENSIONS]


def sample_traits(
    index: int,
    seed: int = 42,
    dimensions: Optional[list[tuple[str, str]]] = None,
) -> dict[str, float]:
    """Sample stable 0-1 latent trait scores for persona ``index``.

    Deterministic for a given (seed, index), so runs are reproducible and each
    persona keeps the same traits across longitudinal waves.
    """
    dims = dimensions or DEFAULT_TRAIT_DIMENSIONS
    rng = random.Random()
    rng.seed(f"{seed}:{index}")
    return {name: round(rng.random(), 2) for name, _ in dims}


def traits_directive(
    traits: dict[str, float],
    dimensions: Optional[list[tuple[str, str]]] = None,
) -> str:
    """Render a latent-trait profile for the form-filling system prompt."""
    if not traits:
        return ""
    desc = dict(dimensions or DEFAULT_TRAIT_DIMENSIONS)
    lines = [
        f"  - {name} = {value:.2f}  ({desc.get(name, name)})"
        for name, value in traits.items()
    ]
    return (
        "LATENT TRAIT PROFILE (0 = low, 1 = high). Let these hidden dispositions "
        "drive your answers so that related questions are internally consistent "
        "and correlated, the way a real person's would be:\n"
        + "\n".join(lines)
    )
