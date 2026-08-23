"""A/B stimulus (conjoint) planning.

For concept/ad/product testing, each persona reacts to a stimulus. With more
than one variant, personas are split evenly across variants (deterministically),
so you can compare how the synthetic audience responds to A vs. B.
"""

from __future__ import annotations

import random
from typing import Optional

from services.quota import largest_remainder


def clean_variants(variants: Optional[list[dict]]) -> list[dict]:
    """Keep only variants with non-empty text; assign default names."""
    cleaned = []
    for idx, v in enumerate(variants or []):
        text = (v.get("text") or "").strip()
        if not text:
            continue
        name = (v.get("name") or f"Variant {chr(65 + idx)}").strip()
        cleaned.append({"name": name, "text": text})
    return cleaned


def build_stimulus_plan(
    variants: Optional[list[dict]],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Return ``n`` per-persona stimulus assignments (empty if no variants)."""
    cleaned = clean_variants(variants)
    if not cleaned or n <= 0:
        return []
    if len(cleaned) == 1:
        return [cleaned[0] for _ in range(n)]

    counts = largest_remainder({v["name"]: 1.0 for v in cleaned}, n)
    by_name = {v["name"]: v for v in cleaned}
    seq: list[dict] = []
    for name, c in counts.items():
        seq.extend([by_name[name]] * c)
    random.Random(seed).shuffle(seq)
    return seq
