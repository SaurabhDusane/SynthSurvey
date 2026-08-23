"""Fidelity scoring: how closely synthetic responses match real reference data.

Given a synthetic DataFrame, a reference (real) DataFrame, and the form schema,
we score each shared question by comparing its answer distribution. The metric
is Total Variation Distance (TVD) between the two normalized distributions;
similarity = 1 - TVD, so 1.0 means identical distributions and 0.0 means no
overlap. The overall fidelity is the mean similarity across scored questions,
reported 0-100.

Dependency-light: pandas/numpy only.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from models.form_schema import FormSchema, QuestionType

# Question types whose answers form a comparable distribution.
_CATEGORICAL = {
    QuestionType.MULTIPLE_CHOICE,
    QuestionType.DROPDOWN,
    QuestionType.CHECKBOX,
    QuestionType.LINEAR_SCALE,
}
_TEXT = {QuestionType.SHORT_TEXT, QuestionType.PARAGRAPH}


def _tvd(ref: pd.Series, syn: pd.Series) -> float:
    """Total Variation Distance between two categorical distributions (0..1)."""
    r = ref.value_counts(normalize=True)
    s = syn.value_counts(normalize=True)
    categories = set(r.index) | set(s.index)
    return 0.5 * sum(abs(float(r.get(c, 0.0)) - float(s.get(c, 0.0))) for c in categories)


def compute_fidelity(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    schema: FormSchema,
) -> dict:
    """Score how well ``synthetic_df`` reproduces ``reference_df``.

    Returns a dict with:
        overall_score: float 0-100, or None if nothing could be scored
        scored / skipped: counts
        per_question: list of {question, type, similarity, tvd, n_ref, n_syn}
        notes: list of human-readable notes
    """
    per_question: list[dict] = []
    skipped: list[str] = []
    notes: list[str] = []

    for q in schema.questions:
        col = q.question_text
        if col not in synthetic_df.columns or col not in reference_df.columns:
            continue

        if q.question_type in _TEXT:
            skipped.append(col)
            continue
        if q.question_type not in _CATEGORICAL:
            skipped.append(col)
            continue

        ref = reference_df[col].dropna().astype(str)
        syn = synthetic_df[col].dropna().astype(str)
        if len(ref) == 0 or len(syn) == 0:
            skipped.append(col)
            continue

        tvd = _tvd(ref, syn)
        per_question.append({
            "question": col,
            "type": q.question_type.value,
            "similarity": round(max(0.0, 1.0 - tvd) * 100, 1),
            "tvd": round(tvd, 3),
            "n_ref": int(len(ref)),
            "n_syn": int(len(syn)),
        })

    if not per_question:
        notes.append(
            "No comparable questions were scored. Fidelity needs shared "
            "choice/scale questions present in both datasets."
        )
        return {
            "overall_score": None,
            "scored": 0,
            "skipped": len(skipped),
            "per_question": [],
            "notes": notes,
        }

    overall = sum(p["similarity"] for p in per_question) / len(per_question)
    # Weakest questions first, so users see where the gap is.
    per_question.sort(key=lambda p: p["similarity"])
    if skipped:
        notes.append(
            f"{len(skipped)} free-text or empty question(s) were not scored "
            "(distributions aren't comparable)."
        )
    return {
        "overall_score": round(overall, 1),
        "scored": len(per_question),
        "skipped": len(skipped),
        "per_question": per_question,
        "notes": notes,
    }


def fidelity_verdict(score: Optional[float]) -> str:
    """Map a fidelity score to a short qualitative label."""
    if score is None:
        return "Not scored"
    if score >= 85:
        return "Excellent match"
    if score >= 70:
        return "Good match"
    if score >= 50:
        return "Fair match"
    return "Poor match"
