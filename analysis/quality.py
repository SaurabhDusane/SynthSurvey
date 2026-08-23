"""Quality / bias checks for a synthetic response set.

Flags the classic tells of low-quality survey data so users can trust (or fix)
what they generated:

- Straightlining: respondents giving the same value across many scale questions.
- Acquiescence / disagreement skew: scale answers systematically high or low.
- Low-variance questions: answers overly concentrated on one option.
- Duplicate rows: identical answer sets.

Each check returns a finding with a severity (pass / info / warn), a headline
metric, a detail sentence, and a suggested fix. Pure pandas/numpy.
"""

from __future__ import annotations

import math

import pandas as pd

from models.form_schema import FormSchema, QuestionType

_CATEGORICAL = {
    QuestionType.MULTIPLE_CHOICE,
    QuestionType.DROPDOWN,
    QuestionType.CHECKBOX,
    QuestionType.LINEAR_SCALE,
}


def _finding(check, severity, metric, detail, suggestion=""):
    return {
        "check": check,
        "severity": severity,  # "pass" | "info" | "warn"
        "metric": metric,
        "detail": detail,
        "suggestion": suggestion,
    }


def _normalized_entropy(series: pd.Series) -> float:
    """Shannon entropy of a categorical distribution, normalized to 0..1."""
    counts = series.value_counts()
    k = len(counts)
    if k <= 1:
        return 0.0
    probs = counts / counts.sum()
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return ent / math.log(k)


def analyze_quality(df: pd.DataFrame, schema: FormSchema) -> dict:
    """Run all quality checks over a (synthetic) response DataFrame.

    Returns {"findings": [...], "verdict": str, "warn_count": int}.
    """
    findings: list[dict] = []

    scale_qs = [
        q for q in schema.questions
        if q.question_type == QuestionType.LINEAR_SCALE and q.question_text in df.columns
    ]
    scale_cols = [q.question_text for q in scale_qs]
    answer_cols = [q.question_text for q in schema.questions if q.question_text in df.columns]

    # ── Straightlining ────────────────────────────────────────────────────
    if len(scale_cols) >= 3:
        sub = df[scale_cols].apply(pd.to_numeric, errors="coerce")
        straight = 0
        considered = 0
        for _, row in sub.iterrows():
            vals = row.dropna()
            if len(vals) >= 3:
                considered += 1
                if vals.nunique() == 1:
                    straight += 1
        pct = (straight / considered * 100) if considered else 0.0
        findings.append(_finding(
            "Straightlining",
            "warn" if pct > 15 else "pass",
            f"{pct:.0f}%",
            f"{straight} of {considered} respondents gave the same value across all "
            f"scale questions.",
            "Raise temperature or diversify persona engagement levels to reduce "
            "identical-answer patterns." if pct > 15 else "",
        ))

    # ── Acquiescence / skew ───────────────────────────────────────────────
    positions: list[float] = []
    for q in scale_qs:
        s = pd.to_numeric(df[q.question_text], errors="coerce").dropna()
        if s.empty:
            continue
        lo = q.scale_config.min_value if q.scale_config else float(s.min())
        hi = q.scale_config.max_value if q.scale_config else float(s.max())
        if hi > lo:
            positions.extend(((s - lo) / (hi - lo)).tolist())
    if positions:
        mean_pos = sum(positions) / len(positions)
        if mean_pos > 0.7:
            sev, detail = "warn", "Scale answers skew high (acquiescence / agreeableness bias)."
            sug = "Add skeptical or low-engagement personas, or lower temperature."
        elif mean_pos < 0.3:
            sev, detail = "warn", "Scale answers skew low (systematic disagreement)."
            sug = "Balance persona attitudes toward the topic."
        else:
            sev, detail, sug = "pass", "Scale answers are centered without strong skew.", ""
        findings.append(_finding(
            "Response skew", sev, f"{mean_pos * 100:.0f}% of scale",
            detail, sug,
        ))

    # ── Low-variance questions ────────────────────────────────────────────
    concentrated: list[str] = []
    for q in schema.questions:
        if q.question_type not in _CATEGORICAL or q.question_text not in df.columns:
            continue
        s = df[q.question_text].dropna().astype(str)
        if len(s) < 5:
            continue
        if _normalized_entropy(s) < 0.4 and s.nunique() > 1:
            concentrated.append(q.question_text)
    if concentrated:
        findings.append(_finding(
            "Answer variety", "info", f"{len(concentrated)} question(s)",
            "Some questions have answers concentrated on one option: "
            + ", ".join(f'"{c[:40]}"' for c in concentrated[:5])
            + ("…" if len(concentrated) > 5 else "") + ".",
            "This can be realistic, but if unexpected, broaden persona diversity.",
        ))

    # ── Duplicate rows ────────────────────────────────────────────────────
    if answer_cols:
        dups = int(df[answer_cols].astype(str).duplicated().sum())
        pct = (dups / len(df) * 100) if len(df) else 0.0
        findings.append(_finding(
            "Duplicate responses",
            "warn" if dups > 0 else "pass",
            str(dups),
            f"{dups} response(s) ({pct:.0f}%) are exact duplicates of another."
            if dups else "No exact duplicate responses.",
            "Increase persona diversity or temperature." if dups else "",
        ))

    warn_count = sum(1 for f in findings if f["severity"] == "warn")
    if warn_count == 0:
        verdict = "No quality issues detected"
    elif warn_count == 1:
        verdict = "1 quality issue to review"
    else:
        verdict = f"{warn_count} quality issues to review"

    return {"findings": findings, "verdict": verdict, "warn_count": warn_count}
