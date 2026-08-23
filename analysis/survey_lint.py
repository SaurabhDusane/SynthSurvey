"""Survey-design linting.

Reviews a parsed form for common question-writing problems *before* generation,
so SynthSurvey doubles as a lightweight survey-QA tool. Pure function over the
schema — no LLM, no network.
"""

from __future__ import annotations

from models.form_schema import FormQuestion, FormSchema, QuestionType

_LEADING = [
    "don't you", "do you agree", "wouldn't you", "how great", "isn't it",
    "aren't they", "surely", "obviously", "how much do you love", "everyone knows",
    "as you know", "clearly",
]
_ABSOLUTES = ["always", "never", " all ", " every ", " none ", " everyone", " no one"]
_OPT_OUT = [
    "prefer not", "n/a", "not applicable", "none", "other", "decline",
    "no opinion", "neutral", "unsure", "don't know",
]
_SINGLE_CHOICE = {QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN}


def _finding(question, issue, severity, suggestion):
    return {"question": question, "issue": issue, "severity": severity, "suggestion": suggestion}


def _lint_question(q: FormQuestion) -> list[dict]:
    text = (q.question_text or "").lower()
    padded = f" {text} "
    out: list[dict] = []

    if any(p in text for p in _LEADING):
        out.append(_finding(q.question_text, "Leading / loaded wording", "warn",
                             "Rephrase neutrally so it doesn't push respondents toward an answer."))

    if any(a in padded for a in _ABSOLUTES):
        out.append(_finding(q.question_text, "Absolute wording (always/never/all)", "info",
                             "Absolutes can force inaccurate answers; consider softer phrasing."))

    if q.question_type in (QuestionType.LINEAR_SCALE, *_SINGLE_CHOICE, QuestionType.CHECKBOX):
        if " and " in text or " or " in text:
            out.append(_finding(q.question_text, "Possibly double-barreled", "warn",
                                "Ask about one thing at a time; split 'X and Y' into two questions."))

    if q.question_type in _SINGLE_CHOICE and q.options:
        opts = " ".join(o.lower() for o in q.options)
        if not any(k in opts for k in _OPT_OUT) and not q.has_other_option:
            out.append(_finding(q.question_text, "No opt-out / neutral option", "info",
                                "Add a 'Prefer not to say' or 'Other' option to avoid forced answers."))

    if q.question_type == QuestionType.LINEAR_SCALE and q.scale_config:
        if not q.scale_config.min_label or not q.scale_config.max_label:
            out.append(_finding(q.question_text, "Scale lacks endpoint labels", "info",
                                "Label both ends of the scale so ratings are interpreted consistently."))

    if len(q.question_text or "") > 200:
        out.append(_finding(q.question_text, "Long / complex question", "info",
                            "Shorten it; long questions increase misreading and dropout."))

    if q.question_type == QuestionType.UNSUPPORTED:
        out.append(_finding(q.question_text, "Unsupported question type", "warn",
                            "This question can't be reliably generated (e.g. image-based)."))

    return out


def lint_survey(schema: FormSchema) -> dict:
    """Return {"findings": [...], "warn_count": int, "verdict": str}."""
    findings: list[dict] = []
    for q in schema.questions:
        findings.extend(_lint_question(q))

    warn_count = sum(1 for f in findings if f["severity"] == "warn")
    if not findings:
        verdict = "No design issues detected"
    elif warn_count:
        verdict = f"{warn_count} issue(s) worth fixing"
    else:
        verdict = f"{len(findings)} minor suggestion(s)"
    return {"findings": findings, "warn_count": warn_count, "verdict": verdict}
