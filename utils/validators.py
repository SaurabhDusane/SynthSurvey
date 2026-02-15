"""Validate generated responses against the form schema."""

from typing import Any

from models.form_schema import FormQuestion, FormSchema, QuestionType


class ResponseValidator:
    """Validates LLM-generated responses against the form schema."""

    def __init__(self, schema: FormSchema):
        self.schema = schema
        self.errors: list[str] = []

    def validate(self, answers: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate all answers against the form schema.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        self.errors = []

        for question in self.schema.questions:
            qid = question.question_id
            answer = answers.get(qid)

            if answer is None or answer == "":
                if question.is_required:
                    self.errors.append(
                        f"Required question '{question.question_text}' ({qid}) has no answer."
                    )
                continue

            self._validate_question(question, answer)

        return len(self.errors) == 0, self.errors

    def _validate_question(self, question: FormQuestion, answer: Any) -> None:
        """Validate a single answer against its question definition."""
        qt = question.question_type

        if qt == QuestionType.MULTIPLE_CHOICE or qt == QuestionType.DROPDOWN:
            self._validate_single_choice(question, answer)
        elif qt == QuestionType.CHECKBOX:
            self._validate_checkbox(question, answer)
        elif qt == QuestionType.LINEAR_SCALE:
            self._validate_scale(question, answer)
        elif qt == QuestionType.SHORT_TEXT:
            self._validate_short_text(question, answer)
        elif qt == QuestionType.PARAGRAPH:
            self._validate_paragraph(question, answer)
        elif qt == QuestionType.MULTIPLE_CHOICE_GRID:
            self._validate_grid(question, answer, multi_select=False)
        elif qt == QuestionType.CHECKBOX_GRID:
            self._validate_grid(question, answer, multi_select=True)
        elif qt == QuestionType.DATE:
            self._validate_date(question, answer)
        elif qt == QuestionType.TIME:
            self._validate_time(question, answer)

    def _validate_single_choice(self, question: FormQuestion, answer: Any) -> None:
        """Validate multiple choice / dropdown answer."""
        if not isinstance(answer, str):
            self.errors.append(
                f"Question '{question.question_text}': expected string, got {type(answer).__name__}."
            )
            return
        valid_options = [opt.lower().strip() for opt in question.options]
        if answer.lower().strip() not in valid_options:
            if question.has_other_option:
                return  # "Other" responses are allowed
            self.errors.append(
                f"Question '{question.question_text}': "
                f"'{answer}' is not a valid option. "
                f"Valid options: {question.options}"
            )

    def _validate_checkbox(self, question: FormQuestion, answer: Any) -> None:
        """Validate checkbox answer (must be a list of valid options)."""
        if isinstance(answer, str):
            answer = [answer]
        if not isinstance(answer, list):
            self.errors.append(
                f"Question '{question.question_text}': expected list, got {type(answer).__name__}."
            )
            return
        valid_options = [opt.lower().strip() for opt in question.options]
        for item in answer:
            if str(item).lower().strip() not in valid_options:
                if question.has_other_option:
                    continue
                self.errors.append(
                    f"Question '{question.question_text}': "
                    f"'{item}' is not a valid checkbox option. "
                    f"Valid options: {question.options}"
                )

    def _validate_scale(self, question: FormQuestion, answer: Any) -> None:
        """Validate linear scale answer."""
        try:
            value = int(answer)
        except (ValueError, TypeError):
            self.errors.append(
                f"Question '{question.question_text}': "
                f"scale answer must be a number, got '{answer}'."
            )
            return

        if question.scale_config:
            min_val = question.scale_config.min_value
            max_val = question.scale_config.max_value
            if value < min_val or value > max_val:
                self.errors.append(
                    f"Question '{question.question_text}': "
                    f"value {value} is outside range [{min_val}, {max_val}]."
                )

    def _validate_short_text(self, question: FormQuestion, answer: Any) -> None:
        """Validate short text answer."""
        if not isinstance(answer, str):
            self.errors.append(
                f"Question '{question.question_text}': expected string, got {type(answer).__name__}."
            )

    def _validate_paragraph(self, question: FormQuestion, answer: Any) -> None:
        """Validate paragraph answer."""
        if not isinstance(answer, str):
            self.errors.append(
                f"Question '{question.question_text}': expected string, got {type(answer).__name__}."
            )

    def _validate_grid(
        self, question: FormQuestion, answer: Any, multi_select: bool
    ) -> None:
        """Validate grid answer (dict mapping row labels to selected column(s))."""
        if not isinstance(answer, dict):
            self.errors.append(
                f"Question '{question.question_text}': "
                f"grid answer must be a dict, got {type(answer).__name__}."
            )
            return

        if question.grid_config:
            valid_rows = [r.lower().strip() for r in question.grid_config.rows]
            valid_cols = [c.lower().strip() for c in question.grid_config.columns]

            for row_key, col_val in answer.items():
                if row_key.lower().strip() not in valid_rows:
                    self.errors.append(
                        f"Question '{question.question_text}': "
                        f"'{row_key}' is not a valid row."
                    )
                if multi_select:
                    if isinstance(col_val, list):
                        for cv in col_val:
                            if str(cv).lower().strip() not in valid_cols:
                                self.errors.append(
                                    f"Question '{question.question_text}': "
                                    f"'{cv}' is not a valid column option."
                                )
                    else:
                        self.errors.append(
                            f"Question '{question.question_text}': "
                            f"checkbox grid row values must be lists."
                        )
                else:
                    if str(col_val).lower().strip() not in valid_cols:
                        self.errors.append(
                            f"Question '{question.question_text}': "
                            f"'{col_val}' is not a valid column option."
                        )

    def _validate_date(self, question: FormQuestion, answer: Any) -> None:
        """Validate date answer (expected format: YYYY-MM-DD)."""
        if not isinstance(answer, str):
            self.errors.append(
                f"Question '{question.question_text}': date must be a string."
            )
            return
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", answer):
            self.errors.append(
                f"Question '{question.question_text}': "
                f"date '{answer}' must be in YYYY-MM-DD format."
            )

    def _validate_time(self, question: FormQuestion, answer: Any) -> None:
        """Validate time answer (expected format: HH:MM)."""
        if not isinstance(answer, str):
            self.errors.append(
                f"Question '{question.question_text}': time must be a string."
            )
            return
        import re
        if not re.match(r"^\d{2}:\d{2}$", answer):
            self.errors.append(
                f"Question '{question.question_text}': "
                f"time '{answer}' must be in HH:MM format."
            )


def fix_response(
    question: FormQuestion, answer: Any
) -> Any:
    """Attempt to fix a response that failed validation.

    Returns the fixed answer or None if unfixable.
    """
    qt = question.question_type

    if qt in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN):
        if isinstance(answer, str) and question.options:
            # Try case-insensitive match
            for opt in question.options:
                if opt.lower().strip() == answer.lower().strip():
                    return opt
            # Try partial match
            for opt in question.options:
                if answer.lower().strip() in opt.lower().strip():
                    return opt
        return None

    if qt == QuestionType.CHECKBOX:
        if isinstance(answer, str):
            return [answer]
        if isinstance(answer, list):
            fixed = []
            for item in answer:
                for opt in question.options:
                    if str(item).lower().strip() == opt.lower().strip():
                        fixed.append(opt)
                        break
            return fixed if fixed else None
        return None

    if qt == QuestionType.LINEAR_SCALE:
        try:
            value = int(float(str(answer)))
            if question.scale_config:
                value = max(question.scale_config.min_value, min(question.scale_config.max_value, value))
            return value
        except (ValueError, TypeError):
            return None

    return answer
