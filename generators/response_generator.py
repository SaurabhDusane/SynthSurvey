"""Persona + Form → filled response generation using LLM."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import Settings
from models.form_schema import FormSchema, FormQuestion, QuestionType
from models.persona import Persona
from models.response import GeneratedResponse
from utils.llm_client import LLMClient
from utils.validators import ResponseValidator, fix_response


class ResponseGenerator:
    """Generates survey responses by having the LLM fill forms in character as a persona."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the form fill system prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "form_fill_system.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_system_prompt(self, persona: Persona, form_schema: FormSchema) -> str:
        """Build the system prompt for form filling."""
        return self._prompt_template.format(
            persona_name=persona.name,
            persona_year=persona.year,
            persona_major=persona.major,
            persona_university=persona.university,
            persona_age=persona.age,
            persona_gender=persona.gender,
            persona_interests=", ".join(persona.interests),
            persona_personality_traits=", ".join(persona.personality_traits),
            persona_engagement_level=persona.engagement_level,
            persona_attitude_toward_topic=persona.attitude_toward_topic,
            persona_background_context=persona.background_context,
            form_title=form_schema.form_title,
        )

    def _build_questions_prompt(self, form_schema: FormSchema) -> str:
        """Build the user prompt containing the form questions."""
        questions_data = []
        for q in form_schema.questions:
            q_info: dict[str, Any] = {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "question_type": q.question_type.value,
                "is_required": q.is_required,
            }

            if q.options:
                q_info["options"] = q.options

            if q.scale_config:
                q_info["scale"] = {
                    "min": q.scale_config.min_value,
                    "max": q.scale_config.max_value,
                }
                if q.scale_config.min_label:
                    q_info["scale"]["min_label"] = q.scale_config.min_label
                if q.scale_config.max_label:
                    q_info["scale"]["max_label"] = q.scale_config.max_label

            if q.grid_config:
                q_info["grid"] = {
                    "rows": q.grid_config.rows,
                    "columns": q.grid_config.columns,
                }

            if q.has_other_option:
                q_info["has_other_option"] = True

            questions_data.append(q_info)

        return (
            "Here are the form questions. Respond with a JSON object mapping "
            "each question_id to your answer:\n\n"
            + json.dumps(questions_data, indent=2)
        )

    def _attempt_fix(
        self, form_schema: FormSchema, answers: dict[str, Any], errors: list[str]
    ) -> dict[str, Any]:
        """Attempt to fix validation errors in the response."""
        fixed = dict(answers)
        question_map = {q.question_id: q for q in form_schema.questions}

        for error in errors:
            # Try to identify which question the error is about
            for qid, question in question_map.items():
                if question.question_text in error and qid in fixed:
                    fixed_answer = fix_response(question, fixed[qid])
                    if fixed_answer is not None:
                        fixed[qid] = fixed_answer

        return fixed

    def generate_one(
        self,
        persona: Persona,
        form_schema: FormSchema,
    ) -> GeneratedResponse:
        """Generate a single survey response for a given persona.

        Args:
            persona: The persona to answer as.
            form_schema: The parsed form structure.

        Returns:
            A GeneratedResponse object.
        """
        system_prompt = self._build_system_prompt(persona, form_schema)
        user_prompt = self._build_questions_prompt(form_schema)
        validator = ResponseValidator(form_schema)

        for attempt in range(self.settings.max_retries + 1):
            try:
                answers = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=getattr(self.settings, 'temperature', 0.8),
                )

                # Handle case where LLM wraps answers in a key
                if "answers" in answers and isinstance(answers["answers"], dict):
                    answers = answers["answers"]
                elif "responses" in answers and isinstance(answers["responses"], dict):
                    answers = answers["responses"]

                # Validate
                is_valid, errors = validator.validate(answers)

                if not is_valid:
                    # Try to fix
                    answers = self._attempt_fix(form_schema, answers, errors)
                    is_valid, errors = validator.validate(answers)

                if is_valid or attempt == self.settings.max_retries:
                    return GeneratedResponse(
                        persona_id=persona.persona_id,
                        persona_summary=persona.summary(),
                        answers=answers,
                        generation_success=is_valid,
                        retry_count=attempt,
                    )

            except (json.JSONDecodeError, TypeError) as e:
                if attempt >= self.settings.max_retries:
                    # Return a failed response
                    return GeneratedResponse(
                        persona_id=persona.persona_id,
                        persona_summary=persona.summary(),
                        answers={},
                        generation_success=False,
                        retry_count=attempt + 1,
                    )

        # Should not reach here, but just in case
        return GeneratedResponse(
            persona_id=persona.persona_id,
            persona_summary=persona.summary(),
            answers={},
            generation_success=False,
            retry_count=self.settings.max_retries + 1,
        )
