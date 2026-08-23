"""JSON export for survey datasets."""

import json
from typing import Any

from models.form_schema import FormSchema
from models.response import SurveyDataset


class JSONExporter:
    """Export survey dataset to JSON format."""

    @staticmethod
    def to_dict(dataset: SurveyDataset, form_schema: FormSchema) -> dict[str, Any]:
        """Convert a SurveyDataset to a JSON-serializable dictionary.

        Args:
            dataset: The generated survey dataset.
            form_schema: The form schema for metadata.

        Returns:
            A dictionary ready for JSON serialization.
        """
        q_map = {q.question_id: q.question_text for q in form_schema.questions}

        responses = []
        for resp in dataset.responses:
            # Map question IDs to question text for readability
            readable_answers = {}
            for qid, answer in resp.answers.items():
                col_name = q_map.get(qid, qid)
                readable_answers[col_name] = answer

            entry = {
                "persona_id": resp.persona_id,
                "persona_summary": resp.persona_summary,
                "is_synthetic": resp.is_synthetic,
                "generation_timestamp": resp.generation_timestamp,
                "generation_success": resp.generation_success,
                "answers": readable_answers,
            }
            if getattr(resp, "wave", 1) != 1:
                entry["wave"] = resp.wave
            if getattr(resp, "stimulus_variant", None):
                entry["stimulus_variant"] = resp.stimulus_variant
            if getattr(resp, "latent_traits", None):
                entry["latent_traits"] = resp.latent_traits
            responses.append(entry)

        return {
            "metadata": {
                "form_title": dataset.form_title,
                "form_url": dataset.form_url,
                "total_generated": dataset.total_generated,
                "total_failed": dataset.total_failed,
                "generation_started": dataset.generation_started,
                "generation_completed": dataset.generation_completed,
            },
            "questions": [
                {
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "question_type": q.question_type.value,
                    "options": q.options,
                    "is_required": q.is_required,
                }
                for q in form_schema.questions
            ],
            "responses": responses,
        }

    @staticmethod
    def to_json_string(dataset: SurveyDataset, form_schema: FormSchema) -> str:
        """Convert to a formatted JSON string."""
        data = JSONExporter.to_dict(dataset, form_schema)
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def to_json_bytes(dataset: SurveyDataset, form_schema: FormSchema) -> bytes:
        """Convert to JSON bytes for download."""
        return JSONExporter.to_json_string(dataset, form_schema).encode("utf-8")
