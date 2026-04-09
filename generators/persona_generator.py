"""Unique persona card generation using LLM."""

import json
import os
from pathlib import Path
from typing import Optional

from config import Settings
from models.form_schema import FormSchema
from models.persona import Persona
from utils.llm_client import LLMClient


class PersonaGenerator:
    """Generates unique personas for survey response simulation."""

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.generated_personas: list[Persona] = []
        self._seen_attribute_tuples: set[tuple] = set()
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the persona system prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "persona_system.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_prompt(
        self,
        form_schema: FormSchema,
        user_constraints: Optional[str] = None,
    ) -> str:
        """Build the system prompt for persona generation."""
        # Collect summaries of previously generated personas (last 20 to keep prompt size manageable)
        recent_summaries = [p.summary() for p in self.generated_personas[-20:]]
        previous_str = (
            "\n".join(f"- {s}" for s in recent_summaries)
            if recent_summaries
            else "None yet — you are generating the first persona."
        )

        return self._prompt_template.format(
            form_title=form_schema.form_title,
            form_description=form_schema.form_description or "No description provided",
            user_constraints=user_constraints or "Generate a diverse, realistic respondent.",
            previous_personas=previous_str,
            default_university=self.settings.default_university,
        )

    def generate_one(
        self,
        form_schema: FormSchema,
        user_constraints: Optional[str] = None,
    ) -> Persona:
        """Generate a single unique persona.

        Args:
            form_schema: The parsed form structure.
            user_constraints: Optional constraints for persona generation.

        Returns:
            A unique Persona object.

        Raises:
            ValueError: If unable to generate a unique persona after retries.
        """
        system_prompt = self._build_prompt(form_schema, user_constraints)
        user_prompt = (
            "Generate one unique persona as a JSON object. "
            "Remember: it must be different from all previously generated personas."
        )

        for attempt in range(self.settings.max_retries + 1):
            try:
                # Persona generation benefits from slightly higher temperature for diversity
                persona_temp = min(getattr(self.settings, 'temperature', 0.9) + 0.1, 1.5)
                data = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=persona_temp,
                )

                # Handle case where LLM wraps in a key
                if "persona" in data and isinstance(data["persona"], dict):
                    data = data["persona"]

                persona = Persona(**data)

                # Check uniqueness
                attr_tuple = persona.attribute_tuple()
                if attr_tuple in self._seen_attribute_tuples:
                    if attempt < self.settings.max_retries:
                        continue
                    # On last attempt, modify slightly to force uniqueness
                    persona.persona_id = persona.persona_id + "_r"

                self._seen_attribute_tuples.add(attr_tuple)
                self.generated_personas.append(persona)
                return persona

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                if attempt >= self.settings.max_retries:
                    raise ValueError(
                        f"Failed to generate valid persona after {self.settings.max_retries + 1} attempts: {e}"
                    )

        raise ValueError("Failed to generate unique persona")

    def generate_batch(
        self,
        form_schema: FormSchema,
        count: int,
        user_constraints: Optional[str] = None,
    ) -> list[Persona]:
        """Generate multiple unique personas.

        Args:
            form_schema: The parsed form structure.
            count: Number of personas to generate.
            user_constraints: Optional constraints.

        Returns:
            List of unique Persona objects.
        """
        personas = []
        for _ in range(count):
            persona = self.generate_one(form_schema, user_constraints)
            personas.append(persona)
        return personas

    def reset(self) -> None:
        """Reset the generator state for a new generation run."""
        self.generated_personas.clear()
        self._seen_attribute_tuples.clear()
