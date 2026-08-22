"""Configuration and environment variables for SynthSurvey."""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

from providers import PROVIDERS

_PROJECT_DIR = Path(__file__).parent
_ENV_FILE = _PROJECT_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: Optional[str] = Field(default=None, alias="MISTRAL_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")

    # Default LLM provider
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")

    # Model names (defaults come from the central provider registry)
    openai_model: str = Field(default=PROVIDERS["openai"].default_model, alias="OPENAI_MODEL")
    anthropic_model: str = Field(default=PROVIDERS["anthropic"].default_model, alias="ANTHROPIC_MODEL")
    gemini_model: str = Field(default=PROVIDERS["gemini"].default_model, alias="GEMINI_MODEL")
    groq_model: str = Field(default=PROVIDERS["groq"].default_model, alias="GROQ_MODEL")
    mistral_model: str = Field(default=PROVIDERS["mistral"].default_model, alias="MISTRAL_MODEL")
    cohere_model: str = Field(default=PROVIDERS["cohere"].default_model, alias="COHERE_MODEL")

    # Generation settings
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # Rate limiting
    api_call_delay: float = Field(default=0.5, alias="API_CALL_DELAY")

    # Number of responses generated in parallel (paced by api_call_delay)
    concurrency: int = Field(default=4, alias="CONCURRENCY")

    # Google Sheets (optional)
    google_sheets_credentials_file: Optional[str] = Field(
        default=None, alias="GOOGLE_SHEETS_CREDENTIALS_FILE"
    )

    # Generation defaults
    default_response_count: int = 50
    max_response_count: int = 500
    max_retries: int = 2
    batch_persona_size: int = 5

    # Default university
    default_university: str = "Arizona State University"

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
