"""Single source of truth for the supported LLM providers.

Everything provider-specific — labels, the Settings attribute names for keys and
models, the Streamlit session-state keys, environment variable names, and the
selectable model list — lives here so config, the LLM client, and the UI can't
drift apart.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of one LLM provider."""

    id: str
    label: str            # full label, e.g. "Google Gemini"
    short_label: str      # compact label, e.g. "Gemini"
    key_attr: str         # Settings attribute holding the API key
    model_attr: str       # Settings attribute holding the model name
    session_key: str      # st.session_state key for the API key
    env_var: str          # environment variable for the API key
    key_placeholder: str  # placeholder text for the key input
    models: tuple         # selectable models; the first is the default

    @property
    def default_model(self) -> str:
        return self.models[0]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai", "OpenAI", "OpenAI",
        "openai_api_key", "openai_model", "api_key_openai", "OPENAI_API_KEY",
        "sk-...",
        ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"),
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic", "Anthropic",
        "anthropic_api_key", "anthropic_model", "api_key_anthropic", "ANTHROPIC_API_KEY",
        "sk-ant-...",
        ("claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"),
    ),
    "gemini": ProviderSpec(
        "gemini", "Google Gemini", "Gemini",
        "gemini_api_key", "gemini_model", "api_key_gemini", "GEMINI_API_KEY",
        "AIza...",
        ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"),
    ),
    "groq": ProviderSpec(
        "groq", "Groq (Llama)", "Groq (Llama)",
        "groq_api_key", "groq_model", "api_key_groq", "GROQ_API_KEY",
        "gsk_...",
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"),
    ),
    "mistral": ProviderSpec(
        "mistral", "Mistral AI", "Mistral",
        "mistral_api_key", "mistral_model", "api_key_mistral", "MISTRAL_API_KEY",
        "...",
        ("mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mixtral-8x22b"),
    ),
    "cohere": ProviderSpec(
        "cohere", "Cohere", "Cohere",
        "cohere_api_key", "cohere_model", "api_key_cohere", "COHERE_API_KEY",
        "...",
        ("command-r-plus", "command-r", "command-light"),
    ),
}

# Ordered list of provider ids (dict preserves insertion order).
PROVIDER_IDS: list[str] = list(PROVIDERS.keys())


def get_provider(provider_id: str) -> ProviderSpec:
    """Return the spec for a provider id, raising a clear error if unknown."""
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise ValueError(f"Unsupported LLM provider: {provider_id}")
