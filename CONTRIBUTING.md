# Contributing to SynthSurvey

Thanks for your interest in improving SynthSurvey! This guide covers the
basics for local development.

## Development setup

```bash
git clone https://github.com/SaurabhDusane/SynthSurvey.git
cd SynthSurvey
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # add at least one API key
streamlit run app.py
```

## Before you open a PR

Run the same checks CI runs:

```bash
ruff check .          # lint
ruff format .         # auto-format (black-compatible)
mypy .                # type check (advisory)
pytest -q             # tests
```

- **Lint & format** are configured in `pyproject.toml` (`ruff`, `black`, `mypy`).
- **Tests** live in `tests/`. Please add tests for new behavior — the LLM
  client, parser, and generation service all have examples of mocking so you
  don't need real API keys.
- Keep the [`CHANGELOG.md`](CHANGELOG.md) updated under an `Unreleased` heading.

## Project layout

| Path | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI (thin adapter over the services) |
| `providers.py` | Single source of truth for supported LLM providers/models |
| `services/generation.py` | Framework-agnostic generation orchestration |
| `generators/` | Persona + response generation |
| `parsers/` | Google Form → `FormSchema` |
| `exporters/` | CSV / JSON / Sheets output |
| `models/` | Pydantic data models |
| `utils/` | LLM client, validators, logging |

## Adding a new LLM provider

Add a `ProviderSpec` entry to `providers.py`, then implement a
`_generate_<provider>` method in `utils/llm_client.py` and wire it into the
dispatch map. Everything else (config fields, sidebar, cost estimation)
derives from the registry.

## Guidelines

- Match the surrounding code style; keep changes focused.
- Never commit real API keys or `.env` files.
- All generated data must remain clearly marked as synthetic.
