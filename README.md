# SynthSurvey — Synthetic Survey Data Generator

Generate realistic synthetic survey responses from any Google Form using LLM-powered unique persona generation.

## Live Demo

No hosted demo is currently published. See [Deployment](#deployment) to run
your own in a couple of minutes on Streamlit Community Cloud, Render, or
locally. Once deployed, add your URL here.

## Features

### Trustworthy synthetic data

- **Distribution / Quota Controls**: Enforce an exact demographic mix (gender, academic year, engagement level, age band). Targets are assigned deterministically and reconciled onto each persona — the requested marginals are hit exactly, not just requested of the model.
- **Fidelity Scoring**: Upload real responses and get a 0–100 score of how closely the synthetic distribution matches, per question and overall (Total Variation Distance based).
- **Bias & Quality Checks**: Automatic detection of straightlining, acquiescence/disagreement skew, low-variance answers, and duplicate responses — with suggested fixes.

### Research use cases

- **Correlated Latent-Trait Personas**: Each persona carries stable hidden traits (openness, skepticism, price sensitivity, tech savviness, …) that drive internally consistent, *correlated* answers across related questions — the way a real respondent's would. Trait scores are exported for analysis.
- **A/B Stimulus (Conjoint) Testing**: Give personas a concept/ad/product to react to. With 2+ variants, respondents are split evenly and you get a side-by-side A/B comparison of how the synthetic audience responds.
- **Longitudinal Wave Simulation**: Have the *same* personas answer across multiple simulated waves with realistic drift, and visualize how responses shift over time.

### Core

- **Google Form Parsing**: Automatically extracts form structure (questions, types, options, validation rules) from any public Google Form URL
- **Unique Persona Generation**: Each synthetic response comes from a distinct, LLM-generated persona with realistic demographics, personality traits, and engagement levels
- **In-Character Form Filling**: The LLM fills out each form as the generated persona, producing realistic and diverse responses
- **Response Validation**: Every generated answer is validated against the form schema (valid options, scale ranges, required fields)
- **6 LLM Providers**: OpenAI, Anthropic, Google Gemini, Groq (Llama), Mistral, and Cohere — switch in one click
- **Smart Retry**: Automatic exponential backoff on rate limits (429) and transient server errors
- **Generation Reports**: Detailed per-run reports with error classification, fix suggestions, and downloadable JSON logs
- **Multiple Export Formats**: CSV, JSON, and optional Google Sheets export
- **Real + Synthetic Data Merging**: Upload existing real survey data and append clearly-marked synthetic responses
- **Interactive Streamlit UI**: 4-page workflow with form preview, live generation progress, charts, and export

## Quick Start

### 1. Install Dependencies

```bash
cd synthsurvey
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API key(s):

```bash
cp .env.example .env
```

Edit `.env` and set at least one of:
- `OPENAI_API_KEY` — for GPT-4o, GPT-4o-mini, GPT-4-turbo
- `ANTHROPIC_API_KEY` — for Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- `GEMINI_API_KEY` — for Gemini 2.0 Flash, Gemini 1.5 Pro
- `GROQ_API_KEY` — for Llama 3.3 70B, Llama 3.1 8B, Gemma2 (fast & free tier)
- `MISTRAL_API_KEY` — for Mistral Large/Medium/Small
- `COHERE_API_KEY` — for Command R+, Command R

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Use the App

1. Paste a **public** Google Form URL
2. Click **Parse Form** to extract the form structure
3. Review the detected questions on the Preview page
4. Configure the number of responses and optional persona constraints
5. Click **Start Generation** and watch the progress
6. Download your synthetic dataset as CSV or JSON

## Project Structure

```
synthsurvey/
├── app.py                    # Streamlit main app (4-page UI)
├── config.py                 # Configuration via pydantic-settings
├── requirements.txt
├── .env.example
├── render.yaml               # Render.com deployment blueprint
├── Procfile                  # Process file for Render/Heroku
├── runtime.txt               # Python version pin
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
├── parsers/
│   └── google_form_parser.py # Google Form URL → structured JSON
├── generators/
│   ├── persona_generator.py  # Unique persona card generation
│   └── response_generator.py # Persona + Form → filled response
├── exporters/
│   ├── csv_exporter.py       # CSV/DataFrame export
│   ├── json_exporter.py      # JSON export
│   └── sheets_exporter.py    # Optional Google Sheets export
├── models/
│   ├── form_schema.py        # Pydantic models for form structure
│   ├── persona.py            # Pydantic model for persona cards
│   └── response.py           # Pydantic model for generated responses
├── prompts/
│   ├── persona_system.txt    # System prompt for persona generation
│   └── form_fill_system.txt  # System prompt for form filling
├── utils/
│   ├── llm_client.py         # Unified client for 6 LLM providers with retry logic
│   └── validators.py         # Validate responses against form schema
└── tests/
    ├── test_parser.py
    ├── test_persona_generator.py
    ├── test_response_generator.py
    └── test_exporters.py
```

## Supported Question Types

| Type | Support |
|------|---------|
| Short text | ✅ |
| Paragraph | ✅ |
| Multiple choice | ✅ |
| Checkboxes | ✅ |
| Dropdown | ✅ |
| Linear scale | ✅ |
| Multiple choice grid | ✅ |
| Checkbox grid | ✅ |
| Date | ✅ |
| Time | ✅ |
| Image-based questions | ⚠️ Flagged as unsupported |

## Persona Constraints

You can guide persona generation with free-text constraints:

- `"ASU undergrad students, mix of STEM and humanities"`
- `"60% undergrad, 40% grad students"`
- `"mostly engineering and design majors, ages 18-25"`
- `"diverse mix of engagement levels"`

## Running Tests

Install the dev dependencies (which include `pytest`), then run the suite:

```bash
cd synthsurvey
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Important Notes

- **All synthetic data is clearly marked** with an `is_synthetic = True` column
- The tool **never submits responses** to the actual Google Form — all data is generated offline
- API keys are loaded from environment variables only — never hardcoded
- The Google Form must be **publicly accessible** (not restricted to organization accounts)

## Deployment

### Option 1: Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Under **Advanced settings**, add your API keys as secrets:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**

> ⚠️ Streamlit Community Cloud apps go to sleep after a period of inactivity and take a few seconds to wake up on the next visit.

### Option 2: Render.com (Free — Always Online)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render will auto-detect `render.yaml`
4. In the Render dashboard, set environment variables:
   - `OPENAI_API_KEY` — your OpenAI key
   - `ANTHROPIC_API_KEY` — your Anthropic key (optional)
5. Click **Apply** — the app deploys automatically

**To prevent Render free-tier spin-down**, use a free cron ping service like [UptimeRobot](https://uptimerobot.com) to hit your Render URL every 14 minutes.

### Option 3: Local

```bash
cd synthsurvey
pip install -r requirements.txt
cp .env.example .env   # then edit with your API keys
streamlit run app.py
```

## Supported LLM Providers

| Provider | Models | Free Tier |
|----------|--------|-----------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo | No |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku | No |
| **Google Gemini** | Gemini 2.0 Flash, 2.0 Flash Lite, 1.5 Pro, 1.5 Flash | Yes (limited) |
| **Groq** | Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B | Yes (generous) |
| **Mistral** | Mistral Large, Medium, Small, Mixtral 8x22B | No |
| **Cohere** | Command R+, Command R, Command Light | Yes (limited) |

> The selectable models are defined in one place — [`providers.py`](providers.py).
> Providers retire and rename models over time; if a model errors with
> "not found," pick another from the sidebar or update the registry.

> **Tip:** Groq offers the most generous free tier — great for testing.

## Cost Estimates

The app displays estimated API costs before generation begins. Rough estimates per response:
- ~800 input tokens (persona + form filling)
- ~350 output tokens
- For 50 responses with GPT-4o: ~$0.05–$0.15

## License

Released under the [MIT License](LICENSE). Free for academic, hackathon,
and commercial use with attribution.
