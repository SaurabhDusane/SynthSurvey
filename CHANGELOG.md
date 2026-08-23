# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

### Added — Trustworthy synthetic data (feature phase 1)
- Distribution / quota controls: deterministic assignment of gender, academic
  year, engagement level, and age band so target marginals are hit exactly
  (`services/quota.py`), reconciled onto each persona regardless of LLM output.
- Fidelity scoring against uploaded real data (`analysis/fidelity.py`) with a
  0–100 overall score and per-question breakdown, shown in a new Fidelity tab.
- Bias & quality checks (`analysis/quality.py`): straightlining, response skew,
  low-variance answers, and duplicate detection, shown in a new Quality tab.

### Added
- Concurrent response generation with a configurable "Parallel Requests"
  control and a shared, thread-safe rate limiter.
- Checkpoint-to-disk with **resume** of interrupted runs (same configuration).
- Central provider registry (`providers.py`) as the single source of truth for
  providers, models, keys, and labels.
- Framework-agnostic `GenerationService` (`services/generation.py`).
- MIT `LICENSE`, `CONTRIBUTING.md`, this changelog, issue/PR templates, and
  Dependabot.
- GitHub Actions CI (compile, lint, tests on Python 3.11 & 3.12).
- Lint/format/type tooling (`ruff`, `black`, `mypy`) via `pyproject.toml`.
- Central logging configuration (`utils/logging_config.py`, `LOG_LEVEL`).
- Warning when a form is parsed via the degraded HTML fallback.

### Changed
- Model-aware cost estimation (priced per selected model, not per provider).
- Gemini migrated from the deprecated `google-generativeai` SDK to `google-genai`.
- All dependencies pinned below their next major for reproducible installs.
- The ~400-line CSS block moved to `static/styles.css`.
- README refreshed (license, deploy notes, provider/model source of truth).

### Fixed
- Replaced decommissioned Groq default models so the provider works out of the box.
- Escaped all LLM/form-derived content rendered into HTML (XSS).
- Restricted the form fetcher to Google Forms hosts (SSRF).
- Friendly, classified errors for parser network failures.
- Persona generation now retries on validation errors.
- "Test Connection" no longer errors due to helper-definition ordering.

## [1.0.0]
- Initial release: synthetic Google Form survey generation across six LLM
  providers, persona generation, validation, reports, and CSV/JSON export.
