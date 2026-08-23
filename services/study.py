"""Saveable / shareable "study" configurations.

A study captures everything about a generation run *except* secrets, so a setup
can be saved, version-controlled, shared, and re-run reproducibly. API keys are
never included.

This module handles the pure build/serialize/parse; mapping a loaded study back
onto Streamlit widget state lives in the app (it knows the widget keys).
"""

from __future__ import annotations

import json
from typing import Any

from providers import PROVIDERS

STUDY_KIND = "synthsurvey_study"
STUDY_VERSION = 1

# Semantic session-state keys that make up a study (no API keys).
STUDY_KEYS = [
    "form_url",
    "num_responses",
    "persona_constraints",
    "llm_provider",
    "temperature",
    "api_call_delay",
    "concurrency",
    "max_retries",
    "quota_targets",
    "enable_traits",
    "waves",
    "stimuli",
    "seed",
]


def build_study_config(state: Any) -> dict:
    """Build a serializable study dict from a session-state-like mapping."""
    cfg: dict = {"kind": STUDY_KIND, "version": STUDY_VERSION}
    for key in STUDY_KEYS:
        cfg[key] = state.get(key)
    # Per-provider selected models (so switching provider keeps the choice).
    cfg["models"] = {
        spec.model_attr: state.get(spec.model_attr) for spec in PROVIDERS.values()
    }
    return cfg


def study_to_json(cfg: dict) -> str:
    return json.dumps(cfg, indent=2)


def study_to_bytes(cfg: dict) -> bytes:
    return study_to_json(cfg).encode("utf-8")


def parse_study_config(raw) -> dict:
    """Parse and validate a study file (str/bytes/dict)."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or data.get("kind") != STUDY_KIND:
        raise ValueError("Not a SynthSurvey study file.")
    return data
