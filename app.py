"""SynthSurvey — Streamlit web application for synthetic survey data generation."""

import json
import os
import sys
import time
from datetime import datetime
from html import escape as _esc
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.fidelity import compute_fidelity, fidelity_verdict
from analysis.quality import analyze_quality
from analysis.stats import crosstab_test, effect_size_label
from analysis.survey_lint import lint_survey
from config import Settings, get_settings
from exporters.csv_exporter import CSVExporter
from exporters.json_exporter import JSONExporter
from models.form_schema import FormSchema, QuestionType
from models.persona import Persona
from models.response import GeneratedResponse, SurveyDataset
from parsers.google_form_parser import GoogleFormParser
from providers import PROVIDER_IDS, PROVIDERS
from services.generation import GenerationService, peek_checkpoint
from services.provenance import (
    APP_VERSION,
    DISCLAIMER,
    build_manifest,
    manifest_to_bytes,
)
from services.quota import SUPPORTED_ATTRS, build_assignment_plan
from services.stimulus import build_stimulus_plan
from services.study import build_study_config, parse_study_config, study_to_bytes
from utils.llm_client import LLMClient
from utils.logging_config import setup_logging

# ─── Logging ──────────────────────────────────────────────────────────────────

setup_logging(get_settings().log_level)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SynthSurvey",
    page_icon="https://em-content.zobj.net/source/twitter/408/chart-increasing_1f4c8.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS & Theme ─────────────────────────────────────────────────────

# Dark palette — re-declares the stylesheet's :root variables. Injected after the
# (light-default) stylesheet when the "Dark mode" toggle is on, so it wins.
_DARK_OVERRIDE = """
:root {
    --bg-primary:#0c0a09;
    --bg-secondary:linear-gradient(180deg,#0f0c09,#12100d);
    --bg-card:rgba(255,245,230,0.02);
    --bg-glass:rgba(255,245,230,0.035);
    --border-glass:rgba(255,200,120,0.08);
    --overlay:rgba(255,255,255,0.05);
    --hairline:rgba(255,255,255,0.08);
    --dl-bg:linear-gradient(135deg,#1a1510,#1c1812);
    --accent:#F59E0B; --accent2:#FB923C; --accent3:#14B8A6; --accent4:#F472B6;
    --text1:#FAF5EF; --text2:#B8A99A; --text3:#8a7f74;
}
.stApp, [data-testid="stHeader"] { background:#0c0a09 !important; }
/* Native widget surfaces that inherit Streamlit's light theme — darken them.
   Solid backgrounds on the actual input elements so no light layer bleeds through. */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
.stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div,
.stSelectbox > div > div, [data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
    background-color:#17120e !important; color:var(--text1) !important;
}
.stNumberInput button, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"],
[data-testid="stFileUploader"] button {
    background:#211a13 !important; color:var(--text1) !important;
    border-color:var(--border-glass) !important;
}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"],
[data-baseweb="popover"] ul, [data-baseweb="popover"] div { background-color:#17120e !important; }
[data-testid="stDataFrame"] { background:#17120e !important; }
"""


@st.cache_data
def _load_css() -> str:
    """Load the stylesheet from the static file (cached across reruns)."""
    return (Path(__file__).parent / "static" / "styles.css").read_text(encoding="utf-8")


def _dark_mode() -> bool:
    return bool(st.session_state.get("dark_mode", False))


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)
if _dark_mode():
    st.markdown(f"<style>{_DARK_OVERRIDE}</style>", unsafe_allow_html=True)
st.markdown(
    '<div class="particles">'
    '<div class="p"></div><div class="p"></div><div class="p"></div>'
    '<div class="p"></div><div class="p"></div><div class="p"></div></div>',
    unsafe_allow_html=True,
)

# ─── Session State ────────────────────────────────────────────────────────────


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "page": "input",
        "form_schema": None,
        "form_url": "",
        "num_responses": get_settings().default_response_count,
        "persona_constraints": "",
        "existing_data": None,
        "llm_provider": "openai",
        "dataset": None,
        "generation_in_progress": False,
        "latest_persona": None,
        "latest_response": None,
        "generated_count": 0,
        "failed_count": 0,
        "results_df": None,
        "temperature": 0.7,
        "api_call_delay": 0.5,
        "concurrency": 4,
        "max_retries": 2,
        "generation_report": [],
        "stop_generation": False,
        "quota_targets": {},
        "enable_traits": False,
        "waves": 1,
        "stimuli": [],
        "seed": 42,
        "dark_mode": False,
    }
    # Per-provider API key + model defaults, from the central registry.
    for spec in PROVIDERS.values():
        defaults[spec.session_key] = ""
        defaults[spec.model_attr] = spec.default_model
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Settings helpers (defined before the sidebar, which calls them) ──────────


def get_llm_settings() -> Settings:
    """Build Settings object from session state."""
    settings = get_settings()
    settings.llm_provider = st.session_state.llm_provider
    # Map all provider keys and models from session state via the registry.
    for spec in PROVIDERS.values():
        if st.session_state.get(spec.session_key):
            setattr(settings, spec.key_attr, st.session_state[spec.session_key])
        if st.session_state.get(spec.model_attr):
            setattr(settings, spec.model_attr, st.session_state[spec.model_attr])
    settings.temperature = st.session_state.temperature
    settings.api_call_delay = st.session_state.api_call_delay
    settings.concurrency = st.session_state.concurrency
    settings.max_retries = st.session_state.max_retries
    settings.seed = int(st.session_state.get("seed", 42))
    return settings


def check_api_key() -> bool:
    """Check if the required API key is set."""
    spec = PROVIDERS.get(st.session_state.llm_provider)
    if spec is None:
        return False
    return bool(st.session_state.get(spec.session_key) or os.environ.get(spec.env_var, ""))


def apply_pending_study() -> None:
    """Apply a loaded study config to widget/session state.

    Runs once at the top of the script (before any widget is instantiated), so
    setting widget-backed keys is safe. A study only sets config, never secrets.
    """
    cfg = st.session_state.pop("_pending_study", None)
    if not cfg:
        return

    def setk(key, value):
        if value is not None:
            st.session_state[key] = value

    # Sidebar/input widgets are controlled (keyed by these semantic names), so
    # setting the semantic key here — before any widget instantiates — restores
    # them without Streamlit's "default + session state" warning.
    prov = cfg.get("llm_provider")
    setk("form_url", cfg.get("form_url"))
    setk("num_responses", cfg.get("num_responses"))
    setk("persona_constraints", cfg.get("persona_constraints"))
    if prov in PROVIDERS:
        setk("llm_provider", prov)
    setk("temperature", cfg.get("temperature"))
    setk("api_call_delay", cfg.get("api_call_delay"))
    setk("concurrency", cfg.get("concurrency"))
    setk("max_retries", cfg.get("max_retries"))
    for mk, mv in (cfg.get("models") or {}).items():
        setk(mk, mv)
    setk("enable_traits", cfg.get("enable_traits"))
    setk("waves", cfg.get("waves"))
    setk("seed", cfg.get("seed"))

    stimuli = cfg.get("stimuli") or []
    st.session_state["n_variants"] = len(stimuli)
    for i, s in enumerate(stimuli):
        setk(f"stim_name_{i}", s.get("name"))
        setk(f"stim_text_{i}", s.get("text"))
    st.session_state["stimuli"] = stimuli

    qt = cfg.get("quota_targets") or {}
    st.session_state["quota_targets"] = qt
    for attr in SUPPORTED_ATTRS:
        st.session_state[f"quota_{attr}_enabled"] = attr in qt
        for bucket, pct in (qt.get(attr) or {}).items():
            st.session_state[f"quota_{attr}_{bucket}"] = pct


apply_pending_study()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .3rem;">
        <div style="font-family:var(--font-display);font-size:1.9rem;font-weight:800;
                    letter-spacing:-0.04em;color:var(--text1)!important;">
            Synth<span style="color:var(--accent)!important;">Survey</span></div>
        <div style="font-size:.66rem;color:var(--text3)!important;text-transform:uppercase;
                    letter-spacing:.18em;font-weight:600;margin-top:4px;">
            Synthetic Data Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.toggle(
        "🌙 Dark mode", key="dark_mode",
        help="Switch between the light and dark theme.",
    )

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

    pages_meta = {
        "input": "1. Input",
        "preview": "2. Preview",
        "generate": "3. Generate",
        "results": "4. Results",
    }

    for key, label in pages_meta.items():
        disabled = False
        if key == "preview" and st.session_state.form_schema is None:
            disabled = True
        if key == "generate" and st.session_state.form_schema is None:
            disabled = True
        if key == "results" and st.session_state.dataset is None:
            disabled = True

        is_active = st.session_state.page == key
        btn_label = f"{'>' if is_active else '  '} {label}"
        if st.button(btn_label, key=f"nav_{key}", disabled=disabled, width="stretch"):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)

    if st.session_state.get("llm_provider") not in PROVIDER_IDS:
        st.session_state["llm_provider"] = PROVIDER_IDS[0]
    provider = st.selectbox(
        "LLM Provider",
        PROVIDER_IDS,
        format_func=lambda x: PROVIDERS[x].label,
        key="llm_provider",
        label_visibility="collapsed",
    )
    spec = PROVIDERS[provider]

    key_attr = spec.session_key
    api_key = st.text_input(
        f"{spec.label} API Key", type="password",
        value=st.session_state[key_attr],
        key=f"sidebar_key_{provider}", placeholder=spec.key_placeholder,
    )
    st.session_state[key_attr] = api_key

    st.markdown('<div class="section-label">Model Settings</div>', unsafe_allow_html=True)

    model_options = list(spec.models)
    model_state_key = spec.model_attr
    if st.session_state.get(model_state_key) not in model_options:
        st.session_state[model_state_key] = spec.default_model
    selected_model = st.selectbox(
        f"{spec.label} Model",
        model_options,
        key=model_state_key,
    )

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.5, step=0.05,
        key="temperature",
        help="Higher = more creative & varied responses. Lower = more consistent.",
    )

    st.markdown('<div class="section-label">Advanced</div>', unsafe_allow_html=True)

    api_delay = st.slider(
        "API Call Delay (s)", min_value=0.0, max_value=3.0, step=0.1,
        key="api_call_delay",
        help="Delay between API calls to avoid rate limits.",
    )

    concurrency = st.slider(
        "Parallel Requests", min_value=1, max_value=8, step=1,
        key="concurrency",
        help="How many responses to generate at once. Higher = faster, but more likely to hit rate limits.",
    )

    max_retries = st.number_input(
        "Max Retries per Response", min_value=0, max_value=5, step=1,
        key="max_retries",
        help="Number of retry attempts if a response fails validation.",
    )

    # Test Connection button
    if st.button("Test Connection", width="stretch", key="sidebar_test_conn"):
        if not check_api_key():
            st.error("Enter an API key first.")
        else:
            with st.spinner("Testing..."):
                try:
                    _settings = get_llm_settings()
                    _llm = LLMClient(_settings)
                    result = _llm.test_connection()
                    if result["success"]:
                        st.success(f"Connected to {result['provider']} ({result['model']})")
                    else:
                        st.error(f"{result.get('error_type', 'Error')}: {result['message'][:120]}")
                        if result.get("suggestion"):
                            st.info(result["suggestion"])
                except Exception as ex:
                    st.error(f"Connection failed: {ex}")

    st.markdown("""
    <div class="disclaimer-v2" style="margin-top:1.2rem;">
        <strong>Disclaimer</strong><br>
        Synthetic data is for testing & development only. Never present it as
        real survey data without clear disclosure.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:1.5rem;padding-top:.8rem;
                border-top:1px solid rgba(255,255,255,.04);">
        <span style="font-size:.68rem;color:#5C5775;font-weight:500;
                     background:rgba(255,255,255,.03);padding:3px 10px;
                     border-radius:16px;border:1px solid rgba(255,255,255,.05);">
            v{APP_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _type_icon(qt: QuestionType) -> str:
    m = {
        QuestionType.SHORT_TEXT: "Aa", QuestionType.PARAGRAPH: "Pg",
        QuestionType.MULTIPLE_CHOICE: "MC", QuestionType.CHECKBOX: "CB",
        QuestionType.DROPDOWN: "DD", QuestionType.LINEAR_SCALE: "LS",
        QuestionType.MULTIPLE_CHOICE_GRID: "MG", QuestionType.CHECKBOX_GRID: "CG",
        QuestionType.DATE: "Dt", QuestionType.TIME: "Tm", QuestionType.UNSUPPORTED: "??",
    }
    return m.get(qt, "??")


def render_question_type_badge(qt: QuestionType) -> str:
    return f"`{_type_icon(qt)}` {qt.value.replace('_',' ').title()}"


def render_survey_review(schema: FormSchema) -> None:
    """Survey-design lint review shown before generation (F3.2)."""
    result = lint_survey(schema)
    warn = result["warn_count"]
    with st.expander(f"Survey Design Review — {result['verdict']}", expanded=warn > 0):
        if not result["findings"]:
            st.success("No common survey-design issues detected.")
            return
        st.caption("Heuristic checks on question wording and options — advisory, not blocking.")
        icons = {"warn": ("&#9888;", "#F59E0B"), "info": ("&#8505;", "#60A5FA")}
        for f in result["findings"]:
            icon, c = icons.get(f["severity"], ("&#8226;", "#9aa"))
            st.markdown(
                f'<div class="glass-card" style="margin-bottom:.5rem;">'
                f'<div style="display:flex;align-items:center;gap:.5rem;">'
                f'<span style="color:{c};">{icon}</span>'
                f'<strong>{_esc(f["issue"])}</strong></div>'
                f'<div style="font-size:.84rem;color:var(--text2);margin-top:3px;">'
                f'Q: {_esc(f["question"][:120])}</div>'
                f'<div style="font-size:.82rem;color:var(--text3);margin-top:3px;">'
                f'{_esc(f["suggestion"])}</div></div>',
                unsafe_allow_html=True,
            )


_QUOTA_LABELS = {
    "gender": "Gender",
    "year": "Academic Year",
    "engagement_level": "Engagement Level",
    "age_range": "Age Range",
}


def render_quota_controls() -> None:
    """Render optional distribution/quota controls; store the spec in state.

    Each enabled attribute's target proportions are honored *exactly* at
    generation time (deterministic pre-assignment), not merely requested of
    the model.
    """
    spec: dict = {}
    active = sum(
        1 for attr in SUPPORTED_ATTRS
        if st.session_state.get(f"quota_{attr}_enabled")
    )
    title = f"Distribution Controls — {active} active" if active else "Distribution Controls (optional)"
    with st.expander(title):
        st.caption(
            "Enforce an exact demographic mix. Enabled attributes are matched "
            "exactly across your responses — the tool assigns them, it doesn't "
            "just ask the model to comply."
        )
        for attr, buckets in SUPPORTED_ATTRS.items():
            label = _QUOTA_LABELS.get(attr, attr)
            if not st.checkbox(f"Constrain {label}", key=f"quota_{attr}_enabled"):
                continue
            default = round(100 / len(buckets))
            cols = st.columns(len(buckets))
            bucket_pcts = {}
            for col, bucket in zip(cols, buckets):
                bkey = f"quota_{attr}_{bucket}"
                st.session_state.setdefault(bkey, default)
                with col:
                    bucket_pcts[bucket] = st.number_input(
                        bucket, min_value=0, max_value=100, step=5, key=bkey,
                    )
            total = sum(bucket_pcts.values())
            if total <= 0:
                st.warning(f"{label}: set at least one bucket above 0%.")
                continue
            if total != 100:
                st.caption(f"{label}: sums to {total}% — will be normalized to 100%.")
            spec[attr] = bucket_pcts
    st.session_state.quota_targets = spec


def render_study_templates() -> None:
    """Save / load the full study configuration (phase 3). No API keys."""
    with st.expander("Study Templates — save or load this configuration"):
        st.caption("Export everything except API keys, so a study setup can be "
                   "shared and re-run reproducibly.")
        st.download_button(
            "Download study config (.json)",
            data=study_to_bytes(build_study_config(st.session_state)),
            file_name="synthsurvey_study.json",
            mime="application/json",
            width="stretch",
            key="study_download",
        )
        up = st.file_uploader("Load a study config", type=["json"], key="study_upload")
        if up is not None and st.button(
            "Apply loaded study", key="study_apply", width="stretch"
        ):
            try:
                st.session_state["_pending_study"] = parse_study_config(up.getvalue())
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load study: {e}")


def render_research_controls() -> None:
    """Latent traits, longitudinal waves, and A/B stimulus controls (phase 2)."""
    with st.expander("Research Controls (latent traits · waves · A/B testing)"):
        st.checkbox(
            "Correlated latent traits",
            key="enable_traits",
            help="Give each persona stable hidden traits so related questions "
                 "correlate the way a real person's answers would.",
        )

        st.number_input(
            "Longitudinal waves", min_value=1, max_value=5, step=1, key="waves",
            help="Each persona answers once per wave with realistic drift. "
                 "Total responses = personas × waves.",
        )

        st.number_input(
            "Random seed", min_value=0, max_value=1_000_000, step=1, key="seed",
            help="Fixes persona/variant assignment and trait sampling so a run is "
                 "reproducible. (LLM answers still vary unless temperature = 0.)",
        )

        st.markdown("**A/B stimulus variants** (optional)")
        st.caption("Give personas a concept/ad/product to react to. With 2+ "
                   "variants, personas are split evenly so you can compare A vs. B.")
        n_variants = st.number_input(
            "Number of variants", min_value=0, max_value=4, step=1, key="n_variants",
        )
        stimuli = []
        for i in range(int(n_variants)):
            nkey = f"stim_name_{i}"
            st.session_state.setdefault(nkey, f"Variant {chr(65 + i)}")
            cols = st.columns([1, 3])
            with cols[0]:
                name = st.text_input("Name", key=nkey)
            with cols[1]:
                text = st.text_area(
                    "Description", key=f"stim_text_{i}", height=70,
                    placeholder="Describe the concept this group evaluates…",
                )
            if text and text.strip():
                stimuli.append({"name": name, "text": text.strip()})
        st.session_state.stimuli = stimuli


# ─── Page: Input ──────────────────────────────────────────────────────────────


def render_input_page():
    st.markdown("""
    <div style="padding:.5rem 0 .3rem;">
        <div class="hero-title">SynthSurvey</div>
        <div class="hero-sub">
            Transform any Google Form into a rich synthetic dataset.
            AI-powered personas generate realistic, validated responses in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_study_templates()
    st.divider()

    col_main, col_side = st.columns([5, 3], gap="large")

    with col_main:
        st.markdown("##### Google Form URL")
        form_url = st.text_input(
            "url",
            placeholder="https://docs.google.com/forms/d/e/.../viewform",
            key="form_url", label_visibility="collapsed",
        )

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("##### Responses")
            _max_n = get_settings().max_response_count
            if st.session_state.get("num_responses", 0) > _max_n:
                st.session_state["num_responses"] = _max_n
            st.number_input(
                "n", min_value=1, max_value=_max_n, step=10,
                key="num_responses", label_visibility="collapsed",
            )
        with sc2:
            st.markdown("##### Provider")
            _spec = PROVIDERS.get(st.session_state.llm_provider)
            _prov_name = _spec.short_label if _spec else "Unknown"
            _model_val = st.session_state.get(_spec.model_attr, "") if _spec else ""
            prov_label = f"{_prov_name} · {_model_val}"
            st.markdown(
                f'<div style="padding:.6rem 1rem;background:var(--bg-glass);'
                f'border:1px solid var(--border-glass);border-radius:var(--radius);'
                f'font-size:.88rem;color:var(--text2);margin-top:2px;">{_esc(prov_label)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("##### Persona Constraints")
        st.text_area(
            "c",
            placeholder="e.g., ASU undergrad students, mix of STEM and humanities, ages 18-25...",
            height=85, key="persona_constraints", label_visibility="collapsed",
        )

        render_quota_controls()
        render_research_controls()

        st.markdown("##### Existing Data (Optional)")
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], key="input_upload", label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                st.session_state.existing_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(st.session_state.existing_data)} existing responses.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with col_side:
        st.markdown("""
        <div class="glass-card">
            <div style="font-weight:700;font-size:.95rem;margin-bottom:.8rem;color:var(--text1);">How It Works</div>
            <div class="step-row"><span class="step-badge">1</span><span>Paste a Google Form URL</span></div>
            <div class="step-row"><span class="step-badge">2</span><span>We parse every question & option</span></div>
            <div class="step-row"><span class="step-badge">3</span><span>AI generates unique personas</span></div>
            <div class="step-row"><span class="step-badge">4</span><span>Each persona fills the form</span></div>
            <div class="step-row"><span class="step-badge">5</span><span>Download your synthetic dataset</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:.8rem;">
            <div style="font-weight:700;font-size:.95rem;margin-bottom:.6rem;color:var(--text1);">Capabilities</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
                <span class="chip">10 Question Types</span>
                <span class="chip">Unique Personas</span>
                <span class="chip">CSV & JSON</span>
                <span class="chip">Validation</span>
                <span class="chip">Cost Estimates</span>
                <span class="chip">Real + Synthetic</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature cards row
    st.divider()
    st.markdown("##### What Makes SynthSurvey Different")
    fc1, fc2, fc3, fc4 = st.columns(4)
    # Inline SVG icons for a professional look
    svg_personas = (
        '<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="16" cy="10" r="5" stroke="#F59E0B" stroke-width="1.8" fill="rgba(245,158,11,.12)"/>'
        '<path d="M6 28c0-5.523 4.477-10 10-10s10 4.477 10 10" stroke="#F59E0B" stroke-width="1.8" stroke-linecap="round" fill="none"/>'
        '<circle cx="24" cy="8" r="3.5" stroke="#14B8A6" stroke-width="1.4" fill="rgba(20,184,166,.1)"/>'
        '<path d="M28 18c0-3.5-2.5-6-5.5-6" stroke="#14B8A6" stroke-width="1.4" stroke-linecap="round" fill="none" opacity=".6"/>'
        '</svg>'
    )
    svg_validation = (
        '<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="3" y="3" width="26" height="26" rx="6" stroke="#14B8A6" stroke-width="1.8" fill="rgba(20,184,166,.08)"/>'
        '<path d="M10 16.5l4 4 8.5-9" stroke="#14B8A6" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="25" cy="7" r="3" fill="#F59E0B" opacity=".7"/>'
        '<path d="M24 7h2M25 6v2" stroke="#fff" stroke-width=".8" stroke-linecap="round"/>'
        '</svg>'
    )
    svg_analytics = (
        '<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="4" y="18" width="5" height="10" rx="1.5" fill="#F59E0B" opacity=".7"/>'
        '<rect x="13.5" y="12" width="5" height="16" rx="1.5" fill="#14B8A6" opacity=".8"/>'
        '<rect x="23" y="6" width="5" height="22" rx="1.5" fill="#FB923C" opacity=".7"/>'
        '<path d="M4 8l8 4 8-6 8 2" stroke="#F472B6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        '<circle cx="4" cy="8" r="1.5" fill="#F472B6"/><circle cx="12" cy="12" r="1.5" fill="#F472B6"/>'
        '<circle cx="20" cy="6" r="1.5" fill="#F472B6"/><circle cx="28" cy="8" r="1.5" fill="#F472B6"/>'
        '</svg>'
    )
    svg_privacy = (
        '<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="7" y="14" width="18" height="14" rx="3" stroke="#F59E0B" stroke-width="1.8" fill="rgba(245,158,11,.08)"/>'
        '<path d="M11 14v-3a5 5 0 0110 0v3" stroke="#F59E0B" stroke-width="1.8" stroke-linecap="round"/>'
        '<circle cx="16" cy="21" r="2" fill="#F59E0B"/>'
        '<line x1="16" y1="23" x2="16" y2="25" stroke="#F59E0B" stroke-width="1.8" stroke-linecap="round"/>'
        '</svg>'
    )
    features = [
        (fc1, svg_personas, "Unique Personas", "Every response comes from a distinct AI-generated character"),
        (fc2, svg_validation, "Smart Validation", "Answers are verified against form rules & auto-corrected"),
        (fc3, svg_analytics, "Rich Analytics", "Interactive charts comparing real vs synthetic data"),
        (fc4, svg_privacy, "Privacy First", "API keys stay local. No data leaves your session"),
    ]
    for col, icon, title, desc in features:
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    bc, _, _ = st.columns([2, 1, 1])
    with bc:
        if st.button("Parse Form & Continue", type="primary", width="stretch"):
            if not form_url:
                st.error("Please enter a Google Form URL.")
                return
            with st.spinner("Analyzing form structure..."):
                try:
                    parser = GoogleFormParser(form_url)
                    schema = parser.parse()
                    if not schema.questions:
                        st.error("No questions found. The form may be private or the URL invalid.")
                        return
                    st.session_state.form_schema = schema
                    st.session_state.page = "preview"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse form: {e}")

    # Keyboard shortcuts hint
    with st.expander("Keyboard Shortcuts & Tips"):
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem .8rem;font-size:.88rem;">'
            '<div><span class="kbd">Ctrl</span> + <span class="kbd">Enter</span> &nbsp; Submit forms</div>'
            '<div><span class="kbd">R</span> &nbsp; Refresh page</div>'
            '<div><span class="kbd">Ctrl</span> + <span class="kbd">S</span> &nbsp; Download CSV (on Results)</div>'
            '<div><span class="kbd">1</span>-<span class="kbd">4</span> &nbsp; Navigate pages via sidebar</div>'
            '</div>'
            '<div style="margin-top:.8rem;font-size:.82rem;color:var(--text3);">'
            'Tip: Use persona constraints to get more targeted demographics. '
            'The more specific you are, the more realistic the output.</div>',
            unsafe_allow_html=True,
        )

    # Footer
    _render_footer()


def _render_footer():
    """Render the app footer."""
    st.markdown(f"""
    <div class="app-footer">
        <div class="footer-links">
            <a href="https://github.com/SaurabhDusane/SynthSurvey" target="_blank">GitHub</a>
            <a href="#">Documentation</a>
            <a href="#">Report a Bug</a>
            <a href="#">Privacy Policy</a>
        </div>
        <div class="footer-copy">
            SynthSurvey v{APP_VERSION} &mdash; Built for academic & hackathon use &mdash;
            Synthetic data should always be clearly disclosed.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Page: Preview ────────────────────────────────────────────────────────────


def render_preview_page():
    schema: FormSchema = st.session_state.form_schema
    if schema is None:
        st.warning("No form parsed yet. Go back to the Input page.")
        return

    st.markdown(f"""
    <div style="padding:.3rem 0;">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;
                    letter-spacing:.12em;font-weight:600;margin-bottom:3px;">Form Preview</div>
        <h1 style="margin:0!important;">{_esc(schema.form_title)}</h1>
    </div>
    """, unsafe_allow_html=True)
    if schema.form_description:
        st.markdown(f"*{schema.form_description}*")

    if getattr(schema, "parse_method", "structured") == "html_fallback":
        st.warning(
            "This form's structured data couldn't be read, so it was parsed "
            "from the page HTML instead. Question types, options, and scales "
            "may be incomplete — review the detected questions below carefully."
        )

    st.divider()

    required = sum(1 for q in schema.questions if q.is_required)
    types = set(q.question_type.value for q in schema.questions)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, schema.total_questions, "Questions"),
        (c2, required, "Required"),
        (c3, len(types), "Types"),
        (c4, len(schema.sections), "Sections"),
    ]:
        with col:
            st.markdown(f'<div class="stat-glow"><div class="sn">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Type chips
    st.markdown("##### Question Types Detected")
    chips = ""
    for qt_val in sorted(types):
        qt = QuestionType(qt_val)
        cnt = sum(1 for q in schema.questions if q.question_type == qt)
        chips += f'<span class="chip">{_type_icon(qt)} {qt_val.replace("_"," ").title()} ({cnt})</span>'
    st.markdown(f'<div style="margin-bottom:.8rem;">{chips}</div>', unsafe_allow_html=True)

    render_survey_review(schema)

    st.markdown("##### All Questions")
    for i, q in enumerate(schema.questions):
        with st.expander(
            f"Q{i+1}. {q.question_text}  |  {render_question_type_badge(q.question_type)}  {'*' if q.is_required else ''}",
            expanded=False,
        ):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Type:** {q.question_type.value.replace('_',' ').title()}")
                st.markdown(f"**Required:** {'Yes' if q.is_required else 'No'}")
                st.markdown(f"**ID:** `{q.question_id}`")
            with cols[1]:
                if q.options:
                    st.markdown("**Options:**")
                    for opt in q.options:
                        st.markdown(f"- {opt}")
                if q.scale_config:
                    sc = q.scale_config
                    st.markdown(f"**Scale:** {sc.min_value} ({sc.min_label or ''}) to {sc.max_value} ({sc.max_label or ''})")
                if q.grid_config:
                    gc = q.grid_config
                    st.markdown(f"**Rows:** {', '.join(gc.rows)}")
                    st.markdown(f"**Columns:** {', '.join(gc.columns)}")
                if q.has_other_option:
                    st.markdown("**Has 'Other' option:** Yes")

    st.divider()

    if check_api_key():
        try:
            settings = get_llm_settings()
            llm = LLMClient(settings)
            cost = llm.estimate_cost(st.session_state.num_responses, schema.total_questions)
            st.markdown("##### Estimated Cost")
            ec1, ec2, ec3 = st.columns(3)
            with ec1: st.metric("Input Tokens", f"{cost['estimated_input_tokens']:,}")
            with ec2: st.metric("Output Tokens", f"{cost['estimated_output_tokens']:,}")
            with ec3: st.metric("Cost", f"${cost['estimated_cost_usd']:.4f}")
            _price_note = (
                "priced for this model"
                if cost.get("priced_by_model")
                else "no per-model price on file — using a provider average"
            )
            st.caption(
                f"Provider: {cost['provider']} | Model: {cost['model']} | "
                f"Approximate estimate ({_price_note})."
            )
        except Exception:
            pass

    n1, n2, _ = st.columns([1, 1, 2])
    with n1:
        if st.button("Back to Input", width="stretch"):
            st.session_state.page = "input"
            st.rerun()
    with n2:
        if st.button("Start Generation", type="primary", width="stretch"):
            if not check_api_key():
                st.error("Please set your API key in the sidebar.")
                return
            st.session_state.page = "generate"
            st.rerun()


# ─── Page: Generation ─────────────────────────────────────────────────────────


def render_generation_page():
    schema: FormSchema = st.session_state.form_schema
    if schema is None:
        st.warning("No form parsed yet. Go back to the Input page.")
        return
    if not check_api_key():
        st.error("Please set your API key in the sidebar before generating.")
        return

    st.markdown(f"""
    <div style="padding:.3rem 0;">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;
                    letter-spacing:.12em;font-weight:600;margin-bottom:3px;">Generation</div>
        <h1 style="margin:0!important;">Generating Responses</h1>
        <div style="color:var(--text2);margin-top:5px;">
            {schema.form_title} &mdash; {st.session_state.num_responses} responses</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.session_state.dataset is not None and not st.session_state.generation_in_progress:
        st.success(f"Generation complete! {st.session_state.dataset.total_generated} responses generated.")
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("Regenerate", width="stretch"):
                st.session_state.dataset = None
                st.session_state.generated_count = 0
                st.session_state.failed_count = 0
                st.rerun()
        with c2:
            if st.button("View Results", type="primary", width="stretch"):
                st.session_state.page = "results"
                st.rerun()
        return

    if not st.session_state.generation_in_progress:
        st.markdown("""
        <div class="glass-card" style="text-align:center;max-width:480px;margin:2rem auto;">
            <div style="font-size:2.5rem;margin-bottom:.4rem;">&#9889;</div>
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:.4rem;color:var(--text1);">Ready to Generate</div>
            <div style="color:var(--text2);font-size:.88rem;margin-bottom:1.2rem;">
                AI will create unique personas and fill the form as each character.</div>
        </div>
        """, unsafe_allow_html=True)
        # U1: offer to resume an interrupted run with the same configuration.
        saved = 0
        if check_api_key():
            try:
                saved = peek_checkpoint(
                    schema, get_llm_settings(),
                    st.session_state.persona_constraints or None,
                )
            except Exception:
                saved = 0
        resume = False
        if 0 < saved < st.session_state.num_responses:
            st.info(
                f"Found {saved} saved responses from an interrupted run with this "
                f"exact configuration. You can resume where you left off."
            )
            resume = st.checkbox("Resume from saved progress", value=True, key="resume_checkbox")
        st.session_state.resume_generation = resume

        bc, _, _ = st.columns([2, 1, 1])
        with bc:
            btn_label = "Resume Generation" if resume else "Begin Generation"
            if st.button(btn_label, type="primary", width="stretch"):
                st.session_state.generation_in_progress = True
                st.rerun()
        return

    # ── Active Generation ──
    num_responses = st.session_state.num_responses
    settings = get_llm_settings()
    provider = settings.llm_provider
    model_name = getattr(settings, f"{provider}_model", "unknown")
    resume = bool(st.session_state.get("resume_generation", False))
    enable_traits = bool(st.session_state.get("enable_traits"))
    waves = int(st.session_state.get("waves", 1) or 1)
    seed = int(st.session_state.get("seed", 42))
    resume = resume and waves == 1  # multi-wave runs are always fresh

    # Deterministic quota + stimulus plans (empty when unconfigured), seeded.
    quota_plan = build_assignment_plan(
        st.session_state.get("quota_targets") or None, num_responses, seed=seed
    )
    stimulus_plan = build_stimulus_plan(
        st.session_state.get("stimuli") or None, num_responses, seed=seed
    )

    try:
        service = GenerationService(
            schema, settings,
            constraints=st.session_state.persona_constraints or None,
            resume=resume,
            quota_plan=quota_plan,
            stimulus_plan=stimulus_plan,
            waves=waves,
            enable_traits=enable_traits,
            traits_seed=seed,
        )
    except ValueError as e:
        st.error(str(e))
        st.session_state.generation_in_progress = False
        return

    progress_bar = st.progress(0, text="Initializing...")
    sc = st.columns(4)
    with sc[0]: gen_m = st.empty()
    with sc[1]: fail_m = st.empty()
    with sc[2]: time_m = st.empty()
    with sc[3]: rate_m = st.empty()

    # Stop button (uses st.empty so it disappears after generation)
    stop_col, _ = st.columns([1, 3])
    with stop_col:
        stop_holder = st.empty()
        if stop_holder.button("Stop Generation", type="secondary", width="stretch", key="stop_gen_btn"):
            st.session_state.stop_generation = True

    st.divider()

    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        st.markdown("##### Latest Persona")
        persona_preview = st.empty()
    with pc2:
        st.markdown("##### Latest Response")
        response_preview = st.empty()

    q_map = {q.question_id: q.question_text for q in schema.questions}

    def _persona_html(persona: Persona) -> str:
        itags = "".join(f'<span class="persona-tag">{_esc(str(x))}</span>' for x in persona.interests)
        ttags = "".join(f'<span class="persona-tag">{_esc(str(x))}</span>' for x in persona.personality_traits)
        return (
            f'<div class="persona-card-v2">'
            f'<div class="persona-name">{_esc(persona.name)}</div>'
            f'<div class="persona-detail">'
            f'{_esc(str(persona.age))} &bull; {_esc(persona.gender)} &bull; {_esc(persona.year)}<br>'
            f'{_esc(persona.major)} @ {_esc(persona.university)}<br><br>'
            f'<strong>Interests:</strong><br>{itags}<br><br>'
            f'<strong>Traits:</strong><br>{ttags}<br><br>'
            f'<strong>Engagement:</strong> {_esc(persona.engagement_level)}<br>'
            f'<strong>Attitude:</strong> {_esc(persona.attitude_toward_topic)}<br><br>'
            f'<em style="color:var(--text3)">{_esc(persona.background_context)}</em>'
            f'</div></div>'
        )

    def _response_html(response: GeneratedResponse) -> str:
        items = list(response.answers.items())[:5]
        html = '<div class="response-card">'
        for qid, ans in items:
            qt = q_map.get(qid, qid)
            if isinstance(ans, list): ans_s = ", ".join(str(a) for a in ans)
            elif isinstance(ans, dict): ans_s = json.dumps(ans, indent=1)
            else: ans_s = str(ans)
            html += (
                f'<div style="margin-bottom:.65rem;">'
                f'<div style="font-size:.75rem;color:var(--text3);font-weight:600;'
                f'text-transform:uppercase;letter-spacing:.04em;">{_esc(str(qt))}</div>'
                f'<div style="color:var(--text1);margin-top:2px;">{_esc(ans_s)}</div></div>'
            )
        if len(response.answers) > 5:
            html += f'<div style="color:var(--text3);font-size:.82rem;font-style:italic;">...and {len(response.answers)-5} more</div>'
        html += '</div>'
        return html

    st.session_state.stop_generation = False  # reset flag
    initial_completed = service.completed_count()
    if resume and initial_completed > 0:
        st.info(
            f"Resuming: {initial_completed} responses already saved — "
            f"generating the remaining {max(0, num_responses - initial_completed)}."
        )

    start_time = time.time()

    def _should_stop() -> bool:
        return st.session_state.stop_generation

    for prog in service.run(num_responses, should_stop=_should_stop):
        if prog.persona is not None:
            persona_preview.markdown(_persona_html(prog.persona), unsafe_allow_html=True)
        if prog.response is not None and prog.response.generation_success:
            response_preview.markdown(_response_html(prog.response), unsafe_allow_html=True)

        elapsed = time.time() - start_time
        total = prog.total or num_responses
        newly = prog.completed - initial_completed
        rate = newly / elapsed if elapsed > 0 else 0
        remaining = (total - prog.completed) / rate if rate > 0 else 0
        pct = prog.completed / total if total else 1.0
        progress_bar.progress(min(pct, 1.0), text=f"Response {prog.completed} / {total}  ({pct*100:.0f}%)")
        gen_m.metric("Generated", prog.generated)
        fail_m.metric("Failed", prog.failed)
        time_m.metric("Elapsed", f"{elapsed:.0f}s")
        rate_m.metric("ETA", f"{remaining:.0f}s" if rate > 0 else "...")

    # Clean up stop button
    stop_holder.empty()

    dataset = service.dataset
    report_log = service.report_log
    generated = sum(1 for r in dataset.responses if r.generation_success)
    failed = len(dataset.responses) - generated
    elapsed = time.time() - start_time

    if service.stopped_reason == "user":
        st.warning(f"Generation stopped by user. {generated} responses completed and saved.")
    elif service.stopped_reason == "failures":
        st.error(
            "Stopped early after repeated failures with no success. "
            "Check the Generation Report below for details."
        )

    progress_bar.progress(1.0, text="Generation complete!")
    gen_m.metric("Generated", generated)
    fail_m.metric("Failed", failed)
    time_m.metric("Total Time", f"{elapsed:.0f}s")
    rate_m.metric("Rate", f"{generated/elapsed:.1f}/s" if elapsed > 0 else "N/A")

    st.session_state.dataset = dataset
    st.session_state.generation_in_progress = False
    st.session_state.generated_count = generated
    st.session_state.failed_count = failed
    st.session_state.generation_report = report_log
    if generated > 0:
        st.session_state.results_df = CSVExporter.to_dataframe(dataset, schema, st.session_state.existing_data)

    if generated > 0:
        st.success(f"Done! Generated {generated} responses ({failed} failed).")
    else:
        st.error(f"All {failed} responses failed. See the Generation Report below.")

    # ── Generation Report ──
    _render_generation_report(report_log, provider, model_name, elapsed, generated, failed)

    if generated > 0:
        if st.button("View Results", type="primary", width="stretch"):
            st.session_state.page = "results"
            st.rerun()


# ─── Generation Report ────────────────────────────────────────────────────────


def _render_generation_report(
    report_log: list[dict],
    provider: str,
    model: str,
    elapsed: float,
    generated: int,
    failed: int,
):
    """Render a detailed generation report with per-response diagnostics."""
    total = generated + failed
    success_rate = (generated / total * 100) if total > 0 else 0

    st.divider()
    st.markdown("##### Generation Report")

    # Summary card
    if failed == 0:
        verdict_color = "#22c55e"
        verdict = "All Successful"
        verdict_icon = "&#10003;"
    elif generated == 0:
        verdict_color = "#ef4444"
        verdict = "All Failed"
        verdict_icon = "&#10007;"
    else:
        verdict_color = "#F59E0B"
        verdict = "Partial Success"
        verdict_icon = "&#9888;"

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:1rem;">
        <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;">
            <span style="font-size:1.6rem;color:{verdict_color};">{verdict_icon}</span>
            <div>
                <div style="font-weight:700;font-size:1.1rem;color:{verdict_color};">{verdict}</div>
                <div style="font-size:.82rem;color:var(--text2);">
                    {generated}/{total} responses generated &bull; {success_rate:.1f}% success rate &bull; {elapsed:.1f}s elapsed
                </div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem .8rem;font-size:.84rem;color:var(--text2);">
            <div><strong>Provider:</strong> {_esc(str(provider))}</div>
            <div><strong>Model:</strong> {_esc(str(model))}</div>
            <div><strong>Temperature:</strong> {st.session_state.temperature}</div>
            <div><strong>API Delay:</strong> {st.session_state.api_call_delay}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Error breakdown (if any failures)
    failures = [e for e in report_log if e["status"] == "failed"]
    if failures:
        # Group by error type
        error_types = {}
        for f in failures:
            et = f["error_type"] or "Unknown"
            if et not in error_types:
                error_types[et] = {"count": 0, "suggestion": f["suggestion"], "sample_msg": f["error_message"]}
            error_types[et]["count"] += 1

        st.markdown("###### Error Breakdown")
        for et, info in error_types.items():
            with st.expander(f"{et}  —  {info['count']} occurrence{'s' if info['count'] > 1 else ''}", expanded=(len(error_types) == 1)):
                st.markdown("**Sample error:**")
                st.code(info["sample_msg"][:500] if info["sample_msg"] else "No message", language=None)
                st.markdown("**How to fix:**")
                st.info(info["suggestion"] or "No suggestion available")

    # Per-response log (collapsible)
    with st.expander(f"Detailed Log ({len(report_log)} entries)", expanded=False):
        log_df = pd.DataFrame(report_log)
        # Color-code status
        def _style_status(val):
            if val == "success":
                return "color: #22c55e"
            return "color: #ef4444"

        display_cols = ["response_num", "status", "persona_name", "error_type"]
        available_cols = [c for c in display_cols if c in log_df.columns]
        styled = log_df[available_cols].style.map(_style_status, subset=["status"])
        st.dataframe(styled, width="stretch", hide_index=True)

    # Downloadable report
    report_json = json.dumps({
        "summary": {
            "provider": provider,
            "model": model,
            "total_attempted": total,
            "generated": generated,
            "failed": failed,
            "success_rate_pct": round(success_rate, 2),
            "elapsed_seconds": round(elapsed, 2),
            "temperature": st.session_state.temperature,
            "api_call_delay": st.session_state.api_call_delay,
            "verdict": verdict,
        },
        "error_breakdown": {
            et: {"count": info["count"], "suggestion": info["suggestion"]}
            for et, info in error_types.items()
        } if failures else {},
        "log": report_log,
    }, indent=2)

    st.download_button(
        "Download Report (JSON)",
        data=report_json,
        file_name=f"synthsurvey_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        width="stretch",
    )


# ─── Page: Results ────────────────────────────────────────────────────────────


def render_results_page():
    schema: FormSchema = st.session_state.form_schema
    dataset: SurveyDataset = st.session_state.dataset
    df: pd.DataFrame = st.session_state.results_df

    if dataset is None or df is None:
        st.warning("No results yet. Generate responses first.")
        return

    synthetic_count = int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else len(df)
    real_count = len(df) - synthetic_count

    st.markdown(f"""
    <div style="padding:.3rem 0;">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;
                    letter-spacing:.12em;font-weight:600;margin-bottom:3px;">Results</div>
        <h1 style="margin:0!important;">{_esc(schema.form_title)}</h1>
        <div style="color:var(--text2);margin-top:5px;">{len(df)} total responses</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="disclaimer-v2" style="margin:.6rem 0;">'
        f'<strong>Synthetic dataset (SynthSurvey v{APP_VERSION}).</strong> '
        f'These are AI-generated responses, not real respondents — see the '
        f'Export tab for the provenance manifest.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df), "Total"), (c2, synthetic_count, "Synthetic"),
        (c3, real_count, "Real"), (c4, schema.total_questions, "Questions"),
    ]:
        with col:
            st.markdown(f'<div class="stat-glow"><div class="sn">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()

    has_reference = real_count > 0
    has_ab = "stimulus_variant" in df.columns and df["stimulus_variant"].dropna().nunique() > 1
    has_waves = "wave" in df.columns and df["wave"].dropna().nunique() > 1
    tab_labels = ["Data Table", "Charts", "Diversity", "Quality", "Insights"]
    if has_ab:
        tab_labels.append("A/B Test")
    if has_waves:
        tab_labels.append("Waves")
    if has_reference:
        tab_labels.append("Fidelity")
    tab_labels += ["Personas", "Export"]
    tabs = dict(zip(tab_labels, st.tabs(tab_labels)))

    with tabs["Data Table"]:
        # Quality score
        if "is_synthetic" in df.columns:
            synth_df = df[df["is_synthetic"].astype(bool)]
            filled = synth_df.notna().sum().sum()
            total_cells = synth_df.shape[0] * synth_df.shape[1]
            fill_rate = (filled / total_cells * 100) if total_cells > 0 else 0
            if fill_rate >= 90:
                badge_cls, badge_label = "quality-high", "Excellent"
            elif fill_rate >= 70:
                badge_cls, badge_label = "quality-med", "Good"
            else:
                badge_cls, badge_label = "quality-low", "Needs Review"
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<span class="quality-badge {badge_cls}">'
                f'{badge_label} &mdash; {fill_rate:.0f}% fill rate</span></div>',
                unsafe_allow_html=True,
            )
        st.dataframe(df, width="stretch", height=500)

    with tabs["Charts"]:
        render_charts(schema, df)

    with tabs["Diversity"]:
        render_diversity_metrics(schema, df, dataset)

    with tabs["Quality"]:
        render_quality_checks(schema, df)

    with tabs["Insights"]:
        render_insights(schema, df)

    if has_ab:
        with tabs["A/B Test"]:
            render_ab_comparison(schema, df)

    if has_waves:
        with tabs["Waves"]:
            render_waves(schema, df)

    if has_reference:
        with tabs["Fidelity"]:
            render_fidelity(schema, df)

    with tabs["Personas"]:
        render_persona_gallery(dataset)

    with tabs["Export"]:
        render_export(schema, dataset, df)

    _render_footer()


def _synthetic_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the synthetic rows (or the whole frame if unmarked)."""
    if "is_synthetic" in df.columns:
        return df[df["is_synthetic"].astype(bool)]
    return df


def render_quality_checks(schema: FormSchema, df: pd.DataFrame):
    """Render bias / straightlining / duplicate quality checks (F1.3)."""
    st.markdown("##### Quality &amp; Bias Checks")
    st.caption("Run on synthetic responses only — the tells of low-quality survey data.")
    synth = _synthetic_only(df)
    if synth.empty:
        st.info("No synthetic responses to analyze.")
        return

    result = analyze_quality(synth, schema)
    warn = result["warn_count"]
    color = "#22c55e" if warn == 0 else "#F59E0B"
    st.markdown(
        f'<div style="margin:.2rem 0 1rem;font-weight:700;color:{color};">'
        f'{_esc(result["verdict"])}</div>',
        unsafe_allow_html=True,
    )

    icons = {"pass": ("&#10003;", "#22c55e"), "warn": ("&#9888;", "#F59E0B"),
             "info": ("&#8505;", "#60A5FA")}
    for f in result["findings"]:
        icon, c = icons.get(f["severity"], ("&#8226;", "#9aa"))
        suggestion = (
            f'<div style="font-size:.82rem;color:var(--text3);margin-top:4px;">'
            f'Fix: {_esc(f["suggestion"])}</div>' if f.get("suggestion") else ""
        )
        st.markdown(
            f'<div class="glass-card" style="margin-bottom:.6rem;">'
            f'<div style="display:flex;align-items:center;gap:.6rem;">'
            f'<span style="color:{c};font-size:1.1rem;">{icon}</span>'
            f'<strong>{_esc(f["check"])}</strong>'
            f'<span style="margin-left:auto;font-family:monospace;color:{c};">{_esc(str(f["metric"]))}</span>'
            f'</div>'
            f'<div style="font-size:.88rem;color:var(--text2);margin-top:4px;">{_esc(f["detail"])}</div>'
            f'{suggestion}</div>',
            unsafe_allow_html=True,
        )


def render_fidelity(schema: FormSchema, df: pd.DataFrame):
    """Compare synthetic vs. uploaded real data distributions (F1.2)."""
    st.markdown("##### Fidelity vs. Real Data")
    if "is_synthetic" not in df.columns:
        st.info("Upload real responses on the Input page to compare distributions.")
        return
    synth = df[df["is_synthetic"].astype(bool)]
    ref = df[~df["is_synthetic"].astype(bool)]
    if synth.empty or ref.empty:
        st.info("Need both synthetic and real responses to score fidelity.")
        return

    result = compute_fidelity(synth, ref, schema)
    score = result["overall_score"]
    verdict = fidelity_verdict(score)

    if score is None:
        for note in result["notes"]:
            st.warning(note)
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall Fidelity", f"{score:.0f}/100")
        st.caption(verdict)
    with c2:
        st.caption(
            f"Scored {result['scored']} question(s) by comparing answer "
            "distributions (100 = identical to real data). Lower-scoring "
            "questions are where synthetic data diverges most."
        )

    pq = pd.DataFrame(result["per_question"])
    if not pq.empty:
        fig = px.bar(
            pq, x="similarity", y="question", orientation="h",
            range_x=[0, 100], title="Distribution match by question (higher = closer)",
            color="similarity", color_continuous_scale=["#EF4444", "#F59E0B", "#14B8A6"],
            template=_chart_template(),
        )
        fig.update_layout(**_chart_layout(
            height=max(300, len(pq) * 38), yaxis_title="", xaxis_title="Similarity",
        ))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Per-question detail"):
            st.dataframe(
                pq.rename(columns={
                    "question": "Question", "type": "Type",
                    "similarity": "Similarity %", "tvd": "Distance",
                    "n_ref": "Real n", "n_syn": "Synthetic n",
                }),
                width="stretch", hide_index=True,
            )
    for note in result["notes"]:
        st.caption(note)


def _chart_template() -> str:
    return "plotly_dark" if _dark_mode() else "plotly_white"


def _chart_layout(**extra) -> dict:
    """Transparent, theme-aware Plotly layout (font readable in both themes)."""
    color = "#E8E6F0" if _dark_mode() else "#1c1917"
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=color),
        title_font=dict(color=color),
    )
    base.update(extra)
    return base


_XTAB_TYPES = (
    QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN,
    QuestionType.LINEAR_SCALE, QuestionType.CHECKBOX,
)


def render_insights(schema: FormSchema, df: pd.DataFrame):
    """Cross-tabulate two variables with a chi-square significance test (F3.1)."""
    st.markdown("##### Cross-Tab &amp; Significance")
    st.caption("Pick two categorical questions to test whether responses are related.")
    synth = _synthetic_only(df)
    cat_qs = [q.question_text for q in schema.questions
              if q.question_type in _XTAB_TYPES and q.question_text in synth.columns]
    row_options = list(cat_qs)
    if "stimulus_variant" in synth.columns and synth["stimulus_variant"].dropna().nunique() > 1:
        row_options = ["stimulus_variant", *row_options]

    if not cat_qs or len(row_options) < 1:
        st.info("Need at least one categorical question to cross-tabulate.")
        return

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Row variable", row_options, key="xtab_a")
    with c2:
        b_options = [x for x in cat_qs if x != a] or cat_qs
        b = st.selectbox("Column variable", b_options, key="xtab_b")
    if a == b:
        st.info("Pick two different variables.")
        return

    res = crosstab_test(synth, a, b)
    if res["table"].empty:
        st.info("Not enough data to cross-tabulate these two.")
        return

    st.dataframe(res["table"], width="stretch")

    p, v = res["p_value"], res["cramers_v"]
    if p is None:
        st.caption("Table too small for a significance test.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("p-value", f"{p:.4f}")
        with m2:
            st.metric("Cramér's V", f"{v:.2f}")
        with m3:
            st.metric("χ² (dof)", f"{res['chi2']:.1f} ({res['dof']})")
        sig = "**significant**" if p < 0.05 else "not significant"
        st.caption(
            f"The association is {sig} at p < 0.05, with a {effect_size_label(v)} "
            f"effect size (n = {res['n']}). Note: this is synthetic data, so "
            "treat significance as illustrative of the generated patterns."
        )

    long = res["table"].reset_index().melt(id_vars=res["table"].index.name, var_name=b, value_name="count")
    fig = px.bar(long, x=res["table"].index.name, y="count", color=b, barmode="group",
                 title=f"{a[:40]} × {b[:40]}", template=_chart_template())
    fig.update_layout(**_chart_layout())
    st.plotly_chart(fig, width="stretch")


def render_ab_comparison(schema: FormSchema, df: pd.DataFrame):
    """Compare answers across A/B stimulus variants (F2.2)."""
    st.markdown("##### A/B Stimulus Comparison")
    d = df[df["stimulus_variant"].notna()] if "stimulus_variant" in df.columns else df.iloc[0:0]
    if d.empty or d["stimulus_variant"].nunique() < 2:
        st.info("Need at least two stimulus variants to compare.")
        return

    counts = d["stimulus_variant"].value_counts()
    st.caption("Respondents per variant — " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    scale_cols = [q.question_text for q in schema.questions
                  if q.question_type == QuestionType.LINEAR_SCALE and q.question_text in d.columns]
    if scale_cols:
        rows = []
        for c in scale_cols:
            means = d.groupby("stimulus_variant")[c].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            for var, m in means.items():
                if pd.notna(m):
                    rows.append({"Question": c[:40], "Variant": var, "Mean rating": round(float(m), 2)})
        if rows:
            fig = px.bar(pd.DataFrame(rows), x="Question", y="Mean rating", color="Variant",
                         barmode="group", title="Mean scale rating by variant", template=_chart_template())
            fig.update_layout(**_chart_layout())
            st.plotly_chart(fig, width="stretch")

    choice_qs = [q for q in schema.questions
                 if q.question_type in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN)
                 and q.question_text in d.columns]
    for q in choice_qs[:5]:
        c = q.question_text
        ct = (d.groupby("stimulus_variant")[c].value_counts(normalize=True)
              .mul(100).round(1).rename("percent").reset_index())
        fig = px.bar(ct, x=c, y="percent", color="stimulus_variant", barmode="group",
                     title=f"{c[:50]} — share by variant", template=_chart_template())
        fig.update_layout(**_chart_layout(), yaxis_title="%", legend_title="Variant")
        st.plotly_chart(fig, width="stretch")


def render_waves(schema: FormSchema, df: pd.DataFrame):
    """Show how answers drift across longitudinal waves (F2.3)."""
    st.markdown("##### Longitudinal Waves")
    d = df[df["wave"].notna()] if "wave" in df.columns else df.iloc[0:0]
    if d.empty or d["wave"].nunique() < 2:
        st.info("Need at least two waves to show drift.")
        return

    counts = d["wave"].value_counts().sort_index()
    st.caption("Responses per wave — " + ", ".join(f"wave {int(k)}: {v}" for k, v in counts.items()))

    scale_cols = [q.question_text for q in schema.questions
                  if q.question_type == QuestionType.LINEAR_SCALE and q.question_text in d.columns]
    if not scale_cols:
        st.info("Add linear-scale questions to visualize drift across waves.")
        return

    rows = []
    for c in scale_cols:
        means = d.groupby("wave")[c].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        for w, m in means.items():
            if pd.notna(m):
                rows.append({"Wave": int(w), "Question": c[:40], "Mean rating": round(float(m), 2)})
    if rows:
        fig = px.line(pd.DataFrame(rows), x="Wave", y="Mean rating", color="Question",
                      markers=True, title="Mean scale rating drift across waves", template=_chart_template())
        fig.update_layout(**_chart_layout())
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")


def render_diversity_metrics(schema: FormSchema, df: pd.DataFrame, dataset: SurveyDataset):
    """Render response diversity analysis."""
    st.markdown("##### Response Diversity Analysis")
    st.markdown(
        '<div style="font-size:.85rem;color:var(--text2);margin-bottom:1rem;">'
        'How varied and realistic are the generated responses?</div>',
        unsafe_allow_html=True,
    )

    colors = ["#F59E0B", "#FB923C", "#14B8A6", "#F472B6", "#FBBF24",
              "#34D399", "#60A5FA", "#A78BFA", "#EF4444", "#C084FC"]
    layout_common = _chart_layout()

    # --- Uniqueness score ---
    question_cols = [q.question_text for q in schema.questions if q.question_text in df.columns]
    if question_cols:
        unique_ratios = []
        for col in question_cols:
            series = df[col].dropna()
            if len(series) > 0:
                unique_ratios.append(series.nunique() / len(series))
        avg_uniqueness = (sum(unique_ratios) / len(unique_ratios) * 100) if unique_ratios else 0

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                f'<div class="stat-glow"><div class="sn">{avg_uniqueness:.0f}%</div>'
                f'<div class="sl">Avg Uniqueness</div></div>',
                unsafe_allow_html=True,
            )
        with mc2:
            total_unique = sum(df[col].nunique() for col in question_cols)
            st.markdown(
                f'<div class="stat-glow"><div class="sn">{total_unique}</div>'
                f'<div class="sl">Unique Values</div></div>',
                unsafe_allow_html=True,
            )
        with mc3:
            # Duplicate rows check (answer columns only)
            dup_count = df[question_cols].duplicated().sum()
            st.markdown(
                f'<div class="stat-glow"><div class="sn">{dup_count}</div>'
                f'<div class="sl">Duplicate Rows</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

    # --- Per-question diversity breakdown ---
    st.markdown("###### Per-Question Diversity")
    diversity_data = []
    for q in schema.questions:
        if q.question_text not in df.columns:
            continue
        series = df[q.question_text].dropna()
        if len(series) == 0:
            continue
        n_unique = series.nunique()
        ratio = n_unique / len(series) * 100 if len(series) > 0 else 0
        diversity_data.append({
            "Question": q.question_text[:50] + ("..." if len(q.question_text) > 50 else ""),
            "Type": q.question_type.value.replace("_", " ").title(),
            "Unique Values": n_unique,
            "Total": len(series),
            "Diversity %": round(ratio, 1),
        })

    if diversity_data:
        div_df = pd.DataFrame(diversity_data)
        # Bar chart of diversity per question
        fig = px.bar(
            div_df, x="Diversity %", y="Question", orientation="h",
            title="Response Diversity by Question",
            color="Diversity %",
            color_continuous_scale=["#EF4444", "#F59E0B", "#14B8A6"],
            template=_chart_template(),
        )
        fig.update_layout(**layout_common, height=max(300, len(diversity_data) * 35))
        fig.update_traces(marker_cornerradius=6)
        st.plotly_chart(fig, width="stretch")

        with st.expander("Detailed Diversity Table"):
            st.dataframe(div_df, width="stretch", hide_index=True)

    # --- Persona trait distributions ---
    if dataset.responses:
        st.divider()
        st.markdown("###### Persona Distributions")

        # Extract persona data from summaries
        engagements = []
        for resp in dataset.responses:
            s = resp.persona_summary.lower()
            if "high" in s:
                engagements.append("High")
            elif "low" in s:
                engagements.append("Low")
            else:
                engagements.append("Medium")

        if engagements:
            pc1, pc2 = st.columns(2)
            with pc1:
                eng_counts = pd.Series(engagements).value_counts().reset_index()
                eng_counts.columns = ["Engagement", "Count"]
                fig = px.pie(eng_counts, values="Count", names="Engagement",
                             title="Engagement Level Distribution",
                             color_discrete_sequence=colors, template=_chart_template())
                fig.update_layout(**layout_common)
                fig.update_traces(textinfo="label+percent", textfont_size=12)
                st.plotly_chart(fig, width="stretch")

            with pc2:
                # Response length distribution for text questions
                text_cols = [
                    q.question_text for q in schema.questions
                    if q.question_type in (QuestionType.SHORT_TEXT, QuestionType.PARAGRAPH)
                    and q.question_text in df.columns
                ]
                if text_cols:
                    lengths = df[text_cols[0]].dropna().astype(str).str.len()
                    fig = px.histogram(lengths, nbins=20,
                                       title=f"Response Length: {text_cols[0][:30]}...",
                                       labels={"value": "Characters", "count": "Frequency"},
                                       template=_chart_template(),
                                       color_discrete_sequence=[colors[2]])
                    fig.update_layout(**layout_common)
                    fig.update_traces(marker_cornerradius=6)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No text questions to analyze response length.")


def render_persona_gallery(dataset: SurveyDataset):
    """Render a gallery of generated personas from the dataset."""
    if not dataset.responses:
        st.info("No personas to display.")
        return

    total = len(dataset.responses)
    success_count = sum(1 for r in dataset.responses if r.generation_success)

    st.markdown("##### Generated Personas")

    # Pagination
    per_page = 12
    total_pages = max(1, (total + per_page - 1) // per_page)
    page_key = "persona_gallery_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    st.markdown(
        f'<div style="font-size:.85rem;color:var(--text2);margin-bottom:.6rem;">'
        f'{success_count} successful / {total} total &bull; '
        f'Page {st.session_state[page_key]+1} of {total_pages}</div>',
        unsafe_allow_html=True,
    )

    # Page controls
    if total_pages > 1:
        pg_cols = st.columns([1, 1, 4])
        with pg_cols[0]:
            if st.button("Previous", disabled=st.session_state[page_key] <= 0, key="pg_prev", width="stretch"):
                st.session_state[page_key] -= 1
                st.rerun()
        with pg_cols[1]:
            if st.button("Next", disabled=st.session_state[page_key] >= total_pages - 1, key="pg_next", width="stretch"):
                st.session_state[page_key] += 1
                st.rerun()

    start = st.session_state[page_key] * per_page
    page_responses = dataset.responses[start:start + per_page]

    # 3-column grid for richer cards
    for i in range(0, len(page_responses), 3):
        cols = st.columns(3, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(page_responses):
                break
            resp = page_responses[idx]
            global_idx = start + idx + 1
            with col:
                success_icon = "&#9679;" if resp.generation_success else "&#9675;"
                success_color = "#14B8A6" if resp.generation_success else "#EF4444"
                # Parse key details from summary
                parts = resp.persona_summary.split(", ") if resp.persona_summary else []
                name = parts[0] if parts else f"Persona #{global_idx}"
                detail_line = ", ".join(parts[1:4]) if len(parts) > 1 else ""
                rest = ". ".join(resp.persona_summary.split(". ")[1:]) if ". " in resp.persona_summary else ""

                st.markdown(
                    f'<div class="persona-card-v2" style="margin-bottom:.4rem;">'
                    f'<div class="persona-name">'
                    f'<span style="color:{success_color};font-size:.6rem;margin-right:6px;">{success_icon}</span>'
                    f'#{global_idx} &mdash; {_esc(name)}</div>'
                    f'<div class="persona-detail">'
                    f'{_esc(detail_line)}<br>'
                    f'<span style="font-size:.78rem;color:var(--text3);line-height:1.5;">{_esc(rest[:150])}{"..." if len(rest)>150 else ""}</span><br>'
                    f'<span style="font-size:.7rem;color:var(--text3);opacity:.6;">Retries: {resp.retry_count}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                _render_why(resp)


def _render_why(resp) -> None:
    """A 'why this answer' trace: the persona drivers behind the responses (F4.3)."""
    ctx = getattr(resp, "persona_context", None) or {}
    traits = getattr(resp, "latent_traits", None) or {}
    if not ctx and not traits:
        return
    with st.expander("Why these answers?"):
        if ctx.get("engagement_level"):
            st.markdown(f"**Engagement:** {_esc(str(ctx['engagement_level']))} "
                        "— sets answer depth and effort.")
        if ctx.get("attitude_toward_topic"):
            st.markdown(f"**Attitude:** {_esc(str(ctx['attitude_toward_topic']))}")
        if ctx.get("background_context"):
            st.markdown(f"**Background:** {_esc(str(ctx['background_context']))}")
        if traits:
            hi = ", ".join(f"{k} {v:.2f}" for k, v in sorted(traits.items(), key=lambda x: -x[1])[:3])
            st.markdown(f"**Top latent traits:** {_esc(hi)} — these drive correlated answers.")
        if resp.stimulus_variant:
            st.markdown(f"**Saw variant:** {_esc(str(resp.stimulus_variant))}")
        if getattr(resp, "wave", 1) != 1:
            st.markdown(f"**Wave:** {resp.wave} (answers reflect drift over time)")


def render_charts(schema: FormSchema, df: pd.DataFrame):
    colors = ["#F59E0B", "#FB923C", "#14B8A6", "#F472B6", "#FBBF24",
              "#34D399", "#60A5FA", "#A78BFA", "#EF4444", "#C084FC"]

    question_options = [
        f"{q.question_text} ({q.question_type.value})"
        for q in schema.questions
        if q.question_type in (
            QuestionType.MULTIPLE_CHOICE, QuestionType.CHECKBOX,
            QuestionType.DROPDOWN, QuestionType.LINEAR_SCALE,
        )
    ]
    if not question_options:
        st.info("No chartable questions found (multiple choice, checkbox, dropdown, or scale).")
        return

    selected = st.selectbox("Select a question to visualize", question_options)
    selected_q = None
    for q in schema.questions:
        if f"{q.question_text} ({q.question_type.value})" == selected:
            selected_q = q
            break
    if selected_q is None:
        return

    col_name = selected_q.question_text
    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found in the data.")
        return

    qt = selected_q.question_type
    layout_common = _chart_layout()

    if qt in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN):
        counts = df[col_name].value_counts().reset_index()
        counts.columns = ["Option", "Count"]
        fig = px.bar(counts, x="Option", y="Count", title=f"Distribution: {col_name}",
                     color="Option", color_discrete_sequence=colors, template=_chart_template())
        fig.update_layout(**layout_common, showlegend=False)
        fig.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig, width="stretch")

        if "is_synthetic" in df.columns and df["is_synthetic"].nunique() > 1:
            st.markdown("##### Real vs Synthetic")
            comp = df.groupby(["is_synthetic", col_name]).size().reset_index(name="Count")
            comp["Source"] = comp["is_synthetic"].map({True: "Synthetic", False: "Real"})
            fig2 = px.bar(comp, x=col_name, y="Count", color="Source", barmode="group",
                          title=f"Real vs Synthetic: {col_name}",
                          color_discrete_map={"Synthetic": "#F59E0B", "Real": "#14B8A6"},
                          template=_chart_template())
            fig2.update_layout(**layout_common)
            fig2.update_traces(marker_cornerradius=8)
            st.plotly_chart(fig2, width="stretch")

    elif qt == QuestionType.LINEAR_SCALE:
        numeric_col = pd.to_numeric(df[col_name], errors="coerce")
        nbins = (selected_q.scale_config.max_value - selected_q.scale_config.min_value + 1
                 if selected_q.scale_config else 10)
        fig = px.histogram(numeric_col.dropna(), nbins=nbins,
                           title=f"Distribution: {col_name}",
                           labels={"value": col_name, "count": "Frequency"},
                           template=_chart_template(), color_discrete_sequence=[colors[0]])
        fig.update_layout(**layout_common)
        fig.update_traces(marker_cornerradius=6)
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"**Mean:** {numeric_col.mean():.2f} | "
            f"**Median:** {numeric_col.median():.1f} | "
            f"**Std Dev:** {numeric_col.std():.2f}"
        )

    elif qt == QuestionType.CHECKBOX:
        exploded = df[col_name].dropna().str.split("; ").explode()
        counts = exploded.value_counts().reset_index()
        counts.columns = ["Option", "Count"]
        fig = px.bar(counts, x="Option", y="Count",
                     title=f"Distribution: {col_name} (Checkboxes)",
                     color="Option", color_discrete_sequence=colors, template=_chart_template())
        fig.update_layout(**layout_common, showlegend=False)
        fig.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig, width="stretch")


def render_export(schema: FormSchema, dataset: SurveyDataset, df: pd.DataFrame):
    st.markdown("##### Download Your Data")
    st.markdown(
        f'<div class="disclaimer-v2" style="margin-bottom:1rem;">'
        f'<strong>Synthetic data.</strong> {_esc(DISCLAIMER)} A provenance '
        f'manifest is embedded in the JSON export and downloadable below.</div>',
        unsafe_allow_html=True,
    )

    # Provenance manifest (embedded in JSON, and downloadable on its own).
    try:
        manifest = build_manifest(
            dataset, schema, get_llm_settings(),
            study_config=build_study_config(st.session_state),
            seed=int(st.session_state.get("seed", 42)),
        )
    except Exception:  # noqa: BLE001
        manifest = None

    safe_title = schema.form_title[:30].replace(' ', '_').replace('/', '_')
    csv_bytes = CSVExporter.to_csv_bytes(df)
    json_bytes = JSONExporter.to_json_bytes(dataset, schema, provenance=manifest)

    # Excel bytes
    import io as _io
    xlsx_buf = _io.BytesIO()
    df.to_excel(xlsx_buf, index=False, engine="openpyxl")
    xlsx_bytes = xlsx_buf.getvalue()

    # File size display
    def _fmt_size(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        if b < 1024 * 1024:
            return f"{b/1024:.1f} KB"
        return f"{b/(1024*1024):.1f} MB"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:1.5rem;">
            <div style="font-size:2rem;margin-bottom:.5rem;">&#128196;</div>
            <div style="font-weight:700;color:var(--text1);margin-bottom:.2rem;">CSV</div>
            <div style="font-size:.75rem;color:var(--text3);margin-bottom:1rem;">
                Spreadsheet-ready &bull; {_fmt_size(len(csv_bytes))}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="Download CSV", data=csv_bytes,
            file_name=f"synthsurvey_{safe_title}.csv",
            mime="text/csv", width="stretch",
        )

    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:1.5rem;">
            <div style="font-size:2rem;margin-bottom:.5rem;">&#128218;</div>
            <div style="font-weight:700;color:var(--text1);margin-bottom:.2rem;">JSON</div>
            <div style="font-size:.75rem;color:var(--text3);margin-bottom:1rem;">
                Structured with metadata &bull; {_fmt_size(len(json_bytes))}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="Download JSON", data=json_bytes,
            file_name=f"synthsurvey_{safe_title}.json",
            mime="application/json", width="stretch",
        )

    with c3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:1.5rem;">
            <div style="font-size:2rem;margin-bottom:.5rem;">&#128202;</div>
            <div style="font-weight:700;color:var(--text1);margin-bottom:.2rem;">Excel</div>
            <div style="font-size:.75rem;color:var(--text3);margin-bottom:1rem;">
                .xlsx workbook &bull; {_fmt_size(len(xlsx_bytes))}</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="Download Excel", data=xlsx_bytes,
            file_name=f"synthsurvey_{safe_title}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    with c4:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:1.5rem;">
            <div style="font-size:2rem;margin-bottom:.5rem;">&#128200;</div>
            <div style="font-weight:700;color:var(--text1);margin-bottom:.2rem;">Google Sheets</div>
            <div style="font-size:.75rem;color:var(--text3);margin-bottom:1rem;">
                Requires credentials</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Export to Sheets", disabled=True, width="stretch",
            help="Configure Google Sheets credentials in .env to enable.",
        )

    st.divider()

    if manifest is not None:
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.download_button(
                "Download provenance manifest (.json)",
                data=manifest_to_bytes(manifest),
                file_name=f"synthsurvey_{safe_title}_manifest.json",
                mime="application/json", width="stretch", key="manifest_download",
            )
        with mc2:
            st.caption(
                f"Auditable record: tool v{manifest['version']}, "
                f"model {manifest['generation']['provider']}/{manifest['generation']['model']}, "
                f"seed {manifest['generation']['seed']}, config hash "
                f"`{manifest['config_hash']}`."
            )

    with st.expander("Raw JSON Preview"):
        json_str = json_bytes.decode("utf-8")
        st.code(json_str[:5000] + ("..." if len(json_str) > 5000 else ""), language="json")


# ─── Router ───────────────────────────────────────────────────────────────────

page = st.session_state.page
if page == "input":
    render_input_page()
elif page == "preview":
    render_preview_page()
elif page == "generate":
    render_generation_page()
elif page == "results":
    render_results_page()
