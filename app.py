"""SynthSurvey — Streamlit web application for synthetic survey data generation."""

import sys
import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings, Settings
from models.form_schema import FormSchema, QuestionType
from models.persona import Persona
from models.response import GeneratedResponse, SurveyDataset
from parsers.google_form_parser import GoogleFormParser
from generators.persona_generator import PersonaGenerator
from generators.response_generator import ResponseGenerator
from exporters.csv_exporter import CSVExporter
from exporters.json_exporter import JSONExporter
from utils.llm_client import LLMClient, classify_error
from utils.validators import ResponseValidator

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SynthSurvey",
    page_icon="https://em-content.zobj.net/source/twitter/408/chart-increasing_1f4c8.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Custom Cursor (Google Forms icon) ───────────────────────────── */
* {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath d='M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z' fill='%23673AB7'/%3E%3Cpath d='M14 2v6h6' fill='%23B39DDB'/%3E%3Cpath d='M14 2l6 6' stroke='%235E35B1' stroke-width='.4' fill='none'/%3E%3Ccircle cx='8.5' cy='12.5' r='1.2' fill='%23fff'/%3E%3Crect x='11' y='11.8' width='5.5' height='1.4' rx='.7' fill='%23fff' opacity='.85'/%3E%3Ccircle cx='8.5' cy='16' r='1.2' fill='%23fff'/%3E%3Crect x='11' y='15.3' width='5.5' height='1.4' rx='.7' fill='%23fff' opacity='.85'/%3E%3C/svg%3E") 3 1, auto !important;
}
a, button, [role="button"], .stButton>button, input, textarea, select,
[data-testid="stFileUploader"], .stSelectbox, .stMultiSelect {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath d='M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z' fill='%237B1FA2'/%3E%3Cpath d='M14 2v6h6' fill='%23CE93D8'/%3E%3Cpath d='M14 2l6 6' stroke='%236A1B9A' stroke-width='.4' fill='none'/%3E%3Ccircle cx='8.5' cy='12.5' r='1.2' fill='%23fff'/%3E%3Crect x='11' y='11.8' width='5.5' height='1.4' rx='.7' fill='%23fff' opacity='.85'/%3E%3Ccircle cx='8.5' cy='16' r='1.2' fill='%23fff'/%3E%3Crect x='11' y='15.3' width='5.5' height='1.4' rx='.7' fill='%23fff' opacity='.85'/%3E%3Cpath d='M12 20h9' stroke='%23fff' stroke-width='1.6' stroke-linecap='round' opacity='.9'/%3E%3C/svg%3E") 3 1, pointer !important;
}

/* ── CSS Variables ─────────────────────────────────────────────────── */
:root {
    --bg-primary: #0c0a09;
    --bg-card: rgba(255,245,230,0.02);
    --bg-glass: rgba(255,245,230,0.035);
    --border-glass: rgba(255,200,120,0.08);
    --accent: #F59E0B;
    --accent2: #FB923C;
    --accent3: #14B8A6;
    --accent4: #F472B6;
    --text1: #FAF5EF;
    --text2: #B8A99A;
    --text3: #6B5E52;
    --radius: 14px;
    --radius-lg: 22px;
    --smooth: all 0.35s cubic-bezier(.25,.46,.45,.94);
}

/* ── Global ────────────────────────────────────────────────────────── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text1) !important;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse at 15% 50%, rgba(245,158,11,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 20%, rgba(251,146,60,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 85%, rgba(20,184,166,0.04) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
    animation: bgPulse 20s ease-in-out infinite alternate;
}
@keyframes bgPulse { 0%{opacity:1} 50%{opacity:.7} 100%{opacity:1} }

.main .block-container {
    animation: fadeIn .45s ease-out;
    padding-top: 2rem !important;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

/* ── Sidebar ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f0c09,#12100d) !important;
    border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text1) !important; }

/* ── Typography ────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6 {
    font-family:'Inter',sans-serif !important;
    font-weight:700 !important;
    color:var(--text1) !important;
    letter-spacing:-0.02em !important;
}
h1 { font-size:2.2rem !important; font-weight:800 !important; }
p,li,span,div { color:var(--text1) !important; }

/* ── Buttons ───────────────────────────────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg,rgba(245,158,11,.12),rgba(20,184,166,.08)) !important;
    color:var(--text1) !important;
    border:1px solid rgba(245,158,11,.18) !important;
    border-radius: var(--radius) !important;
    padding:.65rem 1.5rem !important;
    font-weight:600 !important; font-family:'Inter',sans-serif !important;
    font-size:.9rem !important;
    transition: var(--smooth) !important;
    box-shadow: 0 1px 8px rgba(0,0,0,.15) !important;
    letter-spacing:.01em !important;
    backdrop-filter:blur(8px) !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow: 0 4px 18px rgba(245,158,11,.15) !important;
    background: linear-gradient(135deg,rgba(245,158,11,.2),rgba(20,184,166,.12)) !important;
    border-color:rgba(245,158,11,.3) !important;
}
.stButton>button:active { transform:translateY(0) !important; }
.stButton>button[disabled] {
    background:rgba(255,255,255,.05) !important;
    color:var(--text3) !important; box-shadow:none !important;
}

/* ── Inputs ────────────────────────────────────────────────────────── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stNumberInput>div>div>input {
    background:var(--bg-glass) !important;
    border:1px solid var(--border-glass) !important;
    border-radius:var(--radius) !important;
    color:var(--text1) !important;
    font-family:'Inter',sans-serif !important;
    padding:.75rem 1rem !important;
    transition:var(--smooth) !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus,
.stNumberInput>div>div>input:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(245,158,11,.15) !important;
}
.stTextInput>div>div>input::placeholder,
.stTextArea>div>div>textarea::placeholder { color:var(--text3) !important; }
label,.stTextInput label,.stTextArea label,.stNumberInput label,
.stSelectbox label,.stFileUploader label {
    color:var(--text2) !important; font-weight:500 !important;
    font-size:.82rem !important; letter-spacing:.03em !important;
    text-transform:uppercase !important;
}

/* ── Selectbox ─────────────────────────────────────────────────────── */
.stSelectbox>div>div {
    background:var(--bg-glass) !important;
    border:1px solid var(--border-glass) !important;
    border-radius:var(--radius) !important;
}

/* ── Metrics ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background:var(--bg-glass) !important;
    border:1px solid var(--border-glass) !important;
    border-radius:var(--radius) !important;
    padding:1.1rem 1rem !important;
    transition:var(--smooth) !important;
    backdrop-filter:blur(10px) !important;
}
[data-testid="stMetric"]:hover {
    border-color:rgba(245,158,11,.2) !important;
    box-shadow:0 0 30px rgba(245,158,11,.1) !important;
    transform:translateY(-2px);
}
[data-testid="stMetricValue"] {
    font-family:'Inter',sans-serif !important; font-weight:800 !important;
    font-size:1.7rem !important;
    background:linear-gradient(135deg,var(--accent),var(--accent3)) !important;
    -webkit-background-clip:text !important;
    -webkit-text-fill-color:transparent !important;
    background-clip:text !important;
}
[data-testid="stMetricLabel"] {
    color:var(--text2) !important; font-weight:500 !important;
    text-transform:uppercase !important; font-size:.72rem !important;
    letter-spacing:.08em !important;
}

/* ── Expanders ─────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background:var(--bg-glass) !important;
    border:1px solid var(--border-glass) !important;
    border-radius:var(--radius) !important;
    color:var(--text1) !important; font-weight:500 !important;
    transition:var(--smooth) !important;
}
.streamlit-expanderHeader:hover {
    border-color:rgba(245,158,11,.2) !important;
    background:rgba(255,255,255,.05) !important;
}
.streamlit-expanderContent {
    background:rgba(255,255,255,.02) !important;
    border:1px solid var(--border-glass) !important;
    border-top:none !important;
    border-radius:0 0 var(--radius) var(--radius) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap:6px !important; background:transparent !important;
    border-bottom:1px solid var(--border-glass) !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:var(--text2) !important;
    border-radius:var(--radius) var(--radius) 0 0 !important;
    padding:.7rem 1.4rem !important; font-weight:500 !important;
    transition:var(--smooth) !important; border:none !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--text1) !important; background:var(--bg-glass) !important; }
.stTabs [aria-selected="true"] {
    background:var(--bg-glass) !important; color:var(--accent) !important;
    border-bottom:2px solid var(--accent) !important;
}

/* ── Progress bar ──────────────────────────────────────────────────── */
.stProgress>div>div>div>div {
    background:linear-gradient(90deg,var(--accent),var(--accent3),var(--accent2)) !important;
    background-size:200% 100% !important;
    animation:progGlow 2s ease-in-out infinite !important;
    border-radius:20px !important;
}
@keyframes progGlow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* ── Dividers ──────────────────────────────────────────────────────── */
hr {
    border:none !important; height:1px !important;
    background:linear-gradient(90deg,transparent,var(--border-glass),rgba(245,158,11,.2),var(--border-glass),transparent) !important;
    margin:1.8rem 0 !important;
}

/* ── File uploader ─────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:var(--bg-glass) !important;
    border:1px dashed var(--border-glass) !important;
    border-radius:var(--radius) !important;
    padding:1rem !important; transition:var(--smooth) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:var(--accent) !important;
}

/* ── Download buttons ──────────────────────────────────────────────── */
.stDownloadButton>button {
    background:linear-gradient(135deg,#1a1510,#1c1812) !important;
    border:1px solid var(--border-glass) !important;
    color:var(--text1) !important; border-radius:var(--radius) !important;
    transition:var(--smooth) !important;
}
.stDownloadButton>button:hover {
    border-color:var(--accent3) !important;
    box-shadow:0 4px 15px rgba(20,184,166,.2) !important;
    transform:translateY(-2px) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(245,158,11,.3); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:rgba(245,158,11,.5); }

/* ── Custom Components ─────────────────────────────────────────────── */
.hero-title {
    font-size:3rem; font-weight:900; letter-spacing:-0.03em; line-height:1.1;
    margin-bottom:.4rem;
    background:linear-gradient(135deg,#F59E0B 0%,#FB923C 40%,#14B8A6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; background-size:200% 200%;
    animation:gradText 6s ease-in-out infinite alternate;
}
@keyframes gradText { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }

.hero-sub { font-size:1.1rem; color:var(--text2); line-height:1.6; max-width:580px; }

.glass-card {
    background:var(--bg-glass); backdrop-filter:blur(16px);
    border:1px solid var(--border-glass); border-radius:var(--radius-lg);
    padding:1.8rem; transition:var(--smooth); position:relative; overflow:hidden;
}
.glass-card::before {
    content:''; position:absolute; top:0;left:0;right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
}
.glass-card:hover {
    border-color:rgba(245,158,11,.15);
    box-shadow:0 0 35px rgba(245,158,11,.08);
    transform:translateY(-2px);
}

.step-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:32px; height:32px; border-radius:50%; flex-shrink:0;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#fff; font-weight:700; font-size:.82rem; margin-right:10px;
    box-shadow:0 3px 10px rgba(245,158,11,.3);
}
.step-row { display:flex; align-items:center; padding:.6rem 0; color:var(--text2); font-size:.9rem; }

.persona-card-v2 {
    background:linear-gradient(135deg,rgba(245,158,11,.06),rgba(20,184,166,.04));
    border:1px solid rgba(245,158,11,.1); border-radius:var(--radius-lg);
    padding:1.4rem; margin-bottom:.8rem; position:relative; overflow:hidden;
    transition:var(--smooth);
}
.persona-card-v2::before {
    content:''; position:absolute; top:0;left:0; width:3px; height:100%;
    background:linear-gradient(180deg,var(--accent),var(--accent2));
    border-radius:3px 0 0 3px;
}
.persona-card-v2:hover { border-color:rgba(245,158,11,.2); box-shadow:0 6px 25px rgba(245,158,11,.08); }
.persona-name { font-size:1.1rem; font-weight:700; color:var(--text1); margin-bottom:.4rem; }
.persona-detail { font-size:.85rem; color:var(--text2); line-height:1.65; }
.persona-tag {
    display:inline-block; background:rgba(245,158,11,.1); color:var(--accent);
    padding:2px 9px; border-radius:16px; font-size:.75rem; font-weight:500; margin:2px;
}

.response-card {
    background:rgba(20,184,166,.04); border:1px solid rgba(20,184,166,.1);
    border-radius:var(--radius-lg); padding:1.4rem;
}

.chip {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.1);
    border-radius:16px; padding:3px 12px; font-size:.78rem; font-weight:500;
    color:var(--accent); margin:2px; transition:var(--smooth);
}
.chip:hover { background:rgba(245,158,11,.15); transform:scale(1.03); }

.stat-glow {
    text-align:center; padding:1.3rem; background:var(--bg-glass);
    border:1px solid var(--border-glass); border-radius:var(--radius-lg);
    transition:var(--smooth);
}
.stat-glow:hover { border-color:rgba(245,158,11,.2); box-shadow:0 0 30px rgba(245,158,11,.1); transform:translateY(-2px); }
.stat-glow .sn {
    font-size:2.2rem; font-weight:800;
    background:linear-gradient(135deg,var(--accent),var(--accent3));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.stat-glow .sl {
    font-size:.72rem; color:var(--text3); text-transform:uppercase;
    letter-spacing:.1em; font-weight:600; margin-top:3px;
}

.disclaimer-v2 {
    background:rgba(245,158,11,.05); border:1px solid rgba(245,158,11,.1);
    border-radius:var(--radius); padding:10px 14px; color:var(--accent);
    font-size:.8rem; line-height:1.5;
}

.section-label {
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.14em; color:var(--text3); margin-bottom:.6rem; margin-top:1.2rem;
}

/* ── Particles ─────────────────────────────────────────────────────── */
.particles { position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }
.p { position:absolute; width:2px; height:2px; border-radius:50%; opacity:.15; animation:fl 18s infinite ease-in-out; }
.p:nth-child(1){left:8%;background:var(--accent);animation-duration:16s}
.p:nth-child(2){left:22%;background:var(--accent2);animation-delay:-4s;animation-duration:20s}
.p:nth-child(3){left:40%;background:var(--accent3);animation-delay:-7s;animation-duration:22s}
.p:nth-child(4){left:58%;background:var(--accent);animation-delay:-10s;animation-duration:17s}
.p:nth-child(5){left:75%;background:var(--accent4);animation-delay:-3s;animation-duration:21s}
.p:nth-child(6){left:90%;background:var(--accent2);animation-delay:-8s;animation-duration:19s}
@keyframes fl { 0%{transform:translateY(100vh) scale(0);opacity:0} 10%{opacity:.2} 90%{opacity:.08} 100%{transform:translateY(-5vh) scale(1.5);opacity:0} }

/* ── Quality Score Badge ─────────────────────────────────────────── */
.quality-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:20px; font-size:.8rem; font-weight:600;
    transition:var(--smooth);
}
.quality-high { background:rgba(20,184,166,.12); color:#14B8A6; border:1px solid rgba(20,184,166,.2); }
.quality-med  { background:rgba(245,158,11,.12); color:#F59E0B; border:1px solid rgba(245,158,11,.2); }
.quality-low  { background:rgba(239,68,68,.12); color:#EF4444; border:1px solid rgba(239,68,68,.2); }

/* ── Keyboard Shortcut Keys ─────────────────────────────────────── */
.kbd {
    display:inline-block; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
    border-radius:6px; padding:2px 8px; font-family:'JetBrains Mono',monospace;
    font-size:.72rem; color:var(--text2); font-weight:500;
    box-shadow:0 1px 2px rgba(0,0,0,.2);
}

/* ── Footer ───────────────────────────────────────────────────── */
.app-footer {
    margin-top:3rem; padding:1.5rem 0; text-align:center;
    border-top:1px solid var(--border-glass);
}
.app-footer .footer-links {
    display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;
    margin-bottom:.8rem;
}
.app-footer .footer-links a {
    color:var(--text3) !important; font-size:.8rem; font-weight:500;
    text-decoration:none !important; border:none !important;
    transition:var(--smooth);
}
.app-footer .footer-links a:hover { color:var(--accent) !important; }
.app-footer .footer-copy {
    font-size:.7rem; color:var(--text3); letter-spacing:.03em;
}

/* ── Feature Card ─────────────────────────────────────────────── */
.feature-card {
    background:var(--bg-glass); border:1px solid var(--border-glass);
    border-radius:var(--radius-lg); padding:1.2rem; text-align:center;
    transition:var(--smooth);
}
.feature-card:hover {
    border-color:rgba(245,158,11,.15);
    box-shadow:0 0 25px rgba(245,158,11,.06);
    transform:translateY(-2px);
}
.feature-icon { font-size:1.6rem; margin-bottom:.4rem; }
.feature-title { font-weight:700; font-size:.88rem; color:var(--text1); margin-bottom:.2rem; }
.feature-desc { font-size:.76rem; color:var(--text3); line-height:1.4; }
</style>

<div class="particles"><div class="p"></div><div class="p"></div><div class="p"></div><div class="p"></div><div class="p"></div><div class="p"></div></div>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "page": "input",
        "form_schema": None,
        "form_url": "",
        "num_responses": 50,
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
        "api_key_openai": "",
        "api_key_anthropic": "",
        "api_key_gemini": "",
        "api_key_groq": "",
        "api_key_mistral": "",
        "api_key_cohere": "",
        "openai_model": "gpt-4o",
        "anthropic_model": "claude-3-5-sonnet-20241022",
        "gemini_model": "gemini-2.0-flash",
        "groq_model": "llama-3.1-70b-versatile",
        "mistral_model": "mistral-large-latest",
        "cohere_model": "command-r-plus",
        "temperature": 0.7,
        "api_call_delay": 0.5,
        "max_retries": 2,
        "generation_report": [],
        "stop_generation": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .3rem;">
        <div style="font-size:2rem;font-weight:900;letter-spacing:-0.03em;
                    background:linear-gradient(135deg,#F59E0B,#FB923C);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">SynthSurvey</div>
        <div style="font-size:.7rem;color:#5C5775;text-transform:uppercase;
                    letter-spacing:.14em;font-weight:600;margin-top:2px;">
            Synthetic Data Engine</div>
    </div>
    """, unsafe_allow_html=True)

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
        if st.button(btn_label, key=f"nav_{key}", disabled=disabled, use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)

    _PROVIDERS = ["openai", "anthropic", "gemini", "groq", "mistral", "cohere"]
    _PROVIDER_LABELS = {
        "openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Google Gemini",
        "groq": "Groq (Llama)", "mistral": "Mistral AI", "cohere": "Cohere",
    }
    provider = st.selectbox(
        "LLM Provider",
        _PROVIDERS,
        format_func=lambda x: _PROVIDER_LABELS[x],
        index=_PROVIDERS.index(st.session_state.llm_provider)
        if st.session_state.llm_provider in _PROVIDERS else 0,
        key="sidebar_provider",
        label_visibility="collapsed",
    )
    st.session_state.llm_provider = provider

    _KEY_CONFIG = {
        "openai":    ("api_key_openai",    "OpenAI API Key",    "sk-..."),
        "anthropic": ("api_key_anthropic", "Anthropic API Key", "sk-ant-..."),
        "gemini":    ("api_key_gemini",    "Gemini API Key",    "AIza..."),
        "groq":      ("api_key_groq",      "Groq API Key",      "gsk_..."),
        "mistral":   ("api_key_mistral",   "Mistral API Key",   "..."),
        "cohere":    ("api_key_cohere",    "Cohere API Key",    "..."),
    }
    key_attr, key_label, key_placeholder = _KEY_CONFIG[provider]
    api_key = st.text_input(
        key_label, type="password",
        value=st.session_state[key_attr],
        key=f"sidebar_key_{provider}", placeholder=key_placeholder,
    )
    st.session_state[key_attr] = api_key

    st.markdown('<div class="section-label">Model Settings</div>', unsafe_allow_html=True)

    _MODEL_OPTIONS = {
        "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "gemini":    ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],
        "groq":      ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "mistral":   ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mixtral-8x22b"],
        "cohere":    ["command-r-plus", "command-r", "command-light"],
    }
    _MODEL_STATE_KEY = {
        "openai": "openai_model", "anthropic": "anthropic_model",
        "gemini": "gemini_model", "groq": "groq_model",
        "mistral": "mistral_model", "cohere": "cohere_model",
    }
    model_options = _MODEL_OPTIONS[provider]
    model_state_key = _MODEL_STATE_KEY[provider]
    current_model = st.session_state[model_state_key]
    model_idx = model_options.index(current_model) if current_model in model_options else 0
    selected_model = st.selectbox(
        f"{_PROVIDER_LABELS[provider]} Model",
        model_options,
        index=model_idx,
        key=f"sidebar_model_{provider}",
    )
    st.session_state[model_state_key] = selected_model

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.5, step=0.05,
        value=st.session_state.temperature,
        key="sidebar_temperature",
        help="Higher = more creative & varied responses. Lower = more consistent.",
    )
    st.session_state.temperature = temperature

    st.markdown('<div class="section-label">Advanced</div>', unsafe_allow_html=True)

    api_delay = st.slider(
        "API Call Delay (s)", min_value=0.0, max_value=3.0, step=0.1,
        value=st.session_state.api_call_delay,
        key="sidebar_delay",
        help="Delay between API calls to avoid rate limits.",
    )
    st.session_state.api_call_delay = api_delay

    max_retries = st.number_input(
        "Max Retries per Response", min_value=0, max_value=5, step=1,
        value=st.session_state.max_retries,
        key="sidebar_retries",
        help="Number of retry attempts if a response fails validation.",
    )
    st.session_state.max_retries = max_retries

    # Test Connection button
    if st.button("Test Connection", use_container_width=True, key="sidebar_test_conn"):
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

    st.markdown("""
    <div style="text-align:center;margin-top:1.5rem;padding-top:.8rem;
                border-top:1px solid rgba(255,255,255,.04);">
        <span style="font-size:.68rem;color:#5C5775;font-weight:500;
                     background:rgba(255,255,255,.03);padding:3px 10px;
                     border-radius:16px;border:1px solid rgba(255,255,255,.05);">
            v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def get_llm_settings() -> Settings:
    """Build Settings object from session state."""
    settings = get_settings()
    settings.llm_provider = st.session_state.llm_provider
    # Map all provider keys and models from session state
    _key_map = {
        "openai": ("openai_api_key", "openai_model"),
        "anthropic": ("anthropic_api_key", "anthropic_model"),
        "gemini": ("gemini_api_key", "gemini_model"),
        "groq": ("groq_api_key", "groq_model"),
        "mistral": ("mistral_api_key", "mistral_model"),
        "cohere": ("cohere_api_key", "cohere_model"),
    }
    _session_key_map = {
        "openai": "api_key_openai", "anthropic": "api_key_anthropic",
        "gemini": "api_key_gemini", "groq": "api_key_groq",
        "mistral": "api_key_mistral", "cohere": "api_key_cohere",
    }
    for prov, (settings_key_attr, settings_model_attr) in _key_map.items():
        session_key = _session_key_map[prov]
        if st.session_state.get(session_key):
            setattr(settings, settings_key_attr, st.session_state[session_key])
        model_state = f"{prov}_model"
        if st.session_state.get(model_state):
            setattr(settings, settings_model_attr, st.session_state[model_state])
    settings.temperature = st.session_state.temperature
    settings.api_call_delay = st.session_state.api_call_delay
    settings.max_retries = st.session_state.max_retries
    return settings


def check_api_key() -> bool:
    """Check if the required API key is set."""
    prov = st.session_state.llm_provider
    _env_keys = {
        "openai": ("api_key_openai", "OPENAI_API_KEY"),
        "anthropic": ("api_key_anthropic", "ANTHROPIC_API_KEY"),
        "gemini": ("api_key_gemini", "GEMINI_API_KEY"),
        "groq": ("api_key_groq", "GROQ_API_KEY"),
        "mistral": ("api_key_mistral", "MISTRAL_API_KEY"),
        "cohere": ("api_key_cohere", "COHERE_API_KEY"),
    }
    session_attr, env_var = _env_keys.get(prov, ("", ""))
    return bool(st.session_state.get(session_attr) or os.environ.get(env_var, ""))


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

    st.divider()

    col_main, col_side = st.columns([5, 3], gap="large")

    with col_main:
        st.markdown("##### Google Form URL")
        form_url = st.text_input(
            "url", value=st.session_state.form_url,
            placeholder="https://docs.google.com/forms/d/e/.../viewform",
            key="input_form_url", label_visibility="collapsed",
        )
        st.session_state.form_url = form_url

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("##### Responses")
            num_responses = st.number_input(
                "n", min_value=1, max_value=500,
                value=st.session_state.num_responses, step=10,
                key="input_num_responses", label_visibility="collapsed",
            )
            st.session_state.num_responses = num_responses
        with sc2:
            st.markdown("##### Provider")
            _prov_display = {
                "openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Gemini",
                "groq": "Groq (Llama)", "mistral": "Mistral", "cohere": "Cohere",
            }
            _model_key = f"{st.session_state.llm_provider}_model"
            prov_label = f"{_prov_display.get(st.session_state.llm_provider, 'Unknown')} · {st.session_state.get(_model_key, '')}"
            st.markdown(
                f'<div style="padding:.6rem 1rem;background:var(--bg-glass);'
                f'border:1px solid var(--border-glass);border-radius:var(--radius);'
                f'font-size:.88rem;color:var(--text2);margin-top:2px;">{prov_label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("##### Persona Constraints")
        constraints = st.text_area(
            "c", value=st.session_state.persona_constraints,
            placeholder="e.g., ASU undergrad students, mix of STEM and humanities, ages 18-25...",
            height=85, key="input_constraints", label_visibility="collapsed",
        )
        st.session_state.persona_constraints = constraints

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
        if st.button("Parse Form & Continue", type="primary", use_container_width=True):
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
    st.markdown("""
    <div class="app-footer">
        <div class="footer-links">
            <a href="https://github.com/SaurabhDusane/SynthSurvey" target="_blank">GitHub</a>
            <a href="#">Documentation</a>
            <a href="#">Report a Bug</a>
            <a href="#">Privacy Policy</a>
        </div>
        <div class="footer-copy">
            SynthSurvey v1.0.0 &mdash; Built for academic & hackathon use &mdash;
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
        <h1 style="margin:0!important;">{schema.form_title}</h1>
    </div>
    """, unsafe_allow_html=True)
    if schema.form_description:
        st.markdown(f"*{schema.form_description}*")

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
            st.caption(f"Provider: {cost['provider']} | Model: {cost['model']}")
        except Exception:
            pass

    n1, n2, _ = st.columns([1, 1, 2])
    with n1:
        if st.button("Back to Input", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with n2:
        if st.button("Start Generation", type="primary", use_container_width=True):
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
            if st.button("Regenerate", use_container_width=True):
                st.session_state.dataset = None
                st.session_state.generated_count = 0
                st.session_state.failed_count = 0
                st.rerun()
        with c2:
            if st.button("View Results", type="primary", use_container_width=True):
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
        bc, _, _ = st.columns([2, 1, 1])
        with bc:
            if st.button("Begin Generation", type="primary", use_container_width=True):
                st.session_state.generation_in_progress = True
                st.rerun()
        return

    # ── Active Generation ──
    num_responses = st.session_state.num_responses
    settings = get_llm_settings()
    provider = settings.llm_provider
    model_attr = f"{provider}_model"
    model_name = getattr(settings, model_attr, "unknown")

    try:
        llm = LLMClient(settings)
    except ValueError as e:
        st.error(str(e))
        st.session_state.generation_in_progress = False
        return

    persona_gen = PersonaGenerator(llm, settings)
    response_gen = ResponseGenerator(llm, settings)
    dataset = SurveyDataset(
        form_title=schema.form_title,
        form_url=schema.form_url,
        generation_started=datetime.now(timezone.utc).isoformat(),
    )

    # Report log: list of dicts for every attempt
    report_log = []

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
        if stop_holder.button("Stop Generation", type="secondary", use_container_width=True, key="stop_gen_btn"):
            st.session_state.stop_generation = True

    st.divider()

    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        st.markdown("##### Latest Persona")
        persona_preview = st.empty()
    with pc2:
        st.markdown("##### Latest Response")
        response_preview = st.empty()

    start_time = time.time()
    generated = 0
    failed = 0
    consecutive_failures = 0
    _MAX_CONSECUTIVE_FAIL = 10  # stop early if all failing
    st.session_state.stop_generation = False  # reset flag

    for i in range(num_responses):
        # Check for user-requested stop
        if st.session_state.stop_generation:
            st.warning(f"Generation stopped by user after {i} attempts.")
            break

        elapsed = time.time() - start_time
        rate = generated / elapsed if elapsed > 0 else 0
        remaining = (num_responses - i) / rate if rate > 0 else 0
        pct = (i + 1) / num_responses

        progress_bar.progress(pct, text=f"Response {i+1} / {num_responses}  ({pct*100:.0f}%)")
        gen_m.metric("Generated", generated)
        fail_m.metric("Failed", failed)
        time_m.metric("Elapsed", f"{elapsed:.0f}s")
        rate_m.metric("ETA", f"{remaining:.0f}s" if rate > 0 else "...")

        entry = {
            "response_num": i + 1,
            "status": "success",
            "persona_name": None,
            "error_type": None,
            "error_message": None,
            "suggestion": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            persona = persona_gen.generate_one(schema, st.session_state.persona_constraints or None)
            entry["persona_name"] = persona.name

            itags = "".join(f'<span class="persona-tag">{x}</span>' for x in persona.interests)
            ttags = "".join(f'<span class="persona-tag">{x}</span>' for x in persona.personality_traits)
            persona_preview.markdown(
                f'<div class="persona-card-v2">'
                f'<div class="persona-name">{persona.name}</div>'
                f'<div class="persona-detail">'
                f'{persona.age} &bull; {persona.gender} &bull; {persona.year}<br>'
                f'{persona.major} @ {persona.university}<br><br>'
                f'<strong>Interests:</strong><br>{itags}<br><br>'
                f'<strong>Traits:</strong><br>{ttags}<br><br>'
                f'<strong>Engagement:</strong> {persona.engagement_level}<br>'
                f'<strong>Attitude:</strong> {persona.attitude_toward_topic}<br><br>'
                f'<em style="color:var(--text3)">{persona.background_context}</em>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            response = response_gen.generate_one(persona, schema)
            if response.generation_success:
                generated += 1
                consecutive_failures = 0
                items = list(response.answers.items())[:5]
                q_map = {q.question_id: q.question_text for q in schema.questions}
                html = '<div class="response-card">'
                for qid, ans in items:
                    qt = q_map.get(qid, qid)
                    if isinstance(ans, list): ans_s = ", ".join(str(a) for a in ans)
                    elif isinstance(ans, dict): ans_s = json.dumps(ans, indent=1)
                    else: ans_s = str(ans)
                    html += (
                        f'<div style="margin-bottom:.65rem;">'
                        f'<div style="font-size:.75rem;color:var(--text3);font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:.04em;">{qt}</div>'
                        f'<div style="color:var(--text1);margin-top:2px;">{ans_s}</div></div>'
                    )
                if len(response.answers) > 5:
                    html += f'<div style="color:var(--text3);font-size:.82rem;font-style:italic;">...and {len(response.answers)-5} more</div>'
                html += '</div>'
                response_preview.markdown(html, unsafe_allow_html=True)
            else:
                failed += 1
                consecutive_failures += 1
                entry["status"] = "failed"
                entry["error_type"] = "Validation Failed"
                entry["error_message"] = "Response generated but failed validation checks"
                entry["suggestion"] = "Try lowering temperature or using a more capable model"
            dataset.add_response(response)
        except Exception as e:
            failed += 1
            consecutive_failures += 1
            diag = classify_error(e, provider, model_name)
            entry["status"] = "failed"
            entry["error_type"] = diag["error_type"]
            entry["error_message"] = diag["message"][:300]
            entry["suggestion"] = diag["suggestion"]
            st.warning(f"Response {i+1} failed: {diag['error_type']}")

        report_log.append(entry)

        # Early stop if all recent attempts are failing with the same error
        if consecutive_failures >= _MAX_CONSECUTIVE_FAIL:
            st.error(
                f"Stopped early: {_MAX_CONSECUTIVE_FAIL} consecutive failures. "
                f"Check the Generation Report below for details."
            )
            break

    # Clean up stop button
    stop_holder.empty()

    dataset.generation_completed = datetime.now(timezone.utc).isoformat()
    dataset.total_failed = failed
    progress_bar.progress(1.0, text="Generation complete!")
    gen_m.metric("Generated", generated)
    fail_m.metric("Failed", failed)
    elapsed = time.time() - start_time
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
        if st.button("View Results", type="primary", use_container_width=True):
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
            <div><strong>Provider:</strong> {provider}</div>
            <div><strong>Model:</strong> {model}</div>
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
                st.markdown(f"**Sample error:**")
                st.code(info["sample_msg"][:500] if info["sample_msg"] else "No message", language=None)
                st.markdown(f"**How to fix:**")
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
        st.dataframe(styled, use_container_width=True, hide_index=True)

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
        use_container_width=True,
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
        <h1 style="margin:0!important;">{schema.form_title}</h1>
        <div style="color:var(--text2);margin-top:5px;">{len(df)} total responses</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df), "Total"), (c2, synthetic_count, "Synthetic"),
        (c3, real_count, "Real"), (c4, schema.total_questions, "Questions"),
    ]:
        with col:
            st.markdown(f'<div class="stat-glow"><div class="sn">{val}</div><div class="sl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()

    tab_data, tab_charts, tab_diversity, tab_personas, tab_export = st.tabs(["Data Table", "Charts", "Diversity", "Personas", "Export"])

    with tab_data:
        # Quality score
        if "is_synthetic" in df.columns:
            synth_df = df[df["is_synthetic"] == True]
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
        st.dataframe(df, use_container_width=True, height=500)

    with tab_charts:
        render_charts(schema, df)

    with tab_diversity:
        render_diversity_metrics(schema, df, dataset)

    with tab_personas:
        render_persona_gallery(dataset)

    with tab_export:
        render_export(schema, dataset, df)

    _render_footer()


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
    layout_common = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E8E6F0"),
        title_font=dict(size=14, color="#E8E6F0"),
    )

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
            template="plotly_dark",
        )
        fig.update_layout(**layout_common, height=max(300, len(diversity_data) * 35))
        fig.update_traces(marker_cornerradius=6)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Diversity Table"):
            st.dataframe(div_df, use_container_width=True, hide_index=True)

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
                             color_discrete_sequence=colors, template="plotly_dark")
                fig.update_layout(**layout_common)
                fig.update_traces(textinfo="label+percent", textfont_size=12)
                st.plotly_chart(fig, use_container_width=True)

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
                                       template="plotly_dark",
                                       color_discrete_sequence=[colors[2]])
                    fig.update_layout(**layout_common)
                    fig.update_traces(marker_cornerradius=6)
                    st.plotly_chart(fig, use_container_width=True)
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
            if st.button("Previous", disabled=st.session_state[page_key] <= 0, key="pg_prev", use_container_width=True):
                st.session_state[page_key] -= 1
                st.rerun()
        with pg_cols[1]:
            if st.button("Next", disabled=st.session_state[page_key] >= total_pages - 1, key="pg_next", use_container_width=True):
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
                    f'<div class="persona-card-v2" style="margin-bottom:.6rem;">'
                    f'<div class="persona-name">'
                    f'<span style="color:{success_color};font-size:.6rem;margin-right:6px;">{success_icon}</span>'
                    f'#{global_idx} &mdash; {name}</div>'
                    f'<div class="persona-detail">'
                    f'{detail_line}<br>'
                    f'<span style="font-size:.78rem;color:var(--text3);line-height:1.5;">{rest[:150]}{"..." if len(rest)>150 else ""}</span><br>'
                    f'<span style="font-size:.7rem;color:var(--text3);opacity:.6;">Retries: {resp.retry_count}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


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
    layout_common = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E8E6F0"),
        title_font=dict(size=15, color="#E8E6F0"),
    )

    if qt in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN):
        counts = df[col_name].value_counts().reset_index()
        counts.columns = ["Option", "Count"]
        fig = px.bar(counts, x="Option", y="Count", title=f"Distribution: {col_name}",
                     color="Option", color_discrete_sequence=colors, template="plotly_dark")
        fig.update_layout(**layout_common, showlegend=False)
        fig.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig, use_container_width=True)

        if "is_synthetic" in df.columns and df["is_synthetic"].nunique() > 1:
            st.markdown("##### Real vs Synthetic")
            comp = df.groupby(["is_synthetic", col_name]).size().reset_index(name="Count")
            comp["Source"] = comp["is_synthetic"].map({True: "Synthetic", False: "Real"})
            fig2 = px.bar(comp, x=col_name, y="Count", color="Source", barmode="group",
                          title=f"Real vs Synthetic: {col_name}",
                          color_discrete_map={"Synthetic": "#F59E0B", "Real": "#14B8A6"},
                          template="plotly_dark")
            fig2.update_layout(**layout_common)
            fig2.update_traces(marker_cornerradius=8)
            st.plotly_chart(fig2, use_container_width=True)

    elif qt == QuestionType.LINEAR_SCALE:
        numeric_col = pd.to_numeric(df[col_name], errors="coerce")
        nbins = (selected_q.scale_config.max_value - selected_q.scale_config.min_value + 1
                 if selected_q.scale_config else 10)
        fig = px.histogram(numeric_col.dropna(), nbins=nbins,
                           title=f"Distribution: {col_name}",
                           labels={"value": col_name, "count": "Frequency"},
                           template="plotly_dark", color_discrete_sequence=[colors[0]])
        fig.update_layout(**layout_common)
        fig.update_traces(marker_cornerradius=6)
        st.plotly_chart(fig, use_container_width=True)
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
                     color="Option", color_discrete_sequence=colors, template="plotly_dark")
        fig.update_layout(**layout_common, showlegend=False)
        fig.update_traces(marker_line_width=0, marker_cornerradius=8)
        st.plotly_chart(fig, use_container_width=True)


def render_export(schema: FormSchema, dataset: SurveyDataset, df: pd.DataFrame):
    st.markdown("##### Download Your Data")
    st.markdown("")

    safe_title = schema.form_title[:30].replace(' ', '_').replace('/', '_')
    csv_bytes = CSVExporter.to_csv_bytes(df)
    json_bytes = JSONExporter.to_json_bytes(dataset, schema)

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
            mime="text/csv", use_container_width=True,
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
            mime="application/json", use_container_width=True,
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
            use_container_width=True,
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
            "Export to Sheets", disabled=True, use_container_width=True,
            help="Configure Google Sheets credentials in .env to enable.",
        )

    st.divider()

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
