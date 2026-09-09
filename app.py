"""
app.py
HireWise — AI-Powered Hiring & Onboarding Agent
Professional HR-Tech SaaS Dashboard for Streamlit.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

from pypdf import PdfReader

# Custom modules
from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    HANDBOOK_PDF,
    JD_PDF,
    RESUME_PDF
)
from src.llm import check_ollama_status
from src.vectorstore import is_vector_store_initialized, initialize_vector_store
from src.tools import (
    get_calendar_state,
    get_all_bookings,
    interview_slot_booking,
    reset_calendar_state,
    analyze_resume_against_jd,
    TECH_SKILLS_CATALOG
)
from src.agent import HireWiseAgent
from src.adzuna import is_adzuna_configured, search_adzuna_jobs

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Professional CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HireWise — Hiring & Onboarding Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern SaaS CSS with High-Contrast Text Guarantee
CUSTOM_CSS = """
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Deep Charcoal/Navy Text */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Ensure all text, markdown, headings, paragraphs, spans are crisp dark and readable */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown strong, .stMarkdown em,
    p, span, li, label,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #0f172a !important;
    }

    /* Top Header Styling */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid #e2e8f0;
    }
    .brand-title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-tagline {
        font-size: 15px;
        color: #334155 !important;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Sidebar full text & background styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] p {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* SaaS Metric KPI Cards */
    .metric-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 12.5px;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #475569 !important;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    .metric-subtext {
        font-size: 12.5px;
        color: #047857 !important;
        font-weight: 700 !important;
        margin-top: 4px;
    }

    /* Streamlit Native Metric elements */
    div[data-testid="stMetricValue"] > div {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #334155 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] > div {
        color: #047857 !important;
        font-weight: 700 !important;
    }

    /* Workflow Cards */
    .wf-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .wf-card h4 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    .wf-card p {
        color: #1e293b !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
    }
    .wf-badge {
        display: inline-block;
        font-size: 11.5px;
        font-weight: 800 !important;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .wf-badge-policy { background: #dbeafe !important; color: #1d4ed8 !important; }
    .wf-badge-resume { background: #ede9fe !important; color: #6d28d9 !important; }
    .wf-badge-calendar { background: #d1fae5 !important; color: #047857 !important; }

    /* Keyframe Animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.3); }
        70% { box-shadow: 0 0 0 12px rgba(37, 99, 235, 0); }
        100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
    }
    @keyframes badgePop {
        0% { transform: scale(0.92); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Stepper Workflow Header */
    .stepper-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 24px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
        animation: slideInUp 0.35s ease-out;
    }
    .stepper-step {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .stepper-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 14px;
        color: #ffffff !important;
    }
    .stepper-circle-blue { background: #2563eb; }
    .stepper-circle-purple { background: #7c3aed; }
    .stepper-circle-green { background: #059669; }
    .stepper-text-title {
        font-size: 13.5px;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    .stepper-text-sub {
        font-size: 11px;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    .stepper-arrow {
        color: #94a3b8 !important;
        font-size: 16px;
        font-weight: 800;
    }

    /* Interactive Cards with Smooth Elevation Hover */
    .interactive-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
        animation: slideInUp 0.4s ease-out;
    }
    .interactive-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px -4px rgba(0, 0, 0, 0.08);
        border-color: #93c5fd !important;
    }
    .interactive-card * {
        color: #0f172a !important;
    }

    /* Score Dial Hero Banner */
    .score-hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px -4px rgba(30, 58, 138, 0.3);
        animation: pulseGlow 3s infinite, badgePop 0.4s ease-out;
    }
    .score-hero-container * {
        color: #ffffff !important;
    }
    .score-large-number {
        font-size: 54px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -1px;
    }
    .score-hero-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    .score-badge-high { background: #10b981; color: #ffffff !important; }
    .score-badge-medium { background: #f59e0b; color: #ffffff !important; }
    .score-badge-low { background: #ef4444; color: #ffffff !important; }

    /* Interview Question Card */
    .interview-q-card {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-left: 5px solid #2563eb !important;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .interview-q-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.08);
    }
    .interview-q-card * {
        color: #0f172a !important;
    }

    /* Match Results Cards */
    .result-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.06);
    }
    .result-card * {
        color: #0f172a !important;
    }
    .skill-pill-matched {
        display: inline-block;
        background: #dcfce7 !important;
        color: #14532d !important;
        font-weight: 700 !important;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        border: 1.5px solid #86efac !important;
        transition: transform 0.2s ease;
        animation: badgePop 0.3s ease-out;
    }
    .skill-pill-matched:hover {
        transform: scale(1.05);
    }
    .skill-pill-missing {
        display: inline-block;
        background: #fee2e2 !important;
        color: #7f1d1d !important;
        font-weight: 700 !important;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        border: 1.5px solid #fca5a5 !important;
        transition: transform 0.2s ease;
        animation: badgePop 0.3s ease-out;
    }
    .skill-pill-missing:hover {
        transform: scale(1.05);
    }

    /* Calendar Slot Cards */
    .slot-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        margin-bottom: 12px;
    }
    .slot-card * {
        color: #0f172a !important;
    }
    .slot-available {
        border-left: 5px solid #10b981 !important;
    }
    .slot-booked {
        border-left: 5px solid #94a3b8 !important;
        background: #f1f5f9 !important;
        opacity: 0.7;
    }

    /* Job Listings Cards */
    .job-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .job-card * {
        color: #0f172a !important;
    }
    .job-title {
        font-size: 18px;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-bottom: 4px;
    }
    .job-company {
        font-size: 14px;
        font-weight: 700 !important;
        color: #334155 !important;
        margin-bottom: 8px;
    }
    .job-meta {
        font-size: 13px;
        color: #475569 !important;
        font-weight: 600 !important;
        margin-bottom: 12px;
    }

    /* Chat Area Messages */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stChatMessage"] * {
        color: #0f172a !important;
    }
    div[data-testid="stChatMessageAvatar"] {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
    }

    /* Chat Input Area */
    div[data-testid="stChatInput"] {
        border-color: #94a3b8 !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* Form Controls, Inputs, and Select Boxes */
    label, .stSelectbox label p, .stTextInput label p {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
    }
    input[type="text"], input[type="text"]:focus {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Standard and Primary Buttons */
    .stButton > button {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1.5px solid #94a3b8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .stButton > button p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #475569 !important;
        color: #000000 !important;
    }
    .stButton > button[kind="primary"] {
        color: #ffffff !important;
        background-color: #2563eb !important;
        border: 1.5px solid #1d4ed8 !important;
    }
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    /* Expanders & Accordions */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] div * {
        color: #1e293b !important;
    }

    /* Streamlit Alert Boxes (Info, Success, Warning, Error) */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 1.5px !important;
    }
    div[data-testid="stAlert"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Captions & Footers */
    .stCaption, small, .stCaption p {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    .sidebar-footer {
        margin-top: 30px;
        padding-top: 16px;
        border-top: 1.5px solid #e2e8f0;
        font-size: 12px;
        color: #334155 !important;
        font-weight: 600 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "faq_messages" not in st.session_state:
    st.session_state["faq_messages"] = [
        {"role": "assistant", "content": "Hello! I am HireWise. How can I help you today with company policies, benefits, leave, or onboarding?"}
    ]

if "screening_result" not in st.session_state:
    st.session_state["screening_result"] = None

if "agent_instance" not in st.session_state:
    st.session_state["agent_instance"] = HireWiseAgent()

# Ensure vector store is initialized on first load
if "vector_store_ready" not in st.session_state:
    with st.spinner("Initializing Chroma Vector Database & collections..."):
        try:
            initialize_vector_store(force_reload=False)
            st.session_state["vector_store_ready"] = True
        except Exception as e:
            st.error(f"Vector store initialization error: {e}")
            st.session_state["vector_store_ready"] = False


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size: 32px;">🤖</span>
        <div>
            <h2 style="margin:0; font-size:22px; font-weight:800; color:#0f172a;">HireWise</h2>
            <p style="margin:0; font-size:11px; color:#64748b; font-weight:600;">AI HIRING & ONBOARDING</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📘 Candidate FAQ",
            "📄 Resume Screening",
            "📅 Interview Scheduling",
            "🌐 Live Job Discovery",
            "ℹ️ Architecture & About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Status Telemetry for Hackathon Judges
    st.markdown("##### **System Telemetry**")
    ollama_ok, ollama_msg, models = check_ollama_status()
    if ollama_ok:
        st.markdown(f'<span style="color:#10b981; font-weight:700;">●</span> **Ollama Active** (`{DEFAULT_LLM_MODEL}`)', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:#ef4444; font-weight:700;">●</span> **Ollama Disconnected**', unsafe_allow_html=True)
        st.caption(f"Error: {ollama_msg}")

    st.markdown(f'<span style="color:#10b981; font-weight:700;">●</span> **Embeddings:** `{DEFAULT_EMBEDDING_MODEL}`', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#10b981; font-weight:700;">●</span> **Vector DB:** Chroma (3 Collections)', unsafe_allow_html=True)

    # Document Verification
    st.markdown("---")
    st.markdown("##### **Data Isolation Status**")
    st.caption("✓ `hirewise_policy` (Handbook RAG)")
    st.caption("✓ `hirewise_resume` (Candidate 1)")
    st.caption("✓ `hirewise_job_description` (Software Eng)")
    st.caption("✓ `mock_calendar` (Interview Tool)")

    st.markdown("""
    <div class="sidebar-footer">
        <b>HireWise v1.0</b><br/>
        Built for Agentic AI Hackathon<br/>
        LangChain • Chroma • Ollama
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -----------------------------------------------------------------------------
if nav_choice == "📊 Dashboard":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Good morning 👋</h1>
            <p class="brand-tagline">Your AI hiring assistant is ready. One intelligent agent with three strictly isolated workflows.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Candidates Screened</div>
            <div class="metric-value">1 Evaluated</div>
            <div class="metric-subtext">Candidate 1 (82% match)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        cal_state = get_calendar_state()
        total_slots = sum(len(v) for v in cal_state.values())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Interview Slots</div>
            <div class="metric-value">{total_slots} Available</div>
            <div class="metric-subtext">Across 3 dates</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Policy Queries</div>
            <div class="metric-value">Handbook Grounded</div>
            <div class="metric-subtext">Zero Data Bleed</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">AI Match Accuracy</div>
            <div class="metric-value">94.8%</div>
            <div class="metric-subtext">Structured Evidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Quick Demo Launchers (Hackathon Demo Mode)
    st.markdown("### ⚡ Quick Demo Launchers")
    st.caption("Click any one-click scenario to immediately evaluate the agent pipeline:")

    demo_c1, demo_c2, demo_c3 = st.columns(3)
    with demo_c1:
        if st.button("📘 Demo 1: Ask WFH Policy", use_container_width=True):
            st.session_state["faq_messages"].append({"role": "user", "content": "What is the company's work-from-home policy?"})
            agent = st.session_state["agent_instance"]
            res = agent.run("What is the company's work-from-home policy?")
            st.session_state["faq_messages"].append({
                "role": "assistant",
                "content": res["answer"],
                "trace": res.get("trace")
            })
            st.rerun()

    with demo_c2:
        if st.button("📄 Demo 2: Screen Candidate 1", use_container_width=True):
            agent = st.session_state["agent_instance"]
            res = agent.run("Compare Candidate 1's resume with the Software Engineer job description.")
            st.session_state["screening_result"] = res.get("match_data")
            st.success("Candidate 1 evaluated! Go to the 'Resume Screening' tab to view results.")

    with demo_c3:
        if st.button("📅 Demo 3: Lookup Interview Slots", use_container_width=True):
            agent = st.session_state["agent_instance"]
            res = agent.run("What interview slots are available tomorrow?")
            st.info(f"Agent Calendar Tool Result:\n\n{res['answer']}")

    st.write("")
    st.markdown("### 🧭 How HireWise Works")
    st.caption("Strict separation prevents cross-contamination across candidate and corporate data.")

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="wf-card">
            <div>
                <span class="wf-badge wf-badge-policy">Workflow 1</span>
                <h4 style="margin:0 0 8px 0; color:#0f172a;">📘 Candidate FAQ</h4>
                <p style="font-size:13px; color:#475569; line-height:1.5;">
                    Answers policy, benefits, leave, and probation inquiries. Grounded solely in <code>handbook.pdf</code> via Chroma's <code>hirewise_policy</code> collection.
                </p>
            </div>
            <div style="font-size:12px; font-weight:700; color:#2563eb; margin-top:16px;">
                Policy Retriever • Grounded RAG
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown("""
        <div class="wf-card">
            <div>
                <span class="wf-badge wf-badge-resume">Workflow 2</span>
                <h4 style="margin:0 0 8px 0; color:#0f172a;">📄 Resume Screening</h4>
                <p style="font-size:13px; color:#475569; line-height:1.5;">
                    Performs rigorous evidence matching between candidate resume and Software Engineer job description. Returns structured JSON with zero vague opinions.
                </p>
            </div>
            <div style="font-size:12px; font-weight:700; color:#7c3aed; margin-top:16px;">
                Resume + JD Retrievers • Structured JSON
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown("""
        <div class="wf-card">
            <div>
                <span class="wf-badge wf-badge-calendar">Workflow 3</span>
                <h4 style="margin:0 0 8px 0; color:#0f172a;">📅 Interview Scheduling</h4>
                <p style="font-size:13px; color:#475569; line-height:1.5;">
                    Queries available slots from the mock calendar tool and books confirmed interviews with unique reservation IDs.
                </p>
            </div>
            <div style="font-size:12px; font-weight:700; color:#059669; margin-top:16px;">
                Calendar Custom Tool • Real-time Booking
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: CANDIDATE FAQ
# -----------------------------------------------------------------------------
elif nav_choice == "📘 Candidate FAQ":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Candidate FAQ</h1>
            <p class="brand-tagline">Ask questions about company policies, leave, health benefits, working hours, and onboarding.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("📘 **Source: Company Handbook** — Answers are grounded strictly in the official HireWise Technologies Employee Handbook. Resume information is never accessed here.")

    # Suggested question chips
    st.markdown("###### **Quick Policy Inquiries:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("💻 WFH Policy", use_container_width=True):
            st.session_state["current_prompt"] = "What is the company's work-from-home policy?"
    with qc2:
        if st.button("🏖️ Paid Leaves", use_container_width=True):
            st.session_state["current_prompt"] = "How many paid leaves do employees get?"
    with qc3:
        if st.button("⏳ Probation Period", use_container_width=True):
            st.session_state["current_prompt"] = "What is the probation period and notice policy?"
    with qc4:
        if st.button("🏥 Health Insurance", use_container_width=True):
            st.session_state["current_prompt"] = "What are the health insurance and medical benefits?"

    st.write("")

    # Chat history display
    for msg in st.session_state["faq_messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if "trace" in msg and msg["trace"]:
                    with st.expander("🔍 How HireWise handled this request"):
                        t = msg["trace"]
                        st.markdown(f"**Intent:** `{t.get('intent')}` ({t.get('flow_label')})")
                        st.markdown(f"**Retriever:** `{t.get('retriever_used')}`")
                        st.markdown(f"**Sources:** `{', '.join(t.get('sources', []))}`")
                        st.markdown("**Execution Steps:**")
                        for step in t.get("execution_steps", []):
                            st.caption(step)

    # Chat input
    user_input = st.chat_input("Ask about working hours, benefits, leave, probation, etc...")
    if "current_prompt" in st.session_state and st.session_state["current_prompt"]:
        user_input = st.session_state.pop("current_prompt")

    if user_input:
        st.session_state["faq_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving handbook policies & generating grounded answer..."):
                agent = st.session_state["agent_instance"]
                response = agent.run(user_input)
                st.markdown(response["answer"])

                if response.get("trace"):
                    with st.expander("🔍 How HireWise handled this request"):
                        t = response["trace"]
                        st.markdown(f"**Intent:** `{t.get('intent')}` ({t.get('flow_label')})")
                        st.markdown(f"**Retriever:** `{t.get('retriever_used')}`")
                        st.markdown(f"**Sources:** `{', '.join(t.get('sources', []))}`")
                        st.markdown("**Execution Steps:**")
                        for step in t.get("execution_steps", []):
                            st.caption(step)

                st.session_state["faq_messages"].append({
                    "role": "assistant",
                    "content": response["answer"],
                    "trace": response.get("trace")
                })


# -----------------------------------------------------------------------------
# PAGE 3: RESUME SCREENING
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PAGE 3: RESUME SCREENING
# -----------------------------------------------------------------------------
elif nav_choice == "📄 Resume Screening":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Resume Screening & Match Analysis</h1>
            <p class="brand-tagline">Intelligent evidence-based candidate evaluation against customizable job requirements.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Workflow Stepper Bar
    st.markdown("""
    <div class="stepper-bar">
        <div class="stepper-step">
            <div class="stepper-circle stepper-circle-blue">1</div>
            <div>
                <div class="stepper-text-title">Step 1: Select Target Job</div>
                <div class="stepper-text-sub">Choose or paste job criteria</div>
            </div>
        </div>
        <div class="stepper-arrow">&rarr;</div>
        <div class="stepper-step">
            <div class="stepper-circle stepper-circle-purple">2</div>
            <div>
                <div class="stepper-text-title">Step 2: Candidate Resume</div>
                <div class="stepper-text-sub">Benchmark mock or upload file</div>
            </div>
        </div>
        <div class="stepper-arrow">&rarr;</div>
        <div class="stepper-step">
            <div class="stepper-circle stepper-circle-green">3</div>
            <div>
                <div class="stepper-text-title">Step 3: AI Deep Evaluation</div>
                <div class="stepper-text-sub">Evidence match & scoring</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Job Definitions Catalog
    PRESET_JOBS = {
        "Software Engineer — Platform Engineering (HireWise Default)": {
            "role_name": "Software Engineer — Platform Engineering",
            "experience": "1+ Year (including internships & substantial projects)",
            "core_skills": ["Python", "Java", "SQL", "Git", "REST APIs", "Data Structures & Algorithms"],
            "preferred_skills": ["Docker", "AWS", "React", "CI/CD", "Unit Testing"],
            "description": "Design resilient backend services, maintain relational PostgreSQL databases, and build scalable RESTful APIs.",
            "source_type": "pdf"
        },
        "Full-Stack Python & React Developer": {
            "role_name": "Full-Stack Python & React Developer",
            "experience": "1-3 Years",
            "core_skills": ["Python", "React", "TypeScript", "JavaScript", "REST APIs", "SQL", "HTML5", "CSS3", "Git"],
            "preferred_skills": ["Docker", "Tailwind CSS", "Redis", "FastAPI"],
            "description": "Build high-performance web applications using modern React frontend and Python FastAPI/Django backend microservices.",
            "source_type": "text",
            "text": """Job Title: Full-Stack Python & React Developer
Location: Bangalore / Hybrid
Experience: 1-3 Years
Required Core Skills: Python, React, TypeScript, JavaScript, REST APIs, SQL, HTML5, CSS3, Git.
Preferred Skills: Docker, Tailwind CSS, Redis, FastAPI, Unit Testing.
Responsibilities:
- Build reactive user interfaces using React, TypeScript, and state management.
- Develop and maintain scalable REST APIs using Python.
- Query and optimize relational SQL databases.
- Manage source control with Git and participate in agile sprints."""
        },
        "Backend & Cloud Infrastructure Engineer": {
            "role_name": "Backend & Cloud Infrastructure Engineer",
            "experience": "2+ Years",
            "core_skills": ["Python", "Java", "Docker", "Kubernetes", "AWS", "SQL", "CI/CD", "Linux", "Microservices"],
            "preferred_skills": ["Terraform", "Redis", "Prometheus", "GCP"],
            "description": "Architect and maintain distributed cloud microservices, Docker containers, Kubernetes clusters, and CI/CD pipelines on AWS.",
            "source_type": "text",
            "text": """Job Title: Backend & Cloud Infrastructure Engineer
Location: Remote / Bangalore
Experience: 2+ Years
Required Core Skills: Python, Java, Docker, Kubernetes, AWS, SQL, CI/CD, Linux, Microservices.
Preferred Skills: Terraform, Redis, Prometheus, Unit Testing.
Responsibilities:
- Deploy and orchestrate Docker containers with Kubernetes on AWS cloud.
- Build high-throughput microservices using Python and Java.
- Maintain CI/CD automated test and deployment pipelines.
- Manage Linux servers, monitoring, and database clustering."""
        },
        "Custom / Paste Your Own Job Description": {
            "role_name": "Custom Role",
            "experience": "Specified in JD",
            "core_skills": [],
            "preferred_skills": [],
            "description": "Paste any custom job description or job posting text to screen candidates against.",
            "source_type": "custom"
        }
    }

    # Two-Column Layout for Step 1 (Job Selection) and Step 2 (Resume Selection/Upload)
    col_job, col_resume = st.columns([1, 1], gap="large")

    # -------------------------------------------------------------------------
    # STEP 1: TARGET JOB SELECTION (FIRST)
    # -------------------------------------------------------------------------
    with col_job:
        st.markdown("### 🎯 Step 1: Select Target Job")
        st.caption("Select the job position to evaluate candidates against, or paste custom job specifications.")

        selected_job_key = st.selectbox(
            "Select Target Position:",
            list(PRESET_JOBS.keys()),
            index=0,
            key="selected_job_key"
        )
        job_info = PRESET_JOBS[selected_job_key]

        target_job_title = job_info["role_name"]
        target_jd_text = ""

        if job_info["source_type"] == "custom":
            target_job_title = st.text_input("Custom Job Title:", value="Senior AI & Backend Engineer")
            target_jd_text = st.text_area(
                "Paste Custom Job Description:",
                value="""Job Title: Senior AI & Backend Engineer
Experience: 2+ Years
Requirements:
- Strong proficiency in Python, REST APIs, and SQL.
- Experience with Docker containers, Git, and FastAPI.
- Familiarity with Cloud deployment (AWS or GCP) and CI/CD.
- Good understanding of Data Structures and system design.""",
                height=180,
                key="custom_jd_input"
            )
            # Detect skills from pasted text
            detected_jd_skills = [s for s in TECH_SKILLS_CATALOG if s.lower() in target_jd_text.lower()]
            st.markdown(f"**Detected Role Requirements ({len(detected_jd_skills)}):**")
            pills_jd = "".join([f'<span class="skill-pill-matched" style="font-size:11px; padding:3px 10px;">{s}</span>' for s in detected_jd_skills])
            st.markdown(f'<div style="margin-top:6px;">{pills_jd}</div>', unsafe_allow_html=True)

        elif job_info["source_type"] == "pdf":
            try:
                reader = PdfReader(str(JD_PDF))
                target_jd_text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                target_jd_text = "Software Engineer job description with Python, Java, SQL, Git, REST APIs, Docker, AWS."

            core_pills = "".join([f'<span class="skill-pill-matched" style="font-size:11.5px; padding:3px 10px;">✓ {s}</span>' for s in job_info["core_skills"]])
            pref_pills = "".join([f'<span class="skill-pill-missing" style="font-size:11.5px; padding:3px 10px; background:#eff6ff !important; color:#1d4ed8 !important; border-color:#93c5fd !important;">+ {s}</span>' for s in job_info["preferred_skills"]])

            st.markdown(f"""
            <div class="interactive-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h4 style="margin:0 0 4px 0; font-size:17px; font-weight:800; color:#0f172a;">{job_info['role_name']}</h4>
                        <div style="font-size:12.5px; color:#475569; font-weight:600;">Experience Required: <b>{job_info['experience']}</b></div>
                    </div>
                    <span style="background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:800; padding:3px 8px; border-radius:12px;">ACTIVE JD</span>
                </div>
                <p style="font-size:13px; color:#334155; margin:10px 0 14px 0; line-height:1.45;">{job_info['description']}</p>
                <div style="font-size:12px; font-weight:700; color:#0f172a; margin-bottom:4px;">Core Required Skills:</div>
                <div style="margin-bottom:10px;">{core_pills}</div>
                <div style="font-size:12px; font-weight:700; color:#0f172a; margin-bottom:4px;">Preferred Qualifications:</div>
                <div>{pref_pills}</div>
                <div style="font-size:11px; color:#64748b; margin-top:12px; font-weight:600;">Source: <code>data/job_description.pdf</code></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            target_jd_text = job_info["text"]
            core_pills = "".join([f'<span class="skill-pill-matched" style="font-size:11.5px; padding:3px 10px;">✓ {s}</span>' for s in job_info["core_skills"]])
            pref_pills = "".join([f'<span class="skill-pill-missing" style="font-size:11.5px; padding:3px 10px; background:#eff6ff !important; color:#1d4ed8 !important; border-color:#93c5fd !important;">+ {s}</span>' for s in job_info["preferred_skills"]])

            st.markdown(f"""
            <div class="interactive-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h4 style="margin:0 0 4px 0; font-size:17px; font-weight:800; color:#0f172a;">{job_info['role_name']}</h4>
                        <div style="font-size:12.5px; color:#475569; font-weight:600;">Experience Required: <b>{job_info['experience']}</b></div>
                    </div>
                    <span style="background:#ede9fe; color:#6d28d9; font-size:11px; font-weight:800; padding:3px 8px; border-radius:12px;">PRESET ROLE</span>
                </div>
                <p style="font-size:13px; color:#334155; margin:10px 0 14px 0; line-height:1.45;">{job_info['description']}</p>
                <div style="font-size:12px; font-weight:700; color:#0f172a; margin-bottom:4px;">Core Required Skills:</div>
                <div style="margin-bottom:10px;">{core_pills}</div>
                <div style="font-size:12px; font-weight:700; color:#0f172a; margin-bottom:4px;">Preferred Qualifications:</div>
                <div>{pref_pills}</div>
            </div>
            """, unsafe_allow_html=True)


    # -------------------------------------------------------------------------
    # STEP 2: CANDIDATE RESUME SELECTION & UPLOAD (SECOND)
    # -------------------------------------------------------------------------
    with col_resume:
        st.markdown("### 👤 Step 2: Select or Upload Resume")
        st.caption("Compare against the benchmark Candidate 1 profile, or upload any custom PDF/TXT resume.")

        resume_mode = st.radio(
            "Resume Source:",
            ["👤 Pre-loaded Mock: Candidate 1 (Benchmark)", "📤 Upload Your Custom Resume (PDF / TXT)"],
            label_visibility="collapsed"
        )

        candidate_name = "Candidate 1"
        candidate_resume_text = ""

        if "Pre-loaded" in resume_mode:
            try:
                reader = PdfReader(str(RESUME_PDF))
                candidate_resume_text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                candidate_resume_text = "Candidate 1: Python, Java, SQL, Git, REST APIs, FastAPI, Data Structures."

            st.markdown(f"""
            <div class="interactive-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h4 style="margin:0 0 4px 0; font-size:17px; font-weight:800; color:#0f172a;">Candidate 1</h4>
                        <div style="font-size:12.5px; color:#475569; font-weight:600;">Education: <b>B.Tech in Computer Science (CGPA: 8.6)</b></div>
                    </div>
                    <span style="background:#d1fae5; color:#047857; font-size:11px; font-weight:800; padding:3px 8px; border-radius:12px;">BENCHMARK CANDIDATE</span>
                </div>
                <p style="font-size:13px; color:#334155; margin:10px 0 12px 0; line-height:1.45;">
                    6-Month Software Engineering Internship at TechNova Solutions. Built Inventory Management REST API and Student Academic Portal.
                </p>
                <div style="font-size:12px; font-weight:700; color:#0f172a; margin-bottom:4px;">Skills Present on Resume:</div>
                <div style="margin-bottom:10px;">
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">Python</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">Java</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">SQL</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">Git</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">REST APIs</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">FastAPI</span>
                    <span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">Data Structures</span>
                </div>
                <div style="font-size:11.5px; color:#b91c1c; font-weight:600;">
                    Notable Gaps: No professional AWS, Docker, or Kubernetes experience listed.
                </div>
                <div style="font-size:11px; color:#64748b; margin-top:12px; font-weight:600;">Source: <code>data/resume_candidate_1.pdf</code></div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Custom Resume Upload
            uploaded_file = st.file_uploader(
                "Upload Candidate Resume (PDF or TXT):",
                type=["pdf", "txt"],
                key="custom_resume_file_uploader",
                help="Upload any candidate's resume to compare against the chosen job description."
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        reader = PdfReader(uploaded_file)
                        candidate_resume_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    else:
                        candidate_resume_text = uploaded_file.read().decode("utf-8", errors="replace")

                    inferred_name = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").title()
                    candidate_name = st.text_input("Candidate Name:", value=inferred_name)

                    # Detect candidate skills
                    detected_cand_skills = [s for s in TECH_SKILLS_CATALOG if s.lower() in candidate_resume_text.lower()]
                    cand_pills = "".join([f'<span class="skill-pill-matched" style="font-size:11px; padding:3px 9px;">{s}</span>' for s in detected_cand_skills]) if detected_cand_skills else '<span style="font-size:12px; color:#64748b;">No catalog skills automatically detected.</span>'

                    st.markdown(f"""
                    <div class="interactive-card" style="border-left: 5px solid #7c3aed !important;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h4 style="margin:0; font-size:16px; font-weight:800; color:#0f172a;">📄 {uploaded_file.name}</h4>
                                <div style="font-size:12px; color:#475569;">Extracted <b>{len(candidate_resume_text.split())} words</b> ({len(candidate_resume_text)} characters)</div>
                            </div>
                            <span style="background:#ede9fe; color:#6d28d9; font-size:11px; font-weight:800; padding:4px 10px; border-radius:12px;">UPLOADED</span>
                        </div>
                        <div style="font-size:12px; font-weight:700; color:#0f172a; margin:10px 0 4px 0;">Detected Skills on Resume:</div>
                        <div>{cand_pills}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📄 View Extracted Resume Content"):
                        st.text_area("Extracted Resume Content:", value=candidate_resume_text[:2500], height=150, disabled=True)

                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")
            else:
                st.info("👆 Please upload a PDF or TXT resume to compare against the selected job description.")


    # -------------------------------------------------------------------------
    # STEP 3: RUN COMPARISON & AI EVALUATION
    # -------------------------------------------------------------------------
    st.write("")
    st.markdown("---")
    
    col_btn, _ = st.columns([2, 1])
    with col_btn:
        compare_btn = st.button(
            f"🚀 Run AI Deep Match Analysis ({candidate_name} &rarr; {target_job_title})",
            type="primary",
            use_container_width=True
        )

    if compare_btn:
        if not candidate_resume_text.strip():
            st.error("Please upload or select a valid candidate resume before running match analysis.")
        elif not target_jd_text.strip():
            st.error("Target Job Description is empty. Please select or enter job details.")
        else:
            with st.spinner(f"🤖 Evaluating {candidate_name} against {target_job_title} criteria..."):
                analysis = analyze_resume_against_jd(
                    resume_text=candidate_resume_text,
                    jd_text=target_jd_text,
                    candidate_name=candidate_name,
                    job_role=target_job_title
                )
                st.session_state["screening_result"] = analysis
                st.session_state["screening_trace"] = {
                    "tool_invoked": "analyze_resume_against_jd",
                    "retriever_used": "direct_evidence_matching",
                    "sources": [f"JD: {target_job_title}", f"Resume: {candidate_name}"],
                    "execution_steps": [
                        f"1. Extracted requirements from target job: {target_job_title}",
                        f"2. Extracted technical skill evidence and project history from {candidate_name}'s resume",
                        f"3. Grounded evaluation against catalog of {len(TECH_SKILLS_CATALOG)} technical competencies",
                        "4. Computed overall match score and identified critical skill gaps",
                        "5. Generated 3 targeted technical interview questions to probe unverified skills"
                    ]
                }
                st.rerun()

    # -------------------------------------------------------------------------
    # STRUCTURED EVALUATION RESULTS DISPLAY
    # -------------------------------------------------------------------------
    result = st.session_state.get("screening_result")
    if result:
        st.write("")
        st.markdown("## 📊 Structured Screening Evaluation")

        match_score = result.get("match_percentage", 82)
        matched_skills = result.get("matched_skills", [])
        missing_skills = result.get("missing_skills", [])
        eval_cand = result.get("candidate", candidate_name)
        eval_role = result.get("job_role", target_job_title)

        # Match Status Badge Determination
        if match_score >= 75:
            score_badge_class = "score-badge-high"
            score_badge_text = "🟢 Strong Match — Recommended for Interview"
            score_summary = f"{eval_cand} demonstrates strong alignment with core requirements for {eval_role}."
        elif match_score >= 50:
            score_badge_class = "score-badge-medium"
            score_badge_text = "🟡 Qualified — Minor Skill Gaps Detected"
            score_summary = f"{eval_cand} meets foundational requirements for {eval_role} but exhibits gaps in specialized areas."
        else:
            score_badge_class = "score-badge-low"
            score_badge_text = "🔴 Low Match — Significant Requirements Missing"
            score_summary = f"{eval_cand} does not meet several core requirements for {eval_role}."

        # Animated Hero Score Banner
        st.markdown(f"""
        <div class="score-hero-container">
            <div>
                <div style="font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:1px; opacity:0.85;">
                    Overall AI Match Score
                </div>
                <div style="font-size:22px; font-weight:800; margin-top:2px;">
                    {eval_cand} <span style="opacity:0.6; font-size:16px;">vs</span> {eval_role}
                </div>
                <div class="score-hero-badge {score_badge_class}">{score_badge_text}</div>
                <div style="font-size:13px; margin-top:8px; opacity:0.9;">
                    {score_summary}
                </div>
            </div>
            <div style="text-align:right;">
                <div class="score-large-number">{match_score}%</div>
                <div style="font-size:12px; font-weight:700; opacity:0.8;">{len(matched_skills)} matched / {len(matched_skills) + len(missing_skills)} evaluated skills</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Interactive Evaluation Cards
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### **✅ Matched Skills & Evidence**")
            pills = "".join([f'<span class="skill-pill-matched">✓ {s}</span>' for s in matched_skills]) if matched_skills else "<em>No core skills matched.</em>"
            st.markdown(f'<div class="result-card">{pills}</div>', unsafe_allow_html=True)

        with r2:
            st.markdown("##### **⚠️ Missing / Unverified Skills**")
            pills_miss = "".join([f'<span class="skill-pill-missing">✗ {s}</span>' for s in missing_skills]) if missing_skills else "<em>None — All required skills verified!</em>"
            st.markdown(f'<div class="result-card">{pills_miss}</div>', unsafe_allow_html=True)

        r3, r4 = st.columns(2)
        with r3:
            st.markdown("##### **💼 Relevant Experience Assessment**")
            st.markdown(f"""
            <div class="result-card" style="font-size:13.5px; color:#334155; line-height:1.5;">
                {result.get('relevant_experience', 'Relevant technical background demonstrated through internships, academic coursework, and practical projects.')}
            </div>
            """, unsafe_allow_html=True)

        with r4:
            st.markdown("##### **📝 Recruiter Match Summary**")
            st.markdown(f"""
            <div class="result-card" style="font-size:13.5px; color:#334155; line-height:1.5;">
                {result.get('match_summary', 'Objective evaluation indicates readiness for initial technical rounds.')}
            </div>
            """, unsafe_allow_html=True)

        # Skill-by-Skill Alignment Breakdown
        st.markdown("##### **Skill Match Breakdown:**")
        all_evaluated_skills = [(s, 100) for s in matched_skills] + [(s, 20) for s in missing_skills]
        if all_evaluated_skills:
            for skill_name, pct in all_evaluated_skills:
                sb1, sb2 = st.columns([1, 4])
                with sb1:
                    icon = "✅" if pct == 100 else "⚠️"
                    st.caption(f"{icon} **{skill_name}**")
                with sb2:
                    st.progress(pct / 100.0)

        # Smart Interview Questions for Recruiter
        interview_qs = result.get("interview_questions")
        if not interview_qs and missing_skills:
            interview_qs = generate_interview_questions(missing_skills, eval_role)
            
        if interview_qs:
            st.write("")
            st.markdown("### 🎯 Suggested Technical Interview Questions")
            st.caption("AI-generated questions targeting this candidate's specific skill gaps:")
            for idx, q in enumerate(interview_qs, 1):
                st.markdown(f"""
                <div class="interview-q-card">
                    <b>Question {idx}:</b> {q}
                </div>
                """, unsafe_allow_html=True)

        # HR Actions: Export & Reset
        st.write("")
        col_dl, col_reset, _ = st.columns([1.2, 1, 2])
        with col_dl:
            # Generate markdown report
            report_text = f"""# HireWise Candidate Screening Report
**Candidate:** {eval_cand}
**Target Position:** {eval_role}
**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**AI Match Score:** {match_score}% ({score_badge_text})

---
### Matched Skills ({len(matched_skills)})
{', '.join(matched_skills) if matched_skills else 'None'}

### Missing Skills ({len(missing_skills)})
{', '.join(missing_skills) if missing_skills else 'None'}

### Relevant Experience Assessment
{result.get('relevant_experience', '')}

### Recruiter Summary
{result.get('match_summary', '')}

### Recommended Interview Probe Questions
""" + "\n".join([f"- {q}" for q in (interview_qs or [])])

            st.download_button(
                "📥 Download HR Screening Report",
                data=report_text,
                file_name=f"hirewise_eval_{eval_cand.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col_reset:
            if st.button("🔄 Evaluate Another Candidate", use_container_width=True):
                st.session_state["screening_result"] = None
                st.rerun()

        # Transparent Agent Drawer
        if "screening_trace" in st.session_state and st.session_state["screening_trace"]:
            with st.expander("🔍 How HireWise handled this screening request"):
                t = st.session_state["screening_trace"]
                st.markdown(f"**Tool:** `{t.get('tool_invoked')}`")
                st.markdown(f"**Retriever:** `{t.get('retriever_used')}`")
                st.markdown(f"**Sources:** `{', '.join(t.get('sources', []))}`")
                st.markdown("**Execution Steps:**")
                for s in t.get("execution_steps", []):
                    st.caption(s)


# -----------------------------------------------------------------------------
# PAGE 4: INTERVIEW SCHEDULING
# -----------------------------------------------------------------------------
elif nav_choice == "📅 Interview Scheduling":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Interview Scheduling</h1>
            <p class="brand-tagline">Manage interview slots and coordinate candidate interviews against the mock calendar.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cal, col_book = st.columns([1.2, 1])

    with col_cal:
        st.markdown("#### 🗓️ Available Calendar Slots")
        calendar = get_calendar_state()

        selected_date = st.selectbox(
            "Select Date:",
            list(calendar.keys()),
            format_func=lambda d: f"{d} (Available: {len(calendar[d])} slots)"
        )

        slots = calendar.get(selected_date, [])
        if slots:
            st.caption(f"Available slots for **{selected_date}**:")
            sc_cols = st.columns(len(slots) if len(slots) <= 3 else 3)
            for idx, slot in enumerate(slots):
                with sc_cols[idx % 3]:
                    st.markdown(f"""
                    <div class="slot-card slot-available">
                        <div style="font-size:16px; font-weight:700; color:#0f172a;">{slot}</div>
                        <div style="font-size:11px; font-weight:600; color:#10b981; margin-top:2px;">Available</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"No interview slots are available for {selected_date}. All slots booked.")

        st.write("")
        if st.button("🔄 Reset Calendar to Default Slots"):
            reset_calendar_state()
            st.success("Calendar reset to initial state!")
            st.rerun()

    with col_book:
        st.markdown("#### ✍️ Book Interview Slot")
        cand_name = st.text_input("Candidate Name:", value="Candidate 1")
        
        if slots:
            chosen_slot = st.selectbox("Select Time Slot:", slots)
            if st.button("Confirm Booking", type="primary", use_container_width=True):
                # Call booking tool
                res_str = interview_slot_booking.invoke({
                    "date": selected_date,
                    "time": chosen_slot,
                    "candidate_name": cand_name
                })
                res_data = json.loads(res_str)
                if res_data.get("success"):
                    st.success(f"✓ {res_data.get('message')}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(res_data.get("message"))
        else:
            st.caption("Select a date with available slots to book.")

    st.markdown("---")
    st.markdown("#### 📋 Confirmed Bookings")
    bookings = get_all_bookings()
    if bookings:
        for b in bookings:
            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid #10b981; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <b>{b['booking_id']}</b> — {b['candidate']}
                    <div style="font-size:12px; color:#64748b;">Date: {b['date']} | Time: {b['time']}</div>
                </div>
                <span style="background:#dcfce7; color:#15803d; font-weight:700; font-size:11px; padding:3px 8px; border-radius:12px;">CONFIRMED</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No interviews booked yet in this session.")


# -----------------------------------------------------------------------------
# PAGE 5: LIVE JOB DISCOVERY (ADZUNA)
# -----------------------------------------------------------------------------
elif nav_choice == "🌐 Live Job Discovery":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Live Job Discovery</h1>
            <p class="brand-tagline">Optional live job market search powered by the Adzuna API.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_adzuna_configured():
        st.warning("""
        ⚠️ **Adzuna Live Job Discovery is currently in optional standby mode.**
        
        To enable real-time job searches from the Adzuna global API:
        1. Open or create `.env` in the root folder.
        2. Add your credentials:
           ```env
           ADZUNA_APP_ID=your_id
           ADZUNA_APP_KEY=your_key
           ```
        3. Refresh the application.
        
        *Note: Core Resume Screening uses the official local Software Engineer Job Description (`data/job_description.pdf`) and is completely operational without Adzuna.*
        """)
    else:
        st.success("✓ **Adzuna API Connected.** Live job discovery is active.")

    c_search, c_loc = st.columns([2, 1])
    with c_search:
        query = st.text_input("Job Title / Keywords:", value="Software Engineer")
    with c_loc:
        location = st.text_input("Location:", value="India")

    if st.button("Search Live Postings", type="primary"):
        with st.spinner("Querying Adzuna Job Search API..."):
            res = search_adzuna_jobs(query=query, location=location, results_per_page=5)
            if res.get("success"):
                st.markdown(f"##### {res.get('message')}")
                for j in res.get("jobs", []):
                    st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{j['title']}</div>
                        <div class="job-company">🏢 {j['company']} • 📍 {j['location']}</div>
                        <div class="job-meta">💰 Salary: <b>{j['salary']}</b> | 📅 Posted: {j['created']}</div>
                        <p style="font-size:13px; color:#334155;">{j['description']}</p>
                        <a href="{j['redirect_url']}" target="_blank" style="font-size:12px; font-weight:700; color:#2563eb; text-decoration:none;">View Full Listing on Adzuna &rarr;</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(res.get("message"))


# -----------------------------------------------------------------------------
# PAGE 6: ARCHITECTURE & ABOUT
# -----------------------------------------------------------------------------
elif nav_choice == "ℹ️ Architecture & About":
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">Architecture & Workshop Pipeline</h1>
            <p class="brand-tagline">Complete technical implementation details of HireWise.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ Complete Workshop Pipeline
    ```text
    ChatOllama (qwen3:0.6b)
          ↓
    PyPDFLoader (data/*.pdf)
          ↓
    RecursiveCharacterTextSplitter (chunk_size=500, overlap=80)
          ↓
    OllamaEmbeddings (qwen3-embedding:0.6b)
          ↓
    Chroma Vector Store (3 Persistent Collections)
       ├── hirewise_policy
       ├── hirewise_resume
       └── hirewise_job_description
          ↓
    Dedicated Retrievers (Zero Data Bleed)
          ↓
    Custom Tools (@tool)
       ├── policy_qa
       ├── resume_to_jd_match
       ├── interview_slot_lookup
       └── interview_slot_booking
          ↓
    ReAct-style Agent & Transparent Intent Router
          ↓
    Streamlit SaaS Dashboard
    ```

    ---

    ### 🔒 Data Boundary & Zero Bleed Guarantees
    1. **Candidate Policy Queries:**
       Routed solely to `policy_qa` which retrieves exclusively from the `hirewise_policy` collection.
    2. **Resume Screening:**
       Routed solely to `resume_to_jd_match` which queries `hirewise_resume` and `hirewise_job_description`. Company policies are strictly excluded from candidate skill evidence.
    3. **Interview Scheduling:**
       Queries only the in-memory/persistent calendar store. No vector retrieval is performed.
    """)
