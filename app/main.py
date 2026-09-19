"""AI Bias Laboratory - Main Streamlit Entry Point (Phase 1)

Dark theme matching internal risk tooling aesthetic.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.utils import load_sample_data

st.set_page_config(
    page_title="AI Bias Laboratory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #c9d1d9; font-size: 1.6rem; }
    .stButton button { background-color: #7c5cbf; color: white; border: none; }
    .stButton button:hover { background-color: #9a7cd4; }
    .stDataFrame { background-color: #161b22; }
    h1, h2, h3 { color: #c9d1d9; }
    .stSelectbox, .stSlider, .stTextInput { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 AI Bias Laboratory")
st.caption("Internal Risk & Fairness Auditing Platform — Phase 1")

# 2026-09-19 webapp-parity remediation: these four KPI cards previously
# showed hardcoded fake numbers ("47 models audited", "+3 this week", etc.)
# with zero connection to any real audit activity -- a viewer had no way to
# tell this from a live production metric. Mirrors the same
# fabricated-evidence pattern already fixed (F3/F9) in the MCP server on
# 2026-09-18. Replaced with an explicit demo-mode banner instead of inventing
# a "real-looking" number with no backing data.
st.info(
    "🧪 **Demo mode** — this dashboard shell is not yet wired to a real audit "
    "log. Use the sidebar (Bias Generator / Benchmark Runner) to run a live "
    "computation on the sample dataset; every result there is clearly labeled "
    "with its sample size, whether it's simulated, and any statistical caveats."
)

st.divider()

# Bias by Protected Attribute
st.subheader("Bias Distribution by Protected Attribute")
sample = load_sample_data()
fig = px.bar(
    sample.groupby("gender")["hired"].mean().reset_index(),
    x="gender", y="hired", color="gender",
    color_discrete_sequence=["#7c5cbf", "#58a6ff"],
    labels={"hired": "Hire Rate"}
)
fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Quick Actions → Use sidebar to navigate to Bias Generator, Benchmarks, or Fairness Metrics")