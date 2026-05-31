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

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Models Audited", "47", "+3 this week")
with col2:
    st.metric("Avg Bias Score", "0.31", "-0.04")
with col3:
    st.metric("High-Risk Findings", "12", "3 new")
with col4:
    st.metric("Mitigations Applied", "29", "+7")

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

# Recent Audits
st.subheader("Recent Audits")
recent = pd.DataFrame({
    "Model": ["Hiring-v3", "Claims-Scoring", "Fraud-Detect", "Underwriting-2"],
    "Protected Attr": ["gender", "age", "race", "gender"],
    "Bias Score": [0.28, 0.41, 0.19, 0.33],
    "Status": ["Mitigated", "In Review", "Pass", "Mitigated"]
})
st.dataframe(recent, use_container_width=True, hide_index=True)

st.divider()
st.caption("Quick Actions → Use sidebar to navigate to Bias Generator, Benchmarks, or Fairness Metrics")