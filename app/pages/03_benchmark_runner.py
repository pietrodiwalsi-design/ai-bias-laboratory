"""Benchmark Runner Page — BBQ / CrowS-Pairs / WinoBias."""
import streamlit as st
import pandas as pd
import plotly.express as px
from app.core.benchmarks import run_all_benchmarks

st.set_page_config(page_title="Benchmark Runner", layout="wide")
st.title("📊 Benchmark Runner")
st.caption("Lightweight implementations of standard AI bias benchmarks")

benchmarks = st.multiselect("Select Benchmarks", ["BBQ", "CrowS-Pairs", "WinoBias"], default=["BBQ", "CrowS-Pairs", "WinoBias"])
bias_level = st.slider("Simulated Bias Level", 0.0, 0.6, 0.25, 0.05)

if st.button("▶️ Run Benchmarks", type="primary"):
    results = run_all_benchmarks(bias_level)
    st.session_state["benchmark_results"] = results

if "benchmark_results" in st.session_state:
    res_df = pd.DataFrame(st.session_state["benchmark_results"])
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Bar chart
    fig = px.bar(res_df, x="benchmark", y="score", color="pass_fail",
                 color_discrete_map={"PASS": "#3fb950", "FAIL": "#f85149"},
                 title="Benchmark Scores vs Thresholds")
    fig.add_hline(y=0.65, line_dash="dash", line_color="#d29922", annotation_text="Typical Threshold")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select benchmarks and click Run to execute")