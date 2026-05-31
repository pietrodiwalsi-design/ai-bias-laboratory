"""Bias Generator Page — Synthetic Bias Injection UI."""
import streamlit as st
import pandas as pd
from core.bias_generator import inject_bias
from core.utils import load_sample_data

st.set_page_config(page_title="Bias Generator", layout="wide")
st.title("🧪 Synthetic Bias Generator")
st.caption("Inject controlled bias into datasets for testing and validation")

col1, col2 = st.columns([1, 2])

with col1:
    model_name = st.text_input("Model Name", value="Hiring-Model-v3")
    model_type = st.selectbox("Model Type", ["Classifier", "Regressor", "Ranker"])
    protected_attrs = st.multiselect("Protected Attributes", ["gender", "age", "race", "disability"], default=["gender"])
    bias_intensity = st.slider("Bias Intensity", 0.0, 1.0, 0.35, 0.05)
    injection_method = st.radio("Injection Method", ["label_flip", "feature_skew", "sampling_bias"])
    seed = st.number_input("Random Seed", value=42, step=1)

    if st.button("🚀 Run Injection Test", type="primary"):
        df = load_sample_data()
        results = {}
        for attr in protected_attrs:
            biased_df, meta = inject_bias(df, attr, bias_intensity, injection_method, seed)
            results[attr] = {"df": biased_df, "meta": meta}
        st.session_state["bias_results"] = results
        st.success("Bias injection complete")

with col2:
    if "bias_results" in st.session_state:
        for attr, res in st.session_state["bias_results"].items():
            st.subheader(f"Results: {attr}")
            meta = res["meta"]
            st.json(meta)
            st.dataframe(res["df"].head(8), use_container_width=True)
    else:
        st.info("Run an injection test to see before/after statistics and biased dataset preview")