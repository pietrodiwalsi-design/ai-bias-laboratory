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
        errors = []
        for attr in protected_attrs:
            try:
                biased_df, meta = inject_bias(df, attr, bias_intensity, injection_method, seed)
                results[attr] = {"df": biased_df, "meta": meta}
            except ValueError as e:
                # 2026-09-19 webapp-parity remediation: an unsupported attribute
                # (e.g. race/disability, selectable in the UI but with no
                # injection logic behind them) must surface as a visible error,
                # never as a silently unchanged "no bias found" result.
                errors.append(str(e))
        st.session_state["bias_results"] = results
        st.session_state["bias_errors"] = errors
        if results:
            st.success(f"Bias injection complete for: {', '.join(results.keys())}")
        if errors:
            for err in errors:
                st.error(err)

with col2:
    if "bias_results" in st.session_state and st.session_state["bias_results"]:
        for attr, res in st.session_state["bias_results"].items():
            st.subheader(f"Results: {attr}")
            meta = res["meta"]
            if meta.get("warnings"):
                for w in meta["warnings"]:
                    st.warning(w)
            st.json(meta)
            st.dataframe(res["df"].head(8), use_container_width=True)
    elif st.session_state.get("bias_errors"):
        st.info("No attributes were successfully processed — see errors on the left.")
    else:
        st.info("Run an injection test to see before/after statistics and biased dataset preview")