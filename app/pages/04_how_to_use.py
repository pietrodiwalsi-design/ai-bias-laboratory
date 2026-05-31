"""How To Use — Step-by-step guide for the AI Bias Laboratory."""
import streamlit as st

st.set_page_config(page_title="How To Use", page_icon="📖", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3, h4 { color: #c9d1d9; }
    .stDivider { border-color: #30363d; }
    .step-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #7c5cbf;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .tip-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 12px;
    }
    .warn-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #f0883e;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 How To Use the AI Bias Laboratory")
st.caption("A plain-language step-by-step guide for auditors, risk managers, and AI teams")

st.divider()

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
st.subheader("🔍 What is this tool?")
st.markdown("""
The **AI Bias Laboratory** helps you detect, measure, and reduce unfair behaviour in AI models.

It answers questions like:
- *Is our hiring model treating men and women equally?*
- *Does our fraud-detection system unfairly flag certain age groups?*
- *How biased is our model compared to industry benchmarks?*

You do **not** need to be a data scientist to use it. Just follow the steps below.
""")

st.divider()

# ── NAVIGATION ───────────────────────────────────────────────────────────────
st.subheader("🗺️ Navigating the tool")
st.markdown("""
Use the **sidebar on the left** to move between pages:

| Page | What it does |
|---|---|
| 🔬 **Main / Dashboard** | Overview of recent audits and key metrics |
| 🧪 **Bias Generator** | Create test datasets with controlled bias |
| 📊 **Benchmark Runner** | Run standard industry bias tests |
| 📖 **How To Use** | This guide |
""")

st.divider()

# ── STEP BY STEP ─────────────────────────────────────────────────────────────
st.subheader("📋 Step-by-step walkthrough")

st.markdown('<div class="step-box"><b>Step 1 — Start on the Dashboard</b><br><br>'
    'Open the <b>Main / Dashboard</b> page from the sidebar. '
    'Here you see a summary of all recent model audits, average bias scores, and high-risk findings. '
    'Use this page to get a quick overview before diving into details.</div>', unsafe_allow_html=True)

st.markdown('<div class="step-box"><b>Step 2 — Generate a test dataset</b><br><br>'
    'Go to <b>🧪 Bias Generator</b>.<br>'
    '1. Choose a <b>protected attribute</b> (e.g. gender, age, race).<br>'
    '2. Set how much bias you want to inject using the slider (0 = no bias, 1 = maximum bias).<br>'
    '3. Choose the <b>sample size</b> — how many records to generate.<br>'
    '4. Click <b>Generate Dataset</b>.<br>'
    '5. Download the CSV if you want to use it elsewhere.</div>', unsafe_allow_html=True)

st.markdown('<div class="step-box"><b>Step 3 — Run benchmark tests</b><br><br>'
    'Go to <b>📊 Benchmark Runner</b>.<br>'
    '1. Select which benchmarks to run:<br>'
    '&nbsp;&nbsp;&nbsp;&nbsp;• <b>BBQ</b> — tests bias in question-answering scenarios<br>'
    '&nbsp;&nbsp;&nbsp;&nbsp;• <b>CrowS-Pairs</b> — detects stereotyping in model predictions<br>'
    '&nbsp;&nbsp;&nbsp;&nbsp;• <b>WinoBias</b> — checks gender bias in job-role associations<br>'
    '2. Click <b>Run Benchmarks</b>.<br>'
    '3. Review the results — a score closer to <b>0.5 is fair</b>; far from 0.5 means bias detected.</div>', unsafe_allow_html=True)

st.markdown('<div class="step-box"><b>Step 4 — Interpret the results</b><br><br>'
    'After running benchmarks, look at:<br>'
    '• <b>Bias Score</b> — a number between 0 and 1. Below 0.2 = low risk. Above 0.4 = action needed.<br>'
    '• <b>Protected Attribute</b> — which group is being affected (e.g. women, age 60+).<br>'
    '• <b>Status</b> — Pass / In Review / Mitigated.<br><br>'
    'If a model shows high bias, escalate for review before deploying it in production.</div>', unsafe_allow_html=True)

st.markdown('<div class="step-box"><b>Step 5 — Document your findings</b><br><br>'
    'Use the audit results to:<br>'
    '• Record the model name, test date, bias score, and affected attribute.<br>'
    '• Attach findings to your internal risk register.<br>'
    '• Set a re-audit date (recommended every 3–6 months for high-risk models).<br><br>'
    'Tip: export tables using the download icon on any data table.</div>', unsafe_allow_html=True)

st.divider()

# ── TIPS ─────────────────────────────────────────────────────────────────────
st.subheader("💡 Tips & good practices")

st.markdown('<div class="tip-box">💡 <b>Always test before deploying.</b> Run benchmarks on every new model version, not just once at the start.</div>', unsafe_allow_html=True)
st.markdown('<div class="tip-box">💡 <b>Use multiple protected attributes.</b> A model can be fair on gender but biased on age. Test both.</div>', unsafe_allow_html=True)
st.markdown('<div class="tip-box">💡 <b>Bias score of 0 is not always the goal.</b> Some difference is statistically expected. Focus on scores above 0.3.</div>', unsafe_allow_html=True)
st.markdown('<div class="warn-box">⚠️ <b>This tool uses synthetic/sample data by default.</b> For real audits, connect your own dataset or model output. Contact your IT team to integrate real data.</div>', unsafe_allow_html=True)

st.divider()

# ── FAQ ──────────────────────────────────────────────────────────────────────
st.subheader("❓ Frequently asked questions")

with st.expander("What is a 'protected attribute'?"):
    st.write("A protected attribute is a characteristic that should not influence a model's decision — such as gender, age, race, nationality, or disability. Laws like the EU AI Act require that AI models do not discriminate based on these attributes.")

with st.expander("What does a bias score of 0.41 mean?"):
    st.write("A bias score of 0.41 means the model shows a meaningful difference in outcomes between groups. For example, one gender might be approved for a loan 41% more often than another. Anything above 0.3 should be reviewed. Above 0.5 is high risk and should block deployment.")

with st.expander("Who should use this tool?"):
    st.write("AI Risk Managers, Model Owners, Compliance Officers, and Data Scientists. No coding required for the basic audit workflow.")

with st.expander("How often should I run audits?"):
    st.write("At minimum: before any model goes live, and every 3–6 months after that. For high-risk models (hiring, credit, healthcare), monthly audits are recommended.")

with st.expander("Can I upload my own data?"):
    st.write("Phase 1 uses built-in sample data. Real data integration is planned for Phase 2. Contact your IT team or the tool owner to request this feature.")

st.divider()
st.caption("AI Bias Laboratory — Internal Risk Tool | Questions? Contact your IT Risk team")
