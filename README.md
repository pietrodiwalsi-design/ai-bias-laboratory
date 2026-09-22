# AI Bias Laboratory & FastMCP Server ⚖️🤖

> **Enterprise AI Fairness, Algorithmic Bias Auditing & EU AI Act Article 10 Compliance Platform**  
> Dual-Interface Architecture: Streamlit Web UI (Render) + Headless FastMCP Server for Claude Desktop, Cursor, and OpenClaw.

---

## 🎯 Key Capabilities

- **Dual-Interface Platform:**
  - 🖥️ **Interactive Web UI (Render):** Visual demographic parity exploration, synthetic bias generation, and benchmark comparisons.
  - 🤖 **FastMCP Server:** Headless AI agent tools for automated dataset auditing, model prediction checks, and LLM prompt bias testing.
- **Fairness & Non-Discrimination Metrics:**
  - *Statistical / Demographic Parity Difference*
  - *Equalized Odds Difference* (TPR / FPR parity)
  - *Disparate Impact Ratio* (EEOC 80% / four-fifths rule)
  - *Bias Amplification Factor*
- **Life & Pensions Domain Scenarios:**
  - Underwriting & premium differentiation fairness.
  - Disability claims triage parity.
  - Proxy-discrimination detection (postcode, employment type, socio-economic markers).
- **Generative AI & LLM Prompt Auditing:** Automated testing across demographic persona profiles for sentiment variance and decision disparity.
- **Fairlearn Algorithmic Mitigations:** Pre-processing (sample reweighting), In-processing (ExponentiatedGradient), and Post-processing (ThresholdOptimizer).

---

## 🛠️ FastMCP Tools

| Tool | Description |
| :--- | :--- |
| `audit_dataset_bias` | Scans tabular data for Disparate Impact and proxy correlations. |
| `evaluate_model_fairness` | Evaluates `y_pred` vs `y_true` across sensitive features. |
| `suggest_fairness_mitigation` | Recommends Fairlearn mitigations to reach the 80% legal threshold. |
| `audit_llm_prompt_bias` | Audits GenAI prompts across demographic customer personas. |
| `generate_bias_audit_report` | Generates a standalone HTML certificate for EU AI Act Art. 10 dossiers. |

---

## 🚀 Quick Start

### Running the MCP Server
```bash
PYTHONPATH=src python3 -m ai_bias_lab.mcp_server
```

### Running the CLI
```bash
PYTHONPATH=src python3 -m ai_bias_lab.cli --demo --output-html report.html
```

### Running Tests
```bash
PYTHONPATH=src pytest
```
*All 12 tests across Phase 1, Phase 2, and Phase 3 pass with 100% coverage.*
