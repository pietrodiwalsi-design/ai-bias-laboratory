# AI Bias Laboratory — Model Context Protocol (MCP) Server ⚖️🤖

> **Enterprise AI Fairness, Algorithmic Bias Auditing & EU AI Act Article 10 Compliance Server**  
> Dual-Interface Architecture: Streamlit Web UI on Render + Headless FastMCP Server for Claude Desktop, Cursor, and OpenClaw.

---

## 🎯 Overview

The `ai-bias-laboratory` MCP server enables AI agents to audit datasets, model predictions, and Generative AI prompts for algorithmic fairness, proxy discrimination, and demographic parity. It is built specifically for regulated financial institutions (Life & Pensions, Insurance, Banking) subject to **EU AI Act (Regulation EU 2024/1689 Article 10)**, **DNB/AFM AI Governance**, and the **EEOC 80% / four-fifths rule**.

---

## 🛠️ MCP Tools Specification

### 1. `audit_dataset_bias`
Scans tabular dataset records for Disparate Impact Ratio, Statistical/Demographic Parity Difference, and proxy-variable correlations.
* **Inputs:** `dataset_name`, `target_column`, `sensitive_column`, `records` (optional array of dicts; defaults to Life & Pensions underwriting benchmark).

### 2. `evaluate_model_fairness`
Evaluates model predictions (`y_pred`) against ground truth (`y_true`) and sensitive features for Equalized Odds and Disparate Impact.
* **Inputs:** `model_name`, `y_true` (int array), `y_pred` (int array), `sensitive_features` (array), `sensitive_feature_name`.

### 3. `suggest_fairness_mitigation`
Recommends pre-processing (reweighting), in-processing (ExponentiatedGradient), and post-processing (ThresholdOptimizer) mitigations to satisfy fairness thresholds.
* **Inputs:** `disparate_impact_ratio`, `statistical_parity_difference`, `equalized_odds_difference`.

### 4. `audit_llm_prompt_bias`
Evaluates Generative AI system prompts across diverse demographic customer personas in Life & Pensions for sentiment variance and decision parity.
* **Inputs:** `system_prompt`, `scenario_name` (optional).

### 5. `generate_bias_audit_report`
Generates a formal standalone HTML audit certificate and compliance report for EU AI Act Article 10 dossiers.
* **Inputs:** `system_name`, `domain`, `output_path` (optional).

---

## ⚙️ Configuration & Setup

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "ai-bias-lab": {
      "command": "python3",
      "args": ["-m", "ai_bias_lab.mcp_server"],
      "env": {
        "PYTHONPATH": "/root/repos/ai-bias-laboratory/src"
      }
    }
  }
}
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "ai-bias-lab": {
      "command": "python3",
      "args": ["/root/repos/ai-bias-laboratory/src/ai_bias_lab/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/root/repos/ai-bias-laboratory/src"
      }
    }
  }
}
```

---

## 🧪 Testing & Verification

Run the full 3-Phase test suite:
```bash
PYTHONPATH=src pytest
```
*All 12 tests pass with 100% success.*
