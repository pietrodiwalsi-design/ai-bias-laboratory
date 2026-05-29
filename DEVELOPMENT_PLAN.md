# AI Bias Laboratory - Development Plan

**Project:** AI Bias Laboratory  
**Repository:** https://github.com/pietrodiwalsi-design/ai-bias-laboratory (private)  
**Created:** 2026-05-29  
**Owner:** Peter Van Walsem / Moshe

---

## Executive Summary

The AI Bias Laboratory is a comprehensive platform for testing, measuring, and mitigating bias in AI systems while ensuring compliance with the EU AI Act and other regulatory frameworks. It combines synthetic bias generation, standardized fairness benchmarks, propagation tracking, and mitigation simulation in an interactive "bias laboratory" environment.

---

## Core Features (from Requirements)

### Functional Requirements

| # | Requirement | Priority | Phase |
|---|-------------|----------|-------|
| F1 | Synthetic Bias Generation (5 protected attributes) | High | 1 |
| F2 | Standardized Benchmark Runner (BBQ, CrowS-Pairs, WinoBias, custom) | High | 1 |
| F3 | Bias Propagation Tracking (visual pipeline mapping) | High | 2 |
| F4 | Before/After Mitigation Simulation (4 techniques) | High | 2 |
| F5 | Automated Fairness Metric Calculation | High | 1 |
| F6 | Bias Amplification Factor + Mitigation Effectiveness % | Medium | 2 |
| F7 | Automated Audit Reporting (PDF/JSON) | High | 3 |
| F8 | Explainability & Feature Attribution (LIME/SHAP) | Medium | 3 |
| F9 | Automated Risk Classification (EU AI Act) | High | 2 |
| F10 | Data Input Configuration (reference groups, thresholds) | Medium | 1 |
| F11 | Adversarial Testing for Systemic Risk (GPAI) | Medium | 3 |
| F12 | Algorithmic Impact Assessment (AIA) Workflow | Medium | 3 |

### Non-Functional Requirements

| # | Requirement | Priority | Phase |
|---|-------------|----------|-------|
| NF1 | Rapid Iteration UI (Streamlit/Gradio) | High | 1 |
| NF2 | Library Interoperability (HF, Fairlearn, Aequitas) | High | 1 |
| NF3 | SaaS Monetization Architecture (multi-tenant) | Medium | 4 |
| NF4 | Data Privacy & Confidentiality (GDPR) | High | 2 |
| NF5 | Reproducibility & Statistical Rigor | High | 1 |
| NF6 | Interdisciplinary Readability (legal/compliance stakeholders) | Medium | 2 |
| NF7 | Regulatory Evolvability | Medium | Ongoing |
| NF8 | Security & Robustness | High | 2 |

---

## Development Phases

### Phase 1: Core Bias Laboratory Foundation (Weeks 1-4)

**Goal:** MVP with synthetic bias injection, benchmark execution, and basic metrics

**Deliverables:**
- Streamlit/Gradio UI for rapid iteration
- Synthetic bias generator (gender, race, age, religion, disability)
- Benchmark runner: BBQ, CrowS-Pairs, WinoBias
- Fairness metrics: statistical parity, equalized odds, disparate impact ratio
- Hugging Face + Fairlearn integration
- Reproducible results with seeded experiments

**Tech Stack:**
- Frontend: Streamlit (primary) or Gradio
- Backend: Python 3.11+
- Core libs: `transformers`, `datasets`, `fairlearn`, `aequitas`

**Milestones:**
- Week 1: Project scaffolding + UI skeleton
- Week 2: Synthetic bias generator + benchmark runner
- Week 3: Metrics calculation + reproducibility layer
- Week 4: Polish + internal testing

---

### Phase 2: Propagation, Mitigation & Compliance (Weeks 5-8)

**Goal:** Visual bias tracking, mitigation comparison, EU AI Act risk classification

**Deliverables:**
- Bias propagation visualizer (training → embeddings → output)
- 4 mitigation techniques side-by-side:
  - Adversarial debiasing
  - Counterfactual data augmentation
  - RLHF-style preference tuning
  - RAG-level filtering
- Bias amplification factor + mitigation effectiveness %
- EU AI Act automated risk classifier (Prohibited/High/Limited/Minimal)
- GDPR-compliant data handling + differential privacy options
- Audit-ready visualization for non-technical stakeholders

**Tech Stack Additions:**
- Visualization: Plotly, NetworkX (for propagation graphs)
- XAI: `shap`, `lime`
- EU AI Act logic: rule-based classifier + documentation

---

### Phase 3: Reporting, Explainability & Advanced Testing (Weeks 9-12)

**Goal:** Professional audit outputs and GPAI adversarial capabilities

**Deliverables:**
- PDF/JSON audit report generator
- LIME/SHAP explainability dashboard
- Custom template builder for benchmarks
- Adversarial red-teaming module (GPAI Code of Practice)
- Algorithmic Impact Assessment (AIA) workflow (NIST-aligned)
- Data input configuration panel (reference groups, thresholds)

**Tech Stack Additions:**
- Report generation: `reportlab`, `fpdf2`, or WeasyPrint
- Export: JSON schema validation

---

### Phase 4: SaaS Platform & Enterprise Readiness (Weeks 13-16)

**Goal:** Multi-tenant architecture, billing, enterprise security

**Deliverables:**
- Multi-tenant isolation (user/workspace separation)
- Usage-based billing integration
- Per-model audit pricing
- Enterprise SSO + audit logging
- Model registry integration
- Security hardening + penetration testing

**Tech Stack Additions:**
- Auth: Auth0 / Clerk / Supabase
- Billing: Stripe
- Backend: FastAPI (migrate from Streamlit for scale)
- Deployment: Docker + Kubernetes

---

## Repository Structure

```
ai-bias-laboratory/
├── README.md
├── DEVELOPMENT_PLAN.md
├── REQUIREMENTS.md
├── docs/
│   ├── eu-ai-act-mapping.md
│   ├── fairness-metrics-guide.md
│   └── mitigation-techniques.md
├── src/
│   ├── bias_lab/
│   │   ├── generators/
│   │   ├── benchmarks/
│   │   ├── metrics/
│   │   ├── mitigation/
│   │   ├── propagation/
│   │   ├── reporting/
│   │   └── ui/
│   └── cli.py
├── tests/
├── examples/
├── notebooks/
└── pyproject.toml
```

---

## Key Technical Decisions

1. **UI Framework:** Start with Streamlit for speed (NF1), migrate to FastAPI + React for SaaS phase
2. **Reproducibility:** All experiments must accept `random_seed` and log full config
3. **Model Access:** Support both local HF models and API-based models (OpenAI, Anthropic, etc.)
4. **Data Policy:** Never store user data by default; offer opt-in encrypted storage
5. **Regulatory Updates:** All EU AI Act rules stored in versioned JSON/YAML files

---

## Success Metrics

- Phase 1: Users can run BBQ benchmark + see statistical parity in <5 minutes
- Phase 2: Mitigation comparison shows clear before/after effect with visual proof
- Phase 3: Exportable PDF report passes internal legal review
- Phase 4: First paying enterprise customer completes full audit workflow

---

## Next Immediate Steps

1. ✅ Repository created (private)
2. Initialize Python project with `pyproject.toml`
3. Set up pre-commit hooks + linting
4. Create initial Streamlit skeleton
5. Implement synthetic bias generator (gender first)

---

*Document maintained by Moshe. Update after each phase completion.*