# AI Bias Laboratory - Requirements

Source: Functional_and_non_functional_requirements_for_System---be4bf613-e697-4be7-a3b1-fdd3744648ea.docx

## Functional Requirements

1. **Synthetic Bias Generation**  
   LLM-based synthetic data generator with controllable bias levels (mild → severe) across 5 protected attributes: gender, race/ethnicity, age, religion, disability.

2. **Standardized Benchmark Runner**  
   Execute BBQ, CrowS-Pairs, WinoBias + user-defined custom templates.

3. **Bias Propagation Tracking**  
   Visual mapping of bias flow through the full AI pipeline (training data → embeddings → output distribution).

4. **Before/After Mitigation Simulation**  
   Side-by-side comparison of 4 techniques:
   - Adversarial debiasing
   - Counterfactual data augmentation
   - RLHF-style preference tuning
   - RAG-level filtering

5. **Automated Fairness Metric Calculation**  
   Statistical parity difference, equalized odds, disparate impact ratio (via Fairlearn + Aequitas).

6. **Bias Amplification Factor**  
   Calculate (output bias ÷ input bias) + overall mitigation effectiveness percentage.

7. **Automated Audit Reporting**  
   Exportable PDF + JSON reports for compliance and regulatory submission.

8. **Explainability & Feature Attribution**  
   LIME / SHAP integration to identify drivers of biased outputs.

9. **Automated Risk Classification (EU AI Act)**  
   Classify use-case into Prohibited / High-Risk / Limited-Risk / Minimal-Risk.

10. **Data Input Configuration**  
    Define reference groups, score thresholds, protected attribute columns.

11. **Adversarial Testing for Systemic Risk**  
    GPAI Code of Practice red-teaming capabilities.

12. **Algorithmic Impact Assessment (AIA) Workflow**  
    NIST AI RMF aligned process for documenting purpose, limitations, and community impact.

## Non-Functional Requirements

1. **Rapid Iteration UI** — Streamlit or Gradio for fast experimentation.
2. **Library Interoperability** — Native Hugging Face, Fairlearn, Aequitas integration.
3. **SaaS Monetization Architecture** — Multi-tenant, usage-based billing, enterprise audit pricing.
4. **Data Privacy & Confidentiality** — GDPR compliant, differential privacy support.
5. **Reproducibility & Statistical Rigor** — Seeded experiments, full audit trail.
6. **Interdisciplinary Readability** — Visuals accessible to legal/compliance stakeholders.
7. **Regulatory Evolvability** — Easy updates for CEN-CENELEC standards and GPAI Code of Practice.
8. **Security & Robustness** — Protected testing environment against adversarial manipulation.