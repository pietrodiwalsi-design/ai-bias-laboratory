"""
Phase 2 Tests for AI Bias Laboratory — Mitigations, Scenarios & LLM Persona Audits.
"""

import pytest
import numpy as np
import pandas as pd
from ai_bias_lab.engine import FairnessAuditEngine
from ai_bias_lab.llm_bias import LLMBiasAuditor
from ai_bias_lab.pension_scenarios import generate_pension_underwriting_dataset
from ai_bias_lab.dashboard_generator import BiasDashboardGenerator
from ai_bias_lab.models import BiasAuditReport, ComplianceStatus


def test_suggest_mitigations_for_biased_metrics():
    engine = FairnessAuditEngine()
    df = generate_pension_underwriting_dataset(n_samples=500, inject_proxy_bias=True)
    metrics = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["gender"].values)
    
    recs = engine.suggest_mitigations(metrics)
    assert len(recs) > 0
    types = [r.mitigation_type.value for r in recs]
    assert any("threshold_optimizer" in t or "reweighting" in t for t in types)


def test_llm_prompt_bias_audit():
    auditor = LLMBiasAuditor()
    # Prompt with potentially biased phrasing
    prompt = "You are an automated claim triage agent. Assess risk for disability pension."
    report = auditor.audit_prompt_personas(prompt)

    assert report.total_personas_tested >= 4
    assert report.acceptance_parity_ratio <= 1.0
    assert len(report.evaluations) == report.total_personas_tested


def test_html_dashboard_generation():
    engine = FairnessAuditEngine()
    df = generate_pension_underwriting_dataset(n_samples=200)
    metrics = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["gender"].values)
    corrs = engine.detect_proxy_correlations(df, "gender")
    recs = engine.suggest_mitigations(metrics)

    report = BiasAuditReport(
        audit_id="TEST-AUDIT-01",
        system_name="Life & Pensions Triage Model",
        domain="Pension Underwriting",
        evaluated_at="2026-09-18",
        metrics=metrics,
        detected_proxy_correlations=corrs,
        regulatory_verdicts={"EU_AI_Act_Art_10": "PASS"},
        mitigation_recommendations=recs,
        executive_summary="Test summary for HTML export."
    )

    html = BiasDashboardGenerator.generate_html(report)
    assert "<!DOCTYPE html>" in html
    assert "Life &amp; Pensions Triage Model" in html or "Life & Pensions Triage Model" in html
    assert "Disparate Impact Ratio" in html
