"""
Phase 1 Unit Tests for AI Bias Laboratory — Core Fairness Metrics & Pension Scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from ai_bias_lab.engine import FairnessAuditEngine
from ai_bias_lab.pension_scenarios import generate_pension_underwriting_dataset
from ai_bias_lab.models import ComplianceStatus


@pytest.fixture
def engine():
    return FairnessAuditEngine(random_seed=42)


def test_fairness_metrics_perfect_parity(engine):
    # NOTE (2026-09-18 remediation brief, F2): these 8-row arrays are far
    # below the minimum subgroup sample size (30), so per F2 the verdict
    # must now be INSUFFICIENT_DATA regardless of how clean the ratios look
    # -- a COMPLIANT verdict on n=4-per-group must never be indistinguishable
    # from one backed by real sample sizes. The underlying ratios (DI/SPD)
    # are unchanged and still asserted below; only the verdict status field
    # changed, which is the explicit intent of F2, not a regression.
    # Equal selection rates (50% approved for both groups)
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sens = np.array(["M", "M", "M", "M", "F", "F", "F", "F"])

    metrics = engine.calculate_fairness_metrics(y_true, y_pred, sens)
    assert metrics.disparate_impact_ratio == 1.0
    assert metrics.statistical_parity_difference == 0.0
    assert metrics.disparate_impact_status == ComplianceStatus.INSUFFICIENT_DATA
    assert metrics.eu_ai_act_art10_status == ComplianceStatus.INSUFFICIENT_DATA
    assert any("Insufficient data" in w for w in metrics.warnings)


def test_fairness_metrics_severe_disparity(engine):
    # NOTE (2026-09-18 remediation brief, F2): same rationale as above -- 10
    # rows total is well below the n>=30 minimum, so INSUFFICIENT_DATA now
    # takes precedence over NON_COMPLIANT. See test_bias_phase2/F2 tests for
    # coverage of the >=30-sample case still reaching a real verdict.
    # Group A: 100% approved, Group B: 20% approved -> DI = 0.20 (Violates 80% rule)
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    sens = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

    metrics = engine.calculate_fairness_metrics(y_true, y_pred, sens)
    assert metrics.disparate_impact_ratio == 0.20
    assert metrics.statistical_parity_difference == 0.80
    assert metrics.disparate_impact_status == ComplianceStatus.INSUFFICIENT_DATA
    assert metrics.eu_ai_act_art10_status == ComplianceStatus.INSUFFICIENT_DATA


def test_pension_underwriting_scenario_generation():
    df = generate_pension_underwriting_dataset(n_samples=500, inject_proxy_bias=True, seed=42)
    assert len(df) == 500
    assert "gender" in df.columns
    assert "postcode_cluster" in df.columns
    assert "annual_income" in df.columns
    assert "y_true_eligibility" in df.columns
    assert "y_pred_approval" in df.columns


def test_proxy_correlation_detection(engine):
    # NOTE (2026-09-18 remediation brief, F4): detect_proxy_correlations now
    # returns a ProxyCorrelationResult, not a bare dict, so status/method
    # can be distinguished from an empty result. This also demonstrates the
    # core F4 fix: 'gender' is categorical and previously always returned {}
    # (silently skipped), but now correctly computes and finds the injected
    # proxy bias via Cramer's V / correlation ratio.
    df = generate_pension_underwriting_dataset(n_samples=500, inject_proxy_bias=True)
    result = engine.detect_proxy_correlations(df, sensitive_column="gender", threshold=0.10)
    assert result.status == "computed"
    assert result.method == "cramers_v_and_correlation_ratio"
    assert isinstance(result.correlations, dict)
    assert len(result.correlations) > 0  # gender is no longer silently skipped
