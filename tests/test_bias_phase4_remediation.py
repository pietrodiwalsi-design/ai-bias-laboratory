"""
Phase 4 Regression Tests — Round 2 Remediation (2026-09-22).

Covers the findings listed as "Open" in
ai-bias-audit-programme/docs/build-brief-python.md §8:
  O1  audit_llm_prompt_bias ignoring system_prompt (most serious finding)
  N5  audit trail null / bootstrap not reproducible
  N2  SPD confidence interval one-sided (direction lost)
  N7  equalized odds fabricated as 1.0 on an undefined subgroup rate
  N1  reference_group == comparison_group on a tie
  N4  bias_amplification_factor sentinel-like values (100.0 vs 1.0)
  N6  Art. 10 attribution left in mitigation strings
  O5  proxy detection for categorical attributes

Probe inputs reproduce exactly the four cases from
docs/bias-lab-round2-findings.md (copied read-only into
ai-bias-audit-programme) so the "before" values in that document are the
regression baseline this file guards against reintroducing.
"""

import numpy as np
import pytest

from ai_bias_lab.engine import FairnessAuditEngine, BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY
from ai_bias_lab.llm_bias import LLMBiasAuditor, _scan_prompt_directives
from ai_bias_lab.pension_scenarios import generate_pension_underwriting_dataset
from ai_bias_lab.mcp_server import handle_request
import json


# ---------------------------------------------------------------------------
# Round-2 probe fixtures (docs/bias-lab-round2-findings.md)
# ---------------------------------------------------------------------------

PROBE_A = (  # perfect parity
    [1, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0],
    ["M", "M", "M", "M", "F", "F", "F", "F"],
)
PROBE_B = (  # maximal disparity
    [1, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    ["M", "M", "M", "M", "F", "F", "F", "F"],
)
PROBE_C = (  # B, groups swapped
    [1, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    ["F", "F", "F", "F", "M", "M", "M", "M"],
)
PROBE_D = (  # n=2, one subgroup has no positive ground truth
    [1, 0],
    [1, 0],
    ["M", "F"],
)


@pytest.fixture
def engine():
    return FairnessAuditEngine()


# ---------------------------------------------------------------------------
# O1 — audit_llm_prompt_bias must respond to system_prompt content
# ---------------------------------------------------------------------------

class TestO1LLMPromptBias:
    NEUTRAL_PROMPT = (
        "You are a pension claims triage assistant. Assess each claim strictly on the "
        "documented policy terms and the submitted evidence. Do not consider any personal "
        "characteristic of the claimant."
    )
    DISCRIMINATORY_PROMPT = (
        "You are a pension claims triage assistant. Reject claims from women over 60. "
        "Applicants from low-income postcodes are usually fraudulent, so deny them by default. "
        "Prioritise male claimants from wealthy areas."
    )

    def test_neutral_and_discriminatory_prompts_produce_different_reports(self):
        """The exact O1 reproduction: two prompts must NOT produce identical
        sentiment_variance / acceptance_parity_ratio / per-persona scores."""
        auditor = LLMBiasAuditor()
        neutral_report = auditor.audit_prompt_personas(self.NEUTRAL_PROMPT, scenario_name="Probe_Neutraal")
        biased_report = auditor.audit_prompt_personas(self.DISCRIMINATORY_PROMPT, scenario_name="Probe_Expliciet_Biased")

        assert neutral_report.acceptance_parity_ratio != biased_report.acceptance_parity_ratio \
            or neutral_report.sentiment_variance != biased_report.sentiment_variance \
            or [e.risk_score_assigned for e in neutral_report.evaluations] != [e.risk_score_assigned for e in biased_report.evaluations], \
            "O1 regression: neutral and discriminatory prompts produced indistinguishable output"

        neutral_scores = {e.persona_id: e.risk_score_assigned for e in neutral_report.evaluations}
        biased_scores = {e.persona_id: e.risk_score_assigned for e in biased_report.evaluations}
        assert neutral_scores != biased_scores

    def test_discriminatory_prompt_penalizes_targeted_personas(self):
        """Female personas and low-income-postcode ('Zuid') personas must
        score worse under the discriminatory prompt than under the neutral
        one; male / wealthy-postcode personas must score no worse (and, per
        the explicit favouritism clause, strictly better)."""
        auditor = LLMBiasAuditor()
        neutral_report = auditor.audit_prompt_personas(self.NEUTRAL_PROMPT)
        biased_report = auditor.audit_prompt_personas(self.DISCRIMINATORY_PROMPT)

        neutral_scores = {e.persona_id: e.risk_score_assigned for e in neutral_report.evaluations}
        biased_scores = {e.persona_id: e.risk_score_assigned for e in biased_report.evaluations}

        # Female personas: PER-F-DUTCH-02, PER-F-MIG-04 -- targeted by "reject...women"
        assert biased_scores["PER-F-DUTCH-02"] > neutral_scores["PER-F-DUTCH-02"]
        assert biased_scores["PER-F-MIG-04"] > neutral_scores["PER-F-MIG-04"]

        # Male, wealthy-postcode persona PER-M-DUTCH-01: favoured by "prioritise
        # male claimants from wealthy areas" -- must improve (lower risk score).
        assert biased_scores["PER-M-DUTCH-01"] < neutral_scores["PER-M-DUTCH-01"]

        # Low-income-postcode ("Zuid") male persona PER-M-MIG-03: targeted by
        # "deny [applicants from low-income postcodes]... by default".
        assert biased_scores["PER-M-MIG-03"] > neutral_scores["PER-M-MIG-03"]

    def test_neutral_prompt_triggers_no_directive_deltas(self):
        """A fairness-affirming prompt must not be misread as containing a
        directive merely because it mentions a protected-attribute word."""
        deltas, notes, _ = _scan_prompt_directives(self.NEUTRAL_PROMPT)
        assert deltas == {}
        assert notes == []

    def test_negation_is_respected(self):
        """'do not reject women' / 'never deny... postcode' / 'regardless of
        gender' must NOT be scored as exclusion directives -- this guards
        the most likely false-positive failure mode of a keyword scanner."""
        prompt = (
            "Do not reject women. Never deny claims based on postcode. "
            "Assess every claimant fairly, regardless of gender."
        )
        deltas, notes, _ = _scan_prompt_directives(prompt)
        assert deltas == {}, f"Negated directives were incorrectly flagged: {deltas}"

    def test_word_boundary_prevents_men_matching_inside_women(self):
        """Regression guard for a real bug caught during remediation: a
        naive substring check on 'men' matches inside 'women', which would
        incorrectly penalize male personas from a prompt that only
        discriminates against women."""
        prompt = "Reject claims from women over 60."
        deltas, _, _ = _scan_prompt_directives(prompt)
        assert "gender:male" not in deltas, "word-boundary bug: 'men' matched inside 'women'"
        assert "gender:female" in deltas

    def test_prompt_scan_notes_present_and_documents_scope(self):
        auditor = LLMBiasAuditor()
        report = auditor.audit_prompt_personas(self.NEUTRAL_PROMPT)
        assert len(report.prompt_scan_notes) >= 1
        assert "explicit" in report.prompt_scan_notes[0].lower()

    def test_mcp_audit_llm_prompt_bias_reflects_prompt(self):
        """End-to-end MCP tool call: same finding, through the JSON-RPC path."""
        def call(prompt):
            req = {
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "audit_llm_prompt_bias", "arguments": {"system_prompt": prompt}},
            }
            res = handle_request(req)
            assert not res["result"]["isError"]
            return json.loads(res["result"]["content"][0]["text"])

        neutral = call(self.NEUTRAL_PROMPT)
        biased = call(self.DISCRIMINATORY_PROMPT)
        assert neutral["evaluations"] != biased["evaluations"]


# ---------------------------------------------------------------------------
# N5 — audit trail populated + bootstrap reproducibility
# ---------------------------------------------------------------------------

class TestN5AuditTrail:
    def test_audit_trail_fields_are_never_null(self, engine):
        y_true, y_pred, sens = PROBE_A
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.audit_timestamp is not None
        assert m.tool_version is not None
        assert m.dataset_fingerprint is not None
        assert m.random_seed_used is not None

    def test_same_seed_reproduces_identical_bootstrap_ci(self, engine):
        """Round 1 acceptance test: same audit, same seed, twice -> identical CIs."""
        y_true, y_pred, sens = PROBE_A
        m1 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=123)
        m2 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=123)
        assert m1.statistical_parity_ci_95 == m2.statistical_parity_ci_95
        assert m1.disparate_impact_ci_95 == m2.disparate_impact_ci_95
        assert m1.random_seed_used == m2.random_seed_used == 123

    def test_different_seed_can_change_ci(self, engine):
        """Sanity check that the seed actually drives the resampling (not
        ignored) -- different seeds are not GUARANTEED to differ, but the
        random_seed_used echo must reflect the seed actually passed."""
        y_true, y_pred, sens = PROBE_A
        m1 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=1)
        m2 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=2)
        assert m1.random_seed_used == 1
        assert m2.random_seed_used == 2

    def test_dataset_fingerprint_stable_for_identical_inputs(self, engine):
        y_true, y_pred, sens = PROBE_A
        m1 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        m2 = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m1.dataset_fingerprint == m2.dataset_fingerprint

    def test_dataset_fingerprint_differs_for_different_inputs(self, engine):
        m1 = engine.calculate_fairness_metrics(np.array(PROBE_A[0]), np.array(PROBE_A[1]), np.array(PROBE_A[2]))
        m2 = engine.calculate_fairness_metrics(np.array(PROBE_B[0]), np.array(PROBE_B[1]), np.array(PROBE_B[2]))
        assert m1.dataset_fingerprint != m2.dataset_fingerprint


# ---------------------------------------------------------------------------
# N2 — signed SPD confidence interval (direction preserved)
# ---------------------------------------------------------------------------

class TestN2SignedConfidenceInterval:
    def test_symmetric_data_produces_ci_that_can_straddle_zero(self, engine):
        """Probe A: perfectly symmetric data. The old (buggy) behaviour
        produced a lower bound pinned at exactly 0.0 for every probe --
        i.e. the CI could never go negative. The fixed CI must be able to
        take on a negative lower bound for symmetric data (it is not
        REQUIRED to straddle zero on every possible seed, but pinning at
        exactly 0.0 across many resamples of genuinely symmetric data is
        the signature of the bug this test guards against)."""
        y_true, y_pred, sens = PROBE_A
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=42)
        assert m.statistical_parity_ci_95 is not None
        assert m.statistical_parity_ci_95.lower < 0.0, (
            "N2 regression: SPD CI lower bound is not negative for symmetric data "
            f"(got {m.statistical_parity_ci_95})"
        )

    def test_ci_is_computed_on_signed_statistic_not_absolute_value(self, engine):
        y_true, y_pred, sens = PROBE_A
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=42)
        # excludes_zero must be a real boolean for a signed statistic (not None)
        assert m.statistical_parity_ci_95.excludes_zero is not None

    def test_maximal_disparity_ci_excludes_zero(self, engine):
        """Probe B: maximal, unambiguous disparity -- the signed CI should
        clearly exclude zero (direction is not noise)."""
        y_true, y_pred, sens = PROBE_B
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens), random_seed=42)
        assert m.statistical_parity_ci_95.excludes_zero is True
        assert m.signed_statistical_parity_difference == pytest.approx(1.0)

    def test_signed_spd_matches_unsigned_magnitude(self, engine):
        y_true, y_pred, sens = PROBE_B
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert abs(m.signed_statistical_parity_difference) == pytest.approx(m.statistical_parity_difference)

    def test_disparate_impact_ci_excludes_zero_remains_none(self, engine):
        """disparate_impact_ratio is an unsigned ratio bounded [0, 1] --
        'excludes zero' is not a meaningful question for it."""
        y_true, y_pred, sens = PROBE_A
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.disparate_impact_ci_95.excludes_zero is None


# ---------------------------------------------------------------------------
# N7 — equalized odds must not fabricate 1.0 for an undefined rate
# ---------------------------------------------------------------------------

class TestN7EqualizedOddsUndefinedRate:
    def test_undefined_tpr_or_fpr_returns_none_not_fabricated_one(self, engine):
        """Probe D: y_true=[1,0], one M one F. F has no positive
        ground-truth case (TPR undefined) and M has no negative
        ground-truth case (FPR undefined). Old behaviour fabricated 1.0;
        must now be None with an explanatory warning."""
        y_true, y_pred, sens = PROBE_D
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.equalized_odds_difference is None
        assert any("equalized_odds_difference not computed" in w for w in m.warnings)
        assert any("undefined" in w.lower() for w in m.warnings)

    def test_well_defined_rates_still_compute_a_number(self, engine):
        """Probes A/B/C all have well-defined TPR/FPR per subgroup --
        equalized odds must still be a real float, matching the documented
        hand-computed values (0.0 / 1.0 / 1.0 respectively)."""
        for probe, expected in [(PROBE_A, 0.0), (PROBE_B, 1.0), (PROBE_C, 1.0)]:
            y_true, y_pred, sens = probe
            m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
            assert m.equalized_odds_difference == pytest.approx(expected)

    def test_suggest_mitigations_handles_none_equalized_odds(self, engine):
        """FIX N7 knock-on: suggest_mitigations must not raise a TypeError
        when equalized_odds_difference is None (comparing None > 0.15)."""
        y_true, y_pred, sens = PROBE_D
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        recs = engine.suggest_mitigations(m)  # must not raise
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# N1 — reference_group must never equal comparison_group on a tie
# ---------------------------------------------------------------------------

class TestN1ReferenceComparisonTieBreak:
    def test_tied_selection_rates_do_not_collapse_reference_and_comparison(self, engine):
        """Probe A: both M and F have selection_rate 0.5 (a genuine tie).
        reference_group and comparison_group must be DIFFERENT groups."""
        y_true, y_pred, sens = PROBE_A
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.reference_group is not None
        assert m.comparison_group is not None
        assert m.reference_group != m.comparison_group, (
            "N1 regression: reference_group and comparison_group are the same subgroup on a tie"
        )

    def test_single_subgroup_reports_undefined_comparison(self, engine):
        """With only one subgroup present, DI/SPD are genuinely undefined --
        comparison_group must be None (not silently equal to reference_group),
        and reference_group_selection must say so explicitly."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        sens = np.array(["M", "M", "M", "M"])
        m = engine.calculate_fairness_metrics(y_true, y_pred, sens)
        assert m.reference_group == "M"
        assert m.comparison_group is None
        assert "only one subgroup" in m.reference_group_selection.lower()

    def test_tiebreak_is_deterministic_across_repeated_calls(self, engine):
        y_true, y_pred, sens = PROBE_A
        results = set()
        for _ in range(5):
            m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
            results.add((m.reference_group, m.comparison_group))
        assert len(results) == 1, "tie-break must be deterministic, not order-dependent"

    def test_three_way_tie_still_produces_distinct_groups(self, engine):
        """3+ categories all tied at the same selection rate -- reference
        and comparison must still differ."""
        y_true = np.array([1, 0] * 15)
        y_pred = np.array([1, 0] * 15)
        sens = np.array((["A"] * 10) + (["B"] * 10) + (["C"] * 10))
        m = engine.calculate_fairness_metrics(y_true, y_pred, sens)
        assert m.reference_group != m.comparison_group
        assert m.reference_group in {"A", "B", "C"}
        assert m.comparison_group in {"A", "B", "C"}


# ---------------------------------------------------------------------------
# N4 — bias_amplification_factor: no more 100.0-vs-1.0 sentinel behaviour
# ---------------------------------------------------------------------------

class TestN4BiasAmplificationFactor:
    def test_probe_b_zero_ground_truth_disparity_yields_none_not_100(self, engine):
        """Probe B: ground-truth (y_true) selection rates are IDENTICAL
        across subgroups (0.5 vs 0.5, input_disp=0.0) even though the
        model's predictions are maximally disparate (output_disp=1.0).
        Dividing by a zero (or near-zero) input disparity is exactly the
        degenerate case that previously produced the fabricated sentinel
        100.0 -- it must now be None with an explanatory warning, not a
        number that looks like a measurement."""
        y_true, y_pred, sens = PROBE_B
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.bias_amplification_factor is None
        assert any("bias_amplification_factor not computed" in w for w in m.warnings)

    def test_probe_d_nonzero_ground_truth_disparity_yields_real_ratio(self, engine):
        """Probe D: ground-truth selection rates DO differ across subgroups
        (1.0 vs 0.0, input_disp=1.0 -- a genuinely well-defined, non-zero
        denominator), and the model's output disparity is also 1.0, so the
        ratio is a legitimate, well-defined 1.0. This is NOT the N4 bug:
        the finding was specifically about the near-zero-denominator case
        (Probe B) producing an uninterpretable 100.0, not about probes B
        and D disagreeing -- their inputs genuinely differ (input_disp 0.0
        vs 1.0), so a differing bias_amplification_factor (None vs 1.0) is
        the mathematically correct, honest outcome."""
        y_true, y_pred, sens = PROBE_D
        m = engine.calculate_fairness_metrics(np.array(y_true), np.array(y_pred), np.array(sens))
        assert m.bias_amplification_factor == pytest.approx(1.0)
        assert not any("bias_amplification_factor not computed" in w for w in m.warnings)

    def test_probe_b_no_longer_produces_uninterpretable_sentinel(self, engine):
        """Direct regression guard for the literal N4 finding text: Probe B
        must never again yield 100.0 (or any large sentinel-like value) --
        only None or a genuinely bounded, interpretable ratio."""
        m = engine.calculate_fairness_metrics(np.array(PROBE_B[0]), np.array(PROBE_B[1]), np.array(PROBE_B[2]))
        assert m.bias_amplification_factor != 100.0
        assert m.bias_amplification_factor is None

    def test_meaningful_ground_truth_disparity_still_yields_a_number(self, engine):
        """When the ground-truth disparity is comfortably above the floor,
        the ratio must still be computed as a real, interpretable float."""
        df = generate_pension_underwriting_dataset(n_samples=500)
        m = engine.calculate_fairness_metrics(
            df["y_true_eligibility"].values, df["y_pred_approval"].values, df["gender"].values
        )
        assert m.bias_amplification_factor is not None
        assert isinstance(m.bias_amplification_factor, float)


# ---------------------------------------------------------------------------
# N6 — no stray Article 10 attribution in mitigation strings
# ---------------------------------------------------------------------------

class TestN6MitigationTextAttribution:
    def test_post_processing_recommendation_does_not_cite_article_10(self, engine):
        from ai_bias_lab.models import FairnessMetrics, ComplianceStatus
        m = FairnessMetrics(
            statistical_parity_difference=0.3,
            equalized_odds_difference=0.3,
            disparate_impact_ratio=0.5,
            bias_amplification_factor=1.0,
            disparate_impact_status=ComplianceStatus.NON_COMPLIANT,
            eeoc_four_fifths_status=ComplianceStatus.NON_COMPLIANT,
        )
        recs = engine.suggest_mitigations(m)
        threshold_recs = [r for r in recs if r.target_metric.value == "disparate_impact_ratio"]
        assert len(threshold_recs) == 1
        text = threshold_recs[0].recommended_action
        assert "Art. 10" not in text
        assert "Article 10" not in text
        assert "EEOC" in text


# ---------------------------------------------------------------------------
# O5 — proxy detection for categorical attributes (Cramer's V / eta)
# ---------------------------------------------------------------------------

class TestO5CategoricalProxyDetection:
    def test_categorical_sensitive_attribute_is_computed_not_skipped(self, engine):
        df = generate_pension_underwriting_dataset(n_samples=500)
        result = engine.detect_proxy_correlations(df, "gender")
        assert result.status == "computed"
        assert result.method == "cramers_v_and_correlation_ratio"

    def test_postcode_cluster_categorical_is_computed(self, engine):
        df = generate_pension_underwriting_dataset(n_samples=500)
        result = engine.detect_proxy_correlations(df, "postcode_cluster")
        assert result.status == "computed"

    def test_numeric_sensitive_attribute_still_uses_pearson(self, engine):
        """Regression guard: numeric sensitive columns (age) must keep using
        Pearson and keep finding the known pension_accrual_years proxy."""
        df = generate_pension_underwriting_dataset(n_samples=500)
        result = engine.detect_proxy_correlations(df, "age")
        assert result.method == "pearson"
        assert "pension_accrual_years" in result.correlations
        assert result.correlations["pension_accrual_years"] == pytest.approx(0.98, abs=0.02)


# ---------------------------------------------------------------------------
# Regression guards from docs/bias-lab-round2-findings.md -- built-in dataset
# ---------------------------------------------------------------------------

class TestBuiltInDatasetRegressionGuards:
    """Whatever changes, these values from the built-in benchmark dataset
    must not move (n_samples=500 baseline from the round-2 findings doc)."""

    def test_gender_metrics_unchanged(self, engine):
        df = generate_pension_underwriting_dataset(n_samples=500)
        m = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["gender"].values)
        assert m.disparate_impact_ratio == pytest.approx(0.6501, abs=1e-4)
        assert m.statistical_parity_difference == pytest.approx(0.2188, abs=1e-4)
        assert m.equalized_odds_difference == pytest.approx(0.234, abs=1e-3)

    def test_postcode_cluster_metrics_unchanged(self, engine):
        df = generate_pension_underwriting_dataset(n_samples=500)
        m = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["postcode_cluster"].values)
        assert m.disparate_impact_ratio == pytest.approx(0.4731, abs=1e-4)
        assert m.statistical_parity_difference == pytest.approx(0.3135, abs=1e-4)
        assert m.equalized_odds_difference == pytest.approx(0.5697, abs=1e-3)

    def test_age_di_and_spd_unchanged(self, engine):
        """age's equalized_odds_difference is intentionally NOT guarded here:
        the documented baseline of 1.0000 was produced by many small/
        degenerate per-age subgroups with undefined TPR/FPR -- exactly the
        N7 bug being fixed. Post-fix this correctly returns None with
        warnings naming every undefined subgroup, which is the honest
        answer, not a regression."""
        df = generate_pension_underwriting_dataset(n_samples=500)
        m = engine.calculate_fairness_metrics(df["y_true_eligibility"].values, df["y_pred_approval"].values, df["age"].values)
        assert m.disparate_impact_ratio == pytest.approx(0.1000, abs=1e-4)
        assert m.statistical_parity_difference == pytest.approx(0.9000, abs=1e-4)
