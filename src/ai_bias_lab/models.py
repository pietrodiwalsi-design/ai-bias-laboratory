"""
Data models for AI Bias Laboratory & FastMCP Server.
"""

from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ProtectedAttribute(str, Enum):
    GENDER = "gender"
    AGE = "age"
    POSTCODE_PROXY = "postcode_proxy"
    ETHNICITY_PROXY = "ethnicity_proxy"
    FAMILY_STATUS = "family_status"
    HEALTH_STATUS = "health_status"


class FairnessMetricType(str, Enum):
    STATISTICAL_PARITY = "statistical_parity_difference"
    EQUALIZED_ODDS = "equalized_odds_difference"
    DISPARATE_IMPACT = "disparate_impact_ratio"
    BIAS_AMPLIFICATION = "bias_amplification_factor"


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    NON_COMPLIANT = "NON_COMPLIANT"
    # FIX F2 (2026-09-18 remediation brief, P0): a COMPLIANT verdict on n=8
    # was indistinguishable in shape from one on n=50,000 -- any consumer,
    # human or agent, would treat them identically. This status takes
    # precedence over COMPLIANT / WARNING / NON_COMPLIANT whenever any
    # compared subgroup has fewer than the minimum sample size.
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SubgroupCount(BaseModel):
    """FIX F2 / F6: per-subgroup sample size and selection-rate evidence, so
    a verdict can never be read without also seeing how much data it rests
    on."""
    n: int
    n_favorable: int
    selection_rate: float


class ConfidenceInterval(BaseModel):
    """FIX F2: bootstrap 95% confidence interval for a point estimate.

    FIX N2 (round 2 verification, 2026-09-22): for a SIGNED statistic
    (e.g. signed statistical parity difference = reference_group's
    selection rate minus comparison_group's selection rate), `lower` may
    be negative -- the interval is on the signed statistic, not on its
    absolute value, so direction is preserved and the interval can
    straddle zero. `excludes_zero` is a convenience flag: True means the
    interval does not contain 0, i.e. the disparity's direction is not
    plausibly attributable to sampling noise at the 95% level (a
    disparity-in-either-direction cannot be ruled out when this is False).
    Unsigned statistics (e.g. disparate_impact_ratio, which is a ratio
    bounded in [0, 1] with no direction to lose) leave `excludes_zero` as
    None -- "excludes zero" is not a meaningful question for a ratio whose
    ideal value is 1.0, not 0.
    """
    lower: float
    upper: float
    method: str = "bootstrap_percentile"
    n_resamples: int
    excludes_zero: Optional[bool] = Field(default=None, description="For signed statistics only: True if the 95% CI does not contain 0 (disparity direction is unlikely to be sampling noise). None for unsigned/ratio statistics where this question does not apply.")


class Art10DocumentationStatus(BaseModel):
    """FIX F5 (2026-09-18 remediation brief, P1): Article 10 of Regulation
    (EU) 2024/1689 covers data and data governance for high-risk AI
    systems -- it does NOT specify a numeric fairness threshold. This is a
    documentation-coverage CHECKLIST, not a computed ratio, and every item
    defaults to 'not_assessed'. It must be supplied by the caller
    (e.g. via a future input parameter or manual review) -- it is never
    inferred from a fairness metric, because a metric cannot attest to
    whether governance documentation exists.
    """
    data_governance_practices_documented: str = "not_assessed"
    relevant_design_choices_recorded: str = "not_assessed"
    data_provenance_and_collection_documented: str = "not_assessed"
    examination_for_biases_performed: str = "not_assessed"
    gaps_or_shortcomings_identified: str = "not_assessed"
    mitigation_measures_in_place: str = "not_assessed"


class FairnessMetrics(BaseModel):
    statistical_parity_difference: float = Field(description="Difference in selection rates between groups (ideal: 0.0)")
    # FIX N7 (round 2 verification, 2026-09-22): equalized_odds_difference
    # is nullable. Fairlearn's equalized_odds_difference() silently treats
    # an undefined per-subgroup TPR or FPR (0/0, e.g. a subgroup with no
    # positive -- or no negative -- ground-truth cases) as 0, which makes
    # the reported difference look artificially maximal (observed: 1.0 on
    # a 2-row probe where one subgroup had zero positive labels). None
    # here means "not computable for this input", never "no disparity" --
    # the accompanying `warnings` entry names the affected subgroup(s) and
    # which rate (TPR/FPR) was undefined.
    equalized_odds_difference: Optional[float] = Field(default=None, description="Difference in TPR/FPR between groups (ideal: 0.0). null when a subgroup has an undefined TPR or FPR (no positive, or no negative, ground-truth cases) -- see warnings.")
    disparate_impact_ratio: float = Field(description="Ratio of selection rates (EEOC 80% / four-fifths rule, ideal: >= 0.80)")
    # FIX N4 (round 2 verification, 2026-09-22): bias_amplification_factor
    # is nullable and documented. Formula: abs(max_selection_rate -
    # min_selection_rate) across PREDICTED outcomes, divided by the SAME
    # quantity across GROUND-TRUTH labels. When the ground-truth disparity
    # is at or near zero (input_disparity < BIAS_AMPLIFICATION_MIN_INPUT_DISPARITY,
    # currently 0.01) the ratio is mathematically explosive/undefined --
    # e.g. output disparity 1.0 over input disparity 0.01 previously
    # produced a sentinel-like 100.0 that was indistinguishable from a
    # genuine measurement and differed by 100x from an n=2 case (1.0) with
    # identical headline SPD/EOD/DI values. Now null + a warning in that
    # regime; otherwise the ratio is reported as before (0 = model
    # perfectly equalizes an unequal ground truth; 1.0 = model reproduces
    # ground-truth disparity unchanged; >1.0 = model WIDENS a disparity).
    bias_amplification_factor: Optional[float] = Field(default=None, description="Ratio of output (predicted) selection-rate disparity to input (ground-truth) selection-rate disparity. null when the ground-truth disparity is too close to zero for the ratio to be meaningful -- see warnings. Descriptive only, not a compliance threshold.")
    disparate_impact_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    # FIX F5 (2026-09-18 remediation brief, P1): the four-fifths / 80% rule
    # originates in the US EEOC Uniform Guidelines on Employee Selection
    # Procedures (29 CFR 1607.4(D)) -- NOT EU AI Act Article 10, which
    # covers data governance and does not define a numeric threshold.
    # Renamed the computed field accordingly and added threshold_source.
    # `eu_ai_act_art10_status` is KEPT as a deprecated alias for one release
    # (see model_validator below) so nothing breaks; it mirrors
    # eeoc_four_fifths_status and will be removed in a future release.
    eeoc_four_fifths_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    threshold_source: str = Field(
        default="US EEOC Uniform Guidelines, 29 CFR 1607.4(D) — four-fifths rule",
        description="Regulatory/guidance source of the 80% threshold used for eeoc_four_fifths_status",
    )
    art10_documentation_status: Art10DocumentationStatus = Field(
        default_factory=Art10DocumentationStatus,
        description="EU AI Act Article 10 data-governance documentation checklist. Defaults to not_assessed for every item -- never inferred from a metric.",
    )
    eu_ai_act_art10_status: ComplianceStatus = Field(
        default=ComplianceStatus.COMPLIANT,
        description="DEPRECATED alias for eeoc_four_fifths_status, kept for one release for backward compatibility. This field name was always a mislabel: it reported the US EEOC four-fifths rule, not an EU AI Act Article 10 verdict. Use eeoc_four_fifths_status and art10_documentation_status instead.",
    )
    # FIX F2 (additive fields, no existing field removed/renamed):
    subgroup_counts: Dict[str, SubgroupCount] = Field(default_factory=dict, description="Per-subgroup n, n_favorable, and selection_rate")
    disparate_impact_ci_95: Optional[ConfidenceInterval] = Field(default=None, description="Bootstrap 95% CI for disparate_impact_ratio")
    statistical_parity_ci_95: Optional[ConfidenceInterval] = Field(default=None, description="Bootstrap 95% CI for statistical_parity_difference")
    warnings: List[str] = Field(default_factory=list, description="Plain-language caveats: small subgroup, wide CI, single-category attribute, etc.")
    # FIX F6 (2026-09-18 remediation brief, P1): disparate_impact_ratio and
    # statistical_parity_difference are PAIRWISE comparisons (min/max
    # selection rate). When sensitive_column has >2 categories, the
    # response never stated which two subgroups were actually compared or
    # how they were picked -- silently defaulted to argmax/argmin.
    # FIX N1 (round 2 verification, 2026-09-22): on a tie in selection rate
    # (including the degenerate case of a single subgroup), the naive
    # idxmax/idxmin pick would pick the SAME subgroup as both reference and
    # comparison, silently comparing a group to itself (DI = 1.0, SPD = 0.0
    # by construction, whether or not that is the true answer). The
    # documented deterministic tiebreak is: (1) if the top-rate group is not
    # unique, prefer the tied group with the largest n, then lexically
    # smallest label; (2) comparison_group is chosen from the REMAINING
    # subgroups only (reference_group excluded), with the same n/lexical
    # tiebreak on its own ties. If there is genuinely only one subgroup,
    # reference_group_selection states so explicitly and comparison_group
    # is null -- DI/SPD are undefined with one group, not silently 1.0/0.0.
    reference_group: Optional[str] = Field(default=None, description="Subgroup with the HIGHEST selection rate in this comparison (the implicit baseline). Ties broken by largest n, then lexical order -- see reference_group_selection.")
    comparison_group: Optional[str] = Field(default=None, description="Subgroup with the LOWEST selection rate among the remaining subgroups (reference_group excluded), so it is never equal to reference_group when 2+ subgroups exist.")
    reference_group_selection: Optional[str] = Field(default=None, description="Rule used to pick reference_group, e.g. 'highest selection rate', 'highest selection rate (tie broken by largest n)', 'caller-specified via reference_group parameter', or 'undefined: only one subgroup present'")
    # FIX N2 (round 2 verification, 2026-09-22): signed statistical parity,
    # alongside the existing (unsigned, magnitude-only)
    # statistical_parity_difference. signed_statistical_parity_difference =
    # reference_group's selection rate minus comparison_group's selection
    # rate, so its SIGN carries direction (positive: reference_group is
    # favoured over comparison_group at the point estimate -- this is
    # always >= 0 by construction of how reference/comparison are chosen,
    # but statistical_parity_ci_95 is bootstrapped on this SIGNED quantity,
    # so the interval legitimately straddles zero when resamples
    # occasionally favour the other group). Use this signed value, not
    # abs(), to read which group's CI bound is which direction.
    signed_statistical_parity_difference: Optional[float] = Field(default=None, description="reference_group's selection rate minus comparison_group's selection rate (signed). The magnitude equals statistical_parity_difference; the sign gives direction. statistical_parity_ci_95 is bootstrapped on this signed statistic.")
    # FIX F8 (2026-09-18 remediation brief, P2): audit trail. Without these,
    # no output could be tied to a specific data version or reproduced
    # after the fact.
    audit_timestamp: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of when this computation ran")
    tool_version: Optional[str] = Field(default=None, description="ai-bias-laboratory server version string")
    dataset_fingerprint: Optional[str] = Field(default=None, description="SHA-256 hash of the canonicalised input records (stable for the built-in dataset)")
    random_seed_used: Optional[int] = Field(default=None, description="The random seed actually used for bootstrap CI resampling, echoed back for reproducibility")

    @model_validator(mode="after")
    def _sync_deprecated_art10_alias(self):
        # Keep the deprecated eu_ai_act_art10_status field mirroring the
        # correctly-named eeoc_four_fifths_status, so existing callers who
        # read the old field name during the deprecation window still see
        # the right verdict.
        self.eu_ai_act_art10_status = self.eeoc_four_fifths_status
        return self


class DatasetAuditInput(BaseModel):
    dataset_name: str
    target_column: str
    sensitive_column: str
    data_records: List[Dict[str, Any]]
    favorable_outcome: Any = 1


class ModelFairnessInput(BaseModel):
    model_name: str
    target_domain: str = "life_and_pensions_underwriting"
    y_true: List[int]
    y_pred: List[int]
    sensitive_features: List[Union[str, int]]
    sensitive_feature_name: str = "gender"


class MitigationType(str, Enum):
    PRE_PROCESSING_REWEIGHT = "pre_processing_sample_reweighting"
    IN_PROCESSING_EXPONENTIATED = "in_processing_exponentiated_gradient"
    POST_PROCESSING_THRESHOLD = "post_processing_threshold_optimizer"


class MitigationRecommendation(BaseModel):
    mitigation_type: MitigationType
    target_metric: FairnessMetricType
    current_value: float
    target_value: float
    recommended_action: str
    # FIX F3 (2026-09-18 remediation brief, P1): expected_disparate_impact_improvement
    # was removed. Evidence: across 3 audits with very different disparate
    # impact (0.6501, 0.4731, 0.1000), pre_processing/in_processing
    # 'expected' values were fixed constants (0.15 / 0.12) regardless of the
    # actual data -- an unquantified recommendation is more honest than a
    # fake quantity. Chose option (a) from the brief (remove entirely)
    # rather than (b) rename-with-basis-field, since a heuristic constant
    # dressed up with a 'basis' label still invites being read as a number
    # that means something dataset-specific.
    heuristic_target_gap: float = Field(description="target_value minus current_value; NOT a data-derived estimate, purely the numeric gap to the stated target")
    basis: str = Field(default="fixed heuristic threshold gap, not estimated from this dataset", description="Explicit disclosure that this recommendation is templated, not modelled")


class LLMPromptAuditInput(BaseModel):
    system_prompt: str
    test_personas: List[Dict[str, str]]
    test_scenario: str = "pension_disability_claim_triage"


class PersonaEvaluation(BaseModel):
    persona_id: str
    demographic_attributes: Dict[str, str]
    decision_sentiment: str
    acceptance_recommended: bool
    risk_score_assigned: float
    potential_bias_flags: List[str] = Field(default_factory=list)


class LLMBiasAuditReport(BaseModel):
    scenario_name: str
    total_personas_tested: int
    sentiment_variance: float
    acceptance_parity_ratio: float
    detected_proxy_biases: List[str]
    compliance_verdict: ComplianceStatus
    evaluations: List[PersonaEvaluation]
    # FIX O1 (round 1/2 verification, 2026-09-22, most serious finding):
    # states the documented scope of the deterministic system_prompt text
    # scan that now drives per-persona risk_score adjustments -- what kinds
    # of explicit directives it detects, and that a clean scan is not proof
    # of an unbiased prompt, only that no directive of the checked kinds
    # was found. See ai_bias_lab.llm_bias._scan_prompt_directives.
    prompt_scan_notes: List[str] = Field(default_factory=list, description="Documented scope/limitations of the system_prompt directive scan that fed into these personas' risk scores.")


class ProxyCorrelationResult(BaseModel):
    """FIX F4 (2026-09-18 remediation brief, P1): replaces the bare
    Dict[str, float] which conflated 'no proxy found' with 'proxy detection
    was never run' -- both rendered as an empty {}. Categorical sensitive
    attributes (gender, postcode_cluster) were silently skipped by the old
    Pearson-only implementation and looked identical to a genuinely clean
    numeric attribute with no correlated proxies."""
    status: str = Field(description="'computed' or 'not_computed'")
    reason: Optional[str] = Field(default=None, description="Why computation was skipped, when status='not_computed'")
    correlations: Dict[str, float] = Field(default_factory=dict, description="feature -> association strength, only populated when status='computed'")
    method: Optional[str] = Field(default=None, description="'pearson' (numeric) or 'cramers_v' / 'mutual_information' (categorical)")


class BiasAuditReport(BaseModel):
    audit_id: str
    system_name: str
    domain: str
    evaluated_at: str
    metrics: FairnessMetrics
    # FIX F4: carries the full computed/not_computed status + method, not
    # just a bare (and ambiguous) correlations dict.
    detected_proxy_correlations: ProxyCorrelationResult
    regulatory_verdicts: Dict[str, str] # e.g. {"EU_AI_Act_Art_10": "PASS", "WGBU_Equal_Treatment": "PASS"}
    mitigation_recommendations: List[MitigationRecommendation]
    executive_summary: str
