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
    """FIX F2: bootstrap 95% confidence interval for a point estimate."""
    lower: float
    upper: float
    method: str = "bootstrap_percentile"
    n_resamples: int


class FairnessMetrics(BaseModel):
    statistical_parity_difference: float = Field(description="Difference in selection rates between groups (ideal: 0.0)")
    equalized_odds_difference: float = Field(description="Difference in TPR/FPR between groups (ideal: 0.0)")
    disparate_impact_ratio: float = Field(description="Ratio of selection rates (EEOC 80% / four-fifths rule, ideal: >= 0.80)")
    bias_amplification_factor: float = Field(description="Ratio of output disparity to input data disparity (ideal: <= 1.0)")
    disparate_impact_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    eu_ai_act_art10_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    # FIX F2 (additive fields, no existing field removed/renamed):
    subgroup_counts: Dict[str, SubgroupCount] = Field(default_factory=dict, description="Per-subgroup n, n_favorable, and selection_rate")
    disparate_impact_ci_95: Optional[ConfidenceInterval] = Field(default=None, description="Bootstrap 95% CI for disparate_impact_ratio")
    statistical_parity_ci_95: Optional[ConfidenceInterval] = Field(default=None, description="Bootstrap 95% CI for statistical_parity_difference")
    warnings: List[str] = Field(default_factory=list, description="Plain-language caveats: small subgroup, wide CI, single-category attribute, etc.")


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
