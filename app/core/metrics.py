"""Fairness Metrics Engine using Fairlearn.

Calculates statistical parity, equalized odds, disparate impact, and custom metrics.

2026-09-19 webapp-parity remediation (mirrors F2/F3/F8 fixes already applied
to the MCP server on 2026-09-18): a metric computed on too few rows per
subgroup, or a number that was never actually derived from the data, must
never be presented with the same confident formatting as a real, well-powered
result. See MIN_SUBGROUP_SAMPLE_SIZE in .utils.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference as statistical_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
# NOTE: fairlearn 0.14.0 renamed statistical_parity_difference ->
# demographic_parity_difference (same fairlearn API-rename issue already hit
# and fixed in the MCP server on 2026-09-18). Aliased on import so the rest
# of this module and its public dict key names stay unchanged for callers.
from sklearn.metrics import accuracy_score
from .utils import set_seed, MIN_SUBGROUP_SAMPLE_SIZE

def calculate_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Calculate comprehensive fairness metrics.

    Returns dict with:
    - statistical_parity_difference
    - equalized_odds_difference
    - disparate_impact_ratio
    - bias_amplification_factor
    - subgroup_counts (F2-equivalent: sample size per sensitive group)
    - insufficient_data (bool) + warnings (list[str]) when any subgroup n < 30
    - mitigation_effectiveness_pct is INTENTIONALLY OMITTED (F3-equivalent):
      it was previously a random placeholder (np.random.uniform(35, 72)) with
      no relationship to any actual mitigation being applied, and has been
      removed rather than shown as if it were a real, computed number.
    """
    set_seed(random_seed)

    metrics = {}
    sens_col = sensitive_features.columns[0]
    sens = sensitive_features[sens_col]

    # F2-equivalent: expose subgroup sample sizes and flag statistically
    # unreliable results BEFORE reporting any ratio/verdict, so a clean-looking
    # number backed by a handful of rows can never look identical to one
    # backed by a real sample.
    subgroup_counts = sens.value_counts().to_dict()
    subgroup_counts = {str(k): int(v) for k, v in subgroup_counts.items()}
    metrics["subgroup_counts"] = subgroup_counts
    insufficient_data = any(n < MIN_SUBGROUP_SAMPLE_SIZE for n in subgroup_counts.values())
    metrics["insufficient_data"] = insufficient_data
    warnings = []
    if insufficient_data:
        small = {k: v for k, v in subgroup_counts.items() if v < MIN_SUBGROUP_SAMPLE_SIZE}
        warnings.append(
            f"Insufficient data: subgroup(s) {small} below minimum sample size "
            f"({MIN_SUBGROUP_SAMPLE_SIZE}). Metrics below are still computed for "
            f"transparency but must NOT be read as a reliable pass/fail verdict."
        )

    # Statistical Parity Difference
    spd = statistical_parity_difference(y_true, y_pred, sensitive_features=sens)
    metrics["statistical_parity_difference"] = round(float(spd), 4)

    # Equalized Odds Difference
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sens)
    metrics["equalized_odds_difference"] = round(float(eod), 4)

    # Disparate Impact Ratio (manual calculation)
    pos_rate = pd.DataFrame({"y": y_pred, "s": sens}).groupby("s")["y"].mean()
    if len(pos_rate) >= 2:
        di = pos_rate.min() / max(pos_rate.max(), 1e-6)
    else:
        di = 1.0
        warnings.append("Only one subgroup present in sensitive_features; disparate_impact_ratio defaulted to 1.0 (not computed).")
    metrics["disparate_impact_ratio"] = round(float(di), 4)

    # Bias Amplification (illustrative demo formula, NOT a validated academic
    # metric): compares |mean-0.5| of true labels vs predictions. Explicitly
    # disclosed as a heuristic rather than left unlabeled (F7-equivalent).
    input_disparity = abs(y_true.mean() - 0.5) * 2
    output_disparity = abs(y_pred.mean() - 0.5) * 2
    amp = output_disparity / max(input_disparity, 0.01)
    metrics["bias_amplification_factor"] = round(float(amp), 3)
    metrics["bias_amplification_factor_basis"] = (
        "Illustrative heuristic: ratio of output-vs-input deviation from a 50% "
        "base rate. Not a peer-reviewed fairness metric — for demo/training use only."
    )

    metrics["warnings"] = warnings
    return metrics